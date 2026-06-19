// 1. include + 기본 타입 정의 
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <utility>

#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>
#include <goast/external/vtkIO.h>
#include <SpookyTPE/FastMultipoleEnergy.h>

#include "BoundaryUtils.h"

#include "ScaryTPE/TangentPointEnergy.h"
#include "ScaryTPE/BoundaryCurveTangentPointEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

#include "Optimization/LineSearchNewtonCG.h"
#include "Optimization/Newton.h"

#include "MeshIO.h"
#include "BarycenterPathEnergy.h"

using VectorType = DefaultConfigurator::VectorType;
using MatrixType = DefaultConfigurator::SparseMatrixType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;

using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> >;

// 2. Solver용 선형 연산자 
// 2-1. ShermanMorrisonOperator 
template<typename ConfiguratorType>
class ShermanMorrisonOperator final : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;


  std::unique_ptr<LinearOperator<ConfiguratorType>> m_Ainv;

  VectorType m_u,m_v;
  VectorType m_Ainv_u;
  RealType m_denominator;
  const int m_dim;

public:
  explicit ShermanMorrisonOperator( std::unique_ptr<LinearOperator<ConfiguratorType>> &&Ainv,
                                    const VectorType &u, const VectorType &v,
                                    const std::vector<int> &fixedVariables ) : m_u( u ), m_v( v ),
                                                                               m_Ainv( std::move( Ainv ) ),
                                                                               m_dim( m_Ainv->rows() ) {
    // Assemble system matrix
    applyMaskToVector( fixedVariables, m_v );
    applyMaskToVector( fixedVariables, m_u );

    m_Ainv_u = ( *m_Ainv )( m_u );

    m_denominator = 1 + m_v.dot( m_Ainv_u );
  }


  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( Arg.size());
    Dest = ( *m_Ainv )( Arg );

    Dest -= m_Ainv_u * m_v.dot( Dest ) / m_denominator;
  }

  int rows() const override {
    return m_dim;
  }

  int cols() const override {
    return m_dim;
  }
};

// 2-2. InverseMatrixOperator 
template<typename ConfiguratorType>
class InverseMatrixOperator final : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;



  Eigen::UmfPackLU<SparseMatrixType> m_Solver;
  SparseMatrixType m_localA;
  const int m_dim;

public:
  explicit InverseMatrixOperator( SparseMatrixType &&A, const std::vector<int> &fixedVariables )
          : m_dim( A.rows()) {
    assert( A.rows() == A.cols() && "InverseMatrixOperator: Operator has to be quadratic" );
    m_localA.swap(A);

    // Assemble system matrix
    applyMaskToMatrix( fixedVariables, m_localA );

    // Prepare solver
    m_Solver.compute( m_localA );
  }


  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest = m_Solver.solve( Arg );
  }

  int rows() const override {
    return m_dim;
  }

  int cols() const override {
    return m_dim;
  }
};

// 3. Energy 정의 
// 3-1. DifferenceExp2Energy : exponential map 조건 유지 
template<typename ConfiguratorType>
class DifferenceExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;

  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape0;
  const VectorType &_shape1;
  VectorType _constVecPart;
  RealType _constRealPart;
  int _numDofs;

public:
  DifferenceExp2Energy( const BaseOp<VectorType, RealType> &V,
                        const BaseOp<VectorType, VectorType> &DV,
                        const VectorType &shape0,
                        const VectorType &shape1 ) :
          m_V( V ), m_DV( DV ), _shape0( shape0 ), _shape1( shape1 ), _numDofs( shape0.size()) {

    _constVecPart.resize( _numDofs );
    m_DV.apply( _shape1, _constVecPart );

    _constRealPart = 4 * m_V( _shape1 ) - 2 * m_V( _shape0 );
  }

  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != _numDofs )
      throw std::length_error( "Exp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != _numDofs )
      Dest.resize( _numDofs );

    // add constant partial
    Dest = _constVecPart * ( _constRealPart - 2 * m_V( shape2 ));
  }
};

// 3-2. BarycenterExp2Energy : 전체 이동 제거 (translation invariance)
template<typename ConfiguratorType>
class BarycenterExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;

  const VectorType &m_shape0;
  const VectorType &m_shape1;
  VecType m_bary0, m_bary1;
  int m_numVertices;

  std::vector<int> m_barycenterIndices;

public:
  BarycenterExp2Energy( const VectorType &shape0,
                        const VectorType &shape1,
                        const std::vector<int> &barycenterIndices ) : m_shape0( shape0 ), m_shape1( shape1 ),
                                                                      m_numVertices( shape0.size() / 3 ),
                                                                      m_barycenterIndices( barycenterIndices ) {
//    m_barycenterIndices.resize( m_numVertices, 0 );
//    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0 );

    VecType p;
    for ( int i: m_barycenterIndices ) {
      getXYZCoord( shape0, p, i );
      m_bary0 += p;
      getXYZCoord( shape1, p, i );
      m_bary1 += p;
    }
  }


  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != 3 * m_numVertices )
      throw std::length_error( "BarycenterExp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != 3 * m_numVertices )
      Dest.resize( 3 * m_numVertices );
    Dest.setZero();

    VecType p, bary2;
    for ( int i: m_barycenterIndices ) {
      getXYZCoord( shape2, p, i );
      bary2 += p;
    }

    VecType diff0 = m_bary1 - m_bary0;
    VecType diff1 = bary2 - m_bary1;
    for ( const int i: m_barycenterIndices ) {
      for ( int d: { 0, 1, 2 } ) {
        Dest[d * m_numVertices + i] += diff0[d];
        Dest[d * m_numVertices + i] -= diff1[d];
      }
    }

    Dest *= 4. / m_barycenterIndices.size();
  }
};

// 3-3. DirichletExp2Energy : 특정 vertex 고정 
template<typename ConfiguratorType>
class DirichletExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;

  const VectorType &m_shape0;
  const VectorType &m_shape1;
  VecType m_bary0, m_bary1;
  int m_numVertices;

  const std::vector<int> &m_constrainedVertices;

public:
  DirichletExp2Energy( const VectorType &shape0,
                       const VectorType &shape1,
                       const std::vector<int> &constrainedVertices ) : m_shape0( shape0 ), m_shape1( shape1 ),
                                                                       m_numVertices( shape0.size() / 3 ),
                                                                       m_constrainedVertices( constrainedVertices ) {
//    m_barycenterIndices.resize( m_numVertices, 0 );
//    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0 );

  }


  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != 3 * m_numVertices )
      throw std::length_error( "BarycenterExp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != 3 * m_numVertices )
      Dest.resize( 3 * m_numVertices );
    Dest.setZero();

    Dest.setZero();

    for ( int i: m_constrainedVertices ) {
      VecType p, q, r, displacement;

      // 0
      getXYZCoord( m_shape0, p, i );
      getXYZCoord( m_shape1, q, i );
      getXYZCoord( shape2, r, i );
      displacement = (q - p) - (r - q);

      for ( int j = 0; j < 3; j++ )
        Dest[j * m_numVertices + i] += displacement[j];
    }
  }
};

// 4. Gradient / Hessian 정의 
// 4-1. InverseCombinedExp2Gradient : Hessian inverse 구성 
template<typename ConfiguratorType>
class InverseCombinedExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape1;
  const VectorType _constVecPart;
  const RealType m_elasticWeight, m_tpeWeight,m_barycenterWeight,m_dirichletWeight;
  const std::vector<int> &m_fixedVariables;
  const std::vector<int> &m_barycenterIndices;
  const int m_numVertices;
  VectorType m_oneX, m_oneY, m_oneZ;

public:
  InverseCombinedExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const BaseOp<VectorType, RealType> &V,
                               const BaseOp<VectorType, VectorType> &DV,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables,
                               const std::vector<int> &barycenterIndices,
                               RealType elasticWeight = 1.,
                               RealType tpeWeight = 1.,
                               RealType dirichletWeight = 0.,
                               RealType barycenterWeight = 1. ) : m_W( W ), m_V( V ), m_DV( DV ), _shape1( shape1 ),
                                                                  _constVecPart( -2. * tpeWeight * DV( shape1 ) ),
                                                                  m_elasticWeight( elasticWeight ),
                                                                  m_tpeWeight( tpeWeight ),
                                                                  m_barycenterWeight( barycenterWeight ),
                                                                  m_dirichletWeight( dirichletWeight ),
                                                                  m_fixedVariables( fixedVariables ),
                                                                  m_barycenterIndices( barycenterIndices ),
                                                                  m_numVertices( _shape1.size() / 3 ),
                                                                  m_oneX( 3 * m_numVertices ),
                                                                  m_oneY( 3 * m_numVertices ),
                                                                  m_oneZ( 3 * m_numVertices ) {
    m_oneX.setZero();
    m_oneY.setZero();
    m_oneZ.setZero();

    for ( int i: m_barycenterIndices ) {
      m_oneX[i] = 1.;
      m_oneY[m_numVertices + i] = 1.;
      m_oneZ[2 * m_numVertices + i] = 1.;
    }
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );
    A *= m_elasticWeight;

    std::vector<int> localFixed;

    if (m_fixedVariables.empty()) {
      for (int i = 0; i < A.rows();i++) {
        A.coeffRef(i,i) += 1.e-6;
      }
    }
    else {
      if (m_dirichletWeight == 0.) {
        localFixed = m_fixedVariables;
      }
      else {
        for (int i : m_fixedVariables) {
          A.coeffRef(i,i) -= m_dirichletWeight;
        }
      }
    }

    VectorType changingPart = m_DV( Point );

    // Inverse of elastic part
    auto Ainv = std::make_unique<InverseMatrixOperator<ConfiguratorType>>( std::move( A ), localFixed );

    // Sherman-Morrison for TPE
    auto SMtpe = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( Ainv ), _constVecPart, changingPart, localFixed );

    // Sherman-Morrison for each component of barycenter path energy
    auto SMbX = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMtpe ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneX, m_oneX, localFixed );
    auto SMbY = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMbX ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneY, m_oneY, localFixed );
    return std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMbY ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneZ, m_oneZ, localFixed );
  }

};

// 4-2. CombinedExp2Gradient : 실제 Hessian 구성 
template<typename ConfiguratorType>
class CombinedExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape1;
  VectorType _constVecPart;
  const RealType m_elasticWeight, m_tpeWeight,m_barycenterWeight,m_dirichletWeight;
  const std::vector<int> &m_fixedVariables;
  const std::vector<int> &m_barycenterIndices;
  const int m_numVertices;
public:
  CombinedExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const BaseOp<VectorType, RealType> &V,
                               const BaseOp<VectorType, VectorType> &DV,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables,
                               const std::vector<int> &barycenterIndices,
                               RealType elasticWeight = 1.,
                               RealType tpeWeight = 1.,
                              RealType dirichletWeight = 0.,
                               RealType barycenterWeight = 1. ) :
          m_W( W ), m_V( V ), m_DV( DV ), _shape1( shape1 ),
          _constVecPart( -2. * tpeWeight * DV( shape1 )), m_elasticWeight( elasticWeight ), m_tpeWeight( tpeWeight ), m_barycenterWeight(barycenterWeight),m_dirichletWeight(dirichletWeight),
          m_fixedVariables( fixedVariables ),
          m_barycenterIndices( barycenterIndices ), m_numVertices(_shape1.size() / 3)  {
    applyMaskToVector( m_fixedVariables, _constVecPart );
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );
    A *= m_elasticWeight;



    VectorType changingPart = m_DV( Point );

    std::vector<int> localFixed;

    if (m_fixedVariables.empty()) {
      for (int i = 0; i < A.rows();i++) {
        A.coeffRef(i,i) += 1.e-6;
      }
    }
    else {
      if (m_dirichletWeight == 0.) {
        applyMaskToMatrix( m_fixedVariables, A );
        applyMaskToVector( m_fixedVariables, changingPart );
      }
      else {
        for (int i : m_fixedVariables) {
          A.coeffRef(i,i) -= m_dirichletWeight;
        }
      }
    }


    std::unique_ptr<LinearOperator<ConfiguratorType>> energyOp = std::make_unique<MatrixOperator<ConfiguratorType>>( std::move(A) );
    std::unique_ptr<LinearOperator<ConfiguratorType>> diffOp = std::make_unique<RankOneOperator<ConfiguratorType>>( _constVecPart, changingPart );
    std::unique_ptr<LinearOperator<ConfiguratorType>> baryOp = std::make_unique<BarycenterHessianOperator<ConfiguratorType>>( m_numVertices, m_barycenterIndices, -m_barycenterWeight * 4. / m_barycenterIndices.size() );

    VectorType Weights(3);
    Weights << 1,1,1;
    return std::make_unique<LinearlyCombinedOperator<ConfiguratorType>>( Weights, std::move( energyOp ), std::move( diffOp ), std::move( baryOp ));
  }

};

// 4-3. OperatorExp2Gradient : elastic-only gradient 
template<typename ConfiguratorType>
class OperatorExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &_shape1;
  const std::vector<int> &m_fixedVariables;

public:
  OperatorExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                        const VectorType &shape0,
                        const VectorType &shape1,
                        const std::vector<int> &fixedVariables) :
          m_W( W ), _shape1( shape1 ), m_fixedVariables( fixedVariables ) {
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );

    applyMaskToMatrix( m_fixedVariables, A );

    return std::make_unique<MatrixOperator<ConfiguratorType>>( std::move(A) );
  }

};

// 4-4. InverseExp2Gradient : elastic-only inverse 
template<typename ConfiguratorType>
class InverseExp2Gradient : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &_shape1;
  const std::vector<int> &m_fixedVariables;

public:
  InverseExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables ) :
          m_W( W ), _shape1( shape1 ),
          m_fixedVariables( fixedVariables ) {}


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );

    return std::make_unique<InverseMatrixOperator<ConfiguratorType>>( std::move( A ), m_fixedVariables );
  }

};

// 5. main 함수 
int main( int argc, char *argv[] ) {
  /* [region Config] */
  // Repulsive Energy 계산의 효율성과 정밀도 옵션 
  enum TPEType {
    SPOOKY,         // 멀리 떨어진 점들을 묶어서 근사. 빠르지만 SCARY에 비해서 부정확함. 대규모 mesh에 필수. 
    SCARY           // 모든 점 쌍을 고려하여 계산. 정확하지만 계산량이 많아서 느림. 
  };

  // YAML 파일에 정의된 string을 TPEType으로 대응하기 위한 map 
  std::unordered_map<std::string, TPEType> const TPETypeTable = {
    { "Spooky", TPEType::SPOOKY },
    { "Scary",    TPEType::SCARY },
  };

  // Config struct 
  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    std::string startFile;
    std::string secondFile;
    std::string initFile;
    std::vector<int> dirichletVertices;     // 위치를 고정해놓은 vertices 

    int numSteps = 6;

    struct {
      int maxNumIterations = 50000;         // Newton method 최대 반복 횟수 
      RealType minStepsize = 1.e-12;        
      RealType maxStepsize = 10.;           // step size가 너무 크면 수렴 X. 너무 작으면 계산량 증가 
    } Optimization;

    struct {
      int alpha = 6;
      int beta = 12;
      TPEType Type = SCARY;
      RealType innerWeight = 1.;
      RealType theta = 0.5;
      RealType thetaNear = 10.;
      bool useAdaptivity = true;
      bool useObstacleAdaptivity = true;
    } TPE;

    struct {
      RealType bendingWeight = 1.; 
      RealType elasticWeight = 1.;          // 메시가 구부러지거나 늘어날 때 발생하는 저항력 조절. 이 값이 클수록 부드러운 애니메이션 효과 
      RealType tpeWeight = 1.e-3;           // self-intersection 방지 위한 반발력 강도 
      RealType dirichletWeight = 0.;
      RealType barycenterWeight = 1.;       // 여러 형상의 중간 지점(Barycenter)을 계산할 때 각 형상이 갖는 영향력 조절 or 전체적인 질량 중심 유지하는 데 사용됨 
    } Energy;

    struct {
      bool inspect = false;
      bool saveVisualization = false;
    } Boundary;

    struct {
      bool enabled = false;
      RealType weight = 0.;
      bool useInObjective = false;
      RealType alpha = 6.;
      RealType beta = 12.;

      bool checkGradient = false;
      int checkCurveIndex = -1;
      RealType fdStep = 1.e-6;
    } BoundaryTPE;
  } Config;
  /* endregion */ 

  /* region Read config */ 
  // YAML 파일에 있는 내용을 변수에 매칭해서 저장 
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.startFile = config["Data"]["startFile"].as<std::string>();
    Config.secondFile = config["Data"]["secondFile"].as<std::string>();
    if ( config["Data"]["initFile"] )                                       // initMesh는 옵션. 정의되어 있지 않으면, initMesh = startMesh; 사용
      Config.initFile = config["Data"]["initFile"].as<std::string>();
    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

    Config.outputFilePrefix = config["Output"]["outputFilePrefix"].as<std::string>();
    Config.outputFolder = config["Output"]["outputFolder"].as<std::string>();
    Config.timestampOutput = config["Output"]["timestampOutput"].as<bool>();

    Config.numSteps = config["numSteps"].as<int>();

    Config.TPE.alpha = config["TPE"]["alpha"].as<int>();
    Config.TPE.beta = config["TPE"]["beta"].as<int>();
    Config.TPE.useAdaptivity = config["TPE"]["useAdaptivity"].as<bool>();
    Config.TPE.innerWeight = config["TPE"]["innerWeight"].as<RealType>();
    Config.TPE.theta = config["TPE"]["theta"].as<RealType>();
    Config.TPE.thetaNear = config["TPE"]["thetaNear"].as<RealType>();

    if ( auto ttype_it = TPETypeTable.find( config["TPE"]["Type"].as<std::string>()); ttype_it != TPETypeTable.end())
      Config.TPE.Type = ttype_it->second;
    else
      throw std::runtime_error( "Invalid TPE::Type in Config." );

    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();
    Config.Energy.barycenterWeight = config["Energy"]["barycenterWeight"].as<RealType>();
    Config.Energy.dirichletWeight = config["Energy"]["dirichletWeight"].as<RealType>();

    if (config["Boundary"]) {
      if (config["Boundary"]["inspect"])
        Config.Boundary.inspect = config["Boundary"]["inspect"].as<bool>();
      if (config["Boundary"]["saveVisualization"])
        Config.Boundary.saveVisualization = config["Boundary"]["saveVisualization"].as<bool>();
    }

    if ( config["BoundaryTPE"] ) {
      if ( config["BoundaryTPE"]["enabled"] )
        Config.BoundaryTPE.enabled = config["BoundaryTPE"]["enabled"].as<bool>();
      if ( config["BoundaryTPE"]["weight"] )
        Config.BoundaryTPE.weight = config["BoundaryTPE"]["weight"].as<RealType>();
      if ( config["BoundaryTPE"]["useInObjective"] )
        Config.BoundaryTPE.useInObjective = config["BoundaryTPE"]["useInObjective"].as<bool>();
      if ( config["BoundaryTPE"]["alpha"] )
        Config.BoundaryTPE.alpha = config["BoundaryTPE"]["alpha"].as<RealType>();
      if ( config["BoundaryTPE"]["beta"] )
        Config.BoundaryTPE.beta = config["BoundaryTPE"]["beta"].as<RealType>();
      if ( config["BoundaryTPE"]["checkGradient"] )
        Config.BoundaryTPE.checkGradient = config["BoundaryTPE"]["checkGradient"].as<bool>();
      if ( config["BoundaryTPE"]["checkCurveIndex"] )
        Config.BoundaryTPE.checkCurveIndex = config["BoundaryTPE"]["checkCurveIndex"].as<int>();
      if ( config["BoundaryTPE"]["fdStep"] )
        Config.BoundaryTPE.fdStep = config["BoundaryTPE"]["fdStep"].as<RealType>();
    }
      
    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();
  }
  /* endregion */ 

  /* region Output path */
  // output 폴더 경로가 /로 끝나지 않으면 / 추가. 
  if ( Config.outputFolder.compare( Config.outputFolder.length() - 1, 1, "/" ) != 0 )
    Config.outputFolder += "/";

  // 실행 파일 이름 execName 추출 -> output 폴더명에 추가하기 위함 
  std::string execName( argv[0] );
  execName = execName.substr( execName.find_last_of( '/' ) + 1 );

  // 폴더명에 timestamp를 추가하기 위함 
  if ( Config.timestampOutput ) {
    std::time_t t = std::time( nullptr );
    std::stringstream ss;
    ss << Config.outputFolder;
    ss << std::put_time( std::localtime( &t ), "%Y%m%d_%H%M%S" );
    ss << "_" << execName;
    ss << "/";
    Config.outputFolder = ss.str();
    boost::filesystem::create_directory( Config.outputFolder );
  }

  std::string outputPrefix = Config.outputFolder + Config.outputFilePrefix;

  // 사용한 config 파일을 복사해서 output 폴더 내부에 저장 
  if ( argc == 2 ) {
    boost::filesystem::copy_file( argv[1], Config.outputFolder + "_parameters.conf",
                                  boost::filesystem::copy_options::overwrite_existing );
  }
  /* endregion */ 

  /* region Logging */ 
  // _output.log 로그 파일 열기 
  // cout -> [파일 + 콘솔]
  std::ofstream logFile;
  logFile.open( Config.outputFolder + "/_output.log" );

  std::ostream output_cout( std::cout.rdbuf());
  std::ostream output_cerr( std::cerr.rdbuf());

  using TeeDevice = boost::iostreams::tee_device<std::ofstream, std::ostream>;
  using TeeStream = boost::iostreams::stream<TeeDevice>;
  TeeDevice tee_cout( logFile, output_cout );
  TeeDevice tee_cerr( logFile, output_cerr );

  TeeStream split_cout( tee_cout );
  TeeStream split_cerr( tee_cerr );

  std::cout.rdbuf( split_cout.rdbuf());
  std::cerr.rdbuf( split_cerr.rdbuf());
  /* endregion */

  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  /* Read meshes */
  // startMesh, secondMesh, initMesh 
  TriMesh startMesh, secondMesh, initMesh;
  if ( !OpenMesh::IO::read_mesh( startMesh, Config.startFile ))
    throw std::runtime_error( "Failed to read file: " + Config.startFile );
  if ( !OpenMesh::IO::read_mesh( secondMesh, Config.secondFile ))
    throw std::runtime_error( "Failed to read file: " + Config.secondFile );
  if ( !Config.initFile.empty()) {
    if ( !OpenMesh::IO::read_mesh( initMesh, Config.initFile ))
      throw std::runtime_error( "Failed to read file: " + Config.initFile );
  }
  else {
    initMesh = startMesh;
  }

  /* Topology of the mesh */
  MeshTopologySaver Topology( startMesh );            // Mesh의 연결 상태 정보를 Topology에 저장 
  int numVertices = Topology.getNumVertices();        // startMesh의 vertex 개수 

  /* Boundary Data */
  BoundaryUtils::BoundaryData boundaryData;

  const bool needBoundaryData =
      Config.Boundary.inspect ||
      Config.Boundary.saveVisualization ||
      Config.BoundaryTPE.enabled;

  if ( needBoundaryData ) {
    boundaryData = BoundaryUtils::extractBoundaryData( startMesh );

    if ( Config.Boundary.inspect || Config.BoundaryTPE.enabled ) {
      BoundaryUtils::printBoundaryInfo(
          boundaryData,
          std::cout,
          Config.startFile);
    }

    if ( Config.Boundary.saveVisualization ) {
      BoundaryUtils::writeBoundaryDebugFiles(
          startMesh,
          boundaryData,
          outputPrefix,
          "start");

      std::cout << " .. [Boundary] Saved boundary visualization files."
                << std::endl;
    }
  }

  const bool boundaryTPEAvailable =
      Config.BoundaryTPE.enabled && boundaryData.hasBoundary();

  const bool useBoundaryTPE =
      boundaryTPEAvailable &&
      Config.BoundaryTPE.useInObjective &&
      Config.BoundaryTPE.weight != 0.0;

  std::cout << " .. [BoundaryTPE] enabled        = " << Config.BoundaryTPE.enabled << std::endl;
  std::cout << " .. [BoundaryTPE] weight         = " << Config.BoundaryTPE.weight << std::endl;
  std::cout << " .. [BoundaryTPE] available      = " << boundaryTPEAvailable << std::endl;
  std::cout << " .. [BoundaryTPE] useInObjective = " << Config.BoundaryTPE.useInObjective << std::endl;
  std::cout << " .. [BoundaryTPE] active         = " << useBoundaryTPE << std::endl;

  /* Geometry of the mesh */
  VectorType Vertices_Start, Vertices_Second, Vertices_Init;      // start, second, init의 vertex 개수 계산 
  getGeometry( startMesh, Vertices_Start );
  getGeometry( secondMesh, Vertices_Second );
  getGeometry( initMesh, Vertices_Init );

  std::cout << " .. numVertices = " << numVertices << std::endl;

  // DirichletVertices와 non-DirichletVertices 분류 
  // [Memory Layout] 
      // x좌표들 [x1, x2, ..., xn] : Index 0 ~ N-1 
      // y좌표들 [y1, y2, ..., yn] : Index N ~ 2*N-1 
      // z좌표들 [z1, z2, ..., zn] : Index 2*N ~ 3*N-1 
  // dirichletIndices, nonDirichletIndices 저장 형태 
      // [x1, x2, ..., xn, y1, y2, .., yn, z1, z2, ..., zn] where n = #(vertices)
  std::vector<int> dirichletIndices, nonDirichletIndices;

  dirichletIndices = Config.dirichletVertices;
  const auto numDirichletVertices = Config.dirichletVertices.size();
  Config.dirichletVertices.resize( 3 * numDirichletVertices );
  for ( int i = 0; i < numDirichletVertices; i++ ) {
    Config.dirichletVertices[numDirichletVertices + i] = numVertices + Config.dirichletVertices[i];
    Config.dirichletVertices[2 * numDirichletVertices + i] = 2 * numVertices + Config.dirichletVertices[i];
  }
  for ( int vertexIdx = 0; vertexIdx < numVertices; vertexIdx++ ) {
    if ( std::find( dirichletIndices.begin(), dirichletIndices.end(), vertexIdx ) == dirichletIndices.end())
      nonDirichletIndices.push_back( vertexIdx );
  }

  // elastic shell energy W 
  ShellDeformationType W( Topology, Config.Energy.bendingWeight );

  std::cout << " .. W = " << W( Vertices_Start, Vertices_Second ) << std::endl;       // start -> second shape의 변형 비용 

  // Scary TPE 
  ScaryTPE::TangentPointEnergy<DefaultConfigurator> scTPE( Topology,
                                                           Config.TPE.alpha,
                                                           Config.TPE.beta,
                                                           Config.TPE.innerWeight,
                                                           Config.TPE.useAdaptivity,        // 차이 : adaptive 계산 여부 
                                                           Config.TPE.theta,
                                                           Config.TPE.thetaNear );          // 차이 : near-field 기준 

  // Spooky TPE 
  SpookyTPE::FastMultipoleTangentPointEnergy<DefaultConfigurator> spTPE( Topology,
                                                                         Config.TPE.alpha,
                                                                         Config.TPE.beta,
                                                                         Config.TPE.innerWeight,
                                                                         Config.TPE.theta );

  // FMM(Fast Multipole Method)를 사용하여 멀리 있는 면들 사잉의 연산을 근사 처리 -> 빠른 근사 
  SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy<DefaultConfigurator> adspTPE( Topology,
                                                                                   Config.TPE.alpha,
                                                                                   Config.TPE.beta,
                                                                                   Config.TPE.innerWeight,
                                                                                   Config.TPE.theta,
                                                                                   Config.TPE.thetaNear );

  // 3가지 TPE 
    // SCARY : 정확도 높음. 속도 느림 
    // SPOOKY : 정확도 중간. 속도 빠름 
    // Adaptive : 정확도 높음. 속도 중간 
  ObjectiveFunctional<DefaultConfigurator> *chosenTPE = nullptr;

  if ( Config.TPE.Type == SCARY ) {
    chosenTPE = &scTPE;                       // ScaryTPE 
  }
  else if ( Config.TPE.Type == SPOOKY ) {
    if ( Config.TPE.useAdaptivity ) {
      chosenTPE = &adspTPE;                   // AdaptiveFastMultipoleTPE 
    }
    else {
      chosenTPE = &spTPE;                     // SpookyTPE 
    }
  }

  // TPE 객체를 실제 계산 가능한 함수 형태로 감싸기 (wrapping)
  ObjectiveWrapper TPE( *chosenTPE );               // 에너지 
  ObjectiveGradientWrapper TPG( *chosenTPE );       // 에너지 그래디언트(기울기) 

  std::unique_ptr<
      ScaryTPE::BoundaryCurveTangentPointEnergy<DefaultConfigurator>
  > BoundaryTPE;

  std::unique_ptr<ObjectiveWrapper<DefaultConfigurator>> Ebdry;
  std::unique_ptr<ObjectiveGradientWrapper<DefaultConfigurator>> DEbdry;

  if ( boundaryTPEAvailable ) {
    BoundaryTPE =
        std::make_unique<
            ScaryTPE::BoundaryCurveTangentPointEnergy<DefaultConfigurator>
        >(
            Topology,
            boundaryData.edges,
            Config.BoundaryTPE.alpha,
            Config.BoundaryTPE.beta
        );

    Ebdry = std::make_unique<ObjectiveWrapper<DefaultConfigurator>>( *BoundaryTPE );
    DEbdry = std::make_unique<ObjectiveGradientWrapper<DefaultConfigurator>>( *BoundaryTPE );

    std::cout << " .. [BoundaryTPE] boundary single-shape energy object created."
              << std::endl;
  }

  // start, second Mesh의 TPE 초기값 출력 
  std::cout << " .. TPE(1) = " << TPE( Vertices_Start ) << std::endl;
  std::cout << " .. TPE(2) = " << TPE( Vertices_Second ) << std::endl;

  // Newton method 계산을 위한 shape 3개 초기화 : s0, s1, s2 
  VectorType s0 = Vertices_Start, s1 = Vertices_Second, s2 = Vertices_Second;

  // Topology(연결 정보)와 현재 VectorType 좌표 데이터를 합쳐서 .ply 파일로 저장 
  saveAsPLY<VectorType>( Topology, s0, outputPrefix + "comb_exp_0.ply" );
  saveAsPLY<VectorType>( Topology, s1, outputPrefix + "comb_exp_1.ply" );

  for ( int t = 2; t <= Config.numSteps; t++ ) {
    std::cout << std::endl;
    std::cout << " .. Step " << t << ": " << std::endl;

    if ( !Config.initFile.empty())
      s2 = Vertices_Init;

    /* Vector-Valued Equation (confusingly called energy) */
    // s2가 만족해야 할 물리 방정식을 정의 
    // Goal: find s2 s.t. F(s2) = 0 where F = Weights * (Felast, Ftpe, Fbary, Fdir) 
    // Exp2Energy (scalar 에너지 X. gradient 벡터 에너지 i.e. Jacobian Matrix)
        // Exp: Exponential Map (현재 상태와 속도로부터 다음 상태를 예측하는 기하학적 연산) 
        // 2: 이전 두 프레임을 사용하는 2차(2nd-order) 예측 방식 
        // Energy: 예측 경로가 물리적 관성 및 반발력과 얼마나 일치하는지 수치화한 값 
    Exp2Energy<DefaultConfigurator> Felast( W, s0, s1 );                                // Elastic Inertia (탄성 관성)
    DifferenceExp2Energy<DefaultConfigurator> Ftpe( TPE, TPG, s0, s1 );                 // 충돌 회피를 위한 메트릭 왜곡 
    BarycenterExp2Energy<DefaultConfigurator> Fbary( s0, s1, nonDirichletIndices );     // 질량중심을 유지하여 엉뚱한 방향으로 회전하지 않도록 
    DirichletExp2Energy<DefaultConfigurator> Fdir( s0, s1, dirichletIndices );          // 고정점. 정해진 궤적을 벗어나지 않도록 강한 복원력을 줌 

    // 가중치 벡터 Weights 정의
    std::unique_ptr<AdditionGradient<DefaultConfigurator>> F;
    VectorType Weights;
    if ( useBoundaryTPE ) {
      Weights.resize( 5 );
      Weights << Config.Energy.elasticWeight,
                Config.Energy.tpeWeight,
                Config.Energy.dirichletWeight,
                Config.Energy.barycenterWeight,
                Config.BoundaryTPE.weight;

      F = std::make_unique<AdditionGradient<DefaultConfigurator>>(
          Weights,
          Felast,
          Ftpe,
          Fdir,
          Fbary,
          *DEbdry
      );
    }
    else {
      Weights.resize( 4 );
      Weights << Config.Energy.elasticWeight,
                Config.Energy.tpeWeight,
                Config.Energy.dirichletWeight,
                Config.Energy.barycenterWeight;

      F = std::make_unique<AdditionGradient<DefaultConfigurator>>(
          Weights,
          Felast,
          Ftpe,
          Fdir,
          Fbary
      );
    }

    /* Derivatives (matrix valued) and its inverse */
    // DF: Hessian Matrix 
    CombinedExp2Gradient<DefaultConfigurator> DF( W, TPE, TPG, s0, s1,
                                                  Config.dirichletVertices, nonDirichletIndices,
                                                  Config.Energy.elasticWeight,
                                                  Config.Energy.tpeWeight,
                                                  Config.Energy.dirichletWeight,
                                                  Config.Energy.barycenterWeight );

    // inverse of DF 
    InverseCombinedExp2Gradient<DefaultConfigurator> invDF( W, TPE, TPG, s0, s1,
                                                            Config.dirichletVertices, nonDirichletIndices,
                                                            Config.Energy.elasticWeight,
                                                            Config.Energy.tpeWeight,
                                                            Config.Energy.dirichletWeight,
                                                            Config.Energy.barycenterWeight );

    // Newton Method 
        // s2 <- s2-(DF)^(-1)*F 
    NewOpt::NewtonMethod Solver( *F, DF, invDF,
                                Config.Optimization.maxNumIterations,
                                1e-8,
                                NEWTON_OPTIMAL,
                                SHOW_ALL,
                                0.1,
                                Config.Optimization.minStepsize,
                                Config.Optimization.maxStepsize );
    if ( Config.Energy.dirichletWeight == 0. )
      Solver.setBoundaryMask( Config.dirichletVertices );     // 정점들을 고정 

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( s2, s2 );                                   // find s2 s.t. F(s2)=0 
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
              << " seconds." << std::endl;

    if ( boundaryTPEAvailable ) {
      std::cout << " .... Ebdry(s2)       = "
                << (*Ebdry)( s2 ) << std::endl;
      std::cout << " .... DEbdry(s2).norm = "
                << (*DEbdry)( s2 ).norm() << std::endl;
    }

    saveAsPLY<VectorType>( Topology, s2, outputPrefix + "comb_exp_" + std::to_string(t) + ".ply" );

    s0 = s1;
    s1 = s2;
  }

  std::cout << " - outputPrefix: " << outputPrefix << std::endl;

  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
