#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>
#include <goast/Optimization.h>
#include <goast/external/vtkIO.h>

#include <csignal>    // 외부 신호 처리 -> Ctrl+C 인터럽트 감지
#include <stdexcept>  // 예외 처리
#include <memory>     // 스마트 포인터 사용 

#include "BoundaryUtils.h"

#include "GraphManifold/DifferencePathEnergy.h"

#include "ScaryTPE/TangentPointEnergy.h"
#include "ScaryTPE/TPObstacleEnergy.h"
#include "ScaryTPE/SobolevSlobodeckij.h"
#include "ScaryTPE/BoundaryCurveTangentPointEnergy.h"
#include "ScaryTPE/BoundaryCurvePathEnergy.h"

#include "SpookyTPE/FastMultipoleEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

#include "Optimization/TrustRegionNewton.h"
#include "Optimization/LineSearchNewtonCG.h"

#include "PathMetrics.h"
#include "SoftPointConstraint.h"
#include "CombinedDeformation.h"
#include "OperatorPathEnergyHessian.h"
#include "HessianMetric.h"
#include "BarycenterPathEnergy.h"
#include "RotationPathEnergy.h"

#include "MeshIO.h"

#pragma omp declare reduction (merge : std::vector<DefaultConfigurator::TripletType> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

namespace {
  volatile std::sig_atomic_t g_interruptRequested = 0;

  void handleInterrupt( int ) {
    g_interruptRequested = 1;
  }
}

using VectorType = DefaultConfigurator::VectorType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;
using MatrixType = DefaultConfigurator::SparseMatrixType;

using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> >;


int main( int argc, char *argv[] ) {
  std::signal( SIGINT, handleInterrupt );

  //region Config
  enum TPEType {
    SPOOKY,
    SCARY
  };

  std::unordered_map<std::string, TPEType> const TPETypeTable = {
    { "Spooky", TPEType::SPOOKY },
    { "Scary",  TPEType::SCARY },
  };

  enum OptimizationType {
    GRADIENT_DESCENT,
    BFGS,
    PRECONDITIONED,
    NEWTONCG,
    TRUSTREGION
  };

  std::unordered_map<std::string, OptimizationType> const optTypeTable = {
    { "GradientDescent", OptimizationType::GRADIENT_DESCENT },
    { "BFGS",            OptimizationType::BFGS },
    { "Preconditioned",  OptimizationType::PRECONDITIONED },
    { "NewtonCG",        OptimizationType::NEWTONCG },
    { "TrustRegion",     OptimizationType::TRUSTREGION },
  };

  enum PreconditionerType {
    ELASTIC_HESSIAN,
    L2_ELASTIC_METRIC,
    L2_COMBINED_METRIC,
    ELASTIC_HESSIAN_AND_L2_SSM,
    L2_SSM,
    ELASTIC_HESSIAN_AND_REDUCED,
  };

  std::unordered_map<std::string, PreconditionerType> const PreconditionerTable = {
    { "ElasticHessian",                  PreconditionerType::ELASTIC_HESSIAN },
    { "L2ElasticMetric",                 PreconditionerType::L2_ELASTIC_METRIC },
    { "L2CombinedMetric",                PreconditionerType::L2_COMBINED_METRIC },
    { "ElasticHessianAndL2SSM",          PreconditionerType::ELASTIC_HESSIAN_AND_L2_SSM },
    { "L2SoboSlobo",                     PreconditionerType::L2_SSM },
    { "ElasticHessianAndReducedHessian", PreconditionerType::ELASTIC_HESSIAN_AND_REDUCED },
  };

  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    bool saveIntermediate = false;
    int intermediateEvery = 100;
    std::string intermediateFolder = "intermediate";

    std::string startFile;
    std::string endFile;
    std::string obstacleFile;
    std::vector<std::string> initFiles;
    std::vector<int> dirichletVertices;

    int numSteps = 6;
    int numLevels = 1;

    struct {
      int alpha = 6;
      int beta = 12;
      TPEType Type = SCARY;
      RealType innerWeight = 1.;
      RealType theta = 0.5;
      RealType thetaNear = 10.;
      bool useLowerSSMTerm = false;
      bool useHigherSSMTerm = false;
      bool useAdaptivity = true;
      bool useObstacleAdaptivity = true;
    } TPE;

    struct {
      RealType bendingWeight = 1.;
      RealType elasticWeight = 1.;
      RealType tpeWeight = 1.e-3;
      RealType dirichletWeight = 1.;
      RealType obstacleWeight = 0.;
      RealType barycenterWeight = 0.;
      RealType rotationWeight = 0.;
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

    struct {
      int maxNumIterations = 50000;
      RealType minStepsize = 1.e-12;
      RealType maxStepsize = 10.;
      RealType minReduction = 1.e-14;

      OptimizationType Type = GRADIENT_DESCENT;
      PreconditionerType Preconditioner = ELASTIC_HESSIAN;
    } Optimization;
  } Config;
  //endregion Config

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.startFile = config["Data"]["startFile"].as<std::string>();
    Config.endFile = config["Data"]["endFile"].as<std::string>();
    if ( config["Data"]["obstacleFile"] )
      Config.obstacleFile = config["Data"]["obstacleFile"].as<std::string>();

    Config.numSteps = config["numSteps"].as<int>();
    Config.numLevels = config["numLevels"].as<int>();

    if ( config["Data"]["initFile"] ) {
      if ( config["Data"]["initFile"].IsSequence()) {
        Config.initFiles = config["Data"]["initFile"].as<std::vector<std::string>>();
      }
      else {
        Config.initFiles.resize( Config.numSteps - 1, config["Data"]["initFile"].as<std::string>() );
      }
    }

    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

    Config.outputFilePrefix = config["Output"]["outputFilePrefix"].as<std::string>();
    Config.outputFolder = config["Output"]["outputFolder"].as<std::string>();
    Config.timestampOutput = config["Output"]["timestampOutput"].as<bool>();

    if ( config["Output"]["saveIntermediate"] )
      Config.saveIntermediate = config["Output"]["saveIntermediate"].as<bool>();

    if ( config["Output"]["intermediateEvery"] )
      Config.intermediateEvery = config["Output"]["intermediateEvery"].as<int>();

    if ( config["Output"]["intermediateFolder"] )
      Config.intermediateFolder = config["Output"]["intermediateFolder"].as<std::string>();

    Config.TPE.alpha = config["TPE"]["alpha"].as<int>();
    Config.TPE.beta = config["TPE"]["beta"].as<int>();
    Config.TPE.useAdaptivity = config["TPE"]["useAdaptivity"].as<bool>();
    Config.TPE.innerWeight = config["TPE"]["innerWeight"].as<RealType>();
    Config.TPE.theta = config["TPE"]["theta"].as<RealType>();
    Config.TPE.thetaNear = config["TPE"]["thetaNear"].as<RealType>();
    Config.TPE.useLowerSSMTerm = config["TPE"]["useLowerSSMTerm"].as<bool>();
    Config.TPE.useHigherSSMTerm = config["TPE"]["useHigherSSMTerm"].as<bool>();

    if ( auto ttype_it = TPETypeTable.find( config["TPE"]["Type"].as<std::string>()); ttype_it != TPETypeTable.end())
      Config.TPE.Type = ttype_it->second;
    else
      throw std::runtime_error( "Invalid TPE::Type in Config." );

    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();
    Config.Energy.dirichletWeight = config["Energy"]["dirichletWeight"].as<RealType>();
    if ( config["Energy"]["obstacleWeight"] )
      Config.Energy.obstacleWeight = config["Energy"]["obstacleWeight"].as<RealType>();
    if ( config["Energy"]["barycenterWeight"] )
      Config.Energy.barycenterWeight = config["Energy"]["barycenterWeight"].as<RealType>();
    if ( config["Energy"]["rotationWeight"] )
      Config.Energy.rotationWeight = config["Energy"]["rotationWeight"].as<RealType>();

    if ( config["Boundary"] ) {
      if ( config["Boundary"]["inspect"] )
        Config.Boundary.inspect = config["Boundary"]["inspect"].as<bool>();
      if ( config["Boundary"]["saveVisualization"] )
        Config.Boundary.saveVisualization = config["Boundary"]["saveVisualization"].as<bool>();
    }

    if (config["BoundaryTPE"]) {
      if (config["BoundaryTPE"]["enabled"])
        Config.BoundaryTPE.enabled = config["BoundaryTPE"]["enabled"].as<bool>();
      if (config["BoundaryTPE"]["weight"])
        Config.BoundaryTPE.weight = config["BoundaryTPE"]["weight"].as<RealType>();
      if (config["BoundaryTPE"]["useInObjective"])
        Config.BoundaryTPE.useInObjective = config["BoundaryTPE"]["useInObjective"].as<bool>();
      if (config["BoundaryTPE"]["alpha"])
        Config.BoundaryTPE.alpha = config["BoundaryTPE"]["alpha"].as<RealType>();
      if (config["BoundaryTPE"]["beta"])
        Config.BoundaryTPE.beta = config["BoundaryTPE"]["beta"].as<RealType>();
      if (config["BoundaryTPE"]["checkGradient"])
        Config.BoundaryTPE.checkGradient = config["BoundaryTPE"]["checkGradient"].as<bool>();
      if (config["BoundaryTPE"]["checkCurveIndex"])
        Config.BoundaryTPE.checkCurveIndex = config["BoundaryTPE"]["checkCurveIndex"].as<int>();
      if (config["BoundaryTPE"]["fdStep"])
        Config.BoundaryTPE.fdStep = config["BoundaryTPE"]["fdStep"].as<RealType>();
    }

    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();
    Config.Optimization.minReduction = config["Optimization"]["minReduction"].as<RealType>();

    auto otype_it = optTypeTable.find( config["Optimization"]["Type"].as<std::string>());
    if ( otype_it != optTypeTable.end())
      Config.Optimization.Type = otype_it->second;
    else
      throw std::runtime_error( "Invalid Optimization::Type in Config." );

    auto precond_it = PreconditionerTable.find( config["Optimization"]["Preconditioner"].as<std::string>());
    if ( precond_it != PreconditionerTable.end())
      Config.Optimization.Preconditioner = precond_it->second;
    else
      throw std::runtime_error( "Invalid Optimization::Preconditioner in Config." );
  }
  //endregion

  //region Output path
  if ( Config.outputFolder.compare( Config.outputFolder.length() - 1, 1, "/" ) != 0 )
    Config.outputFolder += "/";

  std::string execName( argv[0] );
  execName = execName.substr( execName.find_last_of( '/' ) + 1 );

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

  if ( argc == 2 ) {
    boost::filesystem::copy_file( argv[1], Config.outputFolder + "_parameters.conf",
                                  boost::filesystem::copy_options::overwrite_existing );
  }
  //endregion

  //region Logging
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
  //endregion

  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  // Read meshes
  TriMesh startMesh, endMesh, obstacleMesh;
  std::vector<TriMesh> initMeshes;
  if ( !OpenMesh::IO::read_mesh( startMesh, Config.startFile ))
    throw std::runtime_error( "Failed to read file: " + Config.startFile );
  if ( !OpenMesh::IO::read_mesh( endMesh, Config.endFile ))
    throw std::runtime_error( "Failed to read file: '" + Config.endFile + "'" );

  if (!Config.obstacleFile.empty()) {
    if ( !OpenMesh::IO::read_mesh( obstacleMesh, Config.obstacleFile ))
      throw std::runtime_error( "Failed to read file: " + Config.obstacleFile );

    OpenMesh::IO::write_mesh( obstacleMesh, outputPrefix + "obstacle.ply" );
  }

  if ( !Config.initFiles.empty()) {
    initMeshes.resize( Config.numSteps - 1 );
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      if ( !OpenMesh::IO::read_mesh( initMeshes[k], Config.initFiles[k] ))
        throw std::runtime_error( "Failed to read file: " + Config.initFiles[k] );
  }
  else {
    initMeshes.resize( Config.numSteps - 1, startMesh );
  }

  //region Setup
  // Topology of the mesh
  MeshTopologySaver Topology( startMesh );
  MeshTopologySaver obstacleTopology( obstacleMesh );
  int numVertices = Topology.getNumVertices();

  std::cout << " .. numVertices = " << numVertices << std::endl;

  // Boundary data extraction and visualization 
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

  const bool boundaryTPEAvailable = Config.BoundaryTPE.enabled && boundaryData.hasBoundary();
  const bool useBoundaryTPE = boundaryTPEAvailable && Config.BoundaryTPE.useInObjective && Config.BoundaryTPE.weight != 0.0;

  std::cout << " .. [BoundaryTPE] enabled        = " << Config.BoundaryTPE.enabled << std::endl;
  std::cout << " .. [BoundaryTPE] weight         = " << Config.BoundaryTPE.weight << std::endl;
  std::cout << " .. [BoundaryTPE] available      = " << boundaryTPEAvailable << std::endl;
  std::cout << " .. [BoundaryTPE] useInObjective = " << Config.BoundaryTPE.useInObjective << std::endl;
  std::cout << " .. [BoundaryTPE] active         = " << useBoundaryTPE << std::endl;
  std::cout << " .. [BoundaryTPE] checkGradient   = " << Config.BoundaryTPE.checkGradient << std::endl;
  std::cout << " .. [BoundaryTPE] checkCurveIndex = " << Config.BoundaryTPE.checkCurveIndex << std::endl;
  std::cout << " .. [BoundaryTPE] fdStep         = " << Config.BoundaryTPE.fdStep << std::endl;

  if ( Config.BoundaryTPE.enabled && !boundaryData.hasBoundary() ) {
    std::cout << " .. [BoundaryTPE] skipped: no boundary detected." << std::endl;
  }
  if ( Config.BoundaryTPE.enabled && Config.BoundaryTPE.weight == 0.0 ) {
    std::cout << " .. [BoundaryTPE] skipped: weight is zero." << std::endl;
  }

  // Geometry of the mesh
  VectorType Vertices_Start, Vertices_End, Vertices_Obstacle;
  std::vector<VectorType> Vertices_Init( Config.numSteps - 1 );
  getGeometry( startMesh, Vertices_Start );
  getGeometry( endMesh, Vertices_End );
  getGeometry( obstacleMesh, Vertices_Obstacle );
  for ( int k = 0; k < Config.numSteps - 1; k++ )
    getGeometry( initMeshes[k], Vertices_Init[k] );

  // Registration
    // dirichletIndices와 nonDirichletIndices 벡터 생성 
    // 변수명 구분 
      // dirichletIndices : 원래 vertex index 목록 
      // Config.dirichletVertices : vertex index 목록이 3배로 늘어난 목록 (x,y,z 각각에 대해 원래 vertex index가 numVertices씩 offset되어 추가됨)
        // x_i : i 
        // y_i : numVertices + i
        // z_i : 2*numVertices + i 
  std::vector<int> dirichletIndices = Config.dirichletVertices;
  std::vector<int> nonDirichletIndices;
  const auto numDirichletVertices = Config.dirichletVertices.size();
  Config.dirichletVertices.resize( 3 * numDirichletVertices );
  for ( int i = 0; i < numDirichletVertices; i++ ) {
    Config.dirichletVertices[numDirichletVertices + i] = numVertices + Config.dirichletVertices[i];
    Config.dirichletVertices[2 * numDirichletVertices + i] = 2 * numVertices + Config.dirichletVertices[i];
  }
  for ( int vertexIdx = 0; vertexIdx < numVertices; vertexIdx++ ) {
    if ( std::find( dirichletIndices.begin(), dirichletIndices.end(), vertexIdx ) == dirichletIndices.end() )
      nonDirichletIndices.push_back( vertexIdx );
  }

  std::cout << " .. dirichletIndices = {";
  for ( auto idx: dirichletIndices )
    std::cout << idx << ", ";
  std::cout << "}" << std::endl;

  // localMask를 globalMask로 변환하는 함수 fillPathMask를 이용하여 fixedVariables 벡터 생성
  std::vector<int> fixedVariables;
  fillPathMask( Config.numSteps - 1, 3 * numVertices, Config.dirichletVertices, fixedVariables );
  //endregion

  //region Basic energies
  // Elastic energy
    // Wmem, Wbend, Welast 정의 
      // Wmem : 표면이 얼마나 늘어나거나 찢어지는지에 관한 에너지 (membrane energy)
      // Wbend : 표면이 얼마나 구부러지는지(접히거나 휘어지는지)에 관한 에너지 (bending energy)
      // Welast : Wmem + bendingWeight * Wbend, 표면의 전체적인 탄성 에너지
  NonlinearMembraneDeformation<DefaultConfigurator> Wmem( Topology, 1. );
  SimpleBendingDeformation<DefaultConfigurator> Wbend( Topology, 1. );
  CombinedDeformation<DefaultConfigurator> Welast( Wmem, Config.Energy.bendingWeight, Wbend );

    // Start와 End 사이의 Wmem, Wbend, Welast 계산 및 출력
  std::cout << " .. Wmem = " << Wmem( Vertices_Start, Vertices_End ) << std::endl;
  std::cout << " .. Wbend = " << Wbend( Vertices_Start, Vertices_End ) << std::endl;
  std::cout << " .. W = " << Welast( Vertices_Start, Vertices_End ) << std::endl;

  // Elastic metric
  HessianMetric<DefaultConfigurator> ElasticMetric( Welast, Config.dirichletVertices );
  OperatorHessianMetric<DefaultConfigurator> opElasticMetric( Welast, Config.dirichletVertices );

  // Tangent Point Energy and SobolevSlobodeckijMetric
    // ScaryTPE와 SpookyTPE의 객체 생성
      // YAML 파일에서 SCARY를 선택 -> scTPE 
      // YAML 파일에서 SPOOKY를 선택 -> spTPE 또는 adspTPE (useAdaptivity 여부에 따라, useAdaptivity가 true면 adspTPE, false면 spTPE)
  ScaryTPE::TangentPointEnergy<DefaultConfigurator> scTPE( Topology,
                                                           Config.TPE.alpha,
                                                           Config.TPE.beta,
                                                           Config.TPE.innerWeight,
                                                           Config.TPE.useAdaptivity,
                                                           Config.TPE.theta,
                                                           Config.TPE.thetaNear );

  SpookyTPE::FastMultipoleTangentPointEnergy<DefaultConfigurator> spTPE( Topology,
                                                                         Config.TPE.alpha,
                                                                         Config.TPE.beta,
                                                                         Config.TPE.innerWeight,
                                                                         Config.TPE.theta );

  SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy<DefaultConfigurator> adspTPE( Topology,
                                                                                   Config.TPE.alpha,
                                                                                   Config.TPE.beta,
                                                                                   Config.TPE.innerWeight,
                                                                                   Config.TPE.theta,
                                                                                   Config.TPE.thetaNear );

  ObjectiveFunctional<DefaultConfigurator> *chosenTPE = nullptr;

  if ( Config.TPE.Type == SCARY ) {
    chosenTPE = &scTPE;
  }
  else if ( Config.TPE.Type == SPOOKY ) {
    if ( Config.TPE.useAdaptivity ) {
      chosenTPE = &adspTPE;
    }
    else {
      chosenTPE = &spTPE;
    }
  }

    // TPE : TPE를 계산하는 wrapper 
    // TPG : TPE의 gradient를 계산하는 wrapper
  ObjectiveWrapper TPE( *chosenTPE );
  ObjectiveGradientWrapper TPG( *chosenTPE );

    // ??? 
  ScaryTPE::SurfaceSobolevSlobodeckijOperatorMap<DefaultConfigurator> SSMop( Topology,
                                                                             Config.TPE.alpha,
                                                                             Config.TPE.beta,
                                                                             Config.TPE.useLowerSSMTerm,
                                                                             Config.TPE.useHigherSSMTerm,
                                                                             Config.TPE.innerWeight,
                                                                             Config.TPE.useAdaptivity );
  //endregion

    // Path : 최적화 과정에서 intermediate shapes (s_1, ..., s_{K-1})의 geometry를 벡터 형태로 저장하는 변수. 크기는 (K-1) * 3 * numVertices.
      // Here, K is the number of steps (Config.numSteps) in the path
      // s_k : 하나의 mesh geometry vector. 크기는 3*numVertices  
        // s_0 = start shape 
        // s_K = end shape
        // s_1, ..., s_{K-1} : intermediate shapes, 최적화 과정에서 업데이트되는 변수. 
  VectorType Path;
  for ( int level = 0; level < Config.numLevels; level++ ) {
    std::cout << std::endl;
    std::cout << " --- Level " << level << " --- " << std::endl;
    // Refine in time
    if ( level > 0 )
      Config.numSteps *= 2;

    // Create Dirichlet mask along time
    fillPathMask( Config.numSteps - 1, 3 * numVertices, Config.dirichletVertices, fixedVariables );

    //region Path energies
      // Weights = weighted sum에서 각각의 에너지 항에 곱해지는 가중치 벡터 (elastic, barycenter, dirichlet, obstacle, tpe, rotation)
    VectorType Weights( 6 );
    Weights << Config.Energy.elasticWeight, Config.Energy.barycenterWeight, Config.Energy.dirichletWeight,
        Config.Energy.obstacleWeight, Config.Energy.tpeWeight, Config.Energy.rotationWeight;

    // [E-1] Elasticity
      // Emem, Ebend : 최적화 후에 step별 energy 항의 값을 출력하기 위한 객체 
      // Eelast : 실제 objective에 들어가는 elastic energy 항 
    DiscretePathEnergy<DefaultConfigurator> Emem( Wmem, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergy<DefaultConfigurator> Ebend( Wbend, Config.numSteps, Vertices_Start, Vertices_End );

    DiscretePathEnergy<DefaultConfigurator> Eelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergyGradient<DefaultConfigurator> DEelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergyHessian<DefaultConfigurator> D2Eelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    OperatorPathEnergyHessian<DefaultConfigurator> D2Eop( Welast, Config.numSteps, Vertices_Start, Vertices_End,
                                                          Config.dirichletVertices );


    // [E-2] TPE
      // TPDE : Tangent Point Difference Energy 
      // YAML에서 정의한 TPE type에 따라서 choseTPDE 객체가 scTPE, spTPE, adspTPE 중 하나로 생성됨.
    std::unique_ptr<ObjectiveFunctional<DefaultConfigurator>> chosenTPDE;
    // ObjectiveFunctional<DefaultConfigurator> *chosenTPDE = &spTPDEn;

    if ( Config.TPE.Type == SCARY ) {
      chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>>(
        Config.numSteps,
        Vertices_Start,
        Vertices_End,
        Topology,
        Config.TPE.alpha,
        Config.TPE.beta,
        Config.TPE.innerWeight,
        Config.TPE.useAdaptivity,
        Config.TPE.theta,
        Config.TPE.thetaNear
      );
    }
    else if ( Config.TPE.Type == SPOOKY ) {
      if ( Config.TPE.useAdaptivity ) {
        chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>>(
          Config.numSteps,
          Vertices_Start,
          Vertices_End,
          Topology,
          Config.TPE.alpha,
          Config.TPE.beta,
          Config.TPE.innerWeight,
          Config.TPE.theta,
          Config.TPE.thetaNear
        );
      }
      else {
        chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>>(
          Config.numSteps,
          Vertices_Start,
          Vertices_End,
          Topology,
          Config.TPE.alpha,
          Config.TPE.beta,
          Config.TPE.innerWeight,
          Config.TPE.theta
        );
      }
    }
      // Etpe : TPE path energy 값 
      // DEtpe : TPE path energy의 gradient
      // RD2Etpe : TPE path energy의 Hessian (matrix 형태)
      // opRD2Etpe : TPE path energy의 Hessian (operator 형태)
    ObjectiveWrapper Etpe( *chosenTPDE );
    ObjectiveGradientWrapper DEtpe( *chosenTPDE );
    ObjectiveHessianWrapper RD2Etpe( *chosenTPDE );
    ObjectiveHessianOperatorWrapper opRD2Etpe( *chosenTPDE );

    // [E-3] RBM : Rigid Body Motion을 고정시키는 에너지 항 -> tracking energy 라고도 불림. 
      // Edir : dirichlet vertices가 path를 따라 부드럽게 이동하도록 유도하는 에너지 항
      // dirichlet weight 값에 따라서  
        // w 값이 크면, dirichlet vertices가 start에서 end로 거의 고정된 채로 움직이게 됨 (즉, rigid하게 움직이게 됨)
        // w 값이 작으면, dirichlet vertices가 path를 따라 자유롭게 움직
    TrackingPathEnergy<DefaultConfigurator> Edir( dirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );
    TrackingPathEnergyGradient<DefaultConfigurator> DEdir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                           Vertices_End );
    TrackingPathEnergyHessian<DefaultConfigurator> D2Edir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                           Vertices_End );
    TrackingPathEnergyHessianOperator<DefaultConfigurator> opD2Edir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                                     Vertices_End );

    // [E-4] Obstacle (only available in scary)
      // Eobs를 정의할 때 TPODE가 parameter로 들어감. 
      /* Etpoe는 코드 상에서 사용되고 있지 않아서 주석처리 함 
    ScaryTPE::TangentPointObstacleEnergy<DefaultConfigurator> Etpoe( Topology, obstacleTopology, Vertices_Obstacle,
                                                                     Config.TPE.alpha, Config.TPE.beta,
                                                                     Config.TPE.innerWeight, Config.TPE.useObstacleAdaptivity,
                                                                     Config.TPE.theta, Config.TPE.thetaNear );
      */
      // ??? 여기서 입력 parameter 5의 의미가 뭐지 
    DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy> TPODE(
      Config.numSteps, Vertices_Start, Vertices_End, Topology, obstacleTopology, Vertices_Obstacle, Config.TPE.alpha,
      Config.TPE.beta, Config.TPE.innerWeight, Config.TPE.useObstacleAdaptivity, Config.TPE.theta, Config.TPE.thetaNear,
      5
    );

    ObjectiveWrapper Eobs( TPODE );
    ObjectiveGradientWrapper DEobs( TPODE );
    ObjectiveHessianWrapper D2Eobs( TPODE );
    ObjectiveHessianOperatorWrapper opD2Eobs( TPODE );

    // [E-5] Barycenter
      // BarycenterPathEnergy는 path를 따라 mesh의 무게중심이 갑자기 이동하는 것을 막는 에너지 항
      // nonDirichletIndices가 parameter로 들어감.
    BarycenterPathEnergy BPE( Topology, nonDirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );

    ObjectiveWrapper Ebary( BPE );
    ObjectiveGradientWrapper DEbary( BPE );
    ObjectiveHessianWrapper D2Ebary( BPE );
    ObjectiveHessianOperatorWrapper opD2Ebary( BPE );

    // [E-6] Rotation
      // RotationPathEnergy는 path를 따라 mesh가 갑자기 회전하는 것을 막는 에너지 항
      // nonDirichletIndices가 parameter로 들어감.
    RotationPathEnergy RPE( Topology, nonDirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );

    ObjectiveWrapper Erot( RPE );
    ObjectiveGradientWrapper DErot( RPE );
    ObjectiveHessianWrapper D2Erot( RPE );
    ObjectiveHessianOperatorWrapper opD2Erot( RPE );

    // [E-7] Boundary Curve TPE (optional, currently zero-energy dummy)
    std::unique_ptr<
        ScaryTPE::BoundaryCurveTangentPointEnergy<DefaultConfigurator>
    > BoundaryTPE;

    std::unique_ptr<
        ScaryTPE::BoundaryCurvePathEnergy<DefaultConfigurator>
    > BoundaryTPDE;

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

      BoundaryTPDE =
        std::make_unique<
            ScaryTPE::BoundaryCurvePathEnergy<DefaultConfigurator>
        >(
            Config.numSteps,
            Topology,
            boundaryData.edges,
            Config.BoundaryTPE.alpha,
            Config.BoundaryTPE.beta
        );

      Ebdry = std::make_unique<ObjectiveWrapper<DefaultConfigurator>>( *BoundaryTPDE );
      DEbdry = std::make_unique<ObjectiveGradientWrapper<DefaultConfigurator>>( *BoundaryTPDE );

      std::cout << " .. [BoundaryTPE] boundary energy object created."
                << std::endl;
    }

    // [E] : 최종적으로 optimization에서 minimize하려는 전체 energy 항. 
      // 6개의 energy 항 (elastic, barycenter, dirichlet, obstacle, tpe, rotation)을 Weights 벡터에 정의된 가중치에 따라서 선형 결합한 형태.
      // Optimizer는 E(Path)와 DE(Path)를 이용해서 Path를 업데이트함.
    std::unique_ptr<AdditionOp<DefaultConfigurator>> E;
    std::unique_ptr<AdditionGradient<DefaultConfigurator>> DE;

    VectorType BoundaryWeights;

    if ( useBoundaryTPE ) {
      BoundaryWeights.resize( 7 );
      BoundaryWeights << Config.Energy.elasticWeight,
                        Config.Energy.barycenterWeight,
                        Config.Energy.dirichletWeight,
                        Config.Energy.obstacleWeight,
                        Config.Energy.tpeWeight,
                        Config.BoundaryTPE.weight,
                        Config.Energy.rotationWeight;

      E = std::make_unique<AdditionOp<DefaultConfigurator>>(
          BoundaryWeights,
          Eelast, Ebary, Edir, Eobs, Etpe, *Ebdry, Erot
      );

      DE = std::make_unique<AdditionGradient<DefaultConfigurator>>(
          BoundaryWeights,
          DEelast, DEbary, DEdir, DEobs, DEtpe, *DEbdry, DErot
      );
    }
    else {
      E = std::make_unique<AdditionOp<DefaultConfigurator>>(
          Weights,
          Eelast, Ebary, Edir, Eobs, Etpe, Erot
      );

      DE = std::make_unique<AdditionGradient<DefaultConfigurator>>(
          Weights,
          DEelast, DEbary, DEdir, DEobs, DEtpe, DErot
      );
    }
    //endregion

    //region 2nd order quadratic models (= Hessian approximations)
    // Elastic metric for path
    L2PathMetric<DefaultConfigurator> L2EM( ElasticMetric, Config.numSteps );
    OperatorL2PathMetric<DefaultConfigurator> opL2EM( opElasticMetric, Config.numSteps );

    // Sobolev-Slobodeckij metric for path
    OperatorL2PathMetric<DefaultConfigurator> opL2SSM( SSMop, Config.numSteps );

    // Tracking metric (a constant matrix since tracking term is quadratic, hence Hessian of penalty to fixed positions is used)
    PointPositionPenaltyHessian<DefaultConfigurator> D2Fdir( Topology, dirichletIndices, Vertices_Start );
    PointPositionPenaltyHessianOperator<DefaultConfigurator> opD2Fdir( Topology, dirichletIndices, Vertices_Start );

    // L2-in-time Elastic + L2-in-time Hs + Dirichlet
    // AdditionHessian<DefaultConfigurator> L2comb( Weights, L2EM, L2SSM, D2Fdir ); -- matrix SSM not available anymore
    LinearlyCombinedMaps<DefaultConfigurator> opL2comb( Weights, opL2EM, opD2Ebary, opD2Edir, opD2Eobs, opL2SSM );

    // Elastic Path Energy Hessian + L2-in-time Hs + Dirichlet
    // AdditionHessian<DefaultConfigurator> EH_L2SSM( Weights, D2Eelast, L2SSM, D2Fdir ); -- matrix SSM not available anymore
    LinearlyCombinedMaps<DefaultConfigurator> opEH_L2SSM( Weights, D2Eop, opD2Ebary, opD2Edir, opD2Eobs, opL2SSM );

    // Elastic Path Energy Hessian + approximate TPE Hessian + Dirichlet
    LinearlyCombinedMaps<DefaultConfigurator> opEH_RD2Etp( Weights, D2Eop, opD2Ebary, opD2Edir, opD2Eobs, opRD2Etpe );
    AdditionHessian<DefaultConfigurator> EH_RD2Etp( Weights, D2Eelast, D2Ebary, D2Edir, D2Eobs, RD2Etpe );

    // Local regularized metric, i.e. elastic metric + Hessian of point constraint energy
    LinearlyCombinedMaps<DefaultConfigurator> regMetric( Weights, opElasticMetric, opD2Fdir );

    // Dirichlet boundary for preconditioners if no soft penalty is used
    std::vector<int> preconditionerMask;
    if ( Config.Energy.dirichletWeight == 0. )
      preconditionerMask = Config.dirichletVertices;

    InverseOperatorL2PathMetric<DefaultConfigurator> Pre( regMetric, Config.numSteps, preconditionerMask );
    //endregion


    //region Initial path
//    const RealType tau = 1. / Config.numSteps;
    if ( level == 0 ) {
      Path.resize(( Config.numSteps - 1 ) * 3 * numVertices );
      if ( Config.initFiles.empty()) {
        for ( int k = 0; k < Config.numSteps - 1; k++ )
          Path.segment( k * 3 * numVertices, 3 * numVertices ) =
                  k < Config.numSteps / 2 ? Vertices_Start : Vertices_End;
      }
      else {
        for ( int k = 0; k < Config.numSteps - 1; k++ )
          Path.segment( k * 3 * numVertices, 3 * numVertices ) = Vertices_Init[k];
      }
    }
    else {
      VectorType newPath(( Config.numSteps - 1 ) * 3 * numVertices );
      for ( int k = 0; k < (Config.numSteps / 2) - 1; k++ ) {
        newPath.segment( 2 * k * 3 * numVertices, 3 * numVertices ) = Path.segment( k * 3 * numVertices,
                                                                                    3 * numVertices );
        newPath.segment(( 2 * k + 1 ) * 3 * numVertices, 3 * numVertices ) = Path.segment( k * 3 * numVertices,
                                                                                           3 * numVertices );
      }
      newPath.tail( 3 * numVertices ) = Path.tail( 3 * numVertices );
      Path.resize(( Config.numSteps - 1 ) * 3 * numVertices );
      Path = newPath;
    }

    auto printBoundaryMaxPairs = [&](const std::string &stageLabel) {
      if (!boundaryTPEAvailable) {
        return;
      }

      const std::string csvFile =
          outputPrefix
          + "boundary_max_pairs_"
          + stageLabel
          + "_level_"
          + std::to_string(level)
          + ".csv";

      std::ofstream csv(csvFile);

      csv << std::scientific << std::setprecision(16);

      csv << "stage,level,curve,curveIndex,valid,"
          << "edgeI,edgeI_v0,edgeI_v1,"
          << "edgeJ,edgeJ_v0,edgeJ_v1,"
          << "midDist,kernel,contribution"
          << std::endl;

      auto printOne = [&](const std::string &curveName,
                          int curveIndex,
                          const VectorType &geometry) {
        auto info = BoundaryTPE->maxPairInfo(geometry);

        if (!info.valid) {
          std::cout << " .. [BoundaryTPE] max pair ["
                    << stageLabel << "][" << curveName
                    << "]: none" << std::endl;

          csv << stageLabel << ","
              << level << ","
              << curveName << ","
              << curveIndex << ","
              << 0 << ","
              << -1 << "," << -1 << "," << -1 << ","
              << -1 << "," << -1 << "," << -1 << ","
              << 0.0 << ","
              << 0.0 << ","
              << 0.0
              << std::endl;

          return;
        }

        std::cout << " .. [BoundaryTPE] max pair ["
                  << stageLabel << "][" << curveName << "] "
                  << "edgeI=" << info.edgeI
                  << "(" << info.edgeI_v0 << "," << info.edgeI_v1 << "), "
                  << "edgeJ=" << info.edgeJ
                  << "(" << info.edgeJ_v0 << "," << info.edgeJ_v1 << "), "
                  << "midDist=" << info.midpointDistance << ", "
                  << "kernel=" << info.kernel << ", "
                  << "contribution=" << info.contribution
                  << std::endl;

        csv << stageLabel << ","
            << level << ","
            << curveName << ","
            << curveIndex << ","
            << 1 << ","
            << info.edgeI << ","
            << info.edgeI_v0 << ","
            << info.edgeI_v1 << ","
            << info.edgeJ << ","
            << info.edgeJ_v0 << ","
            << info.edgeJ_v1 << ","
            << info.midpointDistance << ","
            << info.kernel << ","
            << info.contribution
            << std::endl;
      };

      printOne("curve_0", 0, Vertices_Start);

      for (int k = 0; k < Config.numSteps - 1; ++k) {
        VectorType curveGeometry =
            Path.segment(k * 3 * numVertices, 3 * numVertices);

        printOne(
            "curve_" + std::to_string(k + 1),
            k + 1,
            curveGeometry
        );
      }

      printOne(
          "curve_" + std::to_string(Config.numSteps),
          Config.numSteps,
          Vertices_End
      );

      csv.close();

      std::cout << " .. [BoundaryTPE] max pair CSV saved: "
                << csvFile << std::endl;
    };

    auto getCurveGeometry = [&](int curveIndex) -> VectorType {
      if (curveIndex <= 0) {
        return Vertices_Start;
      }

      if (curveIndex >= Config.numSteps) {
        return Vertices_End;
      }

      return Path.segment(
          (curveIndex - 1) * 3 * numVertices,
          3 * numVertices
      );
    };

    auto findMaxBoundaryTPECurveIndex = [&]() -> int {
      int bestIndex = 0;
      RealType bestValue = -1.;

      for (int curveIndex = 0; curveIndex <= Config.numSteps; ++curveIndex) {
        VectorType geometry = getCurveGeometry(curveIndex);
        RealType value = (*BoundaryTPE)(geometry);

        if (value > bestValue) {
          bestValue = value;
          bestIndex = curveIndex;
        }
      }

      return bestIndex;
    };

    auto printBoundaryPathEnergyCheck = [&](const std::string &stageLabel) {
      if (!boundaryTPEAvailable) {
        return;
      }

      RealType manualSum = 0.;

      for (int k = 0; k < Config.numSteps - 1; ++k) {
        VectorType curve =
            Path.segment(k * 3 * numVertices, 3 * numVertices);

        manualSum += (*BoundaryTPE)(curve);
      }

      const RealType pathEnergy = (*Ebdry)(Path);

      std::cout << " .. [BoundaryTPE] path energy check ["
                << stageLabel << "] "
                << "manualSumIntermediate=" << manualSum
                << ", Ebdry=" << pathEnergy
                << ", diff=" << pathEnergy - manualSum
                << std::endl;
    };

    auto checkBoundaryGradient = [&](const std::string &stageLabel) {
      if (!boundaryTPEAvailable) {
        return;
      }

      if (!Config.BoundaryTPE.checkGradient) {
        return;
      }

      int curveIndex = Config.BoundaryTPE.checkCurveIndex;

      if (curveIndex < 0 || curveIndex > Config.numSteps) {
        curveIndex = findMaxBoundaryTPECurveIndex();
      }

      VectorType geometry = getCurveGeometry(curveIndex);

      RealType energy = (*BoundaryTPE)(geometry);

      VectorType analyticGradient;
      VectorType finiteDiffGradient;

      BoundaryTPE->evaluateGradient(
          geometry,
          analyticGradient
      );

      BoundaryTPE->evaluateFiniteDifferenceGradient(
          geometry,
          finiteDiffGradient,
          Config.BoundaryTPE.fdStep
      );

      VectorType diff = analyticGradient - finiteDiffGradient;

      RealType analyticNorm = analyticGradient.norm();
      RealType finiteDiffNorm = finiteDiffGradient.norm();
      RealType diffNorm = diff.norm();

      RealType denominator = finiteDiffNorm;
      if (denominator < 1.) {
        denominator = 1.;
      }

      RealType relativeError = diffNorm / denominator;

      std::cout << " .. [BoundaryTPE] gradient check ["
                << stageLabel
                << "][curve_" << curveIndex << "]"
                << std::endl;

      std::cout << " .... energy           = "
                << energy << std::endl;

      std::cout << " .... fdStep           = "
                << Config.BoundaryTPE.fdStep << std::endl;

      std::cout << " .... analyticNorm     = "
                << analyticNorm << std::endl;

      std::cout << " .... finiteDiffNorm   = "
                << finiteDiffNorm << std::endl;

      std::cout << " .... differenceNorm   = "
                << diffNorm << std::endl;

      std::cout << " .... relativeError    = "
                << relativeError << std::endl;
    };

    {
      std::cout << " .. Profiling: " << std::endl;
      // -- Energy --
      t_start = std::chrono::high_resolution_clock::now();
      RealType Eval = (*E)( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Energy evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Eelast( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Etpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Edir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Eobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Ebary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Erot( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Rotation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Gradient --
      t_start = std::chrono::high_resolution_clock::now();
      VectorType DEval = (*DE)( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Gradient evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEelast( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEtpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEdir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
      << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEbary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DErot( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Rotation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Hessian --
      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval = opEH_RD2Etp( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian (operator) evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // D2Eop, opRD2Etpe, opD2Edir
      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_elast = D2Eop( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_tpe = opRD2Etpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_dir = opD2Edir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_obs = opD2Eobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_bary = opD2Ebary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;


      t_start = std::chrono::high_resolution_clock::now();
      VectorType Hv = ( *D2Eval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Hv = ( *D2Eval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Preconditioner --
      t_start = std::chrono::high_resolution_clock::now();
      auto Preval = Pre( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Preconditioner (operator) evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      VectorType Pv = ( *Preval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Pre * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Pv = ( *Preval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Pre * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;
    }


    std::cout << " ................................................ " << std::endl;
    std::cout << std::scientific << std::setprecision( 6 );
    std::cout << " .. TPE          = ( ";
    std::cout << TPE( Vertices_Start ) << " ";
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      std::cout << TPE( Path.segment( k * 3 * numVertices, 3 * numVertices )) << " ";
    std::cout << TPE( Vertices_End ) << " )" << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. BoundaryTPE  = ( ";
      std::cout << (*BoundaryTPE)( Vertices_Start ) << " ";
      for ( int k = 0; k < Config.numSteps - 1; k++ ) {
        std::cout << (*BoundaryTPE)(
            Path.segment( k * 3 * numVertices, 3 * numVertices )
        ) << " ";
      }
      std::cout << (*BoundaryTPE)( Vertices_End ) << " )" << std::endl;
    }

    printBoundaryMaxPairs("initial");
    printBoundaryPathEnergyCheck("initial");
    checkBoundaryGradient("initial");
    
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << (*E)( Path ) << std::endl;
    std::cout << " .. Eelast       = " << Eelast( Path ) << std::endl;
    std::cout << " .. Etpe         = " << Etpe( Path ) << std::endl;
    std::cout << " .. Eobs         = " << Eobs( Path ) << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. Ebdry        = " << (*Ebdry)( Path ) << std::endl;
    }
    std::cout << " .. Edir         = " << Edir( Path ) << std::endl;
    std::cout << " .. Ebary        = " << Ebary( Path ) << std::endl;
    std::cout << " .. Erot         = " << Erot( Path ) << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. DE.norm      = " << (*DE)( Path ).norm() << std::endl;
    std::cout << " .. DEelast.norm = " << DEelast( Path ).norm() << std::endl;
    std::cout << " .. DEtpe.norm   = " << DEtpe( Path ).norm() << std::endl;
    std::cout << " .. DEobs.norm   = " << DEobs( Path ).norm() << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. DEbdry.norm  = " << (*DEbdry)( Path ).norm() << std::endl;
    }
    std::cout << " .. DEdir.norm   = " << DEdir( Path ).norm() << std::endl;
    std::cout << " .. DEbary.norm  = " << DEbary( Path ).norm() << std::endl;
    std::cout << " .. DErot.norm   = " << DErot( Path ).norm() << std::endl;
    std::cout << " ................................................ " << std::endl;
    VectorType pathGradient = -(*DE)( Path );
    applyMaskToVector( fixedVariables, pathGradient );

    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_init_0.vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_init_0.ply" );
    }
    for ( int k = 0; k < Config.numSteps - 1; k++ ) {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = pathGradient.segment( k * 3 * numVertices, 3 * numVertices );
      data["TPG"] = -TPG( Path.segment( k * 3 * numVertices, 3 * numVertices ));
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( k + 1 ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( k + 1 ) + ".ply" );
    }
    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( Config.numSteps ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( Config.numSteps ) + ".ply" );
    }

    //endregion

    auto savePathAsPLY = [&]( const VectorType &x,
                              const boost::filesystem::path &folder ) {
      boost::filesystem::create_directories( folder );

      // start mesh
      saveAsPLY<VectorType>(
        Topology,
        Vertices_Start,
        ( folder / "curve_0.ply" ).string()
      );

      // intermediate meshes
      for ( int k = 0; k < Config.numSteps - 1; k++ ) {
        saveAsPLY<VectorType>(
          Topology,
          x.segment( k * 3 * numVertices, 3 * numVertices ),
          ( folder / ( "curve_" + std::to_string( k + 1 ) + ".ply" ) ).string()
        );
      }

      // end mesh
      saveAsPLY<VectorType>(
        Topology,
        Vertices_End,
        ( folder / ( "curve_" + std::to_string( Config.numSteps ) + ".ply" ) ).string()
      );
    };

    //region Optimization
    if ( Config.Optimization.Type == GRADIENT_DESCENT ) {
      GradientDescent<DefaultConfigurator> Solver( *E, *DE, Config.Optimization.maxNumIterations, 1.e-8, ARMIJO, SHOW_ALL,
                                                   0.1, Config.Optimization.minStepsize,
                                                   Config.Optimization.maxStepsize );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == BFGS ) {
      QuasiNewtonBFGS<DefaultConfigurator> Solver( *E, *DE, Config.Optimization.maxNumIterations, 1.e-8, ARMIJO, 50,
                                                   SHOW_ALL, 0.1, Config.Optimization.minStepsize,
                                                   Config.Optimization.maxStepsize );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == PRECONDITIONED ) {
      ObjectiveHessian<DefaultConfigurator> *Preconditioner = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Preconditioner = &D2Eelast;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Preconditioner = &L2EM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Preconditioner = &EH_RD2Etp;
      }

      LineSearchNewton<DefaultConfigurator> Solver( *E, *DE, *Preconditioner, 1.e-8, Config.Optimization.maxNumIterations,
                                                    SHOW_ALL );
      Solver.setParameter( "minimal_stepsize", Config.Optimization.minStepsize );
      Solver.setParameter( "maximal_stepsize", Config.Optimization.maxStepsize );
      Solver.setParameter( "tau_increase", 5. );
      Solver.setParameter( "reduced_direction", false );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == NEWTONCG ) {
      MapToLinOp<DefaultConfigurator> *Hess = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Hess = &D2Eop;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Hess = &opL2EM;
      }
      else if ( Config.Optimization.Preconditioner == L2_COMBINED_METRIC ) {
        Hess = &opL2comb;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_L2_SSM ) {
        Hess = &opEH_L2SSM;
      }
      else if ( Config.Optimization.Preconditioner == L2_SSM ) {
        Hess = &opL2SSM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Hess = &opEH_RD2Etp;
      }

      NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( *E, *DE, *Hess, Pre, 1.e-8,
                                                           Config.Optimization.maxNumIterations,
                                                           SHOW_ALL );
      Solver.setParameter( "cg_iterations", 2000 );
      Solver.setBoundaryMask( Config.dirichletVertices );
      Solver.setParameter( "minimal_stepsize", Config.Optimization.minStepsize );
      Solver.setParameter( "maximal_stepsize", Config.Optimization.maxStepsize );

      if ( Config.Energy.dirichletWeight == 0. )
        Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == TRUSTREGION ) {
      MapToLinOp<DefaultConfigurator> *Hess = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Hess = &D2Eop;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Hess = &opL2EM;
      }
      else if ( Config.Optimization.Preconditioner == L2_COMBINED_METRIC ) {
        Hess = &opL2comb;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_L2_SSM ) {
        Hess = &opEH_L2SSM;
      }
      else if ( Config.Optimization.Preconditioner == L2_SSM ) {
        Hess = &opL2SSM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Hess = &opEH_RD2Etp;
      }

      NewOpt::SteihaugCGMethod<DefaultConfigurator>::resetTimers();
      InverseOperatorL2PathMetric<DefaultConfigurator>::resetTimers();

      if ( Config.TPE.Type == SCARY )
        DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>::resetTimers();
      else if ( Config.TPE.Type == SPOOKY )
        if ( Config.TPE.useAdaptivity )
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>::resetTimers();
        else
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>::resetTimers();

      DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy>::resetTimers();

      DiscretePathEnergy<DefaultConfigurator>::resetTimers();
      DiscretePathEnergyGradient<DefaultConfigurator>::resetTimers();
      DiscretePathEnergyHessian<DefaultConfigurator>::resetTimers();
      OperatorPathEnergyHessian<DefaultConfigurator>::resetTimers();

      NewOpt::TrustRegionNewton<DefaultConfigurator> Solver( *E, *DE, *Hess, 1., 100., 1e-8,
                                                          Config.Optimization.maxNumIterations,
                                                          5000 );
      Solver.setParameter( "minimal_reduction", Config.Optimization.minReduction );
      Solver.setParameter( "trsolver__maximum_iterations", 250 );
      Solver.setParameter( "preconditioner", NewOpt::TrustRegionNewton<DefaultConfigurator>::PROVIDED );
      Solver.setPreconditioner( Pre );

      if ( Config.Energy.dirichletWeight == 0. )
        Solver.setBoundaryMask( fixedVariables );

      std::function<void( int, const VectorType &, const RealType &, const VectorType & )> callbackFct =
      [&]( int i, const VectorType &x, const RealType &F, const VectorType &grad_F ) {

        // Ctrl+C가 들어온 경우: intermediateEvery와 무관하게 즉시 저장
        if ( g_interruptRequested ) {
          boost::filesystem::path interruptFolder =
            boost::filesystem::path( Config.outputFolder )
            / Config.intermediateFolder
            / ( "level_" + std::to_string( level ) )
            / ( "interrupted_iter_" + std::to_string( i ) );

          savePathAsPLY( x, interruptFolder );

          std::ofstream infoFile( ( interruptFolder / "_info.txt" ).string() );
          infoFile << "level: " << level << std::endl;
          infoFile << "iteration: " << i << std::endl;
          infoFile << "energy: " << F << std::endl;
          infoFile << "gradient_norm: " << grad_F.norm() << std::endl;
          infoFile << "reason: interrupted by SIGINT" << std::endl;

          std::cout << std::endl;
          std::cout << " .. Interrupt requested. Saved current path: "
                    << interruptFolder.string() << std::endl;

          Path = x;

          throw std::runtime_error( "USER_INTERRUPT" );
        }

        // 일반 intermediate 저장을 꺼둔 경우에는 여기서 return
        if ( !Config.saveIntermediate )
          return;

        if ( Config.intermediateEvery <= 0 )
          return;

        if ( i % Config.intermediateEvery != 0 )
          return;

        boost::filesystem::path checkpointFolder =
          boost::filesystem::path( Config.outputFolder )
          / Config.intermediateFolder
          / ( "level_" + std::to_string( level ) )
          / ( "iter_" + std::to_string( i ) );

        savePathAsPLY( x, checkpointFolder );

        std::ofstream infoFile( ( checkpointFolder / "_info.txt" ).string() );
        infoFile << "level: " << level << std::endl;
        infoFile << "iteration: " << i << std::endl;
        infoFile << "energy: " << F << std::endl;
        infoFile << "gradient_norm: " << grad_F.norm() << std::endl;

        std::cout << " .. Saved intermediate path: "
                  << checkpointFolder.string() << std::endl;
      };

      Solver.addCallbackFunction( callbackFct );

      t_start = std::chrono::high_resolution_clock::now();
      try {
        Solver.solve( Path, Path );
      }
      catch ( const std::runtime_error &e ) {
        if ( std::string( e.what() ) == "USER_INTERRUPT" ) {
          std::cout << " .. Optimization stopped by user interrupt." << std::endl;
        }
        else {
          throw;
        }
      }
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;


      std::cout << std::scientific << std::setprecision( 6 ) << " .. Result: "
                << Config.numSteps << ","
                << numVertices << ","
                << Solver.Status().Iteration << ","
                << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count() << ","
                << Solver.Status().additionalIterations.at( "Subproblem" ) << ","
                << Solver.Status().additionalTimings.at( "Subproblem" ) << ","
                << Solver.Status().additionalTimings.at( "Preconditioner" ) << ","
                << Solver.Status().additionalTimings.at( "Evaluation" ) << ","
                << std::endl;


      std::cout << " ................................................ " << std::endl;

      NewOpt::SteihaugCGMethod<DefaultConfigurator>::printTimings();
      InverseOperatorL2PathMetric<DefaultConfigurator>::printTimings();

      if ( Config.TPE.Type == SCARY )
        DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>::printTimings();
      else if ( Config.TPE.Type == SPOOKY )
        if ( Config.TPE.useAdaptivity )
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>::printTimings();
        else
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>::printTimings();

      DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy>::printTimings();

      DiscretePathEnergy<DefaultConfigurator>::printTimings();
      DiscretePathEnergyGradient<DefaultConfigurator>::printTimings();
      DiscretePathEnergyHessian<DefaultConfigurator>::printTimings();
      OperatorPathEnergyHessian<DefaultConfigurator>::printTimings();
    }
    //endregion

    //region Evaluation and output
    std::cout << " ................................................ " << std::endl;
    std::cout << std::scientific << std::setprecision( 6 ) << std::endl;
    std::cout << " .. TPE          = ( ";
    std::cout << TPE( Vertices_Start ) << " ";
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      std::cout << TPE( Path.segment( k * 3 * numVertices, 3 * numVertices )) << " ";
    std::cout << TPE( Vertices_End ) << " )" << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. BoundaryTPE  = ( ";
      std::cout << (*BoundaryTPE)( Vertices_Start ) << " ";
      for ( int k = 0; k < Config.numSteps - 1; k++ ) {
        std::cout << (*BoundaryTPE)(
            Path.segment( k * 3 * numVertices, 3 * numVertices )
        ) << " ";
      }
      std::cout << (*BoundaryTPE)( Vertices_End ) << " )" << std::endl;
    }

    printBoundaryMaxPairs("final");
    printBoundaryPathEnergyCheck("final");
    checkBoundaryGradient("final");

    VectorType pathEnergies, pathEnergies_elast, pathEnergies_tpe, pathEnergies_mem, pathEnergies_bend,
        pathEnergies_dir, pathEnergies_obs, pathEnergies_rot, pathEnergies_bary;
    Eelast.evaluateSingleEnergies( Path, pathEnergies_elast );
    Emem.evaluateSingleEnergies( Path, pathEnergies_mem );
    Ebend.evaluateSingleEnergies( Path, pathEnergies_bend );

    if ( Config.TPE.Type == SCARY )
      pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy> *>(
        chosenTPDE.get())->stepEnergies( Path );
    else if ( Config.TPE.Type == SPOOKY )
      if ( Config.TPE.useAdaptivity )
        pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator,
          SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy> *>(chosenTPDE.get())->stepEnergies( Path );
      else
        pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator,
          SpookyTPE::FastMultipoleTangentPointEnergy> *>(chosenTPDE.get())->stepEnergies( Path );

    pathEnergies_obs = TPODE.stepEnergies( Path );

    // Eelast, Ebary, Edir, Eobs, Etpe, Erot
    Edir.evaluateSingleEnergies( Path, pathEnergies_dir );
    pathEnergies_bary = BPE.stepEnergies( Path );
    pathEnergies_rot = RPE.stepEnergies( Path );
    pathEnergies = Config.Energy.elasticWeight * pathEnergies_elast +
                   Config.Energy.tpeWeight * pathEnergies_tpe + Config.Energy.obstacleWeight * pathEnergies_obs +
                   Config.Energy.dirichletWeight * pathEnergies_dir +
                   Config.Energy.barycenterWeight * pathEnergies_bary + Config.Energy.rotationWeight * pathEnergies_rot;

    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << pathEnergies.transpose() << std::endl;
    std::cout << " .. Eelast       = " << pathEnergies_elast.transpose() << std::endl;
    std::cout << " .. Emem         = " << pathEnergies_mem.transpose() << std::endl;
    std::cout << " .. Ebend        = " << pathEnergies_bend.transpose() << std::endl;
    std::cout << " .. Etpe         = " << pathEnergies_tpe.transpose() << std::endl;
    std::cout << " .. Eobs         = " << pathEnergies_obs.transpose() << std::endl;
    std::cout << " .. Edir         = " << pathEnergies_dir.transpose() << std::endl;
    std::cout << " .. Ebary        = " << pathEnergies_bary.transpose() << std::endl;
    std::cout << " .. Erot         = " << pathEnergies_rot.transpose() << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << (*E)( Path ) << std::endl;
    std::cout << " .. Eelast       = " << Eelast( Path ) << std::endl;
    std::cout << " .. Etpe         = " << Etpe( Path ) << std::endl;
    std::cout << " .. Eobs         = " << Eobs( Path ) << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. Ebdry        = " << (*Ebdry)( Path ) << std::endl;
    }
    std::cout << " .. Edir         = " << Edir( Path ) << std::endl;
    std::cout << " .. Ebary        = " << Ebary( Path ) << std::endl;
    std::cout << " .. Erot         = " << Erot( Path ) << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. DE.norm      = " << (*DE)( Path ).norm() << std::endl;
    std::cout << " .. DEelast.norm = " << DEelast( Path ).norm() << std::endl;
    std::cout << " .. DEtpe.norm   = " << DEtpe( Path ).norm() << std::endl;
    std::cout << " .. DEobs.norm   = " << DEobs( Path ).norm() << std::endl;
    if ( boundaryTPEAvailable ) {
      std::cout << " .. DEbdry.norm  = " << (*DEbdry)( Path ).norm() << std::endl;
    }
    std::cout << " .. DEdir.norm   = " << DEdir( Path ).norm() << std::endl;
    std::cout << " .. DEbary.norm  = " << DEbary( Path ).norm() << std::endl;
    std::cout << " .. DErot.norm   = " << DErot( Path ).norm() << std::endl;
    pathGradient = -(*DE)( Path );
    applyMaskToVector( fixedVariables, pathGradient );

    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_0.vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_0.ply" );
    }
    for ( int k = 0; k < Config.numSteps - 1; k++ ) {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = pathGradient.segment( k * 3 * numVertices, 3 * numVertices );
      data["TPG"] = -TPG( Path.segment( k * 3 * numVertices, 3 * numVertices ));
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( k + 1 ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( k + 1 ) + ".ply" );
    }
    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_End );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( Config.numSteps ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( Config.numSteps ) + ".ply" );
    }
    //endregion
  }
  std::cout << " ................................................ " << std::endl;
  std::cout << " - outputPrefix: " << outputPrefix << std::endl;
  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
