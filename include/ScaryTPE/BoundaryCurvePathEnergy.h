#pragma once

#include <stdexcept>
#include <vector>

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include "ScaryTPE/BoundaryCurveTangentPointEnergy.h"
#include "BoundaryUtils.h"

namespace ScaryTPE {

template<typename ConfiguratorType = DefaultConfigurator>
class BoundaryCurvePathEnergy
    : public ObjectiveFunctional<ConfiguratorType> {

public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;

private:
  int m_numSteps;
  int m_numVertices;

  BoundaryCurveTangentPointEnergy<ConfiguratorType> m_curveEnergy;

public:
  BoundaryCurvePathEnergy(
      int numSteps,
      const MeshTopologySaver &Topology,
      const std::vector<BoundaryUtils::BoundaryEdge> &boundaryEdges,
      RealType alpha = 6.,
      RealType beta = 12.)
      : m_numSteps(numSteps),
        m_numVertices(Topology.getNumVertices()),
        m_curveEnergy(Topology, boundaryEdges, alpha, beta) {}

  void evaluate(const VectorType &Path, RealType &Value) const override {
    checkPathSize(Path);

    Value = 0.;

    for (int k = 0; k < m_numSteps - 1; ++k) {
      VectorType curve =
          Path.segment(k * 3 * m_numVertices, 3 * m_numVertices);

      RealType localValue = 0.;
      m_curveEnergy.evaluate(curve, localValue);

      Value += localValue;
    }
  }

  void evaluateGradient(const VectorType &Path, VectorType &Gradient) const override {
    checkPathSize(Path);

    if (Gradient.size() != Path.size()) {
      Gradient.resize(Path.size());
    }

    Gradient.setZero();

    for (int k = 0; k < m_numSteps - 1; ++k) {
      VectorType curve =
          Path.segment(k * 3 * m_numVertices, 3 * m_numVertices);

      VectorType localGradient;
      m_curveEnergy.evaluateGradient(curve, localGradient);

      Gradient.segment(k * 3 * m_numVertices, 3 * m_numVertices) += localGradient;
    }
  }

  VectorType intermediateEnergies(const VectorType &Path) const {
    checkPathSize(Path);

    VectorType values(m_numSteps - 1);

    for (int k = 0; k < m_numSteps - 1; ++k) {
      VectorType curve =
          Path.segment(k * 3 * m_numVertices, 3 * m_numVertices);

      RealType localValue = 0.;
      m_curveEnergy.evaluate(curve, localValue);

      values[k] = localValue;
    }

    return values;
  }

private:
  void checkPathSize(const VectorType &Path) const {
    const int expectedSize = (m_numSteps - 1) * 3 * m_numVertices;

    if (Path.size() != expectedSize) {
      throw std::runtime_error(
          "[BoundaryCurvePathEnergy] Invalid path size.");
    }
  }
};

} // namespace ScaryTPE