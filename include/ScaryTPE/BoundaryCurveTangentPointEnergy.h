#pragma once

#include <cmath>
#include <vector>
#include <limits>

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include "BoundaryUtils.h"

namespace ScaryTPE {

template<typename ConfiguratorType = DefaultConfigurator>
class BoundaryCurveTangentPointEnergy
    : public ObjectiveFunctional<ConfiguratorType> {

public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;

private:
  int m_numVertices;
  std::vector<BoundaryUtils::BoundaryEdge> m_boundaryEdges;

  RealType m_alpha;
  RealType m_beta;
  RealType m_eps;

  struct EdgeSample {
    int edgeIndex = -1;

    int v0 = -1;
    int v1 = -1;

    VecType midpoint;
    VecType tangent;
    RealType length = 0.;
  };

public:
  struct MaxPairInfo {
    bool valid = false;

    int edgeI = -1;
    int edgeJ = -1;

    int edgeI_v0 = -1;
    int edgeI_v1 = -1;
    int edgeJ_v0 = -1;
    int edgeJ_v1 = -1;

    RealType midpointDistance = 0.;
    RealType kernel = 0.;
    RealType contribution = 0.;
  };

public:
  BoundaryCurveTangentPointEnergy(
      const MeshTopologySaver &Topology,
      const std::vector<BoundaryUtils::BoundaryEdge> &boundaryEdges,
      RealType alpha = 6.,
      RealType beta = 12.)
      : m_numVertices(Topology.getNumVertices()),
        m_boundaryEdges(boundaryEdges),
        m_alpha(alpha),
        m_beta(beta),
        m_eps(1.e-12) {}

  void evaluate(const VectorType &Point, RealType &Value) const override {
    Value = computeBoundaryCurveTPE(Point);
  }

  void evaluateGradient(const VectorType &Point, VectorType &Gradient) const override {
    if (Gradient.size() != Point.size()) {
      Gradient.resize(Point.size());
    }

    Gradient.setZero();

    auto samples = buildEdgeSamples(Point);

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
      for (int j = i + 1; j < static_cast<int>(samples.size()); ++j) {
        const auto &a = samples[i];
        const auto &b = samples[j];

        if (shareVertex(a, b)) {
          continue;
        }

        // K_ab = K(x_a, x_b, tangent_a)
        VecType gradKab_dA =
            kernelGradientOneSided(
                a.midpoint,
                b.midpoint,
                a.tangent
            );

        // K_ba = K(x_b, x_a, tangent_b)
        // 이 함수의 반환값은 K_ba를 x_b에 대해 미분한 값
        VecType gradKba_dB =
            kernelGradientOneSided(
                b.midpoint,
                a.midpoint,
                b.tangent
            );

        const RealType pairWeight =
            0.5 * a.length * b.length;

        VecType gradMidA;
        VecType gradMidB;

        for (int d = 0; d < 3; ++d) {
          // E_ij = 0.5 * l_i * l_j * (K_ab + K_ba)
          //
          // dE/d midpoint_a
          // = 0.5*l_i*l_j*( dK_ab/dA - dK_ba/dB )
          gradMidA[d] =
              pairWeight * (gradKab_dA[d] - gradKba_dB[d]);

          gradMidB[d] = -gradMidA[d];
        }

        addMidpointGradientToEdge(Gradient, a, gradMidA);
        addMidpointGradientToEdge(Gradient, b, gradMidB);
      }
    }
  }

  void evaluateFiniteDifferenceGradient(
      const VectorType &Point,
      VectorType &Gradient,
      RealType fdStep = 1.e-6) const {

    Gradient.resize(Point.size());
    Gradient.setZero();

    VectorType xPlus = Point;
    VectorType xMinus = Point;

    for (int i = 0; i < Point.size(); ++i) {
      RealType scale = std::abs(Point[i]);

      if (scale < 1.) {
        scale = 1.;
      }

      RealType h = fdStep * scale;

      xPlus[i] = Point[i] + h;
      xMinus[i] = Point[i] - h;

      RealType fPlus = computeBoundaryCurveTPE(xPlus);
      RealType fMinus = computeBoundaryCurveTPE(xMinus);

      Gradient[i] = (fPlus - fMinus) / (2. * h);

      xPlus[i] = Point[i];
      xMinus[i] = Point[i];
    }
  }

  MaxPairInfo maxPairInfo(const VectorType &Point) const {
    MaxPairInfo info;

    auto samples = buildEdgeSamples(Point);

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
      for (int j = i + 1; j < static_cast<int>(samples.size()); ++j) {
        if (shareVertex(samples[i], samples[j])) {
          continue;
        }

        RealType K = kernelSymmetric(samples[i], samples[j]);

        RealType contribution =
            samples[i].length * samples[j].length * K;

        if (!info.valid || contribution > info.contribution) {
          info.valid = true;

          info.edgeI = samples[i].edgeIndex;
          info.edgeJ = samples[j].edgeIndex;

          info.edgeI_v0 = samples[i].v0;
          info.edgeI_v1 = samples[i].v1;

          info.edgeJ_v0 = samples[j].v0;
          info.edgeJ_v1 = samples[j].v1;

          info.midpointDistance =
              (samples[i].midpoint - samples[j].midpoint).norm();

          info.kernel = K;
          info.contribution = contribution;
        }
      }
    }

    return info;
  }

private:
  RealType dotVec(const VecType &a, const VecType &b) const {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  VecType getPosition(const VectorType &Point, int vertexIndex) const {
    VecType p;

    // GOAST vector layout: X...X | Y...Y | Z...Z
    p[0] = Point[vertexIndex];
    p[1] = Point[m_numVertices + vertexIndex];
    p[2] = Point[2 * m_numVertices + vertexIndex];

    return p;
  }

  bool shareVertex(const EdgeSample &a,
                   const EdgeSample &b) const {
    return a.v0 == b.v0 || a.v0 == b.v1 ||
           a.v1 == b.v0 || a.v1 == b.v1;
  }

  VecType zeroVec() const {
    VecType z;
    z[0] = 0.;
    z[1] = 0.;
    z[2] = 0.;
    return z;
  }

  void addToVertexGradient(
      VectorType &Gradient,
      int vertexIndex,
      const VecType &g,
      RealType scale = 1.) const {

    Gradient[vertexIndex] += scale * g[0];
    Gradient[m_numVertices + vertexIndex] += scale * g[1];
    Gradient[2 * m_numVertices + vertexIndex] += scale * g[2];
  }

  void addMidpointGradientToEdge(
      VectorType &Gradient,
      const EdgeSample &edge,
      const VecType &midpointGradient) const {

    // midpoint = 0.5 * (p0 + p1)
    // 따라서 midpoint에 걸린 gradient를 양 끝 vertex에 절반씩 분배
    addToVertexGradient(Gradient, edge.v0, midpointGradient, 0.5);
    addToVertexGradient(Gradient, edge.v1, midpointGradient, 0.5);
  }

  VecType kernelGradientOneSided(
      const VecType &x,
      const VecType &y,
      const VecType &tangentAtX) const {

    VecType diff = x - y;
    RealType r = diff.norm();

    if (r < m_eps) {
      return zeroVec();
    }

    RealType proj = dotVec(diff, tangentAtX);

    VecType normalProjection = diff;
    for (int d = 0; d < 3; ++d) {
      normalProjection[d] -= proj * tangentAtX[d];
    }

    RealType normalNorm = normalProjection.norm();

    if (normalNorm < m_eps) {
      return zeroVec();
    }

    // K = ||P_perp(diff)||^alpha / ||diff||^beta
    //
    // dK/ddiff =
    // alpha * ||n||^(alpha-2) * n / r^beta
    // - beta * ||n||^alpha * diff / r^(beta+2)

    RealType normalPowAlphaMinus2 =
        std::pow(normalNorm, m_alpha - 2.);

    RealType normalPowAlpha =
        normalPowAlphaMinus2 * normalNorm * normalNorm;

    RealType rPowBeta =
        std::pow(r, m_beta);

    RealType rPowBetaPlus2 =
        rPowBeta * r * r;

    VecType grad;

    for (int d = 0; d < 3; ++d) {
      grad[d] =
          m_alpha * normalPowAlphaMinus2 * normalProjection[d] / rPowBeta
          - m_beta * normalPowAlpha * diff[d] / rPowBetaPlus2;
    }

    return grad;
  }

  RealType kernelOneSided(
      const VecType &x,
      const VecType &y,
      const VecType &tangentAtX) const {

    VecType diff = x - y;
    RealType r = diff.norm();

    if (r < m_eps) {
      return 0.;
    }

    RealType proj = dotVec(diff, tangentAtX);

    VecType normalProjection = diff;
    for (int d = 0; d < 3; ++d) {
      normalProjection[d] -= proj * tangentAtX[d];
    }

    RealType numerator =
        std::pow(normalProjection.norm(), m_alpha);

    RealType denominator =
        std::pow(r, m_beta);

    return numerator / denominator;
  }

  RealType kernelSymmetric(
      const EdgeSample &a,
      const EdgeSample &b) const {

    RealType Kab =
        kernelOneSided(a.midpoint, b.midpoint, a.tangent);

    RealType Kba =
        kernelOneSided(b.midpoint, a.midpoint, b.tangent);

    return 0.5 * (Kab + Kba);
  }

  std::vector<EdgeSample> buildEdgeSamples(const VectorType &Point) const {
    std::vector<EdgeSample> samples;
    samples.reserve(m_boundaryEdges.size());

    for (int edgeIdx = 0;
         edgeIdx < static_cast<int>(m_boundaryEdges.size());
         ++edgeIdx) {

      const auto &edge = m_boundaryEdges[edgeIdx];

      VecType p0 = getPosition(Point, edge.v0);
      VecType p1 = getPosition(Point, edge.v1);

      VecType edgeVector = p1 - p0;
      RealType length = edgeVector.norm();

      if (length < m_eps) {
        continue;
      }

      EdgeSample sample;

      sample.edgeIndex = edgeIdx;
      sample.v0 = edge.v0;
      sample.v1 = edge.v1;

      sample.midpoint = 0.5 * (p0 + p1);
      sample.tangent = edgeVector / length;
      sample.length = length;

      samples.push_back(sample);
    }

    return samples;
  }

  RealType computeBoundaryCurveTPE(const VectorType &Point) const {
    if (m_boundaryEdges.empty()) {
      return 0.;
    }

    auto samples = buildEdgeSamples(Point);

    RealType energy = 0.;

    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
      for (int j = i + 1; j < static_cast<int>(samples.size()); ++j) {
        if (shareVertex(samples[i], samples[j])) {
          continue;
        }

        RealType K = kernelSymmetric(samples[i], samples[j]);

        energy += samples[i].length * samples[j].length * K;
      }
    }

    return energy;
  }
};

} // namespace ScaryTPE