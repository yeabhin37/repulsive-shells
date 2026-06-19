#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/IO/Options.hh>

namespace BoundaryUtils {

struct BoundaryEdge {
  int v0;
  int v1;
};

struct BoundaryData {
  int numMeshVertices = 0;
  int numMeshEdges = 0;
  int numMeshFaces = 0;

  std::vector<BoundaryEdge> edges;
  std::vector<int> vertices;
  std::vector<std::vector<int>> components;

  bool hasBoundary() const {
    return !edges.empty();
  }

  int numBoundaryEdges() const {
    return static_cast<int>(edges.size());
  }

  int numBoundaryVertices() const {
    return static_cast<int>(vertices.size());
  }

  int numBoundaryComponents() const {
    return static_cast<int>(components.size());
  }
};

inline std::vector<std::vector<int>>
computeBoundaryComponents(const std::vector<BoundaryEdge> &edges) {
  std::unordered_map<int, std::vector<int>> adj;

  for (const auto &e : edges) {
    adj[e.v0].push_back(e.v1);
    adj[e.v1].push_back(e.v0);
  }

  std::set<int> visited;
  std::vector<std::vector<int>> components;

  for (const auto &[start, _] : adj) {
    if (visited.count(start))
      continue;

    std::vector<int> component;
    std::vector<int> stack = {start};
    visited.insert(start);

    while (!stack.empty()) {
      int v = stack.back();
      stack.pop_back();
      component.push_back(v);

      for (int nb : adj[v]) {
        if (!visited.count(nb)) {
          visited.insert(nb);
          stack.push_back(nb);
        }
      }
    }

    std::sort(component.begin(), component.end());
    components.push_back(component);
  }

  return components;
}

template <typename MeshType>
BoundaryData extractBoundaryData(const MeshType &mesh) {
  BoundaryData data;

  data.numMeshVertices = static_cast<int>(mesh.n_vertices());
  data.numMeshEdges = static_cast<int>(mesh.n_edges());
  data.numMeshFaces = static_cast<int>(mesh.n_faces());

  std::set<int> boundaryVertexSet;

  for (auto eh : mesh.edges()) {
    if (!mesh.is_boundary(eh))
      continue;

    auto heh0 = mesh.halfedge_handle(eh, 0);
    auto heh1 = mesh.halfedge_handle(eh, 1);

    // boundary halfedge를 우선 사용해서 boundary 방향성을 어느 정도 보존
    auto bheh = mesh.is_boundary(heh0) ? heh0 : heh1;

    int from = mesh.from_vertex_handle(bheh).idx();
    int to = mesh.to_vertex_handle(bheh).idx();

    data.edges.push_back({from, to});
    boundaryVertexSet.insert(from);
    boundaryVertexSet.insert(to);
  }

  data.vertices.assign(boundaryVertexSet.begin(), boundaryVertexSet.end());
  data.components = computeBoundaryComponents(data.edges);

  return data;
}

inline void printBoundaryInfo(
    const BoundaryData &data,
    std::ostream &out = std::cout,
    const std::string &label = "") {

  out << " .. [Boundary]";

  if (!label.empty())
    out << " " << label;

  out << std::endl;

  out << " .. [Boundary] mesh vertices = "
      << data.numMeshVertices << std::endl;

  out << " .. [Boundary] mesh edges    = "
      << data.numMeshEdges << std::endl;

  out << " .. [Boundary] mesh faces    = "
      << data.numMeshFaces << std::endl;

  if (data.hasBoundary()) {
    out << " .. [Boundary] Boundary detected: YES" << std::endl;
    out << " .. [Boundary] boundary edges      = "
        << data.numBoundaryEdges() << std::endl;
    out << " .. [Boundary] boundary vertices   = "
        << data.numBoundaryVertices() << std::endl;
    out << " .. [Boundary] boundary components = "
        << data.numBoundaryComponents() << std::endl;
  } else {
    out << " .. [Boundary] Boundary detected: NO" << std::endl;
    out << " .. [Boundary] boundary edges      = 0" << std::endl;
    out << " .. [Boundary] boundary vertices   = 0" << std::endl;
    out << " .. [Boundary] boundary components = 0" << std::endl;
  }
}

template <typename MeshType>
void writeBoundaryEdgesOBJ(
    const MeshType &mesh,
    const BoundaryData &data,
    const std::string &path) {

  std::ofstream out(path);

  if (!out.is_open())
    throw std::runtime_error("Failed to open file: " + path);

  out << "# Boundary edges extracted from mesh\n";
  out << "# boundary edges: " << data.numBoundaryEdges() << "\n";

  for (auto vh : mesh.vertices()) {
    auto p = mesh.point(vh);
    out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
  }

  for (const auto &e : data.edges) {
    // OBJ index is 1-based
    out << "l " << e.v0 + 1 << " " << e.v1 + 1 << "\n";
  }
}

template <typename MeshType>
void writeBoundaryEdgesCSV(
    const MeshType &mesh,
    const BoundaryData &data,
    const std::string &path) {

  std::ofstream out(path);

  if (!out.is_open())
    throw std::runtime_error("Failed to open file: " + path);

  out << "edge_id,v0,v1,x0,y0,z0,x1,y1,z1\n";

  for (int i = 0; i < static_cast<int>(data.edges.size()); ++i) {
    const auto &e = data.edges[i];

    auto vh0 = typename MeshType::VertexHandle(e.v0);
    auto vh1 = typename MeshType::VertexHandle(e.v1);

    auto p0 = mesh.point(vh0);
    auto p1 = mesh.point(vh1);

    out << i << ","
        << e.v0 << "," << e.v1 << ","
        << p0[0] << "," << p0[1] << "," << p0[2] << ","
        << p1[0] << "," << p1[1] << "," << p1[2] << "\n";
  }
}

template <typename MeshType>
void writeBoundaryVerticesCSV(
    const MeshType &mesh,
    const BoundaryData &data,
    const std::string &path) {

  std::ofstream out(path);

  if (!out.is_open())
    throw std::runtime_error("Failed to open file: " + path);

  out << "vertex_id,x,y,z\n";

  for (int v : data.vertices) {
    auto vh = typename MeshType::VertexHandle(v);
    auto p = mesh.point(vh);

    out << v << ","
        << p[0] << "," << p[1] << "," << p[2] << "\n";
  }
}

template <typename MeshType>
void writeBoundaryMarkedPLY(
    const MeshType &mesh,
    const BoundaryData &data,
    const std::string &path) {

  MeshType outMesh(mesh);
  outMesh.request_vertex_colors();

  using Color = typename MeshType::Color;

  std::set<int> boundaryVertexSet(data.vertices.begin(), data.vertices.end());

  for (auto vh : outMesh.vertices()) {
    if (boundaryVertexSet.count(vh.idx())) {
      outMesh.set_color(vh, Color(255, 0, 0, 255));     // boundary: red
    } else {
      outMesh.set_color(vh, Color(180, 180, 180, 255)); // non-boundary: gray
    }
  }

  OpenMesh::IO::Options wopt = OpenMesh::IO::Options::Default;
  wopt += OpenMesh::IO::Options::VertexColor;

  if (!OpenMesh::IO::write_mesh(outMesh, path, wopt, 20))
    throw std::runtime_error("Failed to write file: " + path);
}

template <typename MeshType>
void writeBoundaryDebugFiles(
    const MeshType &mesh,
    const BoundaryData &data,
    const std::string &outputPrefix,
    const std::string &filePrefix = "") {
    
    std::string fullPrefix = outputPrefix;
    if (!filePrefix.empty()) fullPrefix += filePrefix + "_";

  writeBoundaryEdgesOBJ(
      mesh,
      data,
      fullPrefix + "boundary_edges.obj");

  writeBoundaryEdgesCSV(
      mesh,
      data,
      fullPrefix + "boundary_edges.csv");

  writeBoundaryVerticesCSV(
      mesh,
      data,
      fullPrefix + "boundary_vertices.csv");

  writeBoundaryMarkedPLY(
      mesh,
      data,
      fullPrefix + "boundary_marked.ply");
}

} // namespace BoundaryUtils