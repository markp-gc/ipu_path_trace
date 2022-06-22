// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>
#include <map>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poprand/RandomGen.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <poprand/codelets.hpp>
#include <popops/codelets.hpp>

#include <light/src/light.hpp>
#include <light/src/jobs.hpp>

#include "ipu_utils.hpp"
#include "io_utils.hpp"

#include <boost/program_options.hpp>

struct IpuPathTraceJob;
using ProgramList = std::vector<poplar::program::Program>;
using IpuJobList = std::vector<IpuPathTraceJob>;

// Class that describes and builds the compute graph and programs for a
// path tracing job. Each job traces rays for a small sub-tile of the
// whole image.
struct IpuPathTraceJob {

  using InputMap = std::map<std::string, poplar::Tensor>;
  using CsMap = std::map<std::string, poplar::ComputeSet>;

  TraceTileJob& jobSpec;

  ~IpuPathTraceJob();

  // Constructor only initialises values that are independent of graph construction.
  // The buildGraph() method constructs the Poplar graph components: graph execution and graph
  // construction are completely separated so that buildGraph() can be skipped when loading a
  // pre-compiled executable.
  IpuPathTraceJob(TraceTileJob& spec,
                 const boost::program_options::variables_map& args,
                 std::size_t core);

  void buildGraph(poplar::Graph& graph,
                  const InputMap& inputs,
                  const CsMap& cs,
                  const boost::program_options::variables_map& args);

  poplar::program::Sequence beginTraceJob() const { return beginSeq; }
  poplar::program::Sequence endTraceJob() const { return endSeq; }

  std::size_t getPixelCount() const { return jobSpec.rows() * jobSpec.cols(); }
  std::size_t getTile() const { return ipuCore; }

  // Hard code the maximum number of samples that a single path
  // could need if it reached maximum depth.
  static constexpr std::size_t numChannels = 3;
  static constexpr std::size_t numRayDirComponents = 2;

private:
  const std::size_t ipuCore; // Core instead of 'Tile' to avoid confusion with the image tiles.

  // Member variables below only get assigned during graph construction
  // (which is skipped if we load a precompiled executable):
  poplar::Tensor randForAntiAliasing;
  poplar::Tensor contributionData;

  poplar::VertexRef  rayGenVertex;
  std::vector<poplar::VertexRef> tracerVertices;
  std::vector<poplar::VertexRef> accumulatorVertices;

  poplar::program::Sequence beginSeq;
  poplar::program::Sequence endSeq;

  // Utility to add a scalar constant to the graph and map it to the IPU tile
  // for this job:
  template <typename T>
  poplar::Tensor
  addScalarConstant(poplar::Graph& graph, poplar::VertexRef v, std::string field, poplar::Type type, T value);

  poplar::Tensor addScalar(poplar::Graph& graph, poplar::VertexRef v, std::string field, poplar::Type type);

  /// Set the tile mapping for all variables and vertices:
  void setTileMappings(poplar::Graph& graph);

  std::string jobStringPrefix() {
    return "core_" + std::to_string(ipuCore) + "/";
  }
};
