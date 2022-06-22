// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "IpuPathTraceJob.hpp"

#include <vector>

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

#include "codelets/TraceRecord.hpp"

#include "ipu_utils.hpp"
#include "io_utils.hpp"

#include <boost/program_options.hpp>

using Interval = std::pair<std::size_t, std::size_t>;

/// Compute the start and end indices that can be used to slice the
/// tile's pixels into chunks that each worker will process:
std::vector<Interval> splitTilePixelsOverWorkers(std::size_t rows, std::size_t cols,
                                                 std::size_t workers) {
  const auto rowsPerWorker = rows / workers;
  const auto leftOvers = rows % workers;
  std::vector<std::size_t> work(workers, rowsPerWorker);

  // Distribute leftovers amongst workers:
  for (auto i = 0u; i < leftOvers; ++i) {
    work[i] += 1;
  }

  // Turn list of rows per worker into element intervals:
  std::vector<Interval> intervals;
  intervals.reserve(workers);
  auto start = 0u;
  for (auto w : work) {
    auto end = start + (cols * w);
    intervals.emplace_back(start, end);
    start = end;
  }
  return intervals;
}

template <typename T>
poplar::Tensor
IpuPathTraceJob::addScalarConstant(poplar::Graph& graph, poplar::VertexRef v, std::string field, poplar::Type type, T value) {
  auto t = graph.addConstant(type, {}, value);
  graph.connect(v[field], t);
  graph.setTileMapping(t, ipuCore);
  return t;
}

poplar::Tensor IpuPathTraceJob::addScalar(poplar::Graph& graph, poplar::VertexRef v, std::string field, poplar::Type type) {
  auto t = graph.addVariable(type, {});
  graph.connect(v[field], t);
  graph.setTileMapping(t, ipuCore);
  return t;
}

IpuPathTraceJob::~IpuPathTraceJob() {}

IpuPathTraceJob::IpuPathTraceJob(TraceTileJob& spec,
                const boost::program_options::variables_map& args,
                std::size_t core)
:
  jobSpec(spec),
  ipuCore(core)
{}

void IpuPathTraceJob::buildGraph(poplar::Graph& graph,
                                 const InputMap& inputs,
                                 const CsMap& cs,
                                 const boost::program_options::variables_map& args) {
  const auto prefix = jobStringPrefix();

  auto genRays = cs.at("gen-rays");  
  rayGenVertex = graph.addVertex(genRays, "GenerateCameraRays");
  graph.setPerfEstimate(rayGenVertex, 1); // Fake perf estimate (for IpuModel only).

  auto traceBuffer = inputs.at("tracebuffer");
  auto cameraRays = inputs.at("primary-rays");
  graph.connect(rayGenVertex["rays"], cameraRays);
  graph.connect(rayGenVertex["traceBuffer"], traceBuffer);
  auto imageWidth  = args.at("width").as<std::uint32_t>();
  auto imageHeight = args.at("height").as<std::uint32_t>();
  addScalarConstant<unsigned>(graph, rayGenVertex, "imageWidth", poplar::UNSIGNED_INT, imageWidth);
  addScalarConstant<unsigned>(graph, rayGenVertex, "imageHeight", poplar::UNSIGNED_INT, imageHeight);

  // Make a local copy of AA scale and FOV:
  poplar::Tensor aaScaleTensor = inputs.at("aa-scale");
  poplar::Tensor fovTensor = inputs.at("fov");
  auto localAaScale =
    graph.addVariable(aaScaleTensor.elementType(), aaScaleTensor.shape(), prefix + "antiAliasScale");
  auto localFov = graph.addVariable(fovTensor.elementType(), fovTensor.shape(), prefix + "fov");
  graph.setTileMapping(localAaScale, ipuCore);
  graph.setTileMapping(localFov, ipuCore);
  graph.connect(rayGenVertex["antiAliasScale"], localAaScale);
  graph.connect(rayGenVertex["fov"], localFov);

  // Make a local copy of azimuthal rotation:
  poplar::Tensor rotation = inputs.at("env-map-rotation");
  auto localRotation =
    graph.addVariable(rotation.elementType(), rotation.shape(), prefix + "hdri_azimuth");
  graph.setTileMapping(localRotation, ipuCore);

  contributionData = inputs.at("path-records");

  // Decide which chunks of the image-tile workers will process:
  const auto workers = graph.getTarget().getNumWorkerContexts();
  auto pathTraceCs = cs.at("path-trace");
  auto preProcEscapedRaysCs = cs.at("pre-process-escaped-rays");
  auto accumulateCs = cs.at("accumulate-lighting");
  tracerVertices.reserve(workers);
  accumulatorVertices.reserve(workers);

  auto traceRecordSize = sizeof(TraceRecord);

  const auto intervals = splitTilePixelsOverWorkers(jobSpec.rows(), jobSpec.cols(), workers);
  for (const auto &interval : intervals) {
    tracerVertices.push_back(graph.addVertex(pathTraceCs, "RayTraceKernel"));
    accumulatorVertices.push_back(graph.addVertex(accumulateCs, "AccumulateContributions"));
    auto& v1 = tracerVertices.back();
    auto& v2 = accumulatorVertices.back();

    addScalarConstant(graph, v1, "refractiveIndex", poplar::HALF,
                      args.at("refractive-index").as<float>());
    addScalarConstant(graph, v1, "rouletteDepth", poplar::UNSIGNED_SHORT,
                      args.at("roulette-depth").as<std::uint16_t>());
    addScalarConstant(graph, v1, "stopProb", poplar::HALF,
                      args.at("stop-prob").as<float>());
    graph.connect(v1["cameraRays"], cameraRays.slice(interval.first * numRayDirComponents, interval.second * numRayDirComponents));

    auto contributionWorkerSlice = contributionData.slice(interval.first, interval.second);
    graph.connect(v1["contributionData"], contributionWorkerSlice);
    graph.connect(v2["contributionData"], contributionWorkerSlice);

    auto traceSlice = traceBuffer.slice(interval.first * traceRecordSize, interval.second * traceRecordSize);
    // Trace slice = {tile-width x tile-height/6 * sizeof(TraceRecord)}
    // Contribution slice = {tile-width x tile-height/6, maxContributions * sizeof(Contribution)}
    graph.connect(v2["traceBuffer"], traceSlice);
  }

  poplar::Tensor uvInput = inputs.at("uv-input");
  auto v3 = graph.addVertex(preProcEscapedRaysCs, "PreProcessEscapedRays");
  graph.connect(v3["contributionData"], contributionData);
  graph.connect(v3["azimuthalOffset"], localRotation);
  graph.connect(v3["u"], uvInput[0][0]);
  graph.connect(v3["v"], uvInput[1][0]);
  graph.setTileMapping(v3, ipuCore);
  graph.setPerfEstimate(v3, 1);

  // Environment lighting is not a tile local operation and is done externally so
  // we must build a separate program to apply env mapping result:
  poplar::Tensor envMapResult = inputs.at("env-map-result");
  auto applyEnvLightingCs = cs.at("apply-env-lighting");
  auto v4 = graph.addVertex(applyEnvLightingCs, "PostProcessEscapedRays");
  graph.connect(v4["contributionData"], contributionData);
  graph.connect(v4["bgr"], envMapResult.squeeze({0}));
  graph.setTileMapping(v4, ipuCore);
  graph.setTileMapping(envMapResult, ipuCore);
  graph.setPerfEstimate(v4, 1);

  setTileMappings(graph);

  // Build the programs:

  // Assign modifiable parameters:
  beginSeq.add(poplar::program::Copy(aaScaleTensor, localAaScale));
  beginSeq.add(poplar::program::Copy(fovTensor, localFov));
  beginSeq.add(poplar::program::Copy(rotation, localRotation));

  // Program to generate the anti-aliasing samples:
  auto aaNoiseType = args.at("aa-noise-type").as<std::string>();
  graph.connect(rayGenVertex["antiAliasNoise"], inputs.at("aa-noise"));

  // Create program to generate the path tracing (primary sample space) samples.
  auto randUniform_0_1 = inputs.at("primary-samples");

  // Need to slice the random numbers between vertices. This is simpler than
  // splitting the pixels because we chose num elements to divide exactly:
  if (randUniform_0_1.numElements() % workers != 0) {
    throw std::logic_error("Size of random data must be divisible by number of workers.");
  }
  auto start = 0u;
  const auto inc = randUniform_0_1.numElements() / workers;
  auto end = inc;
  for (auto i = 0u; i < workers; ++i) {
    graph.connect(tracerVertices[i]["uniform_0_1"], randUniform_0_1.slice(start, end));
    start = end;
    end += inc;
  }
}

/// Set the tile mapping for all variables and vertices:
void IpuPathTraceJob::setTileMappings(poplar::Graph& graph) {
  graph.setTileMapping(rayGenVertex, ipuCore);
  graph.setTileMapping(contributionData, ipuCore);
  for (auto& v : tracerVertices) {
    graph.setTileMapping(v, ipuCore);
    graph.setPerfEstimate(v, 1); // Fake perf estimate (for IpuModel only).
  }
  for (auto& v : accumulatorVertices) {
    graph.setTileMapping(v, ipuCore);
    graph.setPerfEstimate(v, 1); // Fake perf estimate (for IpuModel only).
  }
}
