// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "IpuPathTraceJob.hpp"

#include <vector>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

#include <light/src/jobs.hpp>
#include <light/src/light.hpp>

#include "codelets/TraceRecord.hpp"

#include "io_utils.hpp"
#include "ipu_utils.hpp"

#include <boost/program_options.hpp>

using Interval = std::pair<std::size_t, std::size_t>;

/// Compute the start and end indices that can be used to slice the
/// tile's pixels into chunks that each worker will process:
std::vector<Interval> splitTilePixelsOverWorkers(std::size_t pixelCount, std::size_t workers) {
  const auto raysPerWorker = pixelCount / workers;
  const auto leftOvers = pixelCount % workers;
  std::vector<std::size_t> work(workers, raysPerWorker);
  ipu_utils::logger()->trace("Worker split: total rays: {} rays per-worker: {} leftovers: {}", pixelCount, raysPerWorker, leftOvers);

  // Distribute leftovers amongst workers:
  for (auto i = 0u; i < leftOvers; ++i) {
    work[i] += 1;
  }

  // Turn list of rows per worker into element intervals:
  std::vector<Interval> intervals;
  intervals.reserve(workers);
  auto start = 0u;
  for (auto w : work) {
    auto end = start + w;
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

IpuPathTraceJob::IpuPathTraceJob(std::size_t maxRayCount,
                                 const boost::program_options::variables_map& args,
                                 std::size_t core)
    : maxPixelCount(maxRayCount),
      ipuCore(core) {}

void IpuPathTraceJob::buildGraph(poplar::Graph& graph,
                                 const InputMap& inputs,
                                 const CsMap& cs,
                                 const boost::program_options::variables_map& args) {
  const auto prefix = jobStringPrefix();

  auto genRays = cs.at("gen-rays");
  rayGenVertex = graph.addVertex(genRays, "GenerateCameraRays");
  graph.setPerfEstimate(rayGenVertex, 1);  // Fake perf estimate (for IpuModel only).

  auto traceBuffer = inputs.at("tracebuffer");
  auto cameraRays = inputs.at("primary-rays");
  graph.connect(rayGenVertex["rays"], cameraRays);
  graph.connect(rayGenVertex["traceBuffer"], traceBuffer);
  auto imageWidth = args.at("width").as<std::uint32_t>();
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

  // Local copies for exposure settings also:
  poplar::Tensor exposureTensor = inputs.at("exposure");
  poplar::Tensor gammaTensor = inputs.at("gamma");
  auto localExposure = graph.addVariable(exposureTensor.elementType(), exposureTensor.shape(), prefix + "exposure");
  auto localGamma = graph.addVariable(gammaTensor.elementType(), gammaTensor.shape(), prefix + "gamma");
  graph.setTileMapping(localExposure, ipuCore);
  graph.setTileMapping(localGamma, ipuCore);

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
  tracerVertices.reserve(workers);

  const auto intervals = splitTilePixelsOverWorkers(getPixelCount(), workers);
  for (const auto& interval : intervals) {
    tracerVertices.push_back(graph.addVertex(pathTraceCs, "RayTraceKernel"));
    auto& v1 = tracerVertices.back();

    graph.connect(v1["cameraRays"], cameraRays.slice(interval.first * numRayDirComponents, interval.second * numRayDirComponents));

    auto contributionWorkerSlice = contributionData.slice(interval.first, interval.second);
    graph.connect(v1["contributionData"], contributionWorkerSlice);
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
  graph.connect(v4["bgr"], envMapResult.squeeze({0}));
  graph.connect(v4["traceBuffer"], traceBuffer);
  graph.connect(v4["exposure"], localExposure);
  graph.connect(v4["gamma"], localGamma);
  graph.setInitialValue(v4["id"], ipuCore);
  graph.setTileMapping(v4, ipuCore);
  graph.setTileMapping(envMapResult, ipuCore);
  graph.setPerfEstimate(v4, 1);

  setTileMappings(graph);

  // Build the programs:

  // Assign modifiable parameters:
  beginSeq.add(poplar::program::Copy(aaScaleTensor, localAaScale));
  beginSeq.add(poplar::program::Copy(fovTensor, localFov));
  beginSeq.add(poplar::program::Copy(rotation, localRotation));
  beginSeq.add(poplar::program::Copy(exposureTensor, localExposure));
  beginSeq.add(poplar::program::Copy(gammaTensor, localGamma));
}

/// Set the tile mapping for all variables and vertices:
void IpuPathTraceJob::setTileMappings(poplar::Graph& graph) {
  graph.setTileMapping(rayGenVertex, ipuCore);
  graph.setTileMapping(contributionData, ipuCore);
  for (auto& v : tracerVertices) {
    graph.setTileMapping(v, ipuCore);
    graph.setPerfEstimate(v, 1);  // Fake perf estimate (for IpuModel only).
  }
}
