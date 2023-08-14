// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "PathTracerApp.hpp"

#include "AsyncTask.hpp"
#include "codelets/TraceRecord.hpp"
#include "ipu_utils.hpp"
#include "shard_utils.hpp"

#include <poplar/CycleCount.hpp>
#include <popops/Loop.hpp>

#include <light/src/jobs.hpp>

#include <poplar/OptionFlags.hpp>
#include <poplin/MatMul.hpp>

/// Adjust samples per pixel to be multiple of samples per ipu step:
std::size_t roundSamplesPerPixel(std::size_t samplesPerPixel,
                                 std::size_t samplesPerIpuStep) {
  if (samplesPerPixel % samplesPerIpuStep) {
    samplesPerPixel += samplesPerIpuStep - (samplesPerPixel % samplesPerIpuStep);
    ipu_utils::logger()->info("Rounding SPP to next multiple of {}  (Rounded SPP :=  {})",
                              samplesPerIpuStep, samplesPerPixel);
  }
  return samplesPerPixel;
}

poplar::Tensor buildAaNoise(poplar::Graph& graph, poplar::Tensor& layoutTensor, poplar::program::Sequence& prog, std::string aaNoiseType, std::string debugString) {
  if (aaNoiseType == "uniform") {
    return poprand::uniform(
        graph, nullptr, 0u, layoutTensor, poplar::HALF, -1.f, 1.f,
        prog, debugString);
  } else if (aaNoiseType == "normal") {
    return poprand::normal(
        graph, nullptr, 0u, layoutTensor, poplar::HALF, 0.f, 1.f,
        prog, debugString);
  } else if (aaNoiseType == "truncated-normal") {
    return poprand::truncatedNormal(
        graph, nullptr, 0u, layoutTensor, poplar::HALF, 0.f, 1.f, 3.f,
        prog, debugString);
  } else {
    throw std::runtime_error("Invalid AA noise type: " + aaNoiseType);
  }
}

PathTracerApp::PathTracerApp()
    : traceChannel("ipu_path_tracer"),
      seedTensor("seed"),
      aaScaleTensor("anti_alias_scale"),
      fovTensor("field_of_view"),
      azimuthRotation("hdri_azimuth"),
      deviceSampleLimit("on_device_sample_limit"),
      nifCycleCount("nif_cycle_count"),
      pathTraceCycleCount("path_trace_cycle_count"),
      iterationCycles("iter_cycle_count"),
      traceBuffer("trace_buffer"),
      exposureTensor("ipu_exposure"),
      gammaTensor("ipu_gamma")
{}

void PathTracerApp::init(const boost::program_options::variables_map& options) {
  args = options;
  samplesPerPixel = args.at("samples").as<std::uint32_t>();
  samplesPerIpuStep = args.at("samples-per-step").as<std::uint32_t>();
  samplesPerPixel = roundSamplesPerPixel(samplesPerPixel, samplesPerIpuStep);

  // Read the metadata saved with the model:
  auto numIpus = args.at("ipus").as<std::size_t>();
  auto assetPath = args.at("assets").as<std::string>();
  if (!loadNifModels(numIpus, assetPath)) {
    throw std::runtime_error("Could not load NIF model.");
  }
}

ipu_utils::RuntimeConfig PathTracerApp::getRuntimeConfig() const {
  auto exeName = args.at("save-exe").as<std::string>();
  if (exeName.empty()) {
    exeName = args.at("load-exe").as<std::string>();
    ;
  }

  bool compileOnly = args.at("compile-only").as<bool>();
  bool deferAttach = args.at("defer-attach").as<bool>();

  return ipu_utils::RuntimeConfig{
      args.at("ipus").as<std::size_t>(),
      1u,  // Number of replicas
      exeName,
      args.at("model").as<bool>(),
      !args.at("save-exe").as<std::string>().empty(),
      !args.at("load-exe").as<std::string>().empty(),
      compileOnly,
      compileOnly || deferAttach};
}

poplar::Tensor PathTracerApp::createNifInput(poplar::Graph& g, std::size_t numJobsInBatch, std::size_t pixelsPerJob) {
  auto uvInput = g.addVariable(poplar::FLOAT, {2, numJobsInBatch, pixelsPerJob}, "envmap_input_uv");

  // Need to set tile mapping for input before we can use it:
  for (auto j = 0u; j < numJobsInBatch; ++j) {
    auto uvInputSlice = uvInput.slice(j, j + 1, 1);
    g.setTileMapping(uvInputSlice, ipuJobs[j].getTile());
  }
  return uvInput;
}

bool PathTracerApp::loadNifModels(std::size_t numIpus, const std::string& assetPath) {
  try {
    // Load new NIFs and reconnect the streams:
    const auto metaFile = assetPath + "/nif_metadata.txt";
    const auto h5File = assetPath + "/converted.hdf5";
    // The host-side data is shared among replicas:
    auto nifData = std::make_shared<NifModel::Data>(h5File, metaFile);
    models.clear();
    for (auto c = 0u; c < numIpus; ++c) {
      models.push_back(std::make_unique<NifModel>(nifData, "env_nif_ipu" + std::to_string(c)));
    }
  } catch (std::exception& e) {
    ipu_utils::logger()->error("Could not load NIF model from '{}'. Exception: {}", assetPath, e.what());
    return false;
  }

  return true;
}

void PathTracerApp::connectNifStreams(poplar::Engine& engine) {
  // Connect the parameter streams for each NIF:
  for (auto& model : models) {
    model->connectStreams(engine);
  }
}

std::pair<poplar::program::Sequence, poplar::program::Sequence>
PathTracerApp::buildEnvironmentNif(poplar::Graph& g, std::unique_ptr<NifModel>& model, poplar::Tensor input, poplar::Tensor& result) {
  if (!model) {
    throw std::runtime_error("Empty NIF model object.");
  }

  bool optimiseStreamMemory = true;
  auto availableMemoryProportion = args.at("available-memory-proportion").as<float>();
  poplar::OptionFlags matmulOptions{
      {"partialsType", args.at("partials-type").as<std::string>()},
      {"availableMemoryProportion", std::to_string(availableMemoryProportion)},
      {"fullyConnectedPass", "INFERENCE_FWD"},
      {"use128BitConvUnitLoad", "true"},
      {"enableFastReduce", "true"}};

  // We need to serialise the input into smaller batches to save memory.  Keep this
  // part simple and find first divisor below the 'optimal' (empirically determined) batch size.
  // Eventually Poplar will automatically calculate batch serialisation plans so overcomplicating
  // this would be a waste of time.

  // Input shape is {2, image-tiles, rays-per-image-tile}:
  ipu_utils::logger()->debug("NIF input shape: {}", input.shape());
  unsigned fullBatchSize = input[0].numElements();
  float optimalBatchSize = args.at("max-nif-batch-size").as<std::size_t>();
  float optimalFactor = fullBatchSize / optimalBatchSize;
  unsigned closestFactor = std::ceil(optimalFactor);
  while (fullBatchSize % closestFactor) {
    closestFactor += 1;
  }
  std::size_t batchSize = fullBatchSize / closestFactor;
  ipu_utils::logger()->debug("Batch-size serialisation full-size: {} serial-size: {} factor: {}", fullBatchSize, batchSize, closestFactor);
  if (batchSize > optimalBatchSize) {
    throw std::runtime_error("Could not find an efficient batch serialisation.");
  }

  // Make slices of the input for batch serialisation size:
  auto inputSlice = g.addVariable(input.elementType(), {2, batchSize}, poplar::VariableMappingMethod::LINEAR);
  ipu_utils::logger()->debug("Serialised input shape: {}", inputSlice.shape());

  auto nifGraphFunc = g.addFunction(
      model->buildInference(g, matmulOptions, cache, optimiseStreamMemory, inputSlice));

  // Analyse the model for the full batch size per replica:
  model->analyseModel(model->getBatchSize() * closestFactor);

  auto nifResult = model->getOutput();
  ipu_utils::logger()->debug("NIF serialised result tensor shape: {}", nifResult.shape());
  auto nifResultSlice = nifResult.slice(0, batchSize, 0);

  // Need to make a tensor that can be used to pre arrange NIF results back onto correct tiles.
  // (Note: Poplar's automatic rearrangement produces an inefficient result in this case):
  std::vector<std::size_t> outputShape = {input.dim(1), input.dim(2), 3};
  result = g.addVariable(nifResult.elementType(), outputShape);
  ipu_utils::logger()->debug("NIF full result tensor shape: {}", result.shape());

  // Now ready to construct the program. Since the number of serialisation steps will be small
  // we construct the serialisation loop unrolled (slices can be static):
  poplar::program::Sequence unrolledLoop;

  for (auto s = 0u; s < closestFactor; ++s) {
    auto uvSlice = input.reshape({2, fullBatchSize}).slice(s * batchSize, (s + 1) * batchSize, 1);
    auto resultSlice = result.reshape({fullBatchSize, 3}).slice(s * batchSize, (s + 1) * batchSize, 0);

    unrolledLoop.add(poplar::program::Copy(uvSlice, inputSlice));
    unrolledLoop.add(poplar::program::Call(nifGraphFunc));
    unrolledLoop.add(poplar::program::Copy(nifResultSlice, resultSlice));
  }

  auto init = model->buildInit(g, optimiseStreamMemory);

  return std::make_pair(init, unrolledLoop);
}

// When using multiple IPUs we want to have one replica of the environment NIF
// neural network per chip so that there is no inter-ipu exchange of ray data
// and we can utilise all FLOPS for neural network inference.
PathTracerApp::ReplicatedNifs
PathTracerApp::buildNifReplicas(poplar::Graph& g, poplar::Tensor uvInput) {
  // Get virtual graphs for all IPUs:
  auto graphs = createIpuShards(g);
  auto tilesPerIpu = g.getTarget().getTilesPerIPU();

  auto ipus = getIPUMapping(g, uvInput);
  ipu_utils::logger()->debug("UVs are shared over {} IPUs: {}", ipus.size(), ipus);
  if (ipus.size() != graphs.size()) {
    ipu_utils::logger()->error("You have selected {} IPUs but are only utilising {}.", graphs.size(), ipus.size());
    throw std::logic_error("Number of IPUs in graph does not match IPUs used in workload.");
  }

  std::vector<poplar::Tensor> shardResults;
  poplar::program::Sequence initAllNifs;
  poplar::program::Sequence execAllNifs;

  for (std::size_t s = 0u; s < graphs.size(); ++s) {
    // Split the UVs into per chip chunks:
    std::size_t startTile = s * tilesPerIpu;
    std::size_t endTile = startTile + tilesPerIpu;
    endTile = std::min(endTile, uvInput.dim(1));
    auto ipuSlice = uvInput.slice({0, startTile, 0}, {uvInput.dim(0), endTile, uvInput.dim(2)});
    ipu_utils::logger()->info("UV chunk shape in IPU {}: {}", s, ipuSlice.shape());

    // For each shard of UVs build a NIF on corresponding IPU's virtual graph:
    poplar::Tensor result;
    poplar::program::Sequence initNifModel;
    poplar::program::Sequence execNifModel;
#ifdef NO_VIRTUAL_GRAPHS
    std::tie(initNifModel, execNifModel) = buildEnvironmentNif(g, models[s], ipuSlice, result);
#else
    std::tie(initNifModel, execNifModel) = buildEnvironmentNif(graphs[s], models[s], ipuSlice, result);
#endif
    shardResults.push_back(result);
    ipu_utils::logger()->debug("Shard result shape in IPU {}: {}", s, result.shape());
    initAllNifs.add(initNifModel);
    execAllNifs.add(execNifModel);
  }

  auto nifResult = poplar::concat(shardResults, 0);
  ipu_utils::logger()->debug("Concatted NIF result shape {}", nifResult.shape());

  return PathTracerApp::ReplicatedNifs{nifResult, initAllNifs, execAllNifs};
}

/// Set the tile mapping for the given tensor's outer dimension
/// so it is split over jobs.
void PathTracerApp::mapTensorOverJobs(poplar::Graph& g, poplar::Tensor t) {
  if (ipuJobs.size() != t.dim(0)) {
    throw std::logic_error("Dimension of tensor's first axis must match the number of jobs.");
  }
  for (auto j = 0u; j < ipuJobs.size(); ++j) {
    auto slice = t.slice(j, j + 1, 0).flatten();
    g.setTileMapping(slice, ipuJobs[j].getTile());
  }
}

std::pair<poplar::Tensor, poplar::program::Sequence>
PathTracerApp::buildAntiAliasNoise(poplar::Graph& g, const std::string& prefix) {
  // Make a global noise tensor for anti-aliasing. Slices will be passed down to each tile:
  const auto numWorkers = g.getTarget().getNumWorkerContexts();
  auto aaSamplesPerTile = IpuPathTraceJob::numRayDirComponents * ipuJobs.front().getPixelCount();
  // Make sure number of samples per tile is a multiple of the number of workers (number
  // of rows doesn't need to be even but it will maximise utilisation if it is).
  if (aaSamplesPerTile % numWorkers) {
    aaSamplesPerTile += numWorkers - aaSamplesPerTile % numWorkers;
  }
  auto aaNoiseType = args.at("aa-noise-type").as<std::string>();
  auto aaNoise = g.addVariable(poplar::HALF, {ipuJobs.size(), aaSamplesPerTile}, "aa_noise");
  mapTensorOverJobs(g, aaNoise);

  poplar::program::Sequence prog;
  aaNoise = buildAaNoise(g, aaNoise, prog, aaNoiseType, prefix + "generate_aa_noise");
  return std::make_pair(aaNoise, prog);
}

std::pair<poplar::Tensor, poplar::program::Sequence>
PathTracerApp::buildPrimarySamples(poplar::Graph& g, const std::string& prefix) {
  // Make a global noise tensor for primary sample space. Slices will be passed down to each tile:
  auto maxPathLength = args.at("max-path-length").as<std::uint32_t>();
  auto maxSamplesPerRay = 3 * maxPathLength;  // If every ray bounce was diffuse
  ipu_utils::logger()->debug("Max number of primary samples per path: {}", maxSamplesPerRay);
  auto primarySamplesPerTile = maxSamplesPerRay * ipuJobs.front().getPixelCount();

  auto samples = g.addVariable(poplar::HALF, {ipuJobs.size(), primarySamplesPerTile}, "primary_samples");
  mapTensorOverJobs(g, samples);

  poplar::program::Sequence prog;
  samples = poprand::uniform(g, nullptr, 0u, samples, poplar::HALF, 0.f, 1.f, prog, prefix + "generate_uniform_0_1");
  return std::make_pair(samples, prog);
}

poplar::Tensor PathTracerApp::buildPathRecords(poplar::Graph& g, const std::string& prefix) {
  // Make tensors to hold all per-ray paths data and other info:
  const auto numRays = ipuJobs.front().getPixelCount();
  return g.addVariable(poplar::FLOAT, {ipuJobs.size(), numRays, 3}, prefix + "contributions");
}

void PathTracerApp::build(poplar::Graph& g, const poplar::Target& target) {
  using namespace poplar;

  pvti::Tracepoint::begin(&traceChannel, "create_path_tracing_jobs");
  auto imageWidth = args.at("width").as<std::uint32_t>();
  auto imageHeight = args.at("height").as<std::uint32_t>();
  const auto tiles = target.getNumTiles();
  auto raysPerJob = calculateMaxRaysPerTile(imageWidth, imageHeight, target);

  ipuJobs.reserve(tiles);
  for (auto t = 0u; t < tiles; ++t) {
    ipuJobs.emplace_back(raysPerJob, args, t);
  }
  pvti::Tracepoint::end(&traceChannel, "create_path_tracing_jobs");

  poprand::addCodelets(g);
  popops::addCodelets(g);
  g.addCodelets(args.at("codelet-path").as<std::string>() + "/codelets.gp");

  poplar::program::Sequence initRenderSettings;

  // Allow the HW RNG seed to be streamed to the IPU at runtime:
  const bool optimiseCopyMemoryUse = true;
  seedTensor.buildTensor(g, poplar::UNSIGNED_INT, {2});
  g.setTileMapping(seedTensor, 0);
  initRenderSettings.add(seedTensor.buildWrite(g, optimiseCopyMemoryUse));
  poprand::setSeed(g, seedTensor, 1u, initRenderSettings, "set_seed");

  // Allow the anti-alias scale to be streamed to the IPU at runtime:
  aaScaleTensor.buildTensor(g, poplar::HALF, {});
  g.setTileMapping(aaScaleTensor, 1);
  initRenderSettings.add(aaScaleTensor.buildWrite(g, optimiseCopyMemoryUse));

  // Allow FOV to be changed at runtime:
  fovTensor.buildTensor(g, poplar::HALF, {});
  g.setTileMapping(fovTensor, 2);
  initRenderSettings.add(fovTensor.buildWrite(g, optimiseCopyMemoryUse));

  // Allow env map rotation to be a runtime variable also:
  azimuthRotation.buildTensor(g, poplar::FLOAT, {});
  g.setTileMapping(azimuthRotation, 0);
  initRenderSettings.add(azimuthRotation.buildWrite(g, optimiseCopyMemoryUse));

  // Allow runtime update of tonemapping parameters:
  exposureTensor.buildTensor(g, poplar::FLOAT, {});
  g.setTileMapping(exposureTensor, 0);
  initRenderSettings.add(exposureTensor.buildWrite(g, optimiseCopyMemoryUse));
  gammaTensor.buildTensor(g, poplar::FLOAT, {});
  g.setTileMapping(gammaTensor, 0);
  initRenderSettings.add(gammaTensor.buildWrite(g, optimiseCopyMemoryUse));

  deviceSampleLimit.buildTensor(g, poplar::UNSIGNED_INT, {});
  g.setTileMapping(deviceSampleLimit, 0);
  initRenderSettings.add(deviceSampleLimit.buildWrite(g, optimiseCopyMemoryUse));

  pvti::Tracepoint::begin(&traceChannel, "build_nifs");
  auto numJobsInBatch = ipuJobs.size();
  auto pixelsPerJob = ipuJobs.front().getPixelCount();
  auto uvInput = createNifInput(g, numJobsInBatch, pixelsPerJob);
  auto envNifs = buildNifReplicas(g, uvInput);
  pvti::Tracepoint::end(&traceChannel, "build_nifs");

  pvti::Tracepoint::begin(&traceChannel, "build_path_trace_jobs");

  // Make the compute sets for path tracing stages:
  const std::string prefix = "render/";
  const IpuPathTraceJob::CsMap computeSets = {
      {"gen-rays", g.addComputeSet(prefix + "ray_gen")},
      {"path-trace", g.addComputeSet(prefix + "path_trace")},
      {"pre-process-escaped-rays", g.addComputeSet(prefix + "pre_process_escaped_rays")},
      {"apply-env-lighting", g.addComputeSet(prefix + "apply_env_lighting")},
      {"accumulate-lighting", g.addComputeSet(prefix + "accumulate_lighting")}};

  auto pathRecords = buildPathRecords(g, prefix);

  const auto pathsPerTile = ipuJobs.front().getPixelCount();
  traceBuffer = g.addVariable(
      poplar::UNSIGNED_CHAR,
      {ipuJobs.size(), sizeof(TraceRecord) * pathsPerTile},
      prefix + "tracebuffer");
  ipu_utils::logger()->info("Tracebuffer shape: {}", traceBuffer.get().shape());
  auto primaryRays = g.addVariable(
      poplar::HALF,
      {ipuJobs.size(), IpuPathTraceJob::numRayDirComponents * pathsPerTile},
      prefix + "primary_rays");

  mapTensorOverJobs(g, traceBuffer.get());
  mapTensorOverJobs(g, primaryRays);

  for (auto j = 0u; j < ipuJobs.size(); ++j) {
    // Create inputs: input to job on each tile is a slice of the global tensors:
    auto uvInputSlice = uvInput.slice(j, j + 1, 1);
    auto nifResultSlice = envNifs.result.slice(j, j + 1, 0);
    auto pathRecordsSlice = pathRecords.slice(j, j + 1, 0).reshape({pathRecords.dim(1), pathRecords.dim(2)});
    auto traceBufferSlice = traceBuffer.get().slice(j, j + 1, 0).reshape({traceBuffer.get().dim(1)});
    auto primaryRaysSlice = primaryRays.slice(j, j + 1, 0).reshape({primaryRays.dim(1)});
    const IpuPathTraceJob::InputMap jobInputs = {
        {"aa-scale", aaScaleTensor},
        {"fov", fovTensor},
        {"uv-input", uvInputSlice},
        {"env-map-result", nifResultSlice},
        {"env-map-rotation", azimuthRotation},
        {"path-records", pathRecordsSlice},
        {"tracebuffer", traceBufferSlice},
        {"primary-rays", primaryRaysSlice},
        {"exposure", exposureTensor},
        {"gamma", gammaTensor}};
    auto& job = ipuJobs[j];
    job.buildGraph(g, jobInputs, computeSets, args);
  }

  using namespace poplar::program;

  Sequence preTraceInit;
  preTraceInit.add(traceBuffer.buildWrite(g, true));
  for (auto& j : ipuJobs) {
    preTraceInit.add(j.beginTraceJob());
  }

  // Construct the core path tracing program:
  Sequence pathTraceIteration;
  pathTraceIteration.add(Execute(computeSets.at("gen-rays")));

  // Wrap path tracing in cycle counter:
  Sequence execPathTrace;
  execPathTrace.add(Execute(computeSets.at("path-trace")));
  pathTraceCycleCount = poplar::cycleCount(
      g, execPathTrace, 0, poplar::SyncType::EXTERNAL, "path_trace_cycle_count");

  pathTraceIteration.add(execPathTrace);
  pathTraceIteration.add(Execute(computeSets.at("pre-process-escaped-rays")));

  // Do environment map lookups via neural network, count cycles for this also:
  nifCycleCount = poplar::cycleCount(g, envNifs.exec, 0, poplar::SyncType::EXTERNAL, "nif_cycle_count");
  pathTraceIteration.add(envNifs.exec);

  // Environment lighting computed so we can now apply the results:
  pathTraceIteration.add(Execute(computeSets.at("apply-env-lighting")));
  pathTraceIteration.add(Execute(computeSets.at("accumulate-lighting")));
  pathTraceIteration.add(WriteUndef(pathRecords));
  for (auto& j : ipuJobs) {
    pathTraceIteration.add(j.endTraceJob());
  }

  // Record total cycles for one iteration:
  iterationCycles = poplar::cycleCount(
      g, pathTraceIteration, 0, poplar::SyncType::EXTERNAL, "cycles_per_iteration");

  // Repeat the core path tracing program for a number of
  // iterations which is fixed at graph compile time:
  auto sampleCounter = g.addVariable(poplar::UNSIGNED_INT, {});
  g.setTileMapping(sampleCounter, 0);
  auto executeRayTrace = popops::countedForLoop(g, sampleCounter, 0, deviceSampleLimit, 1, pathTraceIteration, "sampling_loop");

  // Program to read back results and stats:
  Sequence readTraceResult;
  readTraceResult.add(traceBuffer.buildRead(g, true));
  readTraceResult.add(nifCycleCount.buildRead(g, true));
  readTraceResult.add(pathTraceCycleCount.buildRead(g, true));
  readTraceResult.add(iterationCycles.buildRead(g, true));

  pvti::Tracepoint::end(&traceChannel, "build_path_trace_jobs");

  programs.add("init_render_settings", initRenderSettings);
  programs.add("init_nif_weights", envNifs.init);
  programs.add("setup", preTraceInit);
  programs.add("path_trace", executeRayTrace);
  programs.add("read_results", readTraceResult);
}

// Initialise the work list (which pixels should be traced on
// which tiles):
void PathTracerApp::initialiseState(std::uint32_t imageWidth, std::uint32_t imageHeight,
                                    poplar::Engine& engine, const poplar::Target& target) {
  auto jobs = createTracingJobs(imageWidth, imageHeight, target);
  ipu_utils::logger()->info("Created worklists for {} tiles", jobs.size());

  // We have two pointers for tracked work: one which is to keep defunct
  // data alive whilst asynchronous host processing completes on it.
  traceState = std::make_unique<PathTracerState>(imageWidth, imageHeight);
  traceState->work.randomiseWorkList(jobs);
  traceState->work.getWork().active() = traceState->work.getWork().inactive();
  connectActiveWorkListStreams(engine);
}

void PathTracerApp::connectActiveWorkListStreams(poplar::Engine& engine) {
  pvti::Tracepoint scopedTrace(&traceChannel, "connect_work_list_streams");
  traceBuffer.connectReadStream(engine, traceState->work.getWork().active());
  traceBuffer.connectWriteStream(engine, traceState->work.getWork().active());
}

// The user interaction invalidates all in progress rendering work but
// we don't want to wait for those defunct jobs to complete before we
// start new work. To achieve this we allocate new tracer state and
// then swap the new state with the defunct state:
void PathTracerApp::defunctState(std::uint32_t imageWidth, std::uint32_t imageHeight, poplar::Engine& engine) {
  if (defunctTraceState) {
    // Avoid reallocation as it is expensive (the workists are large):
    pvti::Tracepoint scopedTrace(&traceChannel, "clear_defunct_worklist");
    defunctTraceState->film.reset();
  } else {
    pvti::Tracepoint scopedTrace(&traceChannel, "allocate_new_worklist");
    defunctTraceState.reset(new PathTracerState(imageWidth, imageHeight));
  }

  // Swap and then copy the up-to-date work from the now defunct worklist:
  pvti::Tracepoint::begin(&traceChannel, "copy_worklists");
  std::swap(traceState, defunctTraceState);
  traceState->work.getWork().active() = defunctTraceState->work.getWork().active();
  traceState->work.getWork().inactive() = defunctTraceState->work.getWork().active();
  pvti::Tracepoint::end(&traceChannel, "copy_worklists");

  connectActiveWorkListStreams(engine);
}

InterfaceServer::Status
PathTracerApp::processUserInput(
    InterfaceServer::State& state,
    std::uint32_t imageWidth,
    std::uint32_t imageHeight,
    poplar::Engine& engine,
    const ipu_utils::ProgramManager& progs) {
  if (state.stop) {
    ipu_utils::logger()->info("Rendering stopped by remote UI");
    return InterfaceServer::Status::Stop;
  }

  if (state.detach) {
    // If the Remote-UI detaches just continue rendering:
    ipu_utils::logger()->info("Remote UI disconnected.");
    return InterfaceServer::Status::Disconnected;
  } else {
    if (!state.newNif.empty()) {
      pvti::Tracepoint scopedTrace2(&traceChannel, "load_nif_file");
      // Load of a new NIF was requested:
      ipu_utils::logger()->info("Loading NIF: {}", state.newNif);
      if (loadNifModels(models.size(), state.newNif)) {
        // Connect new NIF streams and upload the weights:
        connectNifStreams(engine);
        progs.run(engine, "init_nif_weights");
      }
    }

    pvti::Tracepoint scopedTrace3(&traceChannel, "reset_host_render_state");

    return InterfaceServer::Status::Restart;
  }
}

void PathTracerApp::execute(poplar::Engine& engine, const poplar::Device& device) {
  pvti::Tracepoint::begin(&traceChannel, "initialisation");

  auto imageWidth = args.at("width").as<std::uint32_t>();
  auto imageHeight = args.at("height").as<std::uint32_t>();
  auto samplesPerPixel = args.at("samples").as<std::uint32_t>();
  auto seed = args.at("seed").as<std::uint64_t>();
  auto antiAliasingScale = args.at("aa-noise-scale").as<float>();
  float fieldOfView = args.at("fov").as<float>() * (M_PI / 180.f);
  auto configExposure = args.at("exposure").as<float>();
  auto configGamma = args.at("gamma").as<float>();
  auto fileName = args.at("outfile").as<std::string>();
  auto loadBalanceEnabled = args.at("enable-load-balancing").as<bool>();
  auto saveInterval = args.at("save-interval").as<std::uint32_t>();
  samplesPerPixel = roundSamplesPerPixel(samplesPerPixel, samplesPerIpuStep);
  const auto steps = samplesPerPixel / samplesPerIpuStep;
  // Convert env map rotation to radians:
  auto degrees = args.at("env-map-rotation").as<float>();
  float radians = (degrees / 360.f) * (2.0 * M_PI);

  // Connect streams for render state:
  seedTensor.connectWriteStream(engine, &seed);
  std::uint16_t aaScaleHalf;
  std::uint16_t fovHalf;
  poplar::copyFloatToDeviceHalf(device.getTarget(), &antiAliasingScale, &aaScaleHalf, 1);
  poplar::copyFloatToDeviceHalf(device.getTarget(), &fieldOfView, &fovHalf, 1);
  aaScaleTensor.connectWriteStream(engine, &aaScaleHalf);
  fovTensor.connectWriteStream(engine, &fovHalf);
  azimuthRotation.connectWriteStream(engine, &radians);
  deviceSampleLimit.connectWriteStream(engine, &samplesPerIpuStep);

  // Connect streams for cycle counters:
  std::int64_t nifCycles;
  std::int64_t pathTraceCycles;
  std::int64_t totalCycles;
  nifCycleCount.connectReadStream(engine, &nifCycles);
  pathTraceCycleCount.connectReadStream(engine, &pathTraceCycles);
  iterationCycles.connectReadStream(engine, &totalCycles);

  // Record a graph of sample rate for the system analyser:
  pvti::Graph plot("Throughput", "paths/sec");
  auto series = plot.addSeries("Samples/sec");

  const auto& progs = getPrograms();
  auto startTime = std::chrono::steady_clock::now();

  // Setup remote user interface:
  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  auto uiPort = args.at("ui-port").as<int>();
  if (uiPort) {
    uiServer.reset(new InterfaceServer(uiPort));
    uiServer->start();
    uiServer->initialiseVideoStream(imageWidth, imageHeight);
    exposureTensor.connectWriteStream(engine, (void*)&uiServer->getState().exposure);
    gammaTensor.connectWriteStream(engine, (void*)&uiServer->getState().gamma);
  } else {
    // If no remote UI attach set the UI state direct from the options/config:
    state.exposure = configExposure;
    state.gamma = configGamma;
    state.fov = fieldOfView;
    state.envRotationDegrees = degrees;
    state.interactiveSamples = args.at("interactive-samples").as<std::uint32_t>();
    exposureTensor.connectWriteStream(engine, &state.exposure);
    gammaTensor.connectWriteStream(engine, &state.gamma);
  }

  connectNifStreams(engine);
  progs.run(engine, "init_nif_weights");
  progs.run(engine, "init_render_settings");

  // Build the tracing jobs:
  initialiseState(imageWidth, imageHeight, engine, device.getTarget());

  pvti::TraceChannel hostTraceChannel = {"host_processing"};
  AsyncTask hostProcessing;

  pvti::Tracepoint::end(&traceChannel, "initialisation");
  pvti::Tracepoint::begin(&traceChannel, "rendering");
  ipu_utils::logger()->info("Render started");

  constexpr std::size_t sampleCountReversionStep = 5;
  std::size_t totalRays = 0;

  // Loop over the requisite number of steps with each step
  // computing many samples per pixel on IPU.
  for (auto step = 1u; step <= steps; ++step) {
    auto loopStartTime = std::chrono::steady_clock::now();

    // Do the simple thing and restart the entire render if any state changed:
    if (uiServer && uiServer->stateChanged()) {
      pvti::Tracepoint scopedTrace1(&traceChannel, "ui_processing");
      state = uiServer->consumeState();
      auto status = processUserInput(state, imageWidth, imageHeight, engine, progs);

      if (status == InterfaceServer::Status::Stop) {
        uiServer.reset();
        break;
      }

      if (status == InterfaceServer::Status::Disconnected) {
        uiServer.reset();
      } else if (status == InterfaceServer::Status::Restart) {
        startTime = loopStartTime;
        step = 1;
        samplesPerIpuStep = state.interactiveSamples;
      }
    } else {
      if (step == sampleCountReversionStep) {
        // No UI input for a few steps so revert to performant number of samples:
        samplesPerIpuStep = args.at("samples-per-step").as<std::uint32_t>();
        ipu_utils::logger()->debug("Interaction stopped reverting samples per step to: {}", samplesPerIpuStep);
      }
    }

    // Render settings can only updated on these steps:
    if (step == 1 || step == sampleCountReversionStep) {
      // Update the variables that are connected to streams and
      // then stream the new parameters to IPU:
      pvti::Tracepoint::begin(&traceChannel, "update_ipu_settings");
      radians = (state.envRotationDegrees / 360.f) * (2.0 * M_PI);
      poplar::copyFloatToDeviceHalf(device.getTarget(), &state.fov, &fovHalf, 1);
      progs.run(engine, "init_render_settings");
      pvti::Tracepoint::end(&traceChannel, "update_ipu_settings");
    }

    pvti::Tracepoint::begin(&traceChannel, "ipu_render");
    // Run ray tracing on the IPU and read back result (results go into into the active
    // buffer whilst the async host task processes the last result from the inactive buffer
    // so it doesn't matter that sync task is still processing the previous result):
    progs.run(engine, "setup");
    progs.run(engine, "path_trace");
    progs.run(engine, "read_results");
    ipu_utils::logger()->debug("Path-Trace cycle count: {}", pathTraceCycles);
    ipu_utils::logger()->debug("NIF cycle count: {}", nifCycles);
    ipu_utils::logger()->debug("Total cycles per iteration: {}", totalCycles);
    pvti::Tracepoint::end(&traceChannel, "ipu_render");

    // Wait for completion of previous async task before starting the next:
    pvti::Tracepoint::begin(&traceChannel, "wait_for_host");
    ipu_utils::logger()->trace("Waiting for async task to complete.");
    hostProcessing.waitForCompletion();
    ipu_utils::logger()->trace("Async task completed.");
    pvti::Tracepoint::end(&traceChannel, "wait_for_host");

    // Swap the worklist buffers and reconnect new active buffer to engine:
    traceState->work.getWork().swap();
    connectActiveWorkListStreams(engine);

    // This lambda function asynchronously processes the result so far on
    // the host while the IPU continues path tracing. Variables captured
    // by reference must not be used elsewhere until waitForCompletion()
    // has returned. We explicitly capture pointers to the work list and
    // film that we are going to process as these may be made defunct by
    // user interaction if remote-UI is enabled.
    hostProcessing.run([&, step, workPtr = &traceState->work, filmPtr = &traceState->film]() {
      pvti::Tracepoint asyncScopedTrace(&hostTraceChannel, "async_work");

      // We process results from the inactive worklist while the IPU
      // is using the active work list:
      pvti::Tracepoint::begin(&hostTraceChannel, "accumulate_framebuffers");
      filmPtr->accumulate(workPtr->getWork().inactive());
      pvti::Tracepoint::end(&hostTraceChannel, "accumulate_framebuffers");

      if (uiServer) {
        // Send data to update the remote UI:
        {
          pvti::Tracepoint::begin(&hostTraceChannel, "tone_map");
          auto& ldr = filmPtr->updateLdrImage();
          pvti::Tracepoint::end(&hostTraceChannel, "tone_map");
          pvti::Tracepoint scopedTrace(&hostTraceChannel, "ui_encode_video");
          uiServer->sendPreviewImage(ldr);
        }
        pvti::Tracepoint scopedTrace(&hostTraceChannel, "ui_send_events");
        uiServer->updateProgress(step, steps);
      }

      if (loadBalanceEnabled && step > 1) {
        pvti::Tracepoint scopedTrace(&hostTraceChannel, "run_load_balancing");
        workPtr->allocateWorkByPathLength(ipuJobs);
      }

      // If there is a UI server we do not save
      // images as we go (only on the final step):
      if (step % saveInterval == 0 || step == steps) {
        if (uiServer) {
          // If there is a UI server we start transmitting full
          // uncompressed image data at the save interval.
          uiServer->startSendingRawImage(filmPtr->getHdrImage(), step);
        } else {
          pvti::Tracepoint scopedTrace(&hostTraceChannel, "save_images");
          filmPtr->saveImages(fileName);
          ipu_utils::logger()->info("Saved images at step {}", step);
        }
      }
    });

    pvti::Tracepoint::begin(&traceChannel, "log_stats");
    auto loopEndTime = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration<double>(loopEndTime - loopStartTime).count();
    const auto pixelSamplesPerStep = imageWidth * imageHeight * samplesPerIpuStep;
    auto sampleRate = pixelSamplesPerStep / secs;
    auto rayRate = totalRays / secs;
    ipu_utils::logger()->info("Completed render step {}/{} in {} seconds (Samples/sec {}) (Rays/sec {})",
                              step, steps, secs, sampleRate, rayRate);
    series.add(sampleRate);

    if (uiServer) {
      uiServer->updateSampleRate(sampleRate, rayRate);
    }
    pvti::Tracepoint::end(&traceChannel, "log_stats");
  }

  hostProcessing.waitForCompletion();
  pvti::Tracepoint::end(&traceChannel, "rendering");

  auto endTime = std::chrono::steady_clock::now();
  const auto elapsedSecs = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Render finished: {} seconds", elapsedSecs);

  const std::size_t pixelsPerFrame = imageWidth * imageHeight;
  const std::size_t numTiles = ipuJobs.size();
  const double samplesPerSec = (pixelsPerFrame / elapsedSecs) * samplesPerPixel;
  const double samplesPerSecPerTile = samplesPerSec / numTiles;
  ipu_utils::logger()->info("Samples/sec: {}", samplesPerSec);
  ipu_utils::logger()->info("Samples/sec/tile: {}", samplesPerSecPerTile);
}

void PathTracerApp::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("outfile,o", po::value<std::string>()->required(), "Set output file name.")
  ("save-interval", po::value<std::uint32_t>()->default_value(4000))
  ("width,w", po::value<std::uint32_t>()->default_value(256), "Output image width (total pixels).")
  ("height,h", po::value<std::uint32_t>()->default_value(256), "Output image height (total pixels).")
  ("samples,s", po::value<std::uint32_t>()->default_value(1000000), "Total samples to take per pixel.")
  ("samples-per-step", po::value<std::uint32_t>()->default_value(1), "Samples to take per IPU step.")
  ("interactive-samples", po::value<std::uint32_t>()->default_value(1), "Number of samples to take per IPU step during user interaction.")
  ("refractive-index,n", po::value<float>()->default_value(1.5), "Refractive index.")
  ("roulette-depth", po::value<std::uint16_t>()->default_value(3), "Number of bounces before rays are randomly stopped.")
  ("stop-prob", po::value<float>()->default_value(0.3), "Probability of a ray being stopped.")
  ("aa-noise-scale,a", po::value<float>()->default_value(0.1), "Scale of anti-aliasing noise (pixels).")
  ("fov", po::value<float>()->default_value(90.f), "Horizontal field of view (degrees).")
  ("exposure", po::value<float>()->default_value(0.f), "Exposure compensation for tone-mapping.")
  ("gamma", po::value<float>()->default_value(2.2f), "Gamma correction for tone-mapping.")
  ("env-map-rotation",po::value<float>()->default_value(0.f), "Azimuthal rotation for HDRI environment map (degrees)." )
  ("seed", po::value<std::uint64_t>()->default_value(1), "Seed for random number generation.")
  ("aa-noise-type", po::value<std::string>()->default_value("normal"),
  "Choose distribution for anti-aliasing noise ['uniform', 'normal', 'truncated-normal'].")
  ("codelet-path", po::value<std::string>()->default_value("./"), "Path to ray tracing codelets.")
  ("enable-load-balancing", po::bool_switch()->default_value(false), "Run dynamic load balancing algorithm for path tracing.")
  ("max-path-length", po::value<std::uint32_t>()->default_value(10))
  // Neural Environment-map Model Options:
  ("assets", po::value<std::string>()->required(),
    "Path to the 'assets.extra' directory of the saved keras model.")
  ("partials-type", po::value<std::string>()->default_value("half"),
    "Partials type for matrix multiplies.")
  ("available-memory-proportion", po::value<float>()->default_value(0.6),
    "Proportion of on-chip memory that is allowed for matrix multiplies.")
  ("max-nif-batch-size", po::value<std::size_t>()->default_value(30 * 1472),
    "Maximum batch-size for the NIF neural network. If the required batch is larger than this "
    "the batch will be serialised so that this value is not exceeded.")
  ("ui-port", po::value<int>()->default_value(0), "Start a remote user-interface server on the specified port.")
  ;
}
