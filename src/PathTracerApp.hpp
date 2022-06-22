// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>
#include <limits>
#include <chrono>
#include <memory>
#include <numeric>

#include <neural_networks/NifModel.hpp>

#include <boost/program_options.hpp>

#include "IpuPathTraceJob.hpp"
#include "AccumulatedImage.hpp"
#include "LoadBalancer.hpp"
#include "InterfaceServer.hpp"

#include <pvti/pvti.hpp>

// fwd declarations:
struct TraceRecord;

struct PathTracerState {

  PathTracerState(std::uint32_t imageWidth, std::uint32_t imageHeight)
  : work(imageWidth * imageHeight),
    film(imageWidth, imageHeight)
  {}

  LoadBalancer work;
  AccumulatedImage film;
};

/// This is the main application object. It implements the BuilderInterface
/// so that execution can be marshalled by a GraphManager object:
struct PathTracerApp : public ipu_utils::BuilderInterface {

  /// Constructor does nothing of note, conforming to Poplar
  /// explorer tool interface (see init() instead).
  PathTracerApp();
  virtual ~PathTracerApp() {}

  /// Performs all initialisation that doesn't require a graph (and all
  /// init required on execeutable load):
  void init(const boost::program_options::variables_map& args);

  ipu_utils::RuntimeConfig getRuntimeConfig() const override;

  /// Construct the graph and programs.
  void build(poplar::Graph& g, const poplar::Target& target) override;

  ipu_utils::ProgramManager& getPrograms() override { return programs; }

  /// Run the path tracing program.
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  /// Add options specifically for the path tracer:
  void addToolOptions(boost::program_options::options_description& desc);

private:
  // Create a tensor for the UV neural environment map inputs:
  poplar::Tensor createNifInput(poplar::Graph& g, std::size_t numJobsInBatch);

  void loadNifModels(std::size_t numIpus, const std::string& assetPath);
  void connectNifStreams(poplar::Engine& engine);

  std::pair<poplar::program::Sequence, poplar::program::Sequence>
  buildEnvironmentNif(poplar::Graph& g, std::unique_ptr<NifModel>& model,
                      poplar::Tensor input, poplar::Tensor& result);

  void initialiseState(std::uint32_t imageWidth, std::uint32_t imageHeight, poplar::Engine& engine);
  void defunctState(std::uint32_t imageWidth, std::uint32_t imageHeight, poplar::Engine& engine);
  void connectActiveWorkListStreams(poplar::Engine& engine);

  struct ReplicatedNifs {
    poplar::Tensor result;
    poplar::program::Sequence init;
    poplar::program::Sequence exec;
  };

  ReplicatedNifs buildNifReplicas(poplar::Graph& g, poplar::Tensor uvInput);

  void mapTensorOverJobs(poplar::Graph& g, poplar::Tensor t);

  std::pair<poplar::Tensor, poplar::program::Sequence>
  buildAntiAliasNoise(poplar::Graph& g, const std::string& prefix);

  std::pair<poplar::Tensor, poplar::program::Sequence>
  buildPrimarySamples(poplar::Graph& g, const std::string& prefix);

  poplar::Tensor buildPathRecords(poplar::Graph& g, const std::string& prefix);

  void initialiseWorkList(std::vector<TraceRecord>& workList);
  std::vector<TraceRecord> loadBalanceWorkList(const std::vector<TraceRecord>& workList);

  InterfaceServer::Status
  processUserInput(InterfaceServer::State& state,
                   std::uint32_t imageWidth, std::uint32_t imageHeight,
                   poplar::Engine& engine,
                   const ipu_utils::ProgramManager& progs);

  pvti::TraceChannel traceChannel = {"ipu_path_tracer"};
  boost::program_options::variables_map args;
  std::uint32_t samplesPerPixel;
  std::uint32_t samplesPerIpuStep;
  ipu_utils::ProgramManager programs;
  std::vector<TraceTileJob> traceJobs;
  IpuJobList ipuJobs;
  ipu_utils::StreamableTensor seedTensor;
  ipu_utils::StreamableTensor aaScaleTensor;
  ipu_utils::StreamableTensor fovTensor;
  ipu_utils::StreamableTensor azimuthRotation;
  ipu_utils::StreamableTensor deviceSampleLimit;
  ipu_utils::StreamableTensor nifCycleCount;
  ipu_utils::StreamableTensor pathTraceCycleCount;
  ipu_utils::StreamableTensor iterationCycles;
  ipu_utils::StreamableTensor traceBuffer;

  poplin::matmul::PlanningCache cache;
  std::vector<std::unique_ptr<NifModel>> models;

  std::unique_ptr<PathTracerState> traceState;
  std::unique_ptr<PathTracerState> defunctTraceState;
};
