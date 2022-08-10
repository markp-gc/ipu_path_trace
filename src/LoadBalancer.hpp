// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "IpuPathTraceJob.hpp"

// fwd declarations:
struct TraceRecord;

using RecordList = std::vector<TraceRecord>;

/// Calculate the maximum number of rays every tile needs to trace in
/// order to generate one sample per pixel for the whole image of the
/// specified size.
std::size_t calculateMaxRaysPerTile(std::size_t imageWidth, std::size_t imageHeight, const poplar::Target& target);

/// Create a worklist that contains one item for every pixel in the image.
std::vector<TraceRecord> createWorkListForImage(std::size_t imageWidth, std::size_t imageHeight);

/// Return a vector of work items per-tile.
std::vector<RecordList> createTracingJobs(std::size_t imageWidth, std::size_t imageHeight, const poplar::Target& target);

/// A double buffered work list.
struct WorkList {
  WorkList(std::size_t size);

  virtual ~WorkList();

  /// Swap the buffers:
  void swap();

  RecordList& active();
  RecordList& inactive();

private:
  RecordList activeWork;
  RecordList inactiveWork;
};

struct LoadBalancer {
  LoadBalancer(std::size_t workItemCount);
  virtual ~LoadBalancer();

  WorkList& getWork() { return work; }

  void randomiseWorkList(const std::vector<RecordList>& jobs);
  void allocateWorkByPathLength(const IpuJobList& jobs);
  std::size_t clearInactiveAccumulators();
  void clearActiveAccumulators();

private:
  WorkList work;
};
