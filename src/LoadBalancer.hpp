// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>
#include <memory>

#include "IpuPathTraceJob.hpp"

// fwd declarations:
struct TraceRecord;

/// A double buffered work list.
struct WorkList {

  using RecordList = std::vector<TraceRecord>;

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

  void randomiseWorkList(const IpuJobList& jobs);
  void allocateWorkByPathLength(const IpuJobList& jobs);
  void clearInactiveAccumulators();
  void clearActiveAccumulators();
  std::size_t sumTotalInactivePathSegments();

private:
  WorkList work;
};
