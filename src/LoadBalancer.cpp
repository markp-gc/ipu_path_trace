// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "LoadBalancer.hpp"

#include "codelets/TraceRecord.hpp"

#include "io_utils.hpp"
#include "ipu_utils.hpp"

#include <algorithm>
#include <random>

WorkList::WorkList(std::size_t size)
    : activeWork(size),
      inactiveWork(size) {}

WorkList::~WorkList() {}

WorkList::RecordList& WorkList::active() {
  return activeWork;
}

WorkList::RecordList& WorkList::inactive() {
  return inactiveWork;
}

/// Retutn the next batch of work:
void WorkList::swap() {
  std::swap(activeWork, inactiveWork);
  if (activeWork.empty()) {
    throw std::logic_error("The new active worklist is empty.");
  }
}

LoadBalancer::LoadBalancer(std::size_t workItemCount)
    : work(workItemCount) {
}

LoadBalancer::~LoadBalancer() {
}

// Randomise the inactive worklist:
void LoadBalancer::initialiseWorkList(const IpuJobList& jobs) {
  // Take a copy of the active worklist:
  auto workList = work.inactive();

  // Fill the worklist in job order:
  auto ray = 0u;
  for (auto j = 0u; j < jobs.size(); ++j) {
    const auto& spec = jobs[j].jobSpec;
    for (unsigned r = spec.startRow; r < spec.endRow; ++r) {
      for (unsigned c = spec.startCol; c < spec.endCol; ++c, ++ray) {
        workList[ray] = TraceRecord(c, r);
      }
    }
  }

  // Overwrite the inactive worklist:
  work.inactive() = workList;
}

void LoadBalancer::balanceInactiveWorkList(std::size_t totalPaths, std::size_t pathsPerTile) {

  // The inactive worklist is the one used from the last
  // iteration on the device (so the statistics it contains
  // are one step out of date).
  //
  // Each work item in the list is a TraceRecord object
  // (see src/codelets/TraceRecord.cpp). Each IPU tile will
  // process a contiguous chunk of size 'pathsPerTile' the list
  // you return.
  WorkList::RecordList balancedWork = work.inactive();

  // HINT: A reasonable place to start is to sort the worklist by
  // path-length (path-length is proportional to computational load)
  // and then decide how to distribute work evenly across tiles:
  ipu_utils::logger()->trace("Worklist before sort:\n{}", balancedWork);
  std::sort(balancedWork.begin(), balancedWork.end(), [](const TraceRecord& a, const TraceRecord& b) -> bool {
    return a.pathLength < b.pathLength;
  });
  ipu_utils::logger()->trace("Worklist after sort:\n{}", balancedWork);

  // Insert work distribution algorithm here: NOTE: you only need to reorder the existing list (i.e. set pixel coords) - stats will be reset appropriately elsewhere.

  // Record the balanced list:
  work.inactive() = balancedWork;
}

/// Clear the accumulators in the inactive work list:
std::size_t LoadBalancer::sumTotalInactivePathSegments() {
  auto& list = work.inactive();

  std::size_t sum = 0;
#pragma omp parallel for reduction(+ \
                                   : sum) schedule(auto)
  for (std::size_t i = 0; i < list.size(); ++i) {
    sum += list[i].pathLength;
  }

  return sum;
}

/// Clear the accumulators in the inactive work list:
void LoadBalancer::clearInactiveAccumulators() {
  auto& list = work.inactive();
#pragma omp parallel for schedule(auto)
  for (std::size_t i = 0; i < list.size(); ++i) {
    auto& t = list[i];
    t.r = t.g = t.b = 0.f;
    t.pathLength = 0;
    t.sampleCount = 0;
  }
}

/// Clear the accumulators in the inactive work list:
void LoadBalancer::clearActiveAccumulators() {
  auto& list = work.active();
#pragma omp parallel for schedule(auto)
  for (std::size_t i = 0; i < list.size(); ++i) {
    auto& t = list[i];
    t.r = t.g = t.b = 0.f;
    t.pathLength = 0;
    t.sampleCount = 0;
  }
}
