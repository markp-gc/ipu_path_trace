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

  // Lazy but effective solution - randomise the work list:
  // auto workSeed = 142u;
  // std::mt19937 g(workSeed);
  // std::shuffle(balancedWork.begin(), balancedWork.end(), g);
  // work.inactive() = balancedWork;
  // return;

  // Sort a copy of the inactive work list by path length:
  ipu_utils::logger()->trace("Worklist before sort:\n{}", balancedWork);

  std::sort(balancedWork.begin(), balancedWork.end(), [](const TraceRecord& a, const TraceRecord& b) -> bool {
    return a.pathLength < b.pathLength;
  });

  ipu_utils::logger()->trace("Worklist after sort:\n{}", balancedWork);

  // Pre-allocate per tile worklists:
  std::vector<WorkList::RecordList> perTileWork(totalPaths);
  for (auto& t : perTileWork) {
    t.reserve(pathsPerTile);
  }

  // Iterators for longest and shortest paths:
  auto shortItr = balancedWork.begin();
  auto longItr = balancedWork.end() - 1;

  // Allocate work to tiles by taking pairs of items from both ends of the sorted list.
  // I.e. the tile that takes the longest path from the last render step also takes the
  // shortest path, and so on until the worklist is exhausted:
  ipu_utils::logger()->info("Load balancing started ({} work items)", balancedWork.size());
  ipu_utils::logger()->info("Path length min/max: {}/{}", shortItr->pathLength, longItr->pathLength);
  while (true) {
    for (auto& t : perTileWork) {
      // Take 2 work items, one from each end of queue:
      t.push_back(*shortItr);
      t.push_back(*longItr);
      ++shortItr;
      --longItr;
    }
    if (longItr <= shortItr) {
      break;
    }
  }
  ipu_utils::logger()->info("Load balancing finished");

  // Flatten the new worklist by tiles:
  auto itr = balancedWork.begin();
  for (auto& t : perTileWork) {
    for (auto& w : t) {
      *itr = w;
      ++itr;
    }
  }

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
