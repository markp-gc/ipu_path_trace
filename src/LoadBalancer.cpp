// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "LoadBalancer.hpp"

#include "codelets/TraceRecord.hpp"

#include "ipu_utils.hpp"
#include "io_utils.hpp"

#include <random>
#include <algorithm>

WorkList::WorkList(std::size_t size) :
  activeWork(size),
  inactiveWork(size)
{}

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

LoadBalancer::LoadBalancer(std::size_t workItemCount) : work(workItemCount) {
}

LoadBalancer::~LoadBalancer() {
}

// Randomise the inactive worklist:
void LoadBalancer::randomiseWorkList(const IpuJobList& jobs) {
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

  // Random shuffle the work list:
  auto workSeed = 142u;
  std::mt19937 g(workSeed);
  std::shuffle(workList.begin(), workList.end(), g);

  // Overwrite the inactive worklist:
  work.inactive() = workList;
}

void LoadBalancer::allocateWorkByPathLength(const IpuJobList& jobs) {
  // Sort a copy of the inactive work list by path length:
  auto sorted = work.inactive();

  ipu_utils::logger()->trace("Worklist before sort:\n{}", sorted);

  std::sort(sorted.begin(), sorted.end(), [](const TraceRecord& a, const TraceRecord& b) -> bool {
    return a.pathLength < b.pathLength;
  });

  ipu_utils::logger()->trace("Worklist after sort:\n{}", sorted);

  // Pre-allocate per tile worklists:
  std::vector<WorkList::RecordList> perTileWork(jobs.size());
  for (auto &t : perTileWork) {
    t.reserve(jobs[0].getPixelCount());
  }

  // Iterators for longest and shortest paths:
  auto shortItr = sorted.begin();
  auto longItr = sorted.end() - 1;

  // Allocate work to tiles by taking pairs of items from both ends of the sorted list.
  // I.e. the tile that takes the longest path from the last render step also takes the
  // shortest path, and so on until the worklist is exhausted:
  ipu_utils::logger()->info("Load balancing started ({} work items)", sorted.size());
  ipu_utils::logger()->info("Path length min/max: {}/{}", shortItr->pathLength, longItr->pathLength);
  while (true) {
    for (auto &t : perTileWork) {
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
  auto itr = sorted.begin();
  for (auto &t : perTileWork) {
    for (auto& w : t) {
      *itr = w;
      ++itr;
    }
  }

  work.inactive() = sorted;
}

/// Clear the accumulators in the inactive work list:
std::size_t LoadBalancer::sumTotalInactivePathSegments() {
  auto& list = work.inactive();

  std::size_t sum = 0;
  #pragma omp parallel for reduction(+ : sum) schedule(auto)
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
