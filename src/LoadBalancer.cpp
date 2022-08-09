// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "LoadBalancer.hpp"

#include "codelets/TraceRecord.hpp"

#include "io_utils.hpp"
#include "ipu_utils.hpp"

#include <algorithm>
#include <random>
#include <limits>

std::size_t calculateMaxRaysPerTile(std::size_t imageWidth, std::size_t imageHeight, const poplar::Target& target) {
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();

  // Check for performance hint:
  if ((imageWidth * imageHeight) % (numTiles * numWorkers)) {
    ipu_utils::logger()->warn(
      "For best performance number of pixels in image should be divisible by {} x {} (tiles x workers).",
      numTiles, numWorkers);
  }

  const auto totalRayCount = imageWidth * imageHeight;
  const unsigned raysPerTile = std::ceil(totalRayCount / (float)numTiles);

  // We need a minimum number of rays in each tile's worklist
  // to avoid complicating the MultiVertex codelets:
  return std::max(numWorkers, raysPerTile);
}

std::vector<TraceRecord> createWorkListForImage(std::size_t imageWidth, std::size_t imageHeight) {
  std::vector<TraceRecord> workList;

  // Make a worklist that contains every pixel in the image:
  workList.reserve(imageWidth * imageHeight);
  auto allocatedWork = 0u;
  for (std::size_t r = 0; r < imageHeight; ++r) {
    for (std::size_t c = 0; c < imageWidth; ++c) {
      workList.emplace_back(c, r);
      allocatedWork += 1;
    }
  }

  return workList;
}

std::vector<RecordList> createTracingJobs(std::size_t imageWidth, std::size_t imageHeight, const poplar::Target& target) {
  // Calculate number of rays each tile needs to trace
  // to take one sample-per-pixel of the whole image:
  const auto numTiles = target.getNumTiles();
  const auto maxRaysPerTile = calculateMaxRaysPerTile(imageWidth, imageHeight, target);
  auto paddedRayCount = maxRaysPerTile * numTiles;

  // Make a worklist that contains every pixel in the image:
  auto workList = createWorkListForImage(imageWidth, imageHeight);

  // Pad the list with null work (these entries will be
  // ignored during image accumulation):
  const auto dummyCoord = std::numeric_limits<std::uint16_t>::max();
  auto allocatedWork = workList.size();
  while (allocatedWork < paddedRayCount) {
    workList.emplace_back(dummyCoord, dummyCoord);
    allocatedWork += 1;
  }

  // Each tile takes equal chunks from the list:
  auto copyItr = workList.cbegin();
  std::vector<RecordList> perTileWork;
  for (auto t = 0u; t < numTiles; ++t) {
    perTileWork.emplace_back(maxRaysPerTile);
    auto endItr = copyItr + maxRaysPerTile;
    std::copy(copyItr, endItr, perTileWork.back().begin());
    copyItr = endItr;

    ipu_utils::logger()->trace("Initial worklist for tile {}:\n{}", t, perTileWork.back());
  }

  return perTileWork;
}

WorkList::WorkList(std::size_t size)
    : activeWork(size),
      inactiveWork(size) {}

WorkList::~WorkList() {}

RecordList& WorkList::active() {
  return activeWork;
}

RecordList& WorkList::inactive() {
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
void LoadBalancer::randomiseWorkList(const std::vector<RecordList>& jobs) {
  // Take a copy of the active worklist:
  std::vector<TraceRecord> workList;
  workList.reserve(jobs.size() * jobs.front().size());

  ipu_utils::logger()->trace("Work capacity:\n{}", workList.capacity());

  // Fill the worklist in job order:
  for (const auto& j : jobs) {
    for (const auto& w : j) {
      workList.push_back(w);
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
  std::vector<RecordList> perTileWork(jobs.size());
  for (auto& t : perTileWork) {
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
  auto itr = sorted.begin();
  for (auto& t : perTileWork) {
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
