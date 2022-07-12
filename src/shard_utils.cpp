// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "shard_utils.hpp"
#include "io_utils.hpp"

#include <sstream>

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>

std::map<std::size_t, poplar::Interval> getShardInfo(const poplar::Graph& graph) {
  std::map<std::size_t, poplar::Interval> info;
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  std::size_t nextShardStartTile = 0;
  for (auto i = 0u; i < numIpus; ++i) {
    auto nextShardEndTile = nextShardStartTile + tilesPerIpu;
    info.insert(
        std::make_pair(i, poplar::Interval(nextShardStartTile, nextShardEndTile)));
    nextShardStartTile = nextShardEndTile;
  }
  return info;
}

std::vector<poplar::Graph> createIpuShards(poplar::Graph& graph) {
  std::vector<poplar::Graph> shard;
  std::size_t nextShardStartTile = 0;
  for (auto s = 0u; s < graph.getTarget().getNumIPUs(); ++s) {
    auto nextShardEndTile = nextShardStartTile + graph.getTarget().getTilesPerIPU();
    shard.push_back(graph.createVirtualGraph(nextShardStartTile, nextShardEndTile));
    ipu_utils::logger()->debug("Created virtual graph for tiles [{}, {})", nextShardStartTile, nextShardEndTile);
    nextShardStartTile = nextShardEndTile;
  }
  return shard;
}

inline poplar::Interval intersect(const poplar::Interval& a, const poplar::Interval& b) {
  auto begin = std::max(a.begin(), b.begin());
  auto end = std::min(a.end(), b.end());
  if (end < begin) {
    end = begin;
  }
  return poplar::Interval(begin, end);
}

inline bool intersects(const poplar::Interval& a, const poplar::Interval& b) {
  return intersect(a, b).size();
}

poplar::Interval getTileInterval(const poplar::Graph& g, const poplar::Tensor& t) {
  auto mapping = g.getTileMapping(t);
  auto min = std::numeric_limits<std::size_t>::max();
  auto max = std::numeric_limits<std::size_t>::min();
  if (mapping.empty()) {
    throw std::runtime_error("Called getTileInterval() on tensor with no tile mapping.");
  }
  for (auto t = 0u; t < mapping.size(); ++t) {
    if (!mapping[t].empty()) {
      if (t < min) {
        min = t;
      }
      if (t > max) {
        max = t;
      }
    }
  }
  return poplar::Interval(min, max + 1);
}

std::set<std::size_t> getIPUMapping(const poplar::Graph& g, const poplar::Tensor& t) {
  auto shardInfo = getShardInfo(g);
  auto tileInterval = getTileInterval(g, t);

  std::set<std::size_t> shards;
  for (const auto& s : shardInfo) {
    // If any interval in the Tensor's mapping overlaps the tiles for
    // a particular shard then record it as being on that shard:
    auto intersection = intersect(tileInterval, s.second);
    if (intersection.size()) {
      shards.insert(s.first);
    }
  }

  return shards;
}
