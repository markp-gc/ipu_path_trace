// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <poplar/Graph.hpp>

#include <string>
#include <iostream>
#include <set>

std::vector<poplar::Graph> createIpuShards(poplar::Graph& graph);

std::set<std::size_t> getIPUMapping(const poplar::Graph& g, const poplar::Tensor& t);
