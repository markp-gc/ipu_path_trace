// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#ifdef __POPC__
#include <poplar/HalfFloat.hpp>
#else
#include <Eigen/Dense>
using half = Eigen::half;
#endif

/// Record of the final colour contribution for a specific
/// pixel (and any other useful statistics).
struct TraceRecord {
  std::uint16_t u, v; // Image pixel coord.
  std::uint8_t r, g, b; // Final RGB Contribution.

  /// Set pixel coords to trace from and zero everything else.
  TraceRecord(std::uint16_t pixelU, std::uint16_t pixelV)
    : u(pixelU), v(pixelV), r(0), g(0), b(0) {}

  /// Sets entire record to zero:
  TraceRecord() : TraceRecord(0, 0) {}
};
