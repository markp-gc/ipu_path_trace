// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

/// Record of the final colour contribution for a specific
/// pixel (and any other useful statistics).
struct TraceRecord {
  std::uint16_t u, v; // Image pixel coord.
  float r, g, b; // Final RGB Contribution.
  std::uint16_t sampleCount;
  std::uint16_t pathLength;

  /// Set pixel coords to trace from and zero everything else.
  TraceRecord(std::uint16_t pixelU, std::uint16_t pixelV)
    : u(pixelU), v(pixelV), r(0.f), g(0.f), b(0.f), sampleCount(0), pathLength(0) {}

  /// Sets entire record to zero:
  TraceRecord() : TraceRecord(0, 0) {}
};
