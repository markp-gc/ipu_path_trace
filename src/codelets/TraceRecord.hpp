// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

/// Record of the final colour contribution for a specific
/// pixel (and any other useful statistics).
struct TraceRecord {
  std::uint16_t u, v; // The image pixel coord to trace through
                      // is recorded here when a worklist is created.
  float r, g, b; // IPU writes final RGB Contribution here.
  std::uint16_t sampleCount; // IPU records number of samples taken here.
  std::uint16_t pathLength; // IPU records length of the traced path here.

  /// Set pixel coords to trace from and zero everything else.
  TraceRecord(std::uint16_t pixelU, std::uint16_t pixelV)
    : u(pixelU), v(pixelV), r(0.f), g(0.f), b(0.f), sampleCount(0), pathLength(0) {}

  /// Sets entire record to zero:
  TraceRecord() : TraceRecord(0, 0) {}
};
