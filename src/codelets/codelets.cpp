// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <light/src/vector.hpp>
#include <light/src/light.hpp>
#include <light/src/sdf.hpp>

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include "WrappedArray.hpp"
#include "TraceRecord.hpp"

#include <print.h>

// Because intrinsic/vectorised code can not be used with CPU
// or IpuModel targets we need to guard IPU optimised parts of
// the code so we can still support those:
#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>

#define GET_TILE_ID __builtin_ipu_get_tile_id()
#else
#define GET_TILE_ID 0u
#endif // __IPU__

using namespace poplar;
using Vec = light::Vector;

/// Codelet which generates all outgoing (primary) camera rays for
/// a tile. Anti-aliasing noise is added to the rays using random
/// numbers that were generated external to this codelet. Because
/// they are close to normalised the camera rays can be safely
/// stored at half precision which reduces memory requirements.
///
/// This is a multi-vertex that decides how to distribute work
/// over the hardware worker threads inside the compute method
/// itself.
class GenerateCameraRays : public MultiVertex {

public:
  Output<Vector<half>> rays;
  Input<Vector<unsigned char>> traceBuffer;
  Input<unsigned> imageWidth;
  Input<unsigned> imageHeight;
  Input<half> antiAliasScale;
  Input<half> fov;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Get the descriptions of rays to be traced:
    auto rayCount = rays.size();
    const TraceRecord* traces = reinterpret_cast<const TraceRecord*>(&traceBuffer[0]);

    // Each worker will process one sixth of the rays so
    // we simply offset the start address based on worker ID:
    auto workerPtr = traces + workerId;
    float fovf32 = (float)*fov;
    float2 twoowh{2.f / imageWidth, 2.f / imageHeight};
    float aspect = imageWidth / imageHeight;
    float tanTheta = tanf(fovf32 / 2.f);

    for (auto k = 2 * workerId; k < rayCount; k += 2 * workerCount) {
      // Add anti-alias noise in pixel space:
      float2 cr{(float)workerPtr->u, (float)workerPtr->v};
      const float2 noise = __builtin_ipu_f32v2grand();
      cr += noise * (float)*antiAliasScale;
      // Pixel to camera ray transform:
      cr *= twoowh;
      cr -= 1.f;
      cr *= float2{aspect * tanTheta, -tanTheta};
      rays[k]     = cr[0];
      rays[k + 1] = cr[1];
      workerPtr += workerCount;
    }
    return true;
  }
};

/// Codelet which performs ray tracing for the tile. It knows
/// nothing about the image geometry - it just receives a flat
/// buffer of primary rays as input and stores the result of path
/// tracing for that ray in the corresponding position in the output
/// frame buffer. This codelet also receives as input a buffer of
/// uniform noise to use for all MC sampling operations during path
/// tracing.
///
/// For now the scene is hard coded onto the stack of the compute()
/// function but it could be passed in as tensor data (with some extra
/// overhead of manipulating or unpacking the data structure).
class RayTraceKernel : public Vertex {

public:
  Input<Vector<half>> cameraRays;
  Vector<Output<Vector<float>>, poplar::VectorLayout::ONE_PTR> contributionData;

  bool compute() {
    const Vec zero(0.f, 0.f, 0.f);

    // Loop over the camera rays:
    const auto raysSize = (cameraRays.size() >> 1) << 1;
    for (auto r = 0u, c = 0u; r < raysSize; r += 2, c += 1) {
      // Unpack the camera ray directions which are stored as a
      // sequence of x, y coords with implicit z-direction of -1:
      Vec rayDir((float)cameraRays[r], (float)cameraRays[r+1], (float)-1.f);
      light::Ray ray(zero, rayDir);
      contributionData[c][0] = ray.direction.x;
      contributionData[c][1] = ray.direction.y;
      contributionData[c][2] = ray.direction.z;
    } // end loop over camera rays

    return true;
  }
};

// This takes path trace results and calculates UV coords for all
// the escaped rays in order to lookup lighting values from the
// environment map. UVs are calculated using equirectangular
// projection.
class PreProcessEscapedRays : public MultiVertex {
public:
  Vector<Input<Vector<float>>> contributionData;
  Input<float> azimuthalOffset;
  Output<Vector<float>> u;
  Output<Vector<float>> v;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Parallelise over all workers (each worker starts at a different offset):
    for (auto r = workerId; r < contributionData.size(); r += workerCount) {
      // Pre for environment lighting calculation.
      Vec rayDir(contributionData[r][0], contributionData[r][1], contributionData[r][2]);
      // Convert ray direction to UV coords using equirectangular projection.
      // Calc assumes ray-dir was already normalised (note: normalised in Ray constructor).
      auto theta = acosf(rayDir.y);
      auto phi = atan2(rayDir.z, rayDir.x) + azimuthalOffset;
      constexpr auto twoPi = 2.f * light::Pi;
      constexpr auto invPi = 1.f / light::Pi;
      constexpr auto inv2Pi = 1.f / twoPi;
      if (phi < 0.f) {
        phi += twoPi;
      } else if (phi > twoPi) {
        phi -= twoPi;
      }
      auto uCoord = theta * invPi;
      auto vCoord = phi * inv2Pi;
      u[r] = uCoord;
      v[r] = vCoord;
    }

    return true;
  }
};

// Quickly compute x^y using log and exp.
//
// This is not a general purpose powf implementation
// but will work for range of values typically used in
// gamma correction. There is no special case handling.
// Absolute errors can be very high outside of intended
// use case.
inline
float ipu_powf(float x, float y) {
  return __builtin_ipu_exp(y * __builtin_ipu_ln(x));
}

inline
half2 ipu_powh(half2 x, half y) {
  return __builtin_ipu_exp(y * __builtin_ipu_ln(x));
}

// Compute 2^y using dedicated HW instruction:
inline
float ipu_exp2(float y) {
  return __builtin_ipu_exp2(y);
}

// Update escaped rays with the result of env-map lighting lookup:
class PostProcessEscapedRays : public MultiVertex {
public:
  Vector<Input<Vector<float>>> bgr;
  InOut<Vector<unsigned char>> traceBuffer;
  Input<float> exposure;
  Input<float> gamma;
  unsigned id;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    const half exposureScale = ipu_exp2(exposure);
    const half invGamma = 1.f / gamma;

    TraceRecord* traces = reinterpret_cast<TraceRecord*>(&traceBuffer[0]) + workerId;
    for (auto r = workerId; r < bgr.size(); r += workerCount, traces += workerCount) {
      const auto v = bgr[r];

      // We don't care about the 4th component but repeating a
      // component is more efficient than materialising a constant:
      half4 total{(half)v[0], (half)v[1], (half)v[2], (half)v[2]};

      // Apply tone-mapping:
      total *= exposureScale;
      half2 total_xy{total[0], total[1]};
      half2 total_zw{total[2], total[3]};
      total_xy = ipu_powh(total_xy, invGamma);
      total_zw = ipu_powh(total_zw, invGamma);
      total[0] = total_xy[0];
      total[1] = total_xy[1];
      total[2] = total_zw[0];
      total[3] = total_zw[1]; // Value not needed but compiler makes a mess if we don't do this

      // Scale and clip result into an unsigned byte range:
      constexpr half scale = 255.f;
      constexpr half2 validRange {0.f, scale};
      total *= scale;
      total = __builtin_ipu_clamp(total, validRange);

      traces->r = std::uint8_t(total[0]);
      traces->g = std::uint8_t(total[1]);
      traces->b = std::uint8_t(total[2]);
    } // end loop over camera rays

    return true;
  }

};
