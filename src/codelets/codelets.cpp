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
  Input<Vector<half>> antiAliasNoise;
  Output<Vector<half>> rays;
  Input<Vector<unsigned char>> traceBuffer;
  Input<unsigned> imageWidth;
  Input<unsigned> imageHeight;
  Input<half> antiAliasScale;
  Input<half> fov;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Make a lambda to consume random numbers from the buffer.
    // Each worker consumes random numbers using its ID as an
    // offset into the noise buffer:
    std::size_t randomIndex = workerId;
    auto rng = [&] () {
      const half value = antiAliasNoise[randomIndex];
      randomIndex += workerCount;
      return value;
    };

    // Get the descriptions of rays to be traced:
    auto rayCount = rays.size();
    const TraceRecord* traces = reinterpret_cast<const TraceRecord*>(&traceBuffer[0]);

    // Each worker will process one sixth of the rays so
    // we simply offset the start address based on worker ID:
    auto workerPtr = traces + workerId;
    // Outer loop is parallelised over the worker threads:
    float fovf32 = (float)*fov;
    for (auto k = 2 * workerId; k < rayCount; k += 2 * workerCount) {
      // Add anti-alias noise in pixel space:
      float c = workerPtr->u + (float)(*antiAliasScale * rng());
      float r = workerPtr->v + (float)(*antiAliasScale * rng());
      const Vec cam = light::pixelToRay(c, r, imageWidth, imageHeight, fovf32);
      rays[k]     = cam.x;
      rays[k + 1] = cam.y;
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
  Input<Vector<half>> uniform_0_1;
  Vector<Output<Vector<float>>, poplar::VectorLayout::ONE_PTR> contributionData;
  Input<half> refractiveIndex;
  Input<half> stopProb;
  Input<unsigned short> rouletteDepth;

  bool compute() {
    const Vec zero(0.f, 0.f, 0.f);
    const Vec one(1.f, 1.f, 1.f);

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
  Vector<InOut<Vector<float>>> contributionData;
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

#ifdef __IPU__

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

// Compute 2^y using dedicated HW instruction:
inline
float ipu_exp2(float y) {
  return __builtin_ipu_exp2(y);
}

#else

// Fallbacks for CPU targets:
inline
float ipu_powf(float x, float y) {
  return __builtin_powf(x, y);
}

inline
float ipu_exp2(float y) {
  return ipu_powf(2.f, y);
}

#endif

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

    // Note: number of trace records == contributionData.size()
    const float exposureScale = ipu_exp2(exposure);
    const float invGamma = 1.f / gamma;

    TraceRecord* traces = reinterpret_cast<TraceRecord*>(&traceBuffer[0]) + workerId;
    for (auto r = workerId; r < bgr.size(); r += workerCount, traces += workerCount) {
      auto v = bgr[r];
      Vec total(v[2], v[1], v[0]);

      // Apply tone-mapping:
      total.x *= exposureScale;
      total.y *= exposureScale;
      total.z *= exposureScale;
      total.x = ipu_powf(total.x, invGamma);
      total.y = ipu_powf(total.y, invGamma);
      total.z = ipu_powf(total.z, invGamma);

      // Scale and clip result into an unsigned byte range:
      total.x *= 255.f;
      total.y *= 255.f;
      total.z *= 255.f;

#ifdef __IPU__
      total.x = __builtin_ipu_min(total.x, 255.f);
      total.y = __builtin_ipu_min(total.y, 255.f);
      total.z = __builtin_ipu_min(total.z, 255.f);
      // total.x = __builtin_ipu_max(total.x, 0.f);
      // total.y = __builtin_ipu_max(total.y, 0.f);
      // total.z = __builtin_ipu_max(total.z, 0.f);
#else
      total.x = std::min(total.x, 255.f);
      total.y = std::min(total.y, 255.f);
      total.z = std::min(total.z, 255.f);
      // total.x = std::max(total.x, 0.f);
      // total.y = std::max(total.y, 0.f);
      // total.z = std::max(total.z, 0.f);
#endif

      traces->r = std::uint8_t(total.x);
      traces->g = std::uint8_t(total.y);
      traces->b = std::uint8_t(total.z);
    } // end loop over camera rays

    return true;
  }

};
