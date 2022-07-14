// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <light/src/vector.hpp>
#include <light/src/light.hpp>
#include <light/src/sdf.hpp>

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

#include "WrappedArray.hpp"
#include "TraceRecord.hpp"

// Because intrinsic/vectorised code can not be used with CPU
// or IpuModel targets we need to guard IPU optimised parts of
// the code so we can still support those:
#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

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

// NOTE: __builtin_ipu_urand_f32() is an IPU built-in which accesses the HW
// random number generator directly. It is not available in non-IPU compilation
// paths (e.g. for IPU Model or CPU simulation) so we need to put a guard and
// have a different implementation for those cases.
#ifdef __IPU__

inline
bool ipu_glossy_reflect(light::Ray& ray, light::Vector normal, float gloss) {
  light::reflect(ray, normal); // Perfect specular reflection (ray is modifed in place).

  // Basic scattering model:

  // Generate random point on unit sphere:
  auto noise = light::Vector(
    __builtin_ipu_urand_f32(),
    __builtin_ipu_urand_f32(),
    __builtin_ipu_urand_f32()).normalized();
  // Scale the unit noise vector randomly so it lies uniformly
  // in a ball with radius = gloss and add it to the reflected
  // ray direction:
  ray.direction += noise * (__builtin_ipu_urand_f32() * gloss);
  // Normalise the final ray direction (rays are assumed normalised):
  ray.direction = ray.direction.normalized();

  // We need to absorb rays that scatter at grazing angles or
  // below the surface otherwise we can get an unrealistic bright
  // halo for high gloss values:
  return ray.direction.dot(normal) < 0.f;
}

#else

inline
bool ipu_glossy_reflect(light::Ray& ray, light::Vector normal, float gloss) {
  // For the non-IPU version we can not call the built-in that accesses the
  // hardware so we only call the non-glossy specular reflect function
  // and return.
  //
  // We could use random numbers in the same way they are used elsewhere
  // if we wanted to maintain identical functionality (i.e. pre-compute
  // them and and consume them from a buffer).
  light::reflect(ray, normal);
  return false;
}

#endif

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
  Vector<Output<Vector<unsigned char>>, poplar::VectorLayout::ONE_PTR> contributionData;
  Input<half> refractiveIndex;
  Input<half> stopProb;
  Input<unsigned short> rouletteDepth;

  bool compute() {
    const Vec zero(0.f, 0.f, 0.f);
    const Vec one(1.f, 1.f, 1.f);
    const auto X = Vec(1.f, 0.f, 0.f);
    const auto Y = Vec(0.f, 1.f, 0.f);
    const auto Z = Vec(0.f, 0.f, 1.f);

    // The scene is currently hard coded here:
    light::Sphere spheres[] = {
      light::Sphere(Vec(-1.8575f, -0.98714f, -3.6f), 0.6f), // left
      light::Sphere(Vec(0.74795f, -0.55f, -4.3816f), 1.05f), // middle
      light::Sphere(Vec(1.9929f, -1.08666f, -3.23), 0.5f), // right
      light::Sphere(Vec(-0.19931, -1.183f, -2.75f), 0.4f), // front diffuse part
      light::Sphere(Vec(-0.19931, -1.183f, -2.75f), 0.4001f), // front clear-coat part
      //light::Sphere(Vec(12.f, 8.f, -5.3f), .3f) // light
    };

    light::Disc discs[] = {
      light::Disc(Y, Vec(0.f, -1.6f, -5.22f), 3.5f)
    };

    constexpr float lightStrength = 10000.f;
    const auto lightW = Vec(1.f, 1.f, 1.f) * lightStrength;

    const float metalGloss = 0.5f;
    const float colourGain = 2.f;
    const auto sphereColour = Vec(1.f, .89f, .55f) * colourGain;
    const auto clearCoatColour = Vec(.8f, .06f, .391f) * colourGain;
    const auto floorColour = Vec(.98f, .76f, .66f) * colourGain;
    const auto glassTint = Vec(0.75f, 0.75f, 0.75f);
    constexpr auto specular = light::Material::Type::specular;
    constexpr auto refractive = light::Material::Type::refractive;
    constexpr auto diffuse = light::Material::Type::diffuse;
    constexpr auto numObjects = std::size(spheres) + std::size(discs);
    light::Scene<numObjects> scene({
        light::Object(&spheres[0], sphereColour, zero, diffuse),
        light::Object(&spheres[1], one, zero, specular),
        light::Object(&spheres[2], glassTint, zero, refractive),
        light::Object(&spheres[3], clearCoatColour, zero, diffuse),
        light::Object(&spheres[4], one, zero, refractive),
        //light::Object(&spheres[3], zero, lightW, diffuse), // light
        light::Object(&discs[0], floorColour, zero, diffuse), // floor disc
    });

    // Make a lambda to consume random numbers from the buffer:
    std::size_t randomIndex = 0;
    auto rng = [&] () {
      const half value = uniform_0_1[randomIndex];
      randomIndex += 1;
      if (randomIndex == uniform_0_1.size()) {
        randomIndex = 0;
      }
      return value;
    };

    // Loop over the camera rays:
    const auto raysSize = (cameraRays.size() >> 1) << 1;
    for (auto r = 0u, c = 0u; r < raysSize; r += 2, c += 1) {
      // Unpack the camera ray directions which are stored as a
      // sequence of x, y coords with implicit z-direction of -1:
      Vec rayDir((float)cameraRays[r], (float)cameraRays[r+1], (float)-1.f);
      light::Ray ray(zero, rayDir);
      std::uint32_t depth = 0;

      // Store the contributions per ray by wrapping the raw vertex data with a stack data structure:
      const std::size_t maxContributions = contributionData[c].size() / sizeof(light::Contribution);
      light::Contribution* contributionDataPtr = reinterpret_cast<light::Contribution*>(&contributionData[c][0]);
      WrappedArray<light::Contribution> contributions(maxContributions, contributionDataPtr);

      // Trace rays through the scene, recording contribution values and type.
      bool hitEmitter = false;
      while (!contributions.full()) {
        // Russian roulette ray termination:
        float rrFactor = 1.f;
        if (depth >= rouletteDepth) {
          bool stop;
          std::tie(stop, rrFactor) = light::rouletteWeight((float)rng(), (float)*stopProb);
          if (stop) { break; }
        }

        // Intersect the ray with the whole scene, advancing it to the hit point:
        const auto intersection = scene.intersect(ray);
        if (!intersection) {
          // Record ray-direction with escaped rays so that the environment lighting can
          // be deferred until later.
          contributions.push_back({ray.direction, rrFactor, light::Contribution::Type::ESCAPED});
          hitEmitter = true;
          break;
        }

        if (intersection.material->emissive) {
          contributions.push_back({intersection.material->emission, rrFactor, light::Contribution::Type::EMIT});
          hitEmitter = true;
          break;
        }

        // Sample a new ray based on material type:
        if (intersection.material->type == diffuse) {
          const float sample1 = (float)rng();
          const float sample2 = (float)rng();
          const auto result =
            light::diffuse(ray, intersection.normal, intersection, rrFactor, sample1, sample2);
          contributions.push_back(result);
        } else if (intersection.material->type == specular) {
          bool absorbed = ipu_glossy_reflect(ray, intersection.normal, metalGloss);
          if (absorbed) {
            // Ray was absorbed so kill it:
            contributions.push_back({zero, 0.f, light::Contribution::Type::END});
            break;
          } else {
            contributions.push_back({zero, rrFactor, light::Contribution::Type::SPECULAR});
          }
        } else if (intersection.material->type == refractive) {
          const float ri = (float)*refractiveIndex;
          auto refracted = light::refract(ray, intersection.normal, ri, (float)rng());
          auto tint = refracted ? intersection.material->colour : one;
          contributions.push_back({tint, 1.15f * rrFactor, light::Contribution::Type::REFRACT});
        }

        depth += 1;
      }

      // Paths will only have a non-zero contribution if they hit a light source at some point:
      if (hitEmitter == false) {
        // No contribution so overwrite end of array with end marker:
        contributions.back() = light::Contribution{zero, 0.f, light::Contribution::Type::END};
      }
    } // end loop over camera rays

    return true;
  }
};

/// This codelet accumulates the lighting contributions backwards along
/// the ray paths recorded by the ray trace kernel.
///
/// The codelet is templated on the framebuffer type. If using half
/// precision it is the application's responsibility to avoid framebuffer
/// saturation: this avoids extra logic and computation in the codelet.
class AccumulateContributions : public Vertex {

public:
  Vector<Input<Vector<unsigned char>>> contributionData;
  InOut<Vector<unsigned char>> traceBuffer;

  bool compute() {
    const Vec zero(0.f, 0.f, 0.f);
    const auto numRays = contributionData.size();

    // Get the descriptions of rays to be traced.
    // Note: number of trace records == contributionData.size()
    TraceRecord* traces = reinterpret_cast<TraceRecord*>(&traceBuffer[0]);

    for (auto r = 0u; r < numRays; ++r, ++traces) {
      auto contributions = makeArrayWrapper<const light::Contribution>(contributionData[r]);
      const bool pathContributes = resizeContributionArray(contributions);

      traces->pathLength += contributions.size();

      if (pathContributes) {
        bool debug = false;
        Vec total = zero;
        while (!contributions.empty() && !debug) {
          auto c = contributions.back();
          contributions.pop_back();
          switch (c.type) {
          case light::Contribution::Type::DIFFUSE:
            // Diffuse materials modulate the colour being carried back
            // along the light path (scaled by the importance weight):
            total = total.cwiseProduct(c.clr) * c.weight;
            break;
            // Emitters add their colour to the colour being carried back
            // along the light path (scaled by the importance weight):
          case light::Contribution::Type::EMIT:
          case light::Contribution::Type::ESCAPED:
            total += c.clr * c.weight;
            break;
          // For refracted materials modulate by colour to simulate
          // refaction losses and then apply the importance weight:
          case light::Contribution::Type::REFRACT:
            total = total.cwiseProduct(c.clr) * c.weight;
            break;
          // Pure specular reflections have no colour contribution but
          // their importance sampling weights must still be applied:
          case light::Contribution::Type::SPECULAR:
            total *= c.weight;
            break;
          case light::Contribution::Type::DEBUG:
            debug = true;
            total = c.clr;
            break;
          case light::Contribution::Type::END:
          case light::Contribution::Type::SKIP:
          default:
            break;
          }
        }

        // Store the resulting colour contribution:
        traces->r += total.x;
        traces->g += total.y;
        traces->b += total.z;
      }

      traces->sampleCount += 1;
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
  Vector<InOut<Vector<unsigned char>>> contributionData;
  Input<float> azimuthalOffset;
  Output<Vector<float>> u;
  Output<Vector<float>> v;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Parallelise over all workers (each worker starts at a different offset):
    for (auto r = workerId; r < contributionData.size(); r += workerCount) {
      auto contributions = makeArrayWrapper<light::Contribution>(contributionData[r]);
      const bool pathContributes = resizeContributionArray(contributions);
      auto& c = contributions.back();

      if (c.type == light::Contribution::Type::ESCAPED) {
        // Pre for environment lighting calculation.
        auto rayDir = c.clr;
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
        c.clr = Vec(uCoord, vCoord, 0.f); // causes uv values to be rendered for debugging
        u[r] = uCoord;
        v[r] = vCoord;
      } else {
        // Avoid fp exceptions as these could otherwise remain uninitialised:
        u[r] = 0.f;
        v[r] = 0.f;
      }
    }

    return true;
  }

};

// Update escaped rays with the result of env-map lighting lookup:
class PostProcessEscapedRays : public MultiVertex {
public:
  Vector<InOut<Vector<unsigned char>>> contributionData;
  Vector<Input<Vector<float>>> bgr;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Parallelise over all workers (each worker starts at a different offset):
    for (auto r = workerId; r < contributionData.size(); r += workerCount) {
      auto contributions = makeArrayWrapper<light::Contribution>(contributionData[r]);
      const bool pathContributes = resizeContributionArray(contributions);
      auto& c = contributions.back();

      if (c.type == light::Contribution::Type::ESCAPED) {
        // Set the colour of the escaped ray:
        auto v = bgr[r];
        c.clr = Vec(v[2], v[1], v[0]);
      }
    }

    return true;
  }

};
