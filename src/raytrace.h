// Analytic SIMD ray tracer used by the FP32 / FP64 Ray-Trace benchmarks.
//
// The workload follows the shape of AIDA64's JIT ray-trace kernel (the
// "Base512_AVX512F" code dump): every primary ray walks the whole primitive
// list for the nearest hit, each point light fires a shadow ray that repeats
// the same walk against the segment towards the light, surfaces are shaded
// with Lambert plus a pow() Phong highlight, and one recursively traced
// reflection ray is weighted by the surface's mirror term until the depth
// budget runs out.  16-wide single precision and 8-wide double precision use
// AVX-512 when the CPU has it; the AVX2 paths stay as the fallback.
#pragma once

#include <cstdint>
#include <vector>

namespace cst {
namespace rt {

// AIDA64 exposes these exact knobs; defaults mirror the reference kernel:
// 1024x1024, one sample per pixel and a depth budget of 15 generations.
struct Config {
    int width = 1024;        // "RayTraceX"
    int height = 1024;       // "RayTraceY"
    int ss = 1;               // supersampling per axis (ss*ss rays per pixel)
    int depth = 15;           // "RayTraceDepth" - max ray generations
    int scene = 0;            // "RayTraceScene" - 0 = reference layout, 1..8 = sweeps
    uint64_t seed = 0x5eed1234ull;
};

// Sphere soup plus the point lights and pinhole camera that frame it.
//
// The camera maps pixel (x, y) to screen space as
//     sx = (x + 1) / width - ndc_x        (negative-left)
//     sy = ndc_y - (y + 1) / height       (negative-down)
// and shoots rays from `eye` along
//     D = eye + (sx * ray_scale) * right + (sy * ray_scale) * up + focal * fwd.
// D is deliberately not normalised: the hit test accepts only parameters in
// [near_clip, 1], which turns each intersection into a segment test and keeps
// the parameterisation identical to the reference kernel's.
struct Scene {
    int sphere_count = 0;
    std::vector<double> sx, sy, sz, sr;   // centre + radius
    std::vector<double> sr2, invr;        // precomputed r^2 and 1/r
    std::vector<double> shininess;        // Phong exponent
    std::vector<double> cr, cg, cb;       // albedo
    std::vector<double> mirror;           // per-sphere reflectivity, 0 = diffuse

    // Point lights (white).  The reference scene uses exactly two; the array
    // leaves headroom for the complexity sweeps.
    int light_count = 2;
    double lx[4] = {-50.0, 40.0, 0.0, 0.0};
    double ly[4] = {100.0, 40.0, 0.0, 0.0};
    double lz[4] = {-50.0, 150.0, 0.0, 0.0};

    double eye[3] = {0.0, 6.0, -17.0};
    double right[3] = {0.924678, 0.0, 0.0};
    double up[3] = {0.0, 0.855029, 0.352071};
    double fwd[3] = {0.0, -0.38075, 0.924678};
    double focal = 2546.48;      // long focal length: a narrow, deep view
    double ndc_x = 0.5;
    double ndc_y = 0.65;
    double ray_scale = 1000.0;   // screen-space to lateral direction scale
    double near_clip = 5e-06;    // lower end of the accepted parameter window
};

Scene make_scene(int complexity, uint64_t seed);

// Trace scanlines [line_begin, line_end) of the frame into `buf`
// (width*height*3 interleaved channels).  Returns the number of rays traced.
uint64_t trace_frame_f64(const Scene& scene, const Config& cfg, double* buf,
                         int line_begin, int line_end);
uint64_t trace_frame_f32(const Scene& scene, const Config& cfg, float* buf,
                         int line_begin, int line_end);

// As above, but shifts every primary ray by `jitter` sub-pixel units so repeated
// calls perform genuinely different intersection work.  This is how the harness
// prevents the compiler from hoisting or memoising a repeated frame.
uint64_t trace_frame_jittered_f64(const Scene& scene, const Config& cfg, double* buf,
                                  int line_begin, int line_end, double jitter);
uint64_t trace_frame_jittered_f32(const Scene& scene, const Config& cfg, float* buf,
                                  int line_begin, int line_end, double jitter);

const char* f64_method_name();
const char* f32_method_name();

// True when the wide AVX-512 kernels are selected (requires a CPU with
// AVX-512 and CST_FORCE_AVX512=1).  The AVX2+FMA kernels are the default: the
// wide instantiation is 2-3x slower under MSVC's 512-bit register allocation,
// while the two paths produce bit-identical output (tools/verify_rt.cpp).
bool avx512_selected();

} // namespace rt
} // namespace cst
