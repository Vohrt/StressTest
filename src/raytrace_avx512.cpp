// AVX-512 half of the ray tracer, isolated in its own translation unit.
//
// MSVC treats __m512 as a native 512-bit register only when the ISA baseline
// includes AVX-512; compiled under /arch:AVX2 every 512-bit value becomes a
// pair of 256-bit halves that spill to the stack around nearly every operation,
// which throttles this kernel by roughly an order of magnitude.  Building this
// one file with /arch:AVX512 fixes the code generation while the rest of the
// program keeps its /arch:AVX2 baseline.
//
// The entry points are reached exclusively through the CPUID dispatch in
// raytrace.cpp, so no CPU without AVX-512 can execute the code in this file.
#include "raytrace_kernel.h"

namespace cst {
namespace rt {

uint64_t trace_lines_f16_avx512(const Scene& sc, const Config& cfg, float* buf,
                                int width, int height, int y0, int y1, float jitter) {
    uint64_t rays = 0;
    detail::Tracer<simd::F16>::trace_lines(sc, cfg, buf, width, height, y0, y1,
                                           jitter, rays);
    return rays;
}

uint64_t trace_lines_d8_avx512(const Scene& sc, const Config& cfg, double* buf,
                               int width, int height, int y0, int y1, double jitter) {
    uint64_t rays = 0;
    detail::Tracer<simd::D8>::trace_lines(sc, cfg, buf, width, height, y0, y1,
                                          jitter, rays);
    return rays;
}

} // namespace rt
} // namespace cst
