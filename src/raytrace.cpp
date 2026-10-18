// Analytic SIMD ray tracer used by the FP32 / FP64 Ray-Trace benchmarks.
//
// Scene construction and the runtime dispatch between the AVX2 and AVX-512
// kernels live here; the kernel itself is in raytrace_kernel.h.
#include "raytrace.h"

#include "common.h"
#include "raytrace_kernel.h"

#include <cstdlib>

namespace cst {
namespace rt {

namespace {

// CST_FORCE_AVX512=1 selects the wide AVX-512 kernels even though they are not
// the default: measured on the reference machine (Ryzen 7 9800X3D, MSVC 14.51)
// the AVX-512 instantiation of this kernel is 2-3x slower than the AVX2+FMA
// one.  MSVC's register allocator splits 512-bit live ranges and rebuilds them
// through the stack around nearly every operation, while the hardware itself is
// not the limit (zmm div/sqrt run at full rate here).  The AVX2 kernels are
// therefore the default even on CPUs with AVX-512; this switch keeps the wide
// path reachable for experiments and for the bit-for-bit cross-check in
// tools/verify_rt.cpp.
bool force_avx512() {
    static const bool v = [] {
        const char* s = std::getenv("CST_FORCE_AVX512");
        return s != nullptr && s[0] != '0';
    }();
    return v;
}

void add_sphere(Scene& s, double x, double y, double z, double r,
                double cr, double cg, double cb, double shin, double refl) {
    s.sx.push_back(x);
    s.sy.push_back(y);
    s.sz.push_back(z);
    s.sr.push_back(r);
    s.sr2.push_back(r * r);
    s.invr.push_back(1.0 / r);
    s.cr.push_back(cr);
    s.cg.push_back(cg);
    s.cb.push_back(cb);
    s.shininess.push_back(shin);
    s.mirror.push_back(refl);
    s.sphere_count = static_cast<int>(s.sx.size());
}

} // namespace

bool avx512_selected() {
    static const bool v = [] {
        return cpu_info().has_avx512dq && force_avx512();
    }();
    return v;
}

Scene make_scene(int complexity, uint64_t seed) {
    Scene s;

    // The ground is one huge sphere whose surface passes through y = -1, the
    // same construction (and the same shading values) as the reference scene
    // AIDA64's ray-trace kernel is generated for.
    add_sphere(s, 0.0, -1000.0, 2.0, 999.0, 0.1, 0.2, 0.6, 80.0, 0.8);

    if (complexity <= 0) {
        // Reference layout: three glossy spheres floating over the floor,
        // framed by the default camera and lit by the two default point lights.
        add_sphere(s,  0.0,  0.0,  2.0, 1.0, 1.00, 0.50, 0.10, 60.0, 0.7);
        add_sphere(s,  1.5, -0.4,  0.0, 0.6, 0.10, 0.85, 1.00, 50.0, 0.4);
        add_sphere(s, -1.5, -0.3, -1.0, 0.7, 1.00, 0.20, 0.05, 50.0, 0.3);
        return s;
    }

    // Complexity levels 1..8 add four spheres per level, placed from the same
    // splitmix64 stream as the project's other scenes so runs stay
    // reproducible.  Every level keeps the floor and the two point lights.
    uint64_t st = seed ^ (0x9E3779B97F4A7C15ull * static_cast<uint64_t>(complexity));
    auto rnd = [&st]() { return splitmix64(st); };
    auto unit = [&rnd]() {
        return static_cast<double>(rnd() >> 11) * (1.0 / 9007199254740992.0);
    };
    const int extra = 4 * complexity;
    for (int i = 0; i < extra; ++i) {
        const double r = 0.22 + unit() * 0.33;
        const double x = (unit() - 0.5) * 5.0;
        const double z = -1.5 + unit() * 6.0;
        add_sphere(s, x, -1.0 + r, z, r,
                   0.15 + unit() * 0.85, 0.15 + unit() * 0.85, 0.15 + unit() * 0.85,
                   20.0 + unit() * 60.0, 0.15 + unit() * 0.65);
    }
    return s;
}

uint64_t trace_frame_f64(const Scene& scene, const Config& cfg, double* buf,
                         int line_begin, int line_end) {
    return trace_frame_jittered_f64(scene, cfg, buf, line_begin, line_end, 0.0);
}

uint64_t trace_frame_f32(const Scene& scene, const Config& cfg, float* buf,
                         int line_begin, int line_end) {
    return trace_frame_jittered_f32(scene, cfg, buf, line_begin, line_end, 0.0);
}

uint64_t trace_frame_jittered_f64(const Scene& scene, const Config& cfg, double* buf,
                                  int line_begin, int line_end, double jitter) {
    uint64_t rays = 0;
    if (avx512_selected()) {
        rays = trace_lines_d8_avx512(scene, cfg, buf, cfg.width, cfg.height,
                                     line_begin, line_end, static_cast<double>(jitter));
    } else {
        detail::Tracer<simd::D4>::trace_lines(scene, cfg, buf, cfg.width, cfg.height,
                                              line_begin, line_end,
                                              static_cast<double>(jitter), rays);
    }
    return rays;
}

uint64_t trace_frame_jittered_f32(const Scene& scene, const Config& cfg, float* buf,
                                  int line_begin, int line_end, double jitter) {
    uint64_t rays = 0;
    const float j = static_cast<float>(jitter);
    if (avx512_selected()) {
        rays = trace_lines_f16_avx512(scene, cfg, buf, cfg.width, cfg.height,
                                      line_begin, line_end, j);
    } else {
        detail::Tracer<simd::F8>::trace_lines(scene, cfg, buf, cfg.width, cfg.height,
                                              line_begin, line_end, j, rays);
    }
    return rays;
}

const char* f64_method_name() {
    if (avx512_selected()) return "Base512_AVX512F_F64";
    if (cpu_info().has_fma) return "Base256_AVX2_FMA3_F64";
    return "Base256_AVX2_F64";
}

const char* f32_method_name() {
    if (avx512_selected()) return "Base512_AVX512F_F32";
    if (cpu_info().has_fma) return "Base256_AVX2_FMA3_F32";
    return "Base256_AVX2_F32";
}

} // namespace rt
} // namespace cst
