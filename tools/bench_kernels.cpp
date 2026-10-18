// Time each kernel width in isolation, all compiled in one translation unit
// with this file's own /arch setting.
//
//   bench_kernels.exe <seconds> [f32|f64]
//
// This exists to settle whether the AVX-512 path really is slower than the AVX2
// one on this CPU, and to see whether 16-wide beats 8-wide when the ISA baseline
// matches.  The benchmark harness cannot answer that because its dispatch
// compares the same work measured through different code paths.
#include "../src/raytrace.h"
#include "../src/raytrace_kernel.h"
#include "../src/common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <typename V>
double time_one(const cst::rt::Scene& sc, const cst::rt::Config& cfg, double seconds,
                std::vector<typename cst::rt::detail::VT<V>::scalar>& buf) {
    using T = cst::rt::detail::VT<V>;
    using S = typename T::scalar;
    const int W = cfg.width, H = cfg.height;

    // Warm up, then run for the requested wall-clock window.
    uint64_t rays = 0;
    cst::rt::detail::Tracer<V>::trace_lines(sc, cfg, buf.data(), W, H, 0, 1,
                                            static_cast<S>(0), rays);
    const auto t0 = Clock::now();
    uint64_t total = 0;
    int passes = 0;
    for (;;) {
        for (int y = 0; y < H; y += 32) {
            const int y1 = (y + 32 < H) ? y + 32 : H;
            uint64_t r = 0;
            cst::rt::detail::Tracer<V>::trace_lines(sc, cfg, buf.data(), W, H, y, y1,
                                                    static_cast<S>(0), r);
            total += r;
        }
        ++passes;
        const double el = std::chrono::duration<double>(Clock::now() - t0).count();
        if (el >= seconds) break;
    }
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    (void)passes;
    return static_cast<double>(total) / el / 1000.0;   // kRay/s
}

} // namespace

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 3.0;
    const bool f64 = (argc > 2 && std::strcmp(argv[2], "f64") == 0);

    cst::rt::Config cfg;
    cfg.width = 1024;
    cfg.height = 1024;
    cfg.ss = 1;
    cfg.depth = 15;
    cfg.scene = 0;
    const cst::rt::Scene sc = cst::rt::make_scene(cfg.scene, cfg.seed);

    const size_t n = static_cast<size_t>(cfg.width) * cfg.height * 3;
    std::printf("workload 1024x1024 ss=1 depth=15 scene=0, %.1f s per kernel, 1 thread\n\n",
                seconds);

    if (!f64) {
        std::vector<float> b8(n), b16(n);
        std::printf("  F8  (AVX2   8 x float) : %8.2f k KRay/s\n",
                    time_one<cst::simd::F8>(sc, cfg, seconds, b8));
        std::printf("  F16 (AVX-512 16 x float): %8.2f k KRay/s\n",
                    time_one<cst::simd::F16>(sc, cfg, seconds, b16));
        // Do the two agree?
        std::printf("  byte-identical output    : %s\n",
                    std::memcmp(b8.data(), b16.data(), n * sizeof(float)) == 0 ? "yes" : "no");
    } else {
        std::vector<double> b4(n), b8(n);
        std::printf("  D4 (AVX2   4 x double) : %8.2f k KRay/s\n",
                    time_one<cst::simd::D4>(sc, cfg, seconds, b4));
        std::printf("  D8 (AVX-512 8 x double): %8.2f k KRay/s\n",
                    time_one<cst::simd::D8>(sc, cfg, seconds, b8));
        std::printf("  byte-identical output  : %s\n",
                    std::memcmp(b4.data(), b8.data(), n * sizeof(double)) == 0 ? "yes" : "no");
    }
    return 0;
}
