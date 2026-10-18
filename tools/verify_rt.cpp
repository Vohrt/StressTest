// Trace a small patch and report the colour statistics, to confirm the ray
// tracer produces a real image rather than a constant.
#include "../src/raytrace.h"

#include <cstdio>
#include <vector>

int main() {
    cst::rt::Config cfg;
    cfg.width = 320;
    cfg.height = 240;
    cfg.ss = 2;
    cfg.depth = 8;
    cfg.scene = 2;
    const cst::rt::Scene sc = cst::rt::make_scene(cfg.scene, cfg.seed);

    std::vector<double> buf(static_cast<size_t>(cfg.width) * cfg.height * 3, 0.0);
    const uint64_t rays = cst::rt::trace_frame_f64(sc, cfg, buf.data(), 0, cfg.height);

    double mn = 1e30, mx = -1e30, sum = 0.0;
    int nonzero = 0;
    for (double v : buf) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        if (v != 0.0) ++nonzero;
    }
    std::printf("rays=%llu pixels=%zu channels=%zu\n",
                static_cast<unsigned long long>(rays), buf.size() / 3, buf.size());
    std::printf("min=%.6f max=%.6f mean=%.6f nonzero=%d\n", mn, mx, sum / buf.size(), nonzero);

    // A centre row sample.
    const size_t y = cfg.height / 2;
    std::printf("centre row (first 8 px):");
    for (int x = 0; x < 8; ++x) {
        const size_t i = (static_cast<size_t>(y) * cfg.width + x) * 3;
        std::printf(" (%.3f,%.3f,%.3f)", buf[i], buf[i + 1], buf[i + 2]);
    }
    std::printf("\n");
    return 0;
}
