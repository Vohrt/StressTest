// Re-implements the AIDA64-aligned fractal kernels (src/fractal_kernel.h)
// scalar-ly, in the same evaluation order and precision, and compares them
// pixel by pixel with the SIMD implementations the benchmark runs.
#include "../src/fractal_kernel.h"

#include <cmath>
#include <cstdio>
#include <vector>

using cst::fractal::Kind;
using cst::fractal::Params;

namespace {

int scalar_count(const Params& p, double re, double im) {
    switch (p.kind) {
        case Kind::Julia: {
            float zr = static_cast<float>(re);
            float zi = static_cast<float>(im);
            const float cr = static_cast<float>(p.cre);
            const float ci = static_cast<float>(p.cim);
            int count = 0;
            for (int it = 0; it < p.max_iter; ++it) {
                if (!(zr * zr + zi * zi < 4.0f)) break;
                ++count;
                const float nzr = zr * zr - zi * zi + cr;
                zi = 2.0f * zr * zi + ci;
                zr = nzr;
            }
            return count;
        }
        case Kind::Mandel: {
            double zr = re;
            double zi = im;
            const double cr = zr, ci = zi;   // z0 = c = pixel
            int count = 0;
            for (int it = 0; it < p.max_iter; ++it) {
                if (!(zr * zr + zi * zi < 4.0)) break;
                ++count;
                const double nzr = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = nzr;
            }
            return count;
        }
        case Kind::SinJulia: {
            double zr = re;
            double zi = im;
            int count = 0;
            for (int it = 0; it < p.max_iter; ++it) {
                const double wr = std::sin(zr) * std::cosh(zi);
                const double wi = std::cos(zr) * std::sinh(zi);
                if (wr + wi >= 64.0) break;
                const double nre = p.cre * wr - p.cim * wi;
                const double nim = p.cre * wi + p.cim * wr;
                if (std::fabs(nim) >= 708.390625) break;
                zr = nre;
                zi = nim;
                ++count;
            }
            return count;
        }
    }
    return 0;
}

// Mirrors the kernel's x-stepping exactly: f32 accumulation for the Julia
// pipeline, f64 for Mandel and SinJulia.
template <typename T>
void scalar_row(const Params& p, const cst::fractal::View& v, int y,
                std::vector<float>& out) {
    T xr = static_cast<T>(v.cre) + static_cast<T>(y) * static_cast<T>(v.rrx);
    T xi = static_cast<T>(v.cim) + static_cast<T>(y) * static_cast<T>(v.rix);
    const float inv_iter = 1.0f / static_cast<float>(p.max_iter);
    for (int x = 0; x < p.w; ++x) {
        out[x] = static_cast<float>(scalar_count(p, static_cast<double>(xr),
                                                 static_cast<double>(xi))) * inv_iter;
        xr += static_cast<T>(v.crx);
        xi += static_cast<T>(v.cix);
    }
}

int check(const Params& p, int frame) {
    const cst::fractal::View v =
        cst::fractal::make_view(p, cst::fractal::frame_angle(frame));
    const int rows[] = {0, 1, p.h / 3, p.h / 2, p.h - 2, p.h - 1};
    const int nrows = static_cast<int>(sizeof(rows) / sizeof(rows[0]));
    long mismatches = 0;
    for (int i = 0; i < nrows; ++i) {
        std::vector<float> simd(static_cast<size_t>(p.w));
        cst::fractal::render_row(p, v, rows[i], simd.data());
        std::vector<float> ref(static_cast<size_t>(p.w));
        if (p.kind == Kind::Julia) scalar_row<float>(p, v, rows[i], ref);
        else scalar_row<double>(p, v, rows[i], ref);
        for (int x = 0; x < p.w; ++x)
            if (simd[x] != ref[x]) ++mismatches;
    }
    std::printf("  frame %3d: %d rows x %d pixels, mismatches=%ld\n",
                frame, nrows, p.w, mismatches);
    return mismatches == 0 ? 0 : 1;
}

} // namespace

int main() {
    int failures = 0;
    const struct { Kind kind; const char* name; } kinds[] = {
        {Kind::Julia, "julia"},
        {Kind::Mandel, "mandel"},
        {Kind::SinJulia, "sinjulia"},
    };
    for (const auto& k : kinds) {
        const Params p = cst::fractal::make_params(k.kind);
        std::printf("%s (%dx%d iter=%d)\n", k.name, p.w, p.h, p.max_iter);
        failures += check(p, 0);
        failures += check(p, 123);
    }
    std::printf("%s\n", failures ? "FAILED" : "all SIMD rows match the scalar reference");
    return failures ? 1 : 0;
}
