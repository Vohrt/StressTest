#pragma once
// ===========================================================================
// NGFractal kernels, algorithmically aligned with AIDA64 8.40.8500.
//
// Everything below mirrors what aida_bench64.dll's NGFractal method tables
// actually execute (evidence: docs/reverse-engineering.md):
//
//  * Julia    - 1024x1024, 999 iterations, f32 pipeline.  z0 = pixel,
//               c = -0.770885 + 0.1i, escape test |z|^2 < 4 evaluated before
//               each update; the pixel counts how many tests held, so interior
//               pixels reach 999.
//  * Mandel   - 1024x1024, 999 iterations, f64 pipeline.  z0 = c = pixel with
//               the same escape test.  Pixels with |c| >= 2 fail the very
//               first test and count 0 - the black disc outside the radius-2
//               circle in AIDA64's own frame dumps.
//  * SinJulia - 256x256, 70 iterations, x87 extended-precision pipeline
//               (emulated in double here).  z0 = pixel, c = 0.98059 + 0.45075i,
//               z' = c*sin(z); the pixel counts completed iterations, stopping
//               when Re(sin z) + Im(sin z) >= 64 or |Im z'| >= 708.390625.
//
// View geometry (all three methods): a square window of half-extent `ext`
// centred on (cx, cy) is rasterised from its top-left corner with no
// half-pixel offset, then rotated about the complex origin by pi/400 radians
// per frame.  AIDA64 animates every benchmark frame this way; the per-frame
// pixel-iteration count never changes.
// ===========================================================================
#include "simd.h"

#include <cmath>
#include <cstdint>

namespace cst {
namespace fractal {

enum class Kind { Julia, Mandel, SinJulia };

struct Params {
    Kind kind;
    int w = 0, h = 0;      // image size, pixels
    int max_iter = 0;      // per-pixel iteration budget
    double cx = 0.0, cy = 0.0;  // view centre in the complex plane
    double ext = 0.0;      // view half-extent (square window)
    double cre = 0.0, cim = 0.0;  // Julia / SinJulia constant
};

// Exact parameters as decoded from the class init functions (float immediates
// are reproduced bit-for-bit; e.g. Julia's extent is the f32 0x3FC51EB8).
inline Params make_params(Kind kind) {
    Params p;
    p.kind = kind;
    switch (kind) {
        case Kind::Julia:
            p.w = 1024; p.h = 1024; p.max_iter = 999;
            p.cx = 0.0; p.cy = 0.0; p.ext = 1.5399999618530273;
            p.cre = -0.7708849906921387;
            p.cim = 0.10000000149011612;
            break;
        case Kind::Mandel:
            p.w = 1024; p.h = 1024; p.max_iter = 999;
            p.cx = -0.5; p.cy = 0.0; p.ext = 1.5;
            break;
        case Kind::SinJulia:
            p.w = 256; p.h = 256; p.max_iter = 70;
            p.cx = 0.0; p.cy = 0.0; p.ext = 0.005;
            p.cre = 0.98059082;   // 0x3FEF60FFFFD50CE2
            p.cim = 0.450746629;  // 0x3FDCD9086395956D
            break;
    }
    return p;
}

// AIDA64 spins the view by pi/400 radians per frame (the f32 constant 0x3C00ADFD;
// the double pipelines use 0x3F8015BF9217271A, printed below).
inline double frame_angle(int frame) {
    return frame * 0.007853981633974483;
}

// Rotated view frame: the top-left corner plus the per-pixel column/row steps,
// expressed in the complex plane.  Pixel (x, y) maps to
// corner + x*colstep + y*rowstep, with rowstep pointing down.
struct View {
    double cre = 0.0, cim = 0.0;  // top-left corner
    double crx = 0.0, cix = 0.0;  // column step
    double rrx = 0.0, rix = 0.0;  // row step
};

inline View make_view(const Params& p, double theta) {
    const double s = p.ext / (p.w * 0.5);          // per-pixel step
    const double x0 = p.cx - (p.w * 0.5) * s;      // unrotated corner, top-left
    const double y0 = p.cy + (p.h * 0.5) * s;
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    View v;
    v.cre = x0 * ct - y0 * st;
    v.cim = x0 * st + y0 * ct;
    v.crx = s * ct;  v.cix = s * st;
    v.rrx = s * st;  v.rix = -s * ct;
    return v;
}

// ---------------------------------------------------------------------------
// Julia: f32 pipeline (the reference kernel keeps its whole state block in
// floats, so the pixel mapping is float too).
// ---------------------------------------------------------------------------
inline void julia_row(const Params& p, const View& v, int y, float* row) {
    using simd::F8;
    const float cxs = static_cast<float>(v.crx);
    const float cys = static_cast<float>(v.cix);
    const float rsx = static_cast<float>(v.rrx);
    const float rsy = static_cast<float>(v.rix);
    const F8 vcr(static_cast<float>(p.cre));
    const F8 vci(static_cast<float>(p.cim));
    const F8 four(4.0f), one(1.0f), zero(0.0f), two(2.0f);
    const float inv_iter = 1.0f / static_cast<float>(p.max_iter);

    float xr = static_cast<float>(v.cre) + static_cast<float>(y) * rsx;
    float xi = static_cast<float>(v.cim) + static_cast<float>(y) * rsy;
    for (int x = 0; x < p.w; x += 8) {
        float lr[8], li[8];
        for (int k = 0; k < 8; ++k) {
            lr[k] = xr;
            li[k] = xi;
            xr += cxs;
            xi += cys;
        }
        F8 zr(lr[0], lr[1], lr[2], lr[3], lr[4], lr[5], lr[6], lr[7]);
        F8 zi(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7]);
        F8 count = F8::zero();

        for (int it = 0; it < p.max_iter; ++it) {
            const F8 inside = simd::cmp_lt(zr * zr + zi * zi, four);
            if (!simd::any_true(inside)) break;
            count = count + simd::select(inside, one, zero);
            const F8 nzr = zr * zr - zi * zi + vcr;
            zi = two * zr * zi + vci;
            zr = nzr;
        }

        float out[8];
        count.store(out);
        const int n = (p.w - x < 8) ? (p.w - x) : 8;
        for (int k = 0; k < n; ++k) row[x + k] = out[k] * inv_iter;
    }
}

// ---------------------------------------------------------------------------
// Mandel: f64 pipeline, z0 = c = pixel (that is what the AIDA64 kernel loads
// into both its "z" and its "c" registers).
// ---------------------------------------------------------------------------
inline void mandel_row(const Params& p, const View& v, int y, float* row) {
    using simd::D4;
    const D4 four(4.0), one(1.0), zero(0.0), two(2.0);
    const float inv_iter = 1.0f / static_cast<float>(p.max_iter);

    double xr = v.cre + static_cast<double>(y) * v.rrx;
    double xi = v.cim + static_cast<double>(y) * v.rix;
    for (int x = 0; x < p.w; x += 4) {
        double lr[4], li[4];
        for (int k = 0; k < 4; ++k) {
            lr[k] = xr;
            li[k] = xi;
            xr += v.crx;
            xi += v.cix;
        }
        D4 zr(lr[0], lr[1], lr[2], lr[3]);
        D4 zi(li[0], li[1], li[2], li[3]);
        const D4 cr = zr, ci = zi;   // z0 = c = pixel
        D4 count = D4::zero();

        for (int it = 0; it < p.max_iter; ++it) {
            const D4 inside = simd::cmp_lt(zr * zr + zi * zi, four);
            if (!simd::any_true(inside)) break;
            count = count + simd::select(inside, one, zero);
            const D4 nzr = zr * zr - zi * zi + cr;
            zi = two * zr * zi + ci;
            zr = nzr;
        }

        double out[4];
        count.store(out);
        const int n = (p.w - x < 4) ? (p.w - x) : 4;
        for (int k = 0; k < n; ++k)
            row[x + k] = static_cast<float>(out[k]) * inv_iter;
    }
}

// ---------------------------------------------------------------------------
// SinJulia: scalar double transcendental loop (the reference runs the same
// formulas through x87 80-bit arithmetic).
// ---------------------------------------------------------------------------
inline void sinjulia_row(const Params& p, const View& v, int y, float* row) {
    const double t_sum = 64.0;          // Re(sin z) + Im(sin z) escape (f32 64.0f)
    const double t_comp = 708.390625;   // |Im z'| escape (0x4086232000000000)
    const float inv_iter = 1.0f / static_cast<float>(p.max_iter);

    double xr = v.cre + static_cast<double>(y) * v.rrx;
    double xi = v.cim + static_cast<double>(y) * v.rix;
    for (int x = 0; x < p.w; ++x) {
        double re = xr;
        double im = xi;
        xr += v.crx;
        xi += v.cix;

        int count = 0;
        for (int it = 0; it < p.max_iter; ++it) {
            const double wr = std::sin(re) * std::cosh(im);  // sin(z) = sin(x)cosh(y) + i cos(x)sinh(y)
            const double wi = std::cos(re) * std::sinh(im);
            if (wr + wi >= t_sum) break;
            const double nre = p.cre * wr - p.cim * wi;      // z' = c * sin(z)
            const double nim = p.cre * wi + p.cim * wr;
            if (std::fabs(nim) >= t_comp) break;
            re = nre;
            im = nim;
            ++count;
        }
        row[x] = static_cast<float>(count) * inv_iter;
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers.
// ---------------------------------------------------------------------------
inline void render_row(const Params& p, const View& v, int y, float* row) {
    switch (p.kind) {
        case Kind::Julia:    julia_row(p, v, y, row); break;
        case Kind::Mandel:   mandel_row(p, v, y, row); break;
        case Kind::SinJulia: sinjulia_row(p, v, y, row); break;
    }
}

// Renders rows [y0, y1) into `out`, which holds row y0 first (stride floats
// per row).
inline void render_rows(const Params& p, const View& v, int y0, int y1,
                        float* out, int stride) {
    for (int y = y0; y < y1; ++y)
        render_row(p, v, y, out + static_cast<size_t>(y - y0) * stride);
}

// Renders one full frame at the given animation frame index.
inline void render_frame(const Params& p, int frame, float* out, int stride) {
    const View v = make_view(p, frame_angle(frame));
    render_rows(p, v, 0, p.h, out, stride);
}

} // namespace fractal
} // namespace cst
