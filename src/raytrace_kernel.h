// Internal ray-tracer core: vector traits and the templated kernel.
//
// One tracer drives four hardware widths - 4x double and 8x float on AVX2, and
// 8x double / 16x float on AVX-512 - through a small trait layer.  The kernel
// mirrors the structure of AIDA64's JIT ray-trace code dump:
//
//   * every ray sweeps all spheres for the nearest hit inside a [near, 1]
//     parameter window (a segment test, so shadow rays can share the maths);
//   * each white point light gets a shadow ray that repeats the sweep;
//   * shading is Lambert diffuse plus a real pow() Phong highlight;
//   * one reflected ray per bounce, recursively weighted by the mirror term,
//     until the depth budget runs out.
//
// Trait note: D4/F8 use all-ones-lane vector masks (their natural comparison
// output), while the AVX-512 types carry real mask registers.
#pragma once

#include "raytrace.h"
#include "simd.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace cst {
namespace rt {
namespace detail {

// ---------------------------------------------------------------------------
// Hardware traits
// ---------------------------------------------------------------------------
template <typename V> struct VT;

// ---- 4 x double (AVX2) ----------------------------------------------------
template <> struct VT<simd::D4> {
    using scalar = double;
    using V = simd::D4;
    using M = simd::D4;
    static constexpr int lanes = 4;

    static V set1(scalar s) { return V(s); }
    static V zero() { return V::zero(); }
    static M mask_full() { return M(_mm256_castsi256_pd(_mm256_set1_epi64x(-1))); }
    static M mask_zero() { return V::zero(); }
    static V load(const scalar* p) { return V::load(p); }
    static void store(scalar* p, V a) { a.store(p); }

    static V add(V a, V b) { return simd::add(a, b); }
    static V sub(V a, V b) { return simd::sub(a, b); }
    static V mul(V a, V b) { return simd::mul(a, b); }
    static V div(V a, V b) { return simd::div(a, b); }
    static V madd(V a, V b, V c) { return simd::madd(a, b, c); }
    static V msub(V a, V b, V c) { return simd::msub(a, b, c); }
    static V minv(V a, V b) { return simd::minv(a, b); }
    static V maxv(V a, V b) { return simd::maxv(a, b); }
    static V sqrtv(V a) { return simd::sqrtv(a); }
    static V absv(V a) { return simd::absv(a); }
    static V powv(V x, V y) { return simd::powv(x, y); }

    static M lt(V a, V b) { return simd::cmp_lt(a, b); }
    static M gt(V a, V b) { return simd::cmp_gt(a, b); }
    static M ge(V a, V b) { return simd::cmp_ge(a, b); }
    static M eq(V a, V b) { return M(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ)); }
    static M and_(M a, M b) { return simd::mask_and(a, b); }
    static M or_(M a, M b) { return M(_mm256_or_pd(a.v, b.v)); }
    static M andnot(M a, M b) { return M(_mm256_andnot_pd(a.v, b.v)); }  // ~a & b
    static V select(M m, V a, V b) { return simd::select(m, a, b); }
    static bool any(M m) { return simd::any_true(m); }
    static bool all(M m) { return _mm256_movemask_pd(m.v) == 0xF; }
};

// ---- 8 x float (AVX2) -----------------------------------------------------
template <> struct VT<simd::F8> {
    using scalar = float;
    using V = simd::F8;
    using M = simd::F8;
    static constexpr int lanes = 8;

    static V set1(scalar s) { return V(s); }
    static V zero() { return V::zero(); }
    static M mask_full() { return M(_mm256_castsi256_ps(_mm256_set1_epi32(-1))); }
    static M mask_zero() { return V::zero(); }
    static V load(const scalar* p) { return V::load(p); }
    static void store(scalar* p, V a) { a.store(p); }

    static V add(V a, V b) { return simd::add(a, b); }
    static V sub(V a, V b) { return simd::sub(a, b); }
    static V mul(V a, V b) { return simd::mul(a, b); }
    static V div(V a, V b) { return simd::div(a, b); }
    static V madd(V a, V b, V c) { return simd::madd(a, b, c); }
    static V msub(V a, V b, V c) { return simd::msub(a, b, c); }
    static V minv(V a, V b) { return simd::minv(a, b); }
    static V maxv(V a, V b) { return simd::maxv(a, b); }
    static V sqrtv(V a) { return simd::sqrtv(a); }
    static V absv(V a) { return simd::absv(a); }
    static V powv(V x, V y) { return simd::powv(x, y); }

    static M lt(V a, V b) { return simd::cmp_lt(a, b); }
    static M gt(V a, V b) { return simd::cmp_gt(a, b); }
    static M ge(V a, V b) { return simd::cmp_ge(a, b); }
    static M eq(V a, V b) { return M(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)); }
    static M and_(M a, M b) { return simd::mask_and(a, b); }
    static M or_(M a, M b) { return M(_mm256_or_ps(a.v, b.v)); }
    static M andnot(M a, M b) { return M(_mm256_andnot_ps(a.v, b.v)); }
    static V select(M m, V a, V b) { return simd::select(m, a, b); }
    static bool any(M m) { return simd::any_true(m); }
    static bool all(M m) { return _mm256_movemask_ps(m.v) == 0xFF; }
};

// ---- 16 x float (AVX-512F) ------------------------------------------------
template <> struct VT<simd::F16> {
    using scalar = float;
    using V = simd::F16;
    using M = __mmask16;
    static constexpr int lanes = 16;

    static V set1(scalar s) { return V(s); }
    static V zero() { return V::zero(); }
    static M mask_full() { return static_cast<M>(0xFFFF); }
    static M mask_zero() { return static_cast<M>(0); }
    static V load(const scalar* p) { return V::load(p); }
    static void store(scalar* p, V a) { a.store(p); }

    static V add(V a, V b) { return simd::add(a, b); }
    static V sub(V a, V b) { return simd::sub(a, b); }
    static V mul(V a, V b) { return simd::mul(a, b); }
    static V div(V a, V b) { return simd::div(a, b); }
    static V madd(V a, V b, V c) { return simd::madd(a, b, c); }
    static V msub(V a, V b, V c) { return simd::msub(a, b, c); }
    static V minv(V a, V b) { return simd::minv(a, b); }
    static V maxv(V a, V b) { return simd::maxv(a, b); }
    static V sqrtv(V a) { return simd::sqrtv(a); }
    static V absv(V a) { return simd::absv(a); }
    static V powv(V x, V y) { return simd::powv(x, y); }

    static M lt(V a, V b) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ); }
    static M gt(V a, V b) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ); }
    static M ge(V a, V b) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ); }
    static M eq(V a, V b) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ); }
    static M and_(M a, M b) { return static_cast<M>(a & b); }
    static M or_(M a, M b) { return static_cast<M>(a | b); }
    static M andnot(M a, M b) { return static_cast<M>((~a) & b); }
    static V select(M m, V a, V b) { return V(_mm512_mask_blend_ps(m, b.v, a.v)); }
    static bool any(M m) { return m != 0; }
    static bool all(M m) { return m == 0xFFFF; }
};

// ---- 8 x double (AVX-512F/DQ) ---------------------------------------------
template <> struct VT<simd::D8> {
    using scalar = double;
    using V = simd::D8;
    using M = __mmask8;
    static constexpr int lanes = 8;

    static V set1(scalar s) { return V(s); }
    static V zero() { return V::zero(); }
    static M mask_full() { return static_cast<M>(0xFF); }
    static M mask_zero() { return static_cast<M>(0); }
    static V load(const scalar* p) { return V::load(p); }
    static void store(scalar* p, V a) { a.store(p); }

    static V add(V a, V b) { return simd::add(a, b); }
    static V sub(V a, V b) { return simd::sub(a, b); }
    static V mul(V a, V b) { return simd::mul(a, b); }
    static V div(V a, V b) { return simd::div(a, b); }
    static V madd(V a, V b, V c) { return simd::madd(a, b, c); }
    static V msub(V a, V b, V c) { return simd::msub(a, b, c); }
    static V minv(V a, V b) { return simd::minv(a, b); }
    static V maxv(V a, V b) { return simd::maxv(a, b); }
    static V sqrtv(V a) { return simd::sqrtv(a); }
    static V absv(V a) { return simd::absv(a); }
    static V powv(V x, V y) { return simd::powv(x, y); }

    static M lt(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
    static M gt(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
    static M ge(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ); }
    static M eq(V a, V b) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ); }
    static M and_(M a, M b) { return static_cast<M>(a & b); }
    static M or_(M a, M b) { return static_cast<M>(a | b); }
    static M andnot(M a, M b) { return static_cast<M>((~a) & b); }
    static V select(M m, V a, V b) { return V(_mm512_mask_blend_pd(m, b.v, a.v)); }
    static bool any(M m) { return m != 0; }
    static bool all(M m) { return m == 0xFF; }
};

// ---------------------------------------------------------------------------
// Scene fields narrowed to the kernel's scalar type once per frame.
//
// Without this the sweeps pay a double -> float conversion plus a broadcast per
// sphere field, per sphere, per sweep; after packing every gather is a single
// broadcast load.
// ---------------------------------------------------------------------------
template <typename S> struct PackedScene {
    static constexpr int kMaxSpheres = 64;
    int count = 0;
    S cx[kMaxSpheres], cy[kMaxSpheres], cz[kMaxSpheres];
    S r2[kMaxSpheres], invr[kMaxSpheres];
    S alb_r[kMaxSpheres], alb_g[kMaxSpheres], alb_b[kMaxSpheres];
    S shin[kMaxSpheres], refl[kMaxSpheres];
    S lx[4], ly[4], lz[4];
    int light_count = 0;
    S near_clip = 0;

    void pack(const Scene& sc) {
        count = sc.sphere_count < kMaxSpheres ? sc.sphere_count : kMaxSpheres;
        for (int i = 0; i < count; ++i) {
            cx[i] = static_cast<S>(sc.sx[i]);
            cy[i] = static_cast<S>(sc.sy[i]);
            cz[i] = static_cast<S>(sc.sz[i]);
            r2[i] = static_cast<S>(sc.sr2[i]);
            invr[i] = static_cast<S>(sc.invr[i]);
            alb_r[i] = static_cast<S>(sc.cr[i]);
            alb_g[i] = static_cast<S>(sc.cg[i]);
            alb_b[i] = static_cast<S>(sc.cb[i]);
            shin[i] = static_cast<S>(sc.shininess[i]);
            refl[i] = static_cast<S>(sc.mirror[i]);
        }
        light_count = sc.light_count < 4 ? sc.light_count : 4;
        for (int i = 0; i < light_count; ++i) {
            lx[i] = static_cast<S>(sc.lx[i]);
            ly[i] = static_cast<S>(sc.ly[i]);
            lz[i] = static_cast<S>(sc.lz[i]);
        }
        near_clip = static_cast<S>(sc.near_clip);
    }
};

// ---------------------------------------------------------------------------
// The tracer
// ---------------------------------------------------------------------------
template <typename V> struct Tracer {
    using T = VT<V>;
    using S = typename T::scalar;
    using M = typename T::M;
    static constexpr int L = T::lanes;

    // Nearest accepted hit for a batch of rays: surface point, unit normal,
    // unit reflection direction and the material of the sphere that was hit.
    struct Hit {
        V px, py, pz;                  // surface point
        V nx, ny, nz;                  // unit normal
        V rx, ry, rz;                  // unit reflection direction
        V alb_r, alb_g, alb_b;         // albedo
        V shin;                        // Phong exponent
        V refl;                        // mirror weight (0 where nothing was hit)
        M mask;                        // lanes that hit something
    };

    // Brute-force nearest-hit sweep.  a4 = 4*(D.D) and inv2a = 1/(2*(D.D)) are
    // shared by every sphere, and the [near_clip, 1] window rejects both
    // self-intersections and anything behind the segment.
    static Hit nearest(const PackedScene<S>& ps, V ox, V oy, V oz, V dx, V dy, V dz,
                       V a4, V inv2a) {
        const V zero = T::zero();
        const V one = T::set1(static_cast<S>(1));
        const V near_c = T::set1(ps.near_clip);
        V tbest = T::set1(static_cast<S>(1e30));
        V px = zero, py = zero, pz = zero;
        V nx = zero, ny = zero, nz = zero;
        V ar = zero, ag = zero, ab = zero, sh = zero, rf = zero;
        V invr_hit = zero;
        M hit = T::mask_zero();

        for (int i = 0; i < ps.count; ++i) {
            const V ocx = T::sub(ox, T::set1(ps.cx[i]));
            const V ocy = T::sub(oy, T::set1(ps.cy[i]));
            const V ocz = T::sub(oz, T::set1(ps.cz[i]));
            const V r2 = T::set1(ps.r2[i]);
            const V b = T::madd(ocx, dx, T::madd(ocy, dy, T::mul(ocz, dz)));   // oc . D
            const V bb = T::add(b, b);
            const V c = T::sub(T::madd(ocx, ocx, T::madd(ocy, ocy, T::mul(ocz, ocz))), r2);
            const V disc = T::msub(bb, bb, T::mul(a4, c));                      // (2b)^2 - 4ac
            const M ok = T::ge(disc, zero);
            const V sq = T::sqrtv(T::maxv(disc, zero));
            const V tm = T::mul(T::sub(zero, T::add(bb, sq)), inv2a);
            const V tp = T::mul(T::sub(sq, bb), inv2a);
            const M beyond = T::and_(T::gt(tm, one), T::gt(tp, one));
            const M before = T::and_(T::lt(tm, near_c), T::lt(tp, near_c));
            const M win = T::andnot(T::or_(beyond, before), ok);
            // Pick the nearest root in front of the segment start.
            const V ca = T::select(T::lt(tp, near_c), tm, tp);
            const V cb = T::select(T::lt(tm, near_c), tp, tm);
            const V th = T::minv(ca, cb);
            const M take = T::and_(win, T::lt(th, tbest));
            tbest = T::select(take, th, tbest);
            px = T::select(take, T::madd(th, dx, ox), px);
            py = T::select(take, T::madd(th, dy, oy), py);
            pz = T::select(take, T::madd(th, dz, oz), pz);
            nx = T::select(take, T::madd(th, dx, ocx), nx);   // raw P - C
            ny = T::select(take, T::madd(th, dy, ocy), ny);
            nz = T::select(take, T::madd(th, dz, ocz), nz);
            ar = T::select(take, T::set1(ps.alb_r[i]), ar);
            ag = T::select(take, T::set1(ps.alb_g[i]), ag);
            ab = T::select(take, T::set1(ps.alb_b[i]), ab);
            sh = T::select(take, T::set1(ps.shin[i]), sh);
            rf = T::select(take, T::set1(ps.refl[i]), rf);
            invr_hit = T::select(take, T::set1(ps.invr[i]), invr_hit);
            hit = T::or_(hit, take);
        }

        nx = T::mul(nx, invr_hit);
        ny = T::mul(ny, invr_hit);
        nz = T::mul(nz, invr_hit);
        // Reflection direction: D - 2 (N.D) N, normalised.
        const V dn = T::madd(dx, nx, T::madd(dy, ny, T::mul(dz, nz)));
        const V two_dn = T::add(dn, dn);
        V rx = T::sub(dx, T::mul(two_dn, nx));
        V ry = T::sub(dy, T::mul(two_dn, ny));
        V rz = T::sub(dz, T::mul(two_dn, nz));
        const V rl2 = T::madd(rx, rx, T::madd(ry, ry, T::mul(rz, rz)));
        const V tiny = T::set1(static_cast<S>(1e-30));
        const M rdef = T::gt(rl2, tiny);
        const V rinv = T::select(rdef, T::div(one, T::sqrtv(T::maxv(rl2, tiny))), zero);
        rx = T::mul(rx, rinv);
        ry = T::mul(ry, rinv);
        rz = T::mul(rz, rinv);

        Hit h;
        h.px = px; h.py = py; h.pz = pz;
        h.nx = nx; h.ny = ny; h.nz = nz;
        h.rx = rx; h.ry = ry; h.rz = rz;
        h.alb_r = ar; h.alb_g = ag; h.alb_b = ab;
        h.shin = sh; h.refl = rf; h.mask = hit;
        return h;
    }

    // Any-hit sweep for shadow rays: identical quadratic per sphere, but only
    // the blocked mask comes back.  Stops early once every lane is blocked.
    static M occluded(const PackedScene<S>& ps, V ox, V oy, V oz, V dx, V dy, V dz,
                      V a4, V inv2a) {
        const V zero = T::zero();
        const V one = T::set1(static_cast<S>(1));
        const V near_c = T::set1(ps.near_clip);
        M occ = T::mask_zero();
        for (int i = 0; i < ps.count; ++i) {
            const V ocx = T::sub(ox, T::set1(ps.cx[i]));
            const V ocy = T::sub(oy, T::set1(ps.cy[i]));
            const V ocz = T::sub(oz, T::set1(ps.cz[i]));
            const V r2 = T::set1(ps.r2[i]);
            const V b = T::madd(ocx, dx, T::madd(ocy, dy, T::mul(ocz, dz)));
            const V bb = T::add(b, b);
            const V c = T::sub(T::madd(ocx, ocx, T::madd(ocy, ocy, T::mul(ocz, ocz))), r2);
            const V disc = T::msub(bb, bb, T::mul(a4, c));
            const M ok = T::ge(disc, zero);
            const V sq = T::sqrtv(T::maxv(disc, zero));
            const V tm = T::mul(T::sub(zero, T::add(bb, sq)), inv2a);
            const V tp = T::mul(T::sub(sq, bb), inv2a);
            const M beyond = T::and_(T::gt(tm, one), T::gt(tp, one));
            const M before = T::and_(T::lt(tm, near_c), T::lt(tp, near_c));
            occ = T::or_(occ, T::andnot(T::or_(beyond, before), ok));
            if (T::all(occ)) break;
        }
        return occ;
    }

    // Recursive radiance estimate for the batch; black once depth runs out.
    static void trace(const PackedScene<S>& ps, int max_depth,
                      V ox, V oy, V oz, V dx, V dy, V dz, int depth,
                      V& cr, V& cg, V& cb) {
        const V zero = T::zero();
        if (depth >= max_depth) {
            cr = zero; cg = zero; cb = zero;
            return;
        }
        const V one = T::set1(static_cast<S>(1));
        const V tiny = T::set1(static_cast<S>(1e-30));
        const V a = T::madd(dx, dx, T::madd(dy, dy, T::mul(dz, dz)));
        const V a2 = T::add(a, a);
        const V a4 = T::add(a2, a2);
        const V inv2a = T::div(one, a2);
        const Hit h = nearest(ps, ox, oy, oz, dx, dy, dz, a4, inv2a);

        cr = zero; cg = zero; cb = zero;
        if (!T::any(h.mask)) return;   // whole batch missed: no shading work
        for (int li = 0; li < ps.light_count; ++li) {
            const V lx = T::sub(T::set1(ps.lx[li]), h.px);
            const V ly = T::sub(T::set1(ps.ly[li]), h.py);
            const V lz = T::sub(T::set1(ps.lz[li]), h.pz);
            const V la = T::madd(lx, lx, T::madd(ly, ly, T::mul(lz, lz)));
            const V la2 = T::add(la, la);
            const V la4 = T::add(la2, la2);
            const V linv2a = T::div(one, la2);
            const M occ = occluded(ps, h.px, h.py, h.pz, lx, ly, lz, la4, linv2a);
            const M vis = T::and_(h.mask, T::andnot(occ, T::mask_full()));
            if (!T::any(vis)) continue;
            const V linv = T::select(T::gt(la, tiny),
                                     T::div(one, T::sqrtv(T::maxv(la, tiny))), zero);
            const V ux = T::mul(lx, linv);
            const V uy = T::mul(ly, linv);
            const V uz = T::mul(lz, linv);
            const V diff = T::maxv(T::madd(ux, h.nx, T::madd(uy, h.ny, T::mul(uz, h.nz))), zero);
            const V sdot = T::maxv(T::madd(ux, h.rx, T::madd(uy, h.ry, T::mul(uz, h.rz))), zero);
            const V spec = T::powv(sdot, h.shin);
            // Accumulate this light's contribution where the surface is lit
            // and the shadow ray reached the light.
            cr = T::add(cr, T::select(vis, T::add(T::mul(h.alb_r, diff), spec), zero));
            cg = T::add(cg, T::select(vis, T::add(T::mul(h.alb_g, diff), spec), zero));
            cb = T::add(cb, T::select(vis, T::add(T::mul(h.alb_b, diff), spec), zero));
        }

        // One reflection ray per bounce, weighted by the surface mirror term.
        const M refl = T::gt(h.refl, zero);
        if (T::any(refl)) {
            const V k = T::set1(static_cast<S>(1000));
            V sr, sg, sb;
            trace(ps, max_depth, h.px, h.py, h.pz,
                  T::mul(h.rx, k), T::mul(h.ry, k), T::mul(h.rz, k),
                  depth + 1, sr, sg, sb);
            cr = T::madd(h.refl, sr, cr);
            cg = T::madd(h.refl, sg, cg);
            cb = T::madd(h.refl, sb, cb);
        }
    }

    // Batch scanline driver: builds primary rays, traces them and writes the
    // clamped, averaged colour back into the interleaved frame buffer.
    static void trace_lines(const Scene& sc, const Config& cfg, S* buf, int W, int H,
                            int y0, int y1, S jitter, uint64_t& rays_out) {
        const int depth_limit = cfg.depth < 1 ? 1 : cfg.depth;
        const int SS = cfg.ss > 0 ? cfg.ss : 1;
        const int samples = SS * SS;

        const V vW = T::set1(static_cast<S>(W));
        const V vH = T::set1(static_cast<S>(H));
        const V ndcx = T::set1(static_cast<S>(sc.ndc_x));
        const V ndcy = T::set1(static_cast<S>(sc.ndc_y));
        const V vone = T::set1(static_cast<S>(1));
        const V vzero = T::zero();
        // D = eye + focal*fwd + sx*ray_scale*right + sy*ray_scale*up
        const V d0x = T::set1(static_cast<S>(sc.eye[0] + sc.focal * sc.fwd[0]));
        const V d0y = T::set1(static_cast<S>(sc.eye[1] + sc.focal * sc.fwd[1]));
        const V d0z = T::set1(static_cast<S>(sc.eye[2] + sc.focal * sc.fwd[2]));
        const V krx = T::set1(static_cast<S>(sc.ray_scale * sc.right[0]));
        const V kry = T::set1(static_cast<S>(sc.ray_scale * sc.right[1]));
        const V krz = T::set1(static_cast<S>(sc.ray_scale * sc.right[2]));
        const V kux = T::set1(static_cast<S>(sc.ray_scale * sc.up[0]));
        const V kuy = T::set1(static_cast<S>(sc.ray_scale * sc.up[1]));
        const V kuz = T::set1(static_cast<S>(sc.ray_scale * sc.up[2]));
        const V ex = T::set1(static_cast<S>(sc.eye[0]));
        const V ey = T::set1(static_cast<S>(sc.eye[1]));
        const V ez = T::set1(static_cast<S>(sc.eye[2]));

        std::vector<S> line(static_cast<size_t>(W) * 3);
        S lane_buf[L];
        uint64_t rays = 0;

        PackedScene<S> ps;
        ps.pack(sc);

        for (int y = y0; y < y1; ++y) {
            for (int x0 = 0; x0 < W; x0 += L) {
                V accr = vzero, accg = vzero, accb = vzero;
                S xs[L];
                for (int i = 0; i < L; ++i) xs[i] = static_cast<S>(x0 + i);
                const V vx = T::load(xs);
                for (int si = 0; si < samples; ++si) {
                    const S jx = static_cast<S>(si % SS) / static_cast<S>(SS);
                    const S jy = static_cast<S>(si / SS) / static_cast<S>(SS);
                    const V sxv = T::sub(T::div(T::add(vx, T::set1(static_cast<S>(1 + jitter + jx))), vW), ndcx);
                    const V syv = T::sub(ndcy, T::div(T::set1(static_cast<S>(y + 1 + jitter + jy)), vH));
                    const V dx = T::madd(sxv, krx, T::madd(syv, kux, d0x));
                    const V dy = T::madd(sxv, kry, T::madd(syv, kuy, d0y));
                    const V dz = T::madd(sxv, krz, T::madd(syv, kuz, d0z));
                    V cr, cg, cb;
                    trace(ps, depth_limit, ex, ey, ez, dx, dy, dz, 0, cr, cg, cb);
                    accr = T::add(accr, cr);
                    accg = T::add(accg, cg);
                    accb = T::add(accb, cb);
                    rays += static_cast<uint64_t>(L);
                }
                const V invs = T::set1(static_cast<S>(1) / static_cast<S>(samples));
                const V r0 = T::minv(T::maxv(T::mul(accr, invs), vzero), vone);
                const V g0 = T::minv(T::maxv(T::mul(accg, invs), vzero), vone);
                const V b0 = T::minv(T::maxv(T::mul(accb, invs), vzero), vone);
                const int n = (x0 + L <= W) ? L : (W - x0);
                T::store(lane_buf, r0);
                for (int i = 0; i < n; ++i) line[(x0 + i) * 3 + 0] = lane_buf[i];
                T::store(lane_buf, g0);
                for (int i = 0; i < n; ++i) line[(x0 + i) * 3 + 1] = lane_buf[i];
                T::store(lane_buf, b0);
                for (int i = 0; i < n; ++i) line[(x0 + i) * 3 + 2] = lane_buf[i];
            }
            std::memcpy(buf + static_cast<size_t>(y) * W * 3, line.data(),
                        static_cast<size_t>(W) * 3 * sizeof(S));
        }
        rays_out = rays;
    }
};

} // namespace detail

// ---------------------------------------------------------------------------
// AVX-512 kernel entry points (implemented in raytrace_avx512.cpp).
//
// That translation unit is compiled with /arch:AVX512: MSVC only keeps __m512
// values in native 512-bit registers when the ISA baseline contains AVX-512.
// Under /arch:AVX2 it models every __m512 as a pair of 256-bit halves and
// round-trips them through the stack around almost every operation, which
// costs roughly an order of magnitude in this kernel.  Everything else stays
// on /arch:AVX2 so the binary still runs on CPUs without AVX-512; these two
// functions are only ever entered behind the runtime CPUID dispatch.
// ---------------------------------------------------------------------------
uint64_t trace_lines_f16_avx512(const Scene& sc, const Config& cfg, float* buf,
                                int width, int height, int y0, int y1, float jitter);
uint64_t trace_lines_d8_avx512(const Scene& sc, const Config& cfg, double* buf,
                               int width, int height, int y0, int y1, double jitter);

} // namespace rt
} // namespace cst
