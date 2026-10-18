// Minimal SIMD math layer for the ray-tracing and fractal kernels.
//
// Rows of 4 doubles (AVX2, 256-bit) / 8 floats (AVX2) are traced simultaneously,
// which is how AIDA64's "SIMD-enhanced ray tracing engine" achieves its
// throughput.  AVX-512 paths are selected at runtime when the CPU has it.
#pragma once

#include <immintrin.h>

#include <limits>
#include <cstdint>

namespace cst {
namespace simd {

// ---------------------------------------------------------------------------
// 4 x double
// ---------------------------------------------------------------------------
struct D4 {
    __m256d v;

    D4() = default;
    explicit D4(__m256d x) : v(x) {}
    explicit D4(double s) : v(_mm256_set1_pd(s)) {}
    D4(double a, double b, double c, double d) : v(_mm256_set_pd(d, c, b, a)) {}

    static D4 zero() { return D4(_mm256_setzero_pd()); }
    static D4 load(const double* p) { return D4(_mm256_loadu_pd(p)); }
    static D4 load_aligned(const double* p) { return D4(_mm256_load_pd(p)); }
    void store(double* p) const { _mm256_storeu_pd(p, v); }
    void store_aligned(double* p) const { _mm256_store_pd(p, v); }

    // Broadcast the n-th lane of `p`.
    static D4 bcast(const double* p, int n) { return D4(_mm256_set1_pd(p[n])); }
};

inline D4 operator+(D4 a, D4 b) { return D4(_mm256_add_pd(a.v, b.v)); }
inline D4 operator-(D4 a, D4 b) { return D4(_mm256_sub_pd(a.v, b.v)); }
inline D4 operator*(D4 a, D4 b) { return D4(_mm256_mul_pd(a.v, b.v)); }
inline D4 operator/(D4 a, D4 b) { return D4(_mm256_div_pd(a.v, b.v)); }
inline D4 operator-(D4 a) { return D4(_mm256_sub_pd(_mm256_setzero_pd(), a.v)); }

// Named forms used by the ray tracer's hardware traits.
inline D4 add(D4 a, D4 b) { return a + b; }
inline D4 sub(D4 a, D4 b) { return a - b; }
inline D4 mul(D4 a, D4 b) { return a * b; }
inline D4 div(D4 a, D4 b) { return a / b; }

// a*b + c  (FMA when available).  MSVC never defines __FMA__ -- it advertises
// FMA support through __AVX2__ -- so gate on both spellings.
inline D4 madd(D4 a, D4 b, D4 c) {
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    return D4(_mm256_fmadd_pd(a.v, b.v, c.v));
#else
    return a * b + c;
#endif
}
// a*b - c
inline D4 msub(D4 a, D4 b, D4 c) { return madd(a, b, -c); }

inline D4 minv(D4 a, D4 b) { return D4(_mm256_min_pd(a.v, b.v)); }
inline D4 maxv(D4 a, D4 b) { return D4(_mm256_max_pd(a.v, b.v)); }
inline D4 sqrtv(D4 a) { return D4(_mm256_sqrt_pd(a.v)); }
inline D4 rcp_approx(D4 a) { return D4(_mm256_div_pd(_mm256_set1_pd(1.0), a.v)); }
// Mask convention used throughout this file
// -----------------------------------------
// A comparison returns an all-ones lane (0xFFFF...) for true and zero for
// false, exactly as _mm256_cmp_pd/_ps produce natively.
//   * select()   - blends on the sign bit, which all-ones has and zero does not,
//                  so the native comparison result works unchanged.
//   * any_true() - tests the sign bits, which is the right test for a mask but
//                  NOT for ordinary values: 1.0 is positive, so any_true(1.0)
//                  is legitimately false.  Kernels must therefore keep the mask
//                  (e.g. cmp_lt(...)) separate from any value they accumulate.
//   * mask_and() - plain bitwise AND, correct for all-ones/zero masks.

// Selects `a` where the mask is true and `b` where it is false.
inline D4 select(D4 mask, D4 a, D4 b) { return D4(_mm256_blendv_pd(b.v, a.v, mask.v)); }

// True when any lane of the mask is set.
inline bool any_true(D4 a) { return _mm256_movemask_pd(a.v) != 0; }

// Replaces NaN and +/-infinity with zero, and leaves finite values untouched.
//
// A ray tracer has degenerate configurations (a ray exactly parallel to a light
// direction, a zero-length reflected direction) where 0 * infinity yields NaN.
// One poisoned lane would otherwise spread through the whole batch on the next
// iteration, so accumulators are sanitised after each update: a pixel that hits
// such a configuration loses one highlight instead of turning the image into
// NaNs.
//
// The implementation must neutralise infinities as well as NaN, and it must not
// rely on an ordered predicate alone: _CMP_ORD_Q is true for an infinity.  The
// compare-and-mask below produces all-ones only for a finite lane, so masking
// with it clears NaN and both infinities.  Finite inputs pass through unchanged
// (x & ~0 == x), and -0.0 is preserved.
inline D4 sanitize(D4 a) {
    const __m256d finite = _mm256_cmp_pd(
        _mm256_and_pd(a.v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFull))),
        _mm256_set1_pd(std::numeric_limits<double>::infinity()), _CMP_LT_OQ);
    return D4(_mm256_and_pd(a.v, finite));
}

inline D4 dot(D4 a, D4 b) {
    // 4-lane broadcast dot.  Uses the shuffle+add form rather than
    // _mm256_hadd_pd, which is slow on AMD Zen.
    const D4 m = a * b;
    const D4 h = D4(_mm256_add_pd(m.v, _mm256_shuffle_pd(m.v, m.v, 0x5)));
    return D4(_mm256_add_pd(h.v, _mm256_permute2f128_pd(h.v, h.v, 0x01)));
}

// Comparisons return the native all-ones form for true (see the convention note
// above), which is what select() and mask_and() expect.
inline D4 cmp_lt(D4 a, D4 b) { return D4(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)); }
inline D4 cmp_gt(D4 a, D4 b) { return D4(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)); }
inline D4 cmp_ge(D4 a, D4 b) { return D4(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)); }
inline D4 absv(D4 a) { return D4(_mm256_andnot_pd(_mm256_set1_pd(-0.0), a.v)); }
inline D4 floorv(D4 a) { return D4(_mm256_floor_pd(a.v)); }
inline D4 mask_and(D4 a, D4 b) { return D4(_mm256_and_pd(a.v, b.v)); }

// ---------------------------------------------------------------------------
// 8 x float
// ---------------------------------------------------------------------------
struct F8 {
    __m256 v;

    F8() = default;
    explicit F8(__m256 x) : v(x) {}
    explicit F8(float s) : v(_mm256_set1_ps(s)) {}
    F8(float a, float b, float c, float d, float e, float f, float g, float h)
        : v(_mm256_set_ps(h, g, f, e, d, c, b, a)) {}

    static F8 zero() { return F8(_mm256_setzero_ps()); }
    static F8 load(const float* p) { return F8(_mm256_loadu_ps(p)); }
    static F8 load_aligned(const float* p) { return F8(_mm256_load_ps(p)); }
    void store(float* p) const { _mm256_storeu_ps(p, v); }
    static F8 bcast(const float* p, int n) { return F8(_mm256_set1_ps(p[n])); }
};

inline F8 operator+(F8 a, F8 b) { return F8(_mm256_add_ps(a.v, b.v)); }
inline F8 operator-(F8 a, F8 b) { return F8(_mm256_sub_ps(a.v, b.v)); }
inline F8 operator*(F8 a, F8 b) { return F8(_mm256_mul_ps(a.v, b.v)); }
inline F8 operator/(F8 a, F8 b) { return F8(_mm256_div_ps(a.v, b.v)); }
inline F8 operator-(F8 a) { return F8(_mm256_sub_ps(_mm256_setzero_ps(), a.v)); }

// Named forms used by the ray tracer's hardware traits.
inline F8 add(F8 a, F8 b) { return a + b; }
inline F8 sub(F8 a, F8 b) { return a - b; }
inline F8 mul(F8 a, F8 b) { return a * b; }
inline F8 div(F8 a, F8 b) { return a / b; }

inline F8 madd(F8 a, F8 b, F8 c) {
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))   // see D4
    return F8(_mm256_fmadd_ps(a.v, b.v, c.v));
#else
    return a * b + c;
#endif
}
inline F8 msub(F8 a, F8 b, F8 c) { return madd(a, b, -c); }
inline F8 minv(F8 a, F8 b) { return F8(_mm256_min_ps(a.v, b.v)); }
inline F8 maxv(F8 a, F8 b) { return F8(_mm256_max_ps(a.v, b.v)); }
inline F8 sqrtv(F8 a) { return F8(_mm256_sqrt_ps(a.v)); }
// See the D4 overload; the comparison helpers already produce sign-bit masks.
inline F8 select(F8 mask, F8 a, F8 b) { return F8(_mm256_blendv_ps(b.v, a.v, mask.v)); }

// True when any lane of the mask is set.  Same convention as the D4 overload.
inline bool any_true(F8 a) { return _mm256_movemask_ps(a.v) != 0; }

// Replaces NaN and +/-infinity with zero; see the D4 overload.
inline F8 sanitize(F8 a) {
    const __m256 finite = _mm256_cmp_ps(
        _mm256_and_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))),
        _mm256_set1_ps(std::numeric_limits<float>::infinity()), _CMP_LT_OQ);
    return F8(_mm256_and_ps(a.v, finite));
}

inline F8 dot(F8 a, F8 b) {
    // 8-lane broadcast dot via the shuffle+add reduction tree.
    const F8 m = a * b;
    const F8 h1 = F8(_mm256_add_ps(m.v, _mm256_shuffle_ps(m.v, m.v, 0xB1)));
    const F8 h2 = F8(_mm256_add_ps(h1.v, _mm256_shuffle_ps(h1.v, h1.v, 0x4E)));
    return F8(_mm256_add_ps(h2.v, _mm256_permute2f128_ps(h2.v, h2.v, 0x01)));
}

inline F8 cmp_lt(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }
inline F8 cmp_gt(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)); }
inline F8 cmp_ge(F8 a, F8 b) { return F8(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)); }
inline F8 absv(F8 a) { return F8(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v)); }
inline F8 floorv(F8 a) { return F8(_mm256_floor_ps(a.v)); }
inline F8 mask_and(F8 a, F8 b) { return F8(_mm256_and_ps(a.v, b.v)); }

// ---------------------------------------------------------------------------
// pow(x, y) for every width: exp2(y * log2(x)) built from a shared polynomial
// core.  The ray tracer calls this once per light per bounce, and the
// reference workload spends a noticeable slice of its FLOPs there, so it is
// vectorised for all four types rather than falling back to libm per lane.
//
// x is expected to be >= 0 (callers clamp specular terms with max(0, ...)).
// Arguments are clamped to the normal-exponent range, so x == 0 degrades to
//2^-126 = 0 instead of producing a NaN, which is what a renderer wants.
// ---------------------------------------------------------------------------
namespace detail_pow {

// 2^f for f in [-0.5, 0.5]; the caller supplies the integer exponent.
inline F8 exp2_core_f8(F8 f) {
    F8 q(1.5252668e-5f);
    q = madd(q, f, F8(1.5403530e-4f));
    q = madd(q, f, F8(1.3333558e-3f));
    q = madd(q, f, F8(9.6181291e-3f));
    q = madd(q, f, F8(5.5504109e-2f));
    q = madd(q, f, F8(2.4022651e-1f));
    q = madd(q, f, F8(6.9314718e-1f));
    q = madd(q, f, F8(1.0f));
    return q;
}

inline D4 exp2_core_d4(D4 f) {
    D4 q(1.525273380405984e-5);
    q = madd(q, f, D4(1.540353039338161e-4));
    q = madd(q, f, D4(1.333355814642844e-3));
    q = madd(q, f, D4(9.618129107628477e-3));
    q = madd(q, f, D4(5.550410866482158e-2));
    q = madd(q, f, D4(2.402265069591007e-1));
    q = madd(q, f, D4(0.6931471805599453));
    q = madd(q, f, D4(1.0));
    return q;
}

} // namespace detail_pow

inline F8 powv(F8 x, F8 y) {
    using namespace detail_pow;
    const __m256i bits = _mm256_castps_si256(x.v);
    const __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(127));
    const F8 m(_mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000))));
    const F8 ef(_mm256_cvtepi32_ps(e));
    const F8 t = div(sub(m, F8(1.0f)), add(m, F8(1.0f)));
    const F8 t2 = mul(t, t);
    F8 p(1.0f / 11.0f);
    p = madd(p, t2, F8(1.0f / 9.0f));
    p = madd(p, t2, F8(1.0f / 7.0f));
    p = madd(p, t2, F8(1.0f / 5.0f));
    p = madd(p, t2, F8(1.0f / 3.0f));
    p = madd(p, t2, F8(1.0f));
    const F8 lg = mul(mul(t, p), F8(2.8853900817779268f));   // 2/ln2
    F8 arg = mul(y, add(lg, ef));
    arg = maxv(arg, F8(-126.0f));
    arg = minv(arg, F8(127.0f));
    const F8 n = sub(add(arg, F8(12582912.0f)), F8(12582912.0f));   // round-to-nearest
    const F8 f = sub(arg, n);
    const __m256i ni = _mm256_cvtps_epi32(n.v);
    const __m256i sc = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return mul(exp2_core_f8(f), F8(_mm256_castsi256_ps(sc)));
}

inline D4 powv(D4 x, D4 y) {
    using namespace detail_pow;
    const __m256i bits = _mm256_castpd_si256(x.v);
    __m256i e64 = _mm256_sub_epi64(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(1023));
    const D4 m(_mm256_castsi256_pd(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFull)),
        _mm256_set1_epi64x(0x3FF0000000000000ull))));
    // Move the four 64-bit exponents into the low 4 dwords and widen to double.
    e64 = _mm256_permutevar8x32_epi32(e64, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0));
    const D4 ef(_mm256_cvtepi32_pd(_mm256_castsi256_si128(e64)));
    const D4 t = div(sub(m, D4(1.0)), add(m, D4(1.0)));
    const D4 t2 = mul(t, t);
    D4 p(1.0 / 13.0);
    p = madd(p, t2, D4(1.0 / 11.0));
    p = madd(p, t2, D4(1.0 / 9.0));
    p = madd(p, t2, D4(1.0 / 7.0));
    p = madd(p, t2, D4(1.0 / 5.0));
    p = madd(p, t2, D4(1.0 / 3.0));
    p = madd(p, t2, D4(1.0));
    const D4 lg = mul(mul(t, p), D4(2.8853900817779268));
    D4 arg = mul(y, add(lg, ef));
    arg = maxv(arg, D4(-1022.0));
    arg = minv(arg, D4(1023.0));
    const D4 n = sub(add(arg, D4(6755399441055744.0)), D4(6755399441055744.0));
    const D4 f = sub(arg, n);
    const __m128i ni32 = _mm256_cvttpd_epi32(n.v);
    const __m256i ni = _mm256_cvtepi32_epi64(ni32);
    const __m256i sc = _mm256_slli_epi64(_mm256_add_epi64(ni, _mm256_set1_epi64x(1023)), 52);
    return mul(exp2_core_d4(f), D4(_mm256_castsi256_pd(sc)));
}

// ---------------------------------------------------------------------------
// 16 x float / 8 x double (AVX-512F/DQ)
//
// These are defined unconditionally - MSVC resolves intrinsics regardless of
// /arch and only emits code for functions that are actually called - while the
// kernels that use them are gated at runtime on cpu_info().has_avx512dq.
// ---------------------------------------------------------------------------
struct F16 {
    __m512 v;

    F16() = default;
    explicit F16(__m512 x) : v(x) {}
    explicit F16(float s) : v(_mm512_set1_ps(s)) {}

    static F16 zero() { return F16(_mm512_setzero_ps()); }
    static F16 load(const float* p) { return F16(_mm512_loadu_ps(p)); }
    void store(float* p) const { _mm512_storeu_ps(p, v); }
};

inline F16 operator+(F16 a, F16 b) { return F16(_mm512_add_ps(a.v, b.v)); }
inline F16 operator-(F16 a, F16 b) { return F16(_mm512_sub_ps(a.v, b.v)); }
inline F16 operator*(F16 a, F16 b) { return F16(_mm512_mul_ps(a.v, b.v)); }
inline F16 operator/(F16 a, F16 b) { return F16(_mm512_div_ps(a.v, b.v)); }
inline F16 operator-(F16 a) { return F16(_mm512_sub_ps(_mm512_setzero_ps(), a.v)); }

// Named forms used by the ray tracer's hardware traits.
inline F16 add(F16 a, F16 b) { return a + b; }
inline F16 sub(F16 a, F16 b) { return a - b; }
inline F16 mul(F16 a, F16 b) { return a * b; }
inline F16 div(F16 a, F16 b) { return a / b; }
inline F16 madd(F16 a, F16 b, F16 c) { return F16(_mm512_fmadd_ps(a.v, b.v, c.v)); }
inline F16 msub(F16 a, F16 b, F16 c) { return madd(a, b, -c); }
inline F16 minv(F16 a, F16 b) { return F16(_mm512_min_ps(a.v, b.v)); }
inline F16 maxv(F16 a, F16 b) { return F16(_mm512_max_ps(a.v, b.v)); }
inline F16 sqrtv(F16 a) { return F16(_mm512_sqrt_ps(a.v)); }
inline F16 absv(F16 a) { return F16(_mm512_andnot_ps(_mm512_set1_ps(-0.0f), a.v)); }

struct D8 {
    __m512d v;

    D8() = default;
    explicit D8(__m512d x) : v(x) {}
    explicit D8(double s) : v(_mm512_set1_pd(s)) {}

    static D8 zero() { return D8(_mm512_setzero_pd()); }
    static D8 load(const double* p) { return D8(_mm512_loadu_pd(p)); }
    void store(double* p) const { _mm512_storeu_pd(p, v); }
};

inline D8 operator+(D8 a, D8 b) { return D8(_mm512_add_pd(a.v, b.v)); }
inline D8 operator-(D8 a, D8 b) { return D8(_mm512_sub_pd(a.v, b.v)); }
inline D8 operator*(D8 a, D8 b) { return D8(_mm512_mul_pd(a.v, b.v)); }
inline D8 operator/(D8 a, D8 b) { return D8(_mm512_div_pd(a.v, b.v)); }
inline D8 operator-(D8 a) { return D8(_mm512_sub_pd(_mm512_setzero_pd(), a.v)); }

// Named forms used by the ray tracer's hardware traits.
inline D8 add(D8 a, D8 b) { return a + b; }
inline D8 sub(D8 a, D8 b) { return a - b; }
inline D8 mul(D8 a, D8 b) { return a * b; }
inline D8 div(D8 a, D8 b) { return a / b; }
inline D8 madd(D8 a, D8 b, D8 c) { return D8(_mm512_fmadd_pd(a.v, b.v, c.v)); }
inline D8 msub(D8 a, D8 b, D8 c) { return madd(a, b, -c); }
inline D8 minv(D8 a, D8 b) { return D8(_mm512_min_pd(a.v, b.v)); }
inline D8 maxv(D8 a, D8 b) { return D8(_mm512_max_pd(a.v, b.v)); }
inline D8 sqrtv(D8 a) { return D8(_mm512_sqrt_pd(a.v)); }
inline D8 absv(D8 a) { return D8(_mm512_andnot_pd(_mm512_set1_pd(-0.0), a.v)); }

namespace detail_pow {

// 2^f for f in [-0.5, 0.5] (single precision, 16-wide).
inline F16 exp2_core_f16(F16 f) {
    F16 q(1.5252668e-5f);
    q = madd(q, f, F16(1.5403530e-4f));
    q = madd(q, f, F16(1.3333558e-3f));
    q = madd(q, f, F16(9.6181291e-3f));
    q = madd(q, f, F16(5.5504109e-2f));
    q = madd(q, f, F16(2.4022651e-1f));
    q = madd(q, f, F16(6.9314718e-1f));
    q = madd(q, f, F16(1.0f));
    return q;
}

// 2^f for f in [-0.5, 0.5] (double precision, 8-wide).
inline D8 exp2_core_d8(D8 f) {
    D8 q(1.525273380405984e-5);
    q = madd(q, f, D8(1.540353039338161e-4));
    q = madd(q, f, D8(1.333355814642844e-3));
    q = madd(q, f, D8(9.618129107628477e-3));
    q = madd(q, f, D8(5.550410866482158e-2));
    q = madd(q, f, D8(2.402265069591007e-1));
    q = madd(q, f, D8(0.6931471805599453));
    q = madd(q, f, D8(1.0));
    return q;
}

} // namespace detail_pow

inline F16 powv(F16 x, F16 y) {
    const __m512i bits = _mm512_castps_si512(x.v);
    const __m512i e = _mm512_sub_epi32(_mm512_srli_epi32(bits, 23), _mm512_set1_epi32(127));
    const F16 m(_mm512_castsi512_ps(_mm512_or_si512(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x007FFFFF)), _mm512_set1_epi32(0x3F800000))));
    const F16 ef(_mm512_cvtepi32_ps(e));
    const F16 t = div(sub(m, F16(1.0f)), add(m, F16(1.0f)));
    const F16 t2 = mul(t, t);
    F16 p(1.0f / 11.0f);
    p = madd(p, t2, F16(1.0f / 9.0f));
    p = madd(p, t2, F16(1.0f / 7.0f));
    p = madd(p, t2, F16(1.0f / 5.0f));
    p = madd(p, t2, F16(1.0f / 3.0f));
    p = madd(p, t2, F16(1.0f));
    const F16 lg = mul(mul(t, p), F16(2.8853900817779268f));
    F16 arg = mul(y, add(lg, ef));
    arg = maxv(arg, F16(-126.0f));
    arg = minv(arg, F16(127.0f));
    const F16 n = sub(add(arg, F16(12582912.0f)), F16(12582912.0f));
    const F16 f = sub(arg, n);
    const __m512i ni = _mm512_cvtps_epi32(n.v);
    const __m512i sc = _mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23);
    return mul(detail_pow::exp2_core_f16(f), F16(_mm512_castsi512_ps(sc)));
}

inline D8 powv(D8 x, D8 y) {
    const __m512i bits = _mm512_castpd_si512(x.v);
    const __m512i e = _mm512_sub_epi64(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(1023));
    const D8 m(_mm512_castsi512_pd(_mm512_or_si512(
        _mm512_and_si512(bits, _mm512_set1_epi64(0x000FFFFFFFFFFFFFull)),
        _mm512_set1_epi64(0x3FF0000000000000ull))));
    const D8 ef(_mm512_cvtepi64_pd(e));
    const D8 t = div(sub(m, D8(1.0)), add(m, D8(1.0)));
    const D8 t2 = mul(t, t);
    D8 p(1.0 / 13.0);
    p = madd(p, t2, D8(1.0 / 11.0));
    p = madd(p, t2, D8(1.0 / 9.0));
    p = madd(p, t2, D8(1.0 / 7.0));
    p = madd(p, t2, D8(1.0 / 5.0));
    p = madd(p, t2, D8(1.0 / 3.0));
    p = madd(p, t2, D8(1.0));
    const D8 lg = mul(mul(t, p), D8(2.8853900817779268));
    D8 arg = mul(y, add(lg, ef));
    arg = maxv(arg, D8(-1022.0));
    arg = minv(arg, D8(1023.0));
    const D8 n = sub(add(arg, D8(6755399441055744.0)), D8(6755399441055744.0));
    const D8 f = sub(arg, n);
    const __m512i ni = _mm512_cvtpd_epi64(n.v);
    const __m512i sc = _mm512_slli_epi64(_mm512_add_epi64(ni, _mm512_set1_epi64(1023)), 52);
    return mul(detail_pow::exp2_core_d8(f), D8(_mm512_castsi512_pd(sc)));
}

} // namespace simd
} // namespace cst
