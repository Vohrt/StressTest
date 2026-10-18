// ===========================================================================
// AVX-512 memory-bandwidth sweeps for the Memory Read / Write / Copy tests.
//
// Same arrangement as raytrace_avx512.cpp: MSVC keeps 512-bit registers whole
// only under /arch:AVX512, so this translation unit is built with that switch
// and the rest of the program enters it only after a CPUID check
// (cpu_info().has_avx512f).
//
// The shapes mirror AIDA64's MTMBW "Mem"-level kernels - the ZMM_DQ64 variant
// the bench itself selects on AVX-512 CPUs (see docs/reverse-engineering.md):
//
//   read : linear sweep of non-temporal 64-byte loads   (vmovntdqa)
//   write: linear sweep of non-temporal 64-byte stores of an all-zero register
//          (vmovntdq; AIDA64 zeroes its vector registers with vpxord first)
//   copy : non-temporal 64-byte loads + stores over an all-zero source
//
// AIDA64's generated kernels run one fully unrolled 8 KiB swing (128 loads or
// 128 stores) per iteration; the swing length does not change the DRAM access
// pattern, so these kernels move 512 B (8 registers) per iteration to keep the
// loop body short.
//
// Digest: read folds every vector it loads.  Write and copy stream all-zero
// data - see bench_memory.cpp for why that is what AIDA64 streams and why it
// matters for the score - which carries nothing to fold, so they finish each
// pass with a sampled read-back: one contiguous kSamplePageBytes page every
// kSampleStride bytes, folded into the same xor/sum accumulators.  The math is
// byte-identical to bench_memory.cpp's 32-byte-vector fold: the 512-bit
// accumulators reduce down (XOR halves / add halves) to exactly the two 256-bit
// accumulators fold_bytes_avx2() would have built, and mix_accums() below is a
// byte-for-byte copy of the one in that file.  The selftest compares the two on
// every run.
// ===========================================================================

#include "mem_kernels.h"

#include <immintrin.h>

#include <cstdint>

namespace cst {
namespace {

// mix_accums() of bench_memory.cpp (keep the two in sync; selftest_memory
// compares kernel folds against that file's independent fold).
inline uint64_t mix_accums(__m256i x, __m256i s) noexcept {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extracti128_si256(x, 1);
    const __m128i sl = _mm256_castsi256_si128(s);
    const __m128i sh = _mm256_extracti128_si256(s, 1);
    const __m128i xr = _mm_xor_si128(xl, xh);
    const __m128i sr = _mm_xor_si128(sl, sh);
    const uint64_t xa = static_cast<uint64_t>(_mm_cvtsi128_si64(xr));
    const uint64_t xb = static_cast<uint64_t>(_mm_extract_epi64(xr, 1));
    const uint64_t sa = static_cast<uint64_t>(_mm_cvtsi128_si64(sr));
    const uint64_t sb = static_cast<uint64_t>(_mm_extract_epi64(sr, 1));
    uint64_t h = xa ^ (xb * 0x9E3779B97F4A7C15ull);
    h = (h ^ sa) * 0x100000001B3ull;
    h ^= sb * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return h;
}

// Reduce a pair of 512-bit accumulators (xor-fold, sum-fold) to the 256-bit
// pair fold_bytes_avx2() would have built from 32-byte vectors, then mix.
inline uint64_t fold_zmm(__m512i x, __m512i s) noexcept {
    const __m256i x2 = _mm256_xor_si256(_mm512_castsi512_si256(x),
                                        _mm512_extracti64x4_epi64(x, 1));
    const __m256i s2 = _mm256_add_epi64(_mm512_castsi512_si256(s),
                                        _mm512_extracti64x4_epi64(s, 1));
    return mix_accums(x2, s2);
}

inline __m512i stream_load64(const unsigned char* q, size_t off) noexcept {
    return _mm512_stream_load_si512(const_cast<unsigned char*>(q + off));
}

// Fold the sampled read-back pages of `p` into a fresh accumulator pair.  Plain
// (cacheable) loads: the point is to observe what the stores left in memory.
inline uint64_t fold_samples(const unsigned char* p, size_t bytes) noexcept {
    __m512i x0 = _mm512_setzero_si512();
    __m512i s0 = _mm512_setzero_si512();
    for (size_t off = 0; off + kSamplePageBytes <= bytes; off += kSampleStride) {
        for (size_t i = 0; i < kSamplePageBytes; i += 64) {
            const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(p + off + i));
            x0 = _mm512_xor_si512(x0, v);
            s0 = _mm512_add_epi64(s0, v);
        }
    }
    return fold_zmm(x0, s0);
}

// Two independent 64-bit folds -> one digest (a kernel that folds both a source
// stream and a sampled read-back).  Mirrors bench_memory.cpp.
inline uint64_t combine2(uint64_t a, uint64_t b) noexcept {
    return a ^ (b * 0x9E3779B97F4A7C15ull);
}

} // namespace

uint64_t mem_read_sweep_avx512(const void* p, size_t bytes) {
    const unsigned char* q = static_cast<const unsigned char*>(p);
    __m512i x0 = _mm512_setzero_si512();
    __m512i s0 = _mm512_setzero_si512();
    for (size_t off = 0; off < bytes; off += 512) {
        for (size_t lane = 0; lane < 8; ++lane) {
            const __m512i v = stream_load64(q, off + lane * 64);
            x0 = _mm512_xor_si512(x0, v);
            s0 = _mm512_add_epi64(s0, v);
        }
    }
    return fold_zmm(x0, s0);
}

uint64_t mem_write_sweep_avx512(void* p, size_t bytes) {
    unsigned char* q = static_cast<unsigned char*>(p);
    const __m512i z = _mm512_setzero_si512();
    for (size_t off = 0; off < bytes; off += 512) {
        for (size_t lane = 0; lane < 8; ++lane) {
            _mm512_stream_si512(reinterpret_cast<__m512i*>(q + off + lane * 64), z);
        }
    }
    // Non-temporal stores are weakly ordered: fence before reading anything
    // back (the digest hook, another repetition, the selftest).
    _mm_sfence();
    return fold_samples(q, bytes);
}

uint64_t mem_copy_sweep_avx512(void* dst, const void* src, size_t bytes) {
    unsigned char* d = static_cast<unsigned char*>(dst);
    const unsigned char* s = static_cast<const unsigned char*>(src);
    __m512i x0 = _mm512_setzero_si512();
    __m512i s0 = _mm512_setzero_si512();
    for (size_t off = 0; off < bytes; off += 512) {
        for (size_t lane = 0; lane < 8; ++lane) {
            const __m512i v = stream_load64(s, off + lane * 64);
            _mm512_stream_si512(reinterpret_cast<__m512i*>(d + off + lane * 64), v);
            x0 = _mm512_xor_si512(x0, v);
            s0 = _mm512_add_epi64(s0, v);
        }
    }
    _mm_sfence();
    return combine2(fold_zmm(x0, s0), fold_samples(d, bytes));
}

} // namespace cst
