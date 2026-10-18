// CPU PhotoWorxx, CPU Queen and CPU CheckMate workload implementations.
//
// These re-implement the integer/image benchmarks of the AIDA64 CPU benchmark
// page from public documentation, for the sustained-load purpose of this tool.
// No FinalWire code is used.
//
// Work units (see the individual info()/params() overrides):
//   photoworxx : one work unit = one pixel of one AIDA64 NGImage2D stage,
//                scaled by the stage count so the reported figure is the sum
//                of the per-stage rates; unit_scale 1e-6 -> MPixel/s.
//   queen      : one work unit = one candidate board checked (one of the 10!
//                column permutations); unit_scale 1e-3 -> kPos/s, the same
//                "checked positions / second" figure AIDA64's CPU Queen uses.
//   checkmate  : one work unit = one checked position of the complete 16!
//                permutation space.  The kernel walks it in 3-rank chunks
//                (a 13-column factoradic prefix plus the 3! = 6 completions
//                of the last three ranks) and counts +6 per chunk, exactly the
//                accounting of AIDA64 v7.35's NGQueen AVX512_VPOPCNTDQ kernel;
//                unit_scale 1e-6 -> MP/s, the unit the GUI prints.
//
// Call contract (see cst.h / main.cpp): the harness owns the clock and calls
// run_once() in a loop until the round ends, handing it a budget in
// `min_seconds`.  Every call performs bounded work - one complete board, one
// batch of chunks, or a bounded run of photo stages - and stays inside that
// budget, so the measured work is never attributed to the wrong round.
#include "bench_image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <immintrin.h>

#include "common.h"

namespace cst {
namespace {

// ===========================================================================
// 1. N-Queens kernels - CPU Queen (10x10) and CPU CheckMate (16x16)
// ===========================================================================
//
// Bitmask backtracking with the classic cols / d1 / d2 occupancy masks: d1
// carries the squares attacked along one diagonal direction (shifted one bit
// per row), d2 the other.  N <= 16, so the entire state lives in three 32-bit
// registers and the search stays branch-prediction bound, which is the
// property AIDA64 documents for CPU Queen.  The first two rows are unrolled
// into a straight double loop so no call/return overhead is paid near the root
// of the tree, where the branch predictor sees the widest fan-out.
template <int N>
struct Queens {
    static_assert(N > 3 && N < 31, "occupancy masks are 32 bits wide");

    static constexpr unsigned kFull = (1u << N) - 1u;

    static constexpr unsigned bit(int col) { return 1u << col; }

    // Number of solutions of the subtree below `row`.  The count is returned
    // (not accumulated through a reference) so it stays in a register.
    static uint64_t search(int row, unsigned cols, unsigned d1, unsigned d2) {
        if (row == N) return 1;
        uint64_t total = 0;
        unsigned avail = ~(cols | d1 | d2) & kFull;
        while (avail != 0) {
            const unsigned b = avail & (0u - avail);   // lowest set bit
            avail ^= b;
            total += search(row + 1, cols | b, ((d1 | b) << 1) & kFull, (d2 | b) >> 1);
        }
        return total;
    }

    // Occupancy left behind by fixing rows 0 and 1 to columns c0 and c1.
    static bool prefix(int c0, int c1, unsigned& cols, unsigned& d1, unsigned& d2) {
        const unsigned b0 = bit(c0);
        const unsigned c = b0;
        const unsigned x1 = (b0 << 1) & kFull;
        const unsigned x2 = b0 >> 1;
        const unsigned b1 = bit(c1);
        if (((c | x1 | x2) & b1) != 0) return false;   // attacked by the row 0 queen
        cols = c | b1;
        d1 = ((x1 | b1) << 1) & kFull;
        d2 = (x2 | b1) >> 1;
        return true;
    }

    // Solutions of the subtree below a fixed two-row prefix.
    static uint64_t count_prefix(int c0, int c1) {
        unsigned cols = 0, d1 = 0, d2 = 0;
        if (!prefix(c0, c1, cols, d1, d2)) return 0;
        return search(2, cols, d1, d2);
    }

    // Complete board enumeration (all N rows).
    static uint64_t count_all() {
        uint64_t total = 0;
        for (int c0 = 0; c0 < N; ++c0) {
            const unsigned b0 = bit(c0);
            const unsigned cols = b0;
            const unsigned d1 = (b0 << 1) & kFull;
            const unsigned d2 = b0 >> 1;
            for (int c1 = 0; c1 < N; ++c1) {
                const unsigned b1 = bit(c1);
                if (((cols | d1 | d2) & b1) != 0) continue;
                total += search(2, cols | b1, ((d1 | b1) << 1) & kFull, (d2 | b1) >> 1);
            }
        }
        return total;
    }
};

// Deliberately independent, naive reference solver (explicit conflict check,
// no bit tricks) used to cross-check the kernels above in the self test.
uint64_t queens_ref_search(std::vector<int>& pos, int row, int n) {
    if (row == n) return 1;
    uint64_t total = 0;
    for (int col = 0; col < n; ++col) {
        bool ok = true;
        for (int r = 0; r < row && ok; ++r) {
            const int d = pos[r] - col;
            if (d == 0 || d == row - r || d == -(row - r)) ok = false;
        }
        if (!ok) continue;
        pos[static_cast<size_t>(row)] = col;
        total += queens_ref_search(pos, row + 1, n);
    }
    return total;
}

uint64_t queens_ref(int n) {
    std::vector<int> pos(static_cast<size_t>(n), 0);
    return queens_ref_search(pos, 0, n);
}

// Reference count for the subtree below a fixed two-row prefix of an n x n board.
uint64_t queens_ref_prefix(int n, int c0, int c1) {
    if (c0 == c1) return 0;
    if (c0 - c1 == 1 || c1 - c0 == 1) return 0;
    std::vector<int> pos(static_cast<size_t>(n), 0);
    pos[0] = c0;
    pos[1] = c1;
    return queens_ref_search(pos, 2, n);
}

constexpr int kQueenN = 10;                             // AIDA64's CPU Queen board
constexpr uint64_t kQueenSolutions10 = 724ull;          // known 10x10 solution count
constexpr uint64_t kQueenSolutions8 = 92ull;            // known 8x8 solution count
constexpr int kCheckMateN = 16;                         // AIDA64 CPU CheckMate board
constexpr uint64_t kCheckMateSolutions16 = 14772512ull; // known 16x16 solution count

// Every legal two-row prefix of the 16x16 board, and the complete enumeration
// of one such bounded subtree.  These are the *reference solver* now: the
// shipped CheckMate kernel walks the permutation space, not pruned subtrees,
// so this pruned bitmask machinery stays only to verify the known 16x16
// solution total (14,772,512) in the self test, where the subtrees are summed
// and checked against the complete board.
const std::vector<std::pair<int, int>>& checkmate_prefixes() {
    static const std::vector<std::pair<int, int>> table = [] {
        std::vector<std::pair<int, int>> v;
        v.reserve(256);
        for (int c0 = 0; c0 < kCheckMateN; ++c0) {
            for (int c1 = 0; c1 < kCheckMateN; ++c1) {
                unsigned cols = 0, d1 = 0, d2 = 0;
                if (Queens<kCheckMateN>::prefix(c0, c1, cols, d1, d2)) {
                    v.emplace_back(c0, c1);
                }
            }
        }
        return v;
    }();
    return table;
}

// Complete 16x16 enumeration, split over `threads` workers by prefix range.
// Integer accumulation is order independent, so the result does not depend on
// the thread count.
uint64_t checkmate_count_full(int threads) {
    const std::vector<std::pair<int, int>>& prefixes = checkmate_prefixes();
    const int count = static_cast<int>(prefixes.size());
    threads = std::max(1, std::min(threads, count));
    std::vector<uint64_t> partial(static_cast<size_t>(threads), 0ull);
    run_workers(threads, [&](int t) {
        const size_t begin = prefixes.size() * static_cast<size_t>(t) / static_cast<size_t>(threads);
        const size_t end = prefixes.size() * static_cast<size_t>(t + 1) / static_cast<size_t>(threads);
        uint64_t acc = 0;
        for (size_t i = begin; i < end; ++i) {
            acc += Queens<kCheckMateN>::count_prefix(prefixes[i].first, prefixes[i].second);
        }
        partial[static_cast<size_t>(t)] = acc;
    });
    uint64_t total = 0;
    for (uint64_t v : partial) total += v;
    return total;
}

// ===========================================================================
// 2. CPU PhotoWorxx - AIDA64 "NGImage2D" photo pipeline
// ===========================================================================
//
// AIDA64's NGImage2D benchmark is a photo pipeline that runs a fixed list of
// stages over 32-bit RGB images.  The stage list, image sizes and per-stage
// buffer counts come from the method descriptor table at 0xBF8978 in
// aida_bench64.dll (docs/reverse-engineering.md, section 11) and were
// confirmed by driving the DLL through its shared-memory settings channel:
//
//   #  stage      image       kernel observed in the DLL                     bufs
//   -  ---------  ----------  --------------------------------------------    ----
//     HWRandom   4096x3072   RDRAND / RDSEED / VIA XSTORE pixel fill           1
//     SWRandom   4096x3072   fixed-point polynomial PRNG pixel fill            1
//     Rot90      3072x2048   dst[x][h-1-y] = src[y][x], tiled, NT stores       2
//     Rot180     3072x2048   dst[i] = src[n-1-i], 64-pixel blocks, NT stores   2
//     Diff       2048x2048   dst = per channel |a-b|, alpha cleared            3
//     RGB2YUV    3072x2048   BT.601 studio YUV 4:2:0                           4
//     VP8YUV     3072x2048   the same conversion with an identity scale        4
//     Pattern    4096x3072   dst = (y<<16) | (x<<8) | ((x+y)&255)              1
//     Access     4096x3072   linear read                                        1
//     Fill       4096x3072   zero fill                                          1
//     Move       3072x2048   linear copy                                        2
//
// HWRandom needs a hardware random generator and the DLL's own automatic
// method selection (StartMethod 0x8000) fails its feature check on hosts
// without one - exactly as it does in the GUI - so this pipeline always folds
// HWRandom into the software fill and does not score it (documented deviation:
// the stage contributes no rate on this class of machine either way).
//
// AIDA64 times every stage in its own window and reports
//     stage rate = completed frames * X * Y / window_seconds
// in MPixel/s; one frame processes exactly X*Y pixels of that stage's image.
// The GUI's "CPU PhotoWorxx" number is the sum of the ten stage rates, so this
// port measures the same way: a wall-clock schedule hands every stage an equal
// window in round-robin order, each worker runs full-image passes of the
// current stage over its slice of the image, and every pass reports its
// slice's pixel count scaled by the stage count.  The harness's single
// total_work / elapsed division then lands on the sum of the stage rates.
constexpr int kPwStageCount = 10;
constexpr double kPwStageSeconds = 0.30;   // per-stage window (DLL minruntime)

struct PwStage {
    const char* name;
    int w;
    int h;
};

constexpr PwStage kPwStages[kPwStageCount] = {
    {"SWRandom", 4096, 3072},
    {"Rot90", 3072, 2048},
    {"Rot180", 3072, 2048},
    {"Diff", 2048, 2048},
    {"RGB2YUV", 3072, 2048},
    {"VP8YUV", 3072, 2048},
    {"Pattern", 4096, 3072},
    {"Access", 4096, 3072},
    {"Fill", 4096, 3072},
    {"Move", 3072, 2048},
};
constexpr size_t kPwMaxPixels = 4096ull * 3072ull;    // 12.58 MPixel
constexpr size_t kPwMidPixels = 3072ull * 2048ull;    // 6.29 MPixel
// Ring factor for the read-only stage inputs: the ring must be large enough
// that a window is evicted from any last level cache before it comes around
// again (8 x 50 MB = 400 MB for the Access ring).
constexpr size_t kPwRingFactor = 8;
constexpr uint64_t kPwSeed = 0x70686f746f77726bull;   // "photowrx"

// AIDA64 stores pixels as 0x00RRGGBB (BGRA byte order in memory).
inline unsigned pw_r(uint32_t p) { return (p >> 16) & 0xFFu; }
inline unsigned pw_g(uint32_t p) { return (p >> 8) & 0xFFu; }
inline unsigned pw_b(uint32_t p) { return p & 0xFFu; }

// SWRandom: random pixel fill.  The DLL's software generator is a four-tap
// fixed-point polynomial recurrence over four 32-bit states (coefficients at
// 0xC1A600 in the DLL).  Any ALU-comparable stream generator is store-bound at
// this image size, so this port uses splitmix64 to generate the same volume of
// pixels (documented deviation: the random values are not part of AIDA64's
// score or of ours).
//
// Every write stage here and below uses non-temporal stores.  The reference
// DLL's buffers are far larger than the CPU's last level cache (its per-run
// working set is ~20 bytes per pixel, 250 MB for this image size), so every
// AIDA64 stage is DRAM bound on any current CPU, and its selected kernels
// carry the matching *_NT suffixes.  Streaming stores keep the port's memory
// behaviour on the same side of the cache and make the per-stage rates
// comparable; the sfence after a stage publishes the writes for later passes.
void pw_swrandom(uint32_t* dst, size_t begin, size_t end, uint64_t seed) {
    uint64_t s = seed ^ (static_cast<uint64_t>(begin) * 0x9E3779B97F4A7C15ull);
    size_t i = begin;
    for (; i + 4 <= end; i += 4) {
        const uint64_t r0 = splitmix64(s);
        const uint64_t r1 = splitmix64(s);
        const __m128i v = _mm_set_epi32(static_cast<int>(r1 >> 32), static_cast<int>(r1),
                                        static_cast<int>(r0 >> 32), static_cast<int>(r0));
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), v);
    }
    for (; i < end; ++i) dst[i] = static_cast<uint32_t>(splitmix64(s));
    _mm_sfence();
}

// Rot90: dst[x][h-1-y] = src[y][x].  4x4 tiles keep both sides of the
// transpose cache resident.  The DLL's selected variant ("16x1_AVX512Quad_NT")
// transposes 16-wide tiles; a 4x4 tile only produces a 16-byte run per
// destination row, and non-temporal stores at that granularity defeat their
// own write combining, so this port keeps the destination cached and lets the
// eviction pressure of the whole cycle push it out, like AIDA64's larger
// buffers do.
void pw_rot90(uint32_t* dst, const uint32_t* src, int w, int h, int y_begin, int y_end) {
    const int dw = h;   // destination row stride
    int y = y_begin;
    for (; y + 4 <= y_end; y += 4) {
        int x = 0;
        for (; x + 4 <= w; x += 4) {
            const uint32_t* base = src + static_cast<size_t>(y) * w + x;
            __m128 r0 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base)));
            __m128 r1 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base + w)));
            __m128 r2 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base + 2 * w)));
            __m128 r3 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base + 3 * w)));
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);   // r_i = source column x+i
            uint32_t* out = dst + static_cast<size_t>(x) * dw + (h - 4 - y);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                             _mm_shuffle_epi32(_mm_castps_si128(r0), 0x1B));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + dw),
                             _mm_shuffle_epi32(_mm_castps_si128(r1), 0x1B));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * dw),
                             _mm_shuffle_epi32(_mm_castps_si128(r2), 0x1B));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 3 * dw),
                             _mm_shuffle_epi32(_mm_castps_si128(r3), 0x1B));
        }
        for (; x < w; ++x) {
            for (int j = 0; j < 4; ++j) {
                dst[static_cast<size_t>(x) * dw + (h - 1 - (y + j))] =
                    src[static_cast<size_t>(y + j) * w + x];
            }
        }
    }
    for (; y < y_end; ++y) {
        for (int x = 0; x < w; ++x) {
            dst[static_cast<size_t>(x) * dw + (h - 1 - y)] = src[static_cast<size_t>(y) * w + x];
        }
    }
}

// Rot180: dst[i] = src[n-1-i] over the linear image; 16-byte chunks are
// reversed with a shuffle and written streaming, mirroring the DLL's 64-pixel
// vpermd blocks with non-temporal stores.
void pw_rot180(uint32_t* dst, const uint32_t* src, size_t n, size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 4 <= end; i += 4) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + (n - i - 4)));
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), _mm_shuffle_epi32(v, 0x1B));
    }
    for (; i < end; ++i) dst[i] = src[n - 1 - i];
    _mm_sfence();
}

// Diff: per channel absolute difference of two images.  The fourth byte of
// the result is cleared, exactly like the DLL's scalar reference kernel
// (out = |B| | |G|<<8 | |R|<<16).  Like every write stage this one streams its
// output (AIDA64's selected Diff variant is X64__64x1_AVX512F_PERM_NT).
void pw_diff(uint32_t* dst, const uint32_t* a, const uint32_t* b, size_t begin, size_t end) {
    const __m256i keep = _mm256_set1_epi32(0x00FFFFFF);
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i d = _mm256_and_si256(
            _mm256_sub_epi8(_mm256_max_epu8(va, vb), _mm256_min_epu8(va, vb)), keep);
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_castsi256_si128(d));
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i + 4), _mm256_extracti128_si256(d, 1));
    }
    for (; i < end; ++i) {
        const uint32_t x = a[i];
        const uint32_t y = b[i];
        uint32_t out = 0;
        for (int shift = 0; shift < 24; shift += 8) {
            const unsigned p = (x >> shift) & 0xFFu;
            const unsigned q = (y >> shift) & 0xFFu;
            out |= (p > q ? p - q : q - p) << shift;
        }
        dst[i] = out;
    }
    _mm_sfence();
}

// RGB2YUV / VP8YUV: BT.601 studio-swing YUV 4:2:0.  The fixed-point
// coefficients are the DLL's and identical in both stages:
//   Y = ( 8421*R + 16515*G +  3211*B + 0x84000 ) >> 15
//   U = ( -4850*R - 9536*G + 14385*B + 0x404000) >> 15
//   V = (14385*R - 12059*G -  2327*B + 0x404000) >> 15
// A chroma sample is the average of a 2x2 block's two column averages, each
// rounded up first - the same (x+1)>>1 chain the DLL's scalar kernel uses.
// VP8YUV's only extra operation in the DLL is a multiply by a per-run scale
// word that the framework sets to 1, so the two stages share this kernel
// (documented deviation).
struct PwYuv {
    uint8_t y;
    int u;
    int v;
};

inline PwYuv pw_yuv_pixel(uint32_t p) {
    const int r = static_cast<int>(pw_r(p));
    const int g = static_cast<int>(pw_g(p));
    const int b = static_cast<int>(pw_b(p));
    PwYuv c;
    c.y = static_cast<uint8_t>((8421 * r + 16515 * g + 3211 * b + 0x84000) >> 15);
    c.u = (-4850 * r - 9536 * g + 14385 * b + 0x404000) >> 15;
    c.v = (14385 * r - 12059 * g - 2327 * b + 0x404000) >> 15;
    return c;
}

void pw_rgb2yuv(const uint32_t* src, uint8_t* y_plane, uint8_t* u_plane, uint8_t* v_plane,
                int w, int h, int y_begin, int y_end) {
    const int hw = w / 2;
    for (int y = y_begin; y + 2 <= y_end; y += 2) {
        const uint32_t* row0 = src + static_cast<size_t>(y) * w;
        const uint32_t* row1 = row0 + w;
        uint8_t* y0 = y_plane + static_cast<size_t>(y) * w;
        uint8_t* y1 = y0 + w;
        uint8_t* up = u_plane + static_cast<size_t>(y >> 1) * hw;
        uint8_t* vp = v_plane + static_cast<size_t>(y >> 1) * hw;
        for (int bx = 0; bx < hw; ++bx) {
            const PwYuv p00 = pw_yuv_pixel(row0[2 * bx]);
            const PwYuv p01 = pw_yuv_pixel(row0[2 * bx + 1]);
            const PwYuv p10 = pw_yuv_pixel(row1[2 * bx]);
            const PwYuv p11 = pw_yuv_pixel(row1[2 * bx + 1]);
            y0[2 * bx] = p00.y;
            y0[2 * bx + 1] = p01.y;
            y1[2 * bx] = p10.y;
            y1[2 * bx + 1] = p11.y;
            up[bx] = static_cast<uint8_t>(((p00.u + p10.u + 1) / 2 + (p01.u + p11.u + 1) / 2 + 1) / 2);
            vp[bx] = static_cast<uint8_t>(((p00.v + p10.v + 1) / 2 + (p01.v + p11.v + 1) / 2 + 1) / 2);
        }
    }
}

// Pattern: the DLL writes the deterministic gradient pixel
//     value = (y << 16) | (x << 8) | ((x + y) & 0xFF)
// (BGRA: B = (x+y)&255, G = x, R = y) - the reference image the Diff stage
// compares against in the real benchmark.  Streamed like the other writes.
void pw_pattern(uint32_t* dst, int w, int y_begin, int y_end) {
    for (int y = y_begin; y < y_end; ++y) {
        uint32_t* row = dst + static_cast<size_t>(y) * w;
        const uint32_t g = static_cast<uint32_t>(y) << 16;
        int x = 0;
        for (; x + 4 <= w; x += 4) {
            const __m128i v = _mm_set_epi32(
                static_cast<int>(g | (static_cast<uint32_t>(x + 3) << 8) |
                                 static_cast<uint32_t>((x + 3 + y) & 0xFF)),
                static_cast<int>(g | (static_cast<uint32_t>(x + 2) << 8) |
                                 static_cast<uint32_t>((x + 2 + y) & 0xFF)),
                static_cast<int>(g | (static_cast<uint32_t>(x + 1) << 8) |
                                 static_cast<uint32_t>((x + 1 + y) & 0xFF)),
                static_cast<int>(g | (static_cast<uint32_t>(x) << 8) |
                                 static_cast<uint32_t>((x + y) & 0xFF)));
            _mm_stream_si128(reinterpret_cast<__m128i*>(row + x), v);
        }
        for (; x < w; ++x) {
            row[x] = g | (static_cast<uint32_t>(x) << 8) |
                     static_cast<uint32_t>((x + y) & 0xFF);
        }
    }
    _mm_sfence();
}

// Access: linear read; the DLL's stage only touches the image, so the port
// folds the words into an XOR sink to keep the reads real.  The DLL's selected
// Access kernel reads one 32-bit word per iteration ("X64_1x1_MOV"), which is
// DRAM latency bound on any machine; the scalar loop below reproduces that
// shape (and therefore its throughput class) instead of issuing vector loads
// that would stream far faster.
uint64_t pw_access(const uint32_t* src, size_t begin, size_t end, uint64_t sink) {
    // volatile keeps this one 32-bit load per iteration; without it the
    // compiler vectorizes the reduction and the stage no longer resembles the
    // reference kernel.
    const volatile uint32_t* p = src + begin;
    const size_t count = end - begin;
    uint32_t acc = static_cast<uint32_t>(sink);
    for (size_t i = 0; i < count; ++i) {
        acc ^= p[i];
    }
    return sink ^ acc;
}

// Fill / Move: the DLL uses plain scalar MOV stores and a scalar copy loop.
// Both stages move 4 bytes per pixel, and the port keeps the same traffic but
// streams the destination so the behaviour stays DRAM bound like the DLL's.
void pw_fill(uint32_t* dst, size_t begin, size_t end) {
    const __m128i zero = _mm_setzero_si128();
    size_t i = begin;
    for (; i + 4 <= end; i += 4) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i), zero);
    }
    for (; i < end; ++i) dst[i] = 0u;
    _mm_sfence();
}

void pw_move(uint32_t* dst, const uint32_t* src, size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 4 <= end; i += 4) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
    }
    for (; i < end; ++i) dst[i] = src[i];
    _mm_sfence();
}

// The image set every stage works on.  AIDA64 reuses one set of buffers for
// all stages (each stage's X/Y decides how much of them is touched) and this
// port does the same: `a` and `b` are full 4096x3072 images, `c` is the Diff
// destination, and the luma/chroma planes are sized for the 3072x2048 stages.
//
// `cold` and `move_src` are read-only inputs placed in rings that are several
// times larger than any current last level cache.  The DLL's pipeline keeps
// no image hot: its per-run footprint (hundreds of megabytes of image and
// workspace buffers) is far larger than L3, so its read stages really do
// fetch from DRAM on every pass.  Sliding each pass's read window through a
// ring reproduces that behaviour on CPUs whose L3 could otherwise hold a
// single stage's image; the window always advances by exactly one image per
// pass, so the bytes touched per pass are unchanged.
struct PwImages {
    std::vector<uint32_t> a;        // 4096x3072, stride 4096 (written by stages)
    std::vector<uint32_t> b;        // 4096x3072; the first 6.29 MP hold a 3072x2048 image
    std::vector<uint32_t> c;        // 2048x2048, the Diff destination
    std::vector<uint32_t> cold_ring;   // 3 x 4096x3072, Access source
    std::vector<uint32_t> move_ring;   // 3 x 3072x2048, Move source
    std::vector<uint8_t> y1;        // 3072x2048 luma
    std::vector<uint8_t> y2;        // 3072x2048 luma
    std::vector<uint8_t> u;         // 1536x1024 chroma
    std::vector<uint8_t> v;         // 1536x1024 chroma
};

struct PwWorkerState {
    uint64_t access_sink = 0;
    size_t access_pos = 0;   // ring offset, advanced one image per pass
    size_t move_pos = 0;
};

// Splits `count` units over worker `worker` of `threads`, rounding both
// boundaries down to a multiple of `align` so the kernels only see whole
// vectors; the union of all workers still covers [0, count] because
// count is a multiple of `align` for every counter used here.
inline std::pair<size_t, size_t> pw_split(size_t count, int worker, int threads, size_t align) {
    const size_t a = (count * static_cast<size_t>(worker) / static_cast<size_t>(threads)) & ~(align - 1);
    const size_t b = (count * static_cast<size_t>(worker + 1) / static_cast<size_t>(threads)) & ~(align - 1);
    return {a, b};
}

// Runs one full-image pass of `stage` on worker `slot`'s slice and returns the
// number of pixels that slice covers (the stage's X*Y divided by the number of
// workers, up to rounding).
size_t pw_stage_slice(int stage, int slot, int threads, PwImages& img, PwWorkerState& ws) {
    switch (stage) {
    case 0: {   // SWRandom: fill image `a`
        const auto band = pw_split(kPwMaxPixels, slot, threads, 8);
        pw_swrandom(img.a.data(), band.first, band.second, kPwSeed + static_cast<uint64_t>(slot));
        return band.second - band.first;
    }
    case 1: {   // Rot90: rows of `a` (stride 4096) -> `b` (3072 rows x 2048)
        const auto band = pw_split(2048, slot, threads, 8);
        pw_rot90(img.b.data(), img.a.data(), 4096, 2048, static_cast<int>(band.first),
                 static_cast<int>(band.second));
        return (band.second - band.first) * 3072u;
    }
    case 2: {   // Rot180: `b` -> first 6.29 MP of `a`
        const auto band = pw_split(kPwMidPixels, slot, threads, 8);
        pw_rot180(img.a.data(), img.b.data(), kPwMidPixels, band.first, band.second);
        return band.second - band.first;
    }
    case 3: {   // Diff: `a`, `b` -> `c` over 2048x2048
        const auto band = pw_split(2048ull * 2048ull, slot, threads, 8);
        pw_diff(img.c.data(), img.a.data(), img.b.data(), band.first, band.second);
        return band.second - band.first;
    }
    case 4: {   // RGB2YUV: `a` -> `y1`/`u`/`v` over 3072x2048
        const auto band = pw_split(2048, slot, threads, 8);
        pw_rgb2yuv(img.a.data(), img.y1.data(), img.u.data(), img.v.data(), 3072, 2048,
                   static_cast<int>(band.first), static_cast<int>(band.second));
        return (band.second - band.first) * 3072u;
    }
    case 5: {   // VP8YUV: `b` -> `y2`/`u`/`v` over 3072x2048
        const auto band = pw_split(2048, slot, threads, 8);
        pw_rgb2yuv(img.b.data(), img.y2.data(), img.u.data(), img.v.data(), 3072, 2048,
                   static_cast<int>(band.first), static_cast<int>(band.second));
        return (band.second - band.first) * 3072u;
    }
    case 6: {   // Pattern: gradient into `a` over 4096x3072
        const auto band = pw_split(3072, slot, threads, 8);
        pw_pattern(img.a.data(), 4096, static_cast<int>(band.first), static_cast<int>(band.second));
        return (band.second - band.first) * 4096u;
    }
    case 7: {   // Access: linear read of the cold ring window
        const auto band = pw_split(kPwMaxPixels, slot, threads, 8);
        const size_t len = band.second - band.first;
        const size_t ring = img.cold_ring.size();
        size_t at = ws.access_pos + band.first;
        size_t remaining = len;
        while (remaining > 0) {
            if (at >= ring) at -= ring;
            const size_t chunk = std::min(remaining, ring - at);
            ws.access_sink = pw_access(img.cold_ring.data() + at, 0, chunk, ws.access_sink);
            at += chunk;
            remaining -= chunk;
        }
        ws.access_pos += len;
        if (ws.access_pos >= ring) ws.access_pos -= ring;
        return len;
    }
    case 8: {   // Fill: zero `a`
        const auto band = pw_split(kPwMaxPixels, slot, threads, 8);
        pw_fill(img.a.data(), band.first, band.second);
        return band.second - band.first;
    }
    default: {  // 9: Move: cold ring window -> `b` over 3072x2048
        const auto band = pw_split(kPwMidPixels, slot, threads, 8);
        const size_t len = band.second - band.first;
        const size_t ring = img.move_ring.size();
        size_t at = ws.move_pos + band.first;
        size_t remaining = len;
        while (remaining > 0) {
            if (at >= ring) at -= ring;
            const size_t chunk = std::min(remaining, ring - at);
            pw_move(img.b.data() + band.first + (len - remaining), img.move_ring.data() + at, 0,
                    chunk);
            at += chunk;
            remaining -= chunk;
        }
        ws.move_pos += len;
        if (ws.move_pos >= ring) ws.move_pos -= ring;
        return len;
    }
    }
}

// Deterministic single-threaded pass over all ten stages; the fingerprint is
// folded from every stage's output, so it covers the whole pipeline.  This is
// the value the recorded reference compares against; it does not depend on
// the number of workers, because the digest pass never splits the image.
uint64_t pw_pipeline_digest(PwImages& img) {
    PwWorkerState ws;
    uint64_t h = 0x70686f746full;   // "phot"
    pw_swrandom(img.a.data(), 0, kPwMaxPixels, kPwSeed);
    h = hash_bytes(img.a.data(), kPwMaxPixels * sizeof(uint32_t), h);
    pw_rot90(img.b.data(), img.a.data(), 4096, 2048, 0, 2048);
    h = hash_bytes(img.b.data(), kPwMidPixels * sizeof(uint32_t), h);
    pw_rot180(img.a.data(), img.b.data(), kPwMidPixels, 0, kPwMidPixels);
    h = hash_bytes(img.a.data(), kPwMidPixels * sizeof(uint32_t), h);
    const size_t diff_pixels = 2048ull * 2048ull;
    pw_diff(img.c.data(), img.a.data(), img.b.data(), 0, diff_pixels);
    h = hash_bytes(img.c.data(), diff_pixels * sizeof(uint32_t), h);
    pw_rgb2yuv(img.a.data(), img.y1.data(), img.u.data(), img.v.data(), 3072, 2048, 0, 2048);
    h = hash_bytes(img.y1.data(), kPwMidPixels, h);
    h = hash_bytes(img.u.data(), img.u.size(), h);
    h = hash_bytes(img.v.data(), img.v.size(), h);
    pw_rgb2yuv(img.b.data(), img.y2.data(), img.u.data(), img.v.data(), 3072, 2048, 0, 2048);
    h = hash_bytes(img.y2.data(), kPwMidPixels, h);
    h = hash_bytes(img.u.data(), img.u.size(), h);
    h = hash_bytes(img.v.data(), img.v.size(), h);
    pw_pattern(img.a.data(), 4096, 0, 3072);
    h = hash_bytes(img.a.data(), kPwMaxPixels * sizeof(uint32_t), h);
    ws.access_sink = pw_access(img.cold_ring.data(), 0, kPwMaxPixels, 0);
    h = hash_bytes(&ws.access_sink, sizeof(ws.access_sink), h);
    pw_fill(img.a.data(), 0, kPwMaxPixels);
    pw_move(img.b.data(), img.move_ring.data(), 0, kPwMidPixels);
    h = hash_bytes(img.b.data(), kPwMidPixels * sizeof(uint32_t), h);
    return h;
}

struct QueenWorker {
    uint64_t positions = 0;      // candidate boards checked (one per permutation)
    uint64_t solutions = 0;
    uint64_t digest_value = 0;   // fingerprint of the last pass's solution set
};

// run_once() is called with a worker index that prepare() has already sized;
// clamp defensively so a mis-sized caller cannot walk off the vector.
inline size_t clamp_slot(int worker, size_t count) {
    if (worker < 0) return 0;
    const size_t slot = static_cast<size_t>(worker);
    return slot < count ? slot : 0;
}

// ===========================================================================
// 3. CPU PhotoWorxx
// ===========================================================================
class PhotoWorxxBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "photoworxx";
        i.label = "CPU PhotoWorxx";
        i.unit = "MPixel/s";
        i.group = "CPU";
        i.unit_scale = 1e-6;   // pixels/s -> MPixel/s
        i.description =
            "AIDA64 NGImage2D photo pipeline: 10 stages on 2048..4096 wide RGB "
            "images (SWRandom/Rot90/Rot180/Diff/RGB2YUV/VP8YUV/Pattern/Access/"
            "Fill/Move); the score is the sum of the per-stage rates";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        threads_ = threads;
        workers_.assign(static_cast<size_t>(threads), PwWorkerState{});
        images_.a.assign(kPwMaxPixels, 0u);
        images_.b.assign(kPwMaxPixels, 0u);
        images_.c.assign(2048ull * 2048ull, 0u);
        images_.cold_ring.assign(kPwRingFactor * kPwMaxPixels, 0u);
        images_.move_ring.assign(kPwRingFactor * kPwMidPixels, 0u);
        images_.y1.assign(kPwMidPixels, 0);
        images_.y2.assign(kPwMidPixels, 0);
        images_.u.assign(1536ull * 1024ull, 0);
        images_.v.assign(1536ull * 1024ull, 0);
        // The read-only rings are written exactly once; every pass afterwards
        // reads a fresh window, so the rest of the cycle always evicts it.
        pw_swrandom(images_.cold_ring.data(), 0, images_.cold_ring.size(), kPwSeed ^ 0xC01DCAFEull);
        pw_swrandom(images_.move_ring.data(), 0, images_.move_ring.size(), kPwSeed ^ 0x0C0FFEE0ull);
        origin_.store(0.0);
        for (auto& counter : stage_pixels_) counter.store(0, std::memory_order_relaxed);
        printed_ = false;
        // One deterministic single-threaded pass: it produces the reference
        // fingerprint and seeds every buffer the timed stages will touch.
        digest_ = pw_pipeline_digest(images_);
    }

    // A call spends up to its budget on the stages the wall-clock schedule
    // selects, running whole image passes over this worker's slice.  Work is
    // reported as slice pixels times the stage count, so the harness's
    // total_work / elapsed division yields the sum of the per-stage rates.
    double run_once(int worker, int thread_count, double min_seconds) override {
        if (workers_.empty()) prepare(thread_count);
        const size_t slot = clamp_slot(worker, workers_.size());
        PwWorkerState& ws = workers_[slot];
        const double now = now_seconds();
        double expected = 0.0;
        origin_.compare_exchange_strong(expected, now);
        const double origin = origin_.load();
        const double call_end = now + std::max(0.0, min_seconds);
        double work = 0.0;
        for (;;) {
            const double t = now_seconds();
            if (t >= call_end) break;
            const long long tick = static_cast<long long>((t - origin) / kPwStageSeconds);
            const int stage = static_cast<int>(((tick % kPwStageCount) + kPwStageCount) % kPwStageCount);
            const size_t pixels = pw_stage_slice(stage, static_cast<int>(slot), threads_, images_, ws);
            stage_pixels_[stage].fetch_add(pixels, std::memory_order_relaxed);
            work += static_cast<double>(pixels) * kPwStageCount;
        }
        return work;
    }

    // Fingerprint of the deterministic digest pass; it cannot change while the
    // timed workers keep running (they never touch the digest), and it does
    // not depend on the worker count.
    uint64_t digest() const override {
        // The harness calls this after the first round; report the per-stage
        // rates once so they can be compared with the DLL's per-stage result
        // tags.  Each stage owns 1/kPwStageCount of the wall clock, so its rate
        // is (pixels / elapsed) * kPwStageCount.
        if (!printed_ && !workers_.empty()) {
            printed_ = true;
            const double elapsed = now_seconds() - origin_.load();
            if (elapsed > 0.0) {
                double sum = 0.0;
                std::printf("  per-stage rates (elapsed %.3f s, %d threads):\n",
                            elapsed, threads_);
                for (int i = 0; i < kPwStageCount; ++i) {
                    const double pixels = static_cast<double>(
                        stage_pixels_[static_cast<size_t>(i)].load(std::memory_order_relaxed));
                    const double rate = pixels * kPwStageCount / elapsed / 1e6;
                    sum += rate;
                    std::printf("    %-8s %10.1f MPixel/s\n", kPwStages[i].name, rate);
                }
                std::printf("    %-8s %10.1f MPixel/s\n", "sum", sum);
            }
        }
        return digest_;
    }
    bool has_digest() const override { return true; }

    uint64_t full_digest() const override { return corrupt_if_testing(digest_); }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        std::string s = "AIDA64 NGImage2D stages:";
        for (int i = 0; i < kPwStageCount; ++i) {
            s += ' ';
            s += kPwStages[i].name;
            s += '(';
            s += std::to_string(kPwStages[i].w);
            s += 'x';
            s += std::to_string(kPwStages[i].h);
            s += ')';
        }
        s += "; work = stage pixels x 10 so the score is the AIDA64 sum of rates";
        return s;
    }

private:
    int threads_ = 1;
    PwImages images_;
    std::vector<PwWorkerState> workers_;
    std::atomic<double> origin_{0.0};
    uint64_t digest_ = 0;
    mutable std::atomic<uint64_t> stage_pixels_[kPwStageCount] = {};
    mutable bool printed_ = false;
};

// ===========================================================================
// 4. CPU Queen - 10x10 N-Queens, AIDA64-aligned permutation scan
// ===========================================================================
//
// AIDA64's CPU Queen (ngqueen kernel, see docs/reverse-engineering.md section 10)
// walks *every* one of the 10! = 3,628,800 column permutations of the board
// instead of a pruned search tree, and screens each complete board with a
// byte-lane SIMD diagonal test.  Each board is represented as a 16-lane
// vector of column indices (one lane per row); two difference vectors
//   d1[row] = col - (15 - row)        // constant on one diagonal family
//   d2[row] = col - row               // constant on the other
// turn "same diagonal" into "equal byte k lanes apart".  Cyclically rotating
// the vector by one lane and comparing against itself therefore detects a
// shared diagonal after k rotations for every lane pair (k = 1 .. N/2 covers
// all N*(N-1)/2 pairs once); no branch, no early exit.  The pass reports its
// position count, so the throughput is directly AIDA64's "checked positions
// per second" figure, and it finds the known 724 solutions of the 10x10 board.
constexpr int kQueenLanes = 16;
constexpr int kQueenRotations = kQueenN / 2;      // cyclic distances 1..N/2
constexpr uint64_t kQueenPerms10 = 3628800ull;    // 10!, positions per pass

// Lane bias tables and the cyclic rotate-left-by-one shuffle of lanes 0..9
// (lanes >= 10 stay in place and are masked out of the result).
alignas(16) const uint8_t kQueenBias1[kQueenLanes] =
    {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
alignas(16) const uint8_t kQueenBias2[kQueenLanes] =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
alignas(16) const uint8_t kQueenRotate[kQueenLanes] =
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15};
alignas(16) const uint8_t kQueenStart[kQueenLanes] =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

// Swap-shuffle masks for Heap's permutation generator: entry [i][j] is the
// identity permutation with lanes i and j exchanged (lanes 0..9 only; lanes
// >= 10 always stay in place).
const std::array<std::array<uint8_t, kQueenLanes>, 100>& queen_swap_masks() {
    static const std::array<std::array<uint8_t, kQueenLanes>, 100> table = [] {
        std::array<std::array<uint8_t, kQueenLanes>, 100> t{};
        for (int i = 0; i < kQueenN; ++i) {
            for (int j = 0; j < kQueenN; ++j) {
                std::array<uint8_t, kQueenLanes>& m = t[static_cast<size_t>(i * 10 + j)];
                for (int k = 0; k < kQueenLanes; ++k) m[static_cast<size_t>(k)] = static_cast<uint8_t>(k);
                m[static_cast<size_t>(i)] = static_cast<uint8_t>(j);
                m[static_cast<size_t>(j)] = static_cast<uint8_t>(i);
            }
        }
        return t;
    }();
    return table;
}

// Order-independent fingerprint of one solution board: the ten 4-bit column
// indices packed from row 0 upwards, run through splitmix64 and summed.  The
// sum is commutative, so the self test can compare it against a reference
// enumeration that visits the solutions in a different order.
uint64_t queen_solution_hash(const uint8_t* lanes) {
    uint64_t packed = 0;
    for (int row = 0; row < kQueenN; ++row) {
        packed = (packed << 4) | static_cast<uint64_t>(lanes[row] & 0x0Fu);
    }
    uint64_t state = packed;
    return splitmix64(state);
}

struct QueenPass {
    uint64_t positions = 0;   // candidate boards checked (10! per pass)
    uint64_t solutions = 0;   // 724 on a correct 10x10 board
    uint64_t solsum = 0;      // sum of queen_solution_hash() over the solutions
};

// One complete pass over the permutation space.  Heap's algorithm yields each
// permutation with exactly one lane swap; applying that swap as a pshufb keeps
// the board vector in a register.
QueenPass queen_simd_pass() {
    QueenPass pass;
    const __m128i bias1 = _mm_load_si128(reinterpret_cast<const __m128i*>(kQueenBias1));
    const __m128i bias2 = _mm_load_si128(reinterpret_cast<const __m128i*>(kQueenBias2));
    const __m128i rot = _mm_load_si128(reinterpret_cast<const __m128i*>(kQueenRotate));
    const std::array<std::array<uint8_t, kQueenLanes>, 100>& masks = queen_swap_masks();
    __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(kQueenStart));
    uint8_t c[kQueenN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (;;) {
        ++pass.positions;
        const __m128i d1 = _mm_sub_epi8(v, bias1);
        const __m128i d2 = _mm_sub_epi8(v, bias2);
        __m128i r1 = d1;
        __m128i r2 = d2;
        unsigned acc = 0;
        for (int k = 0; k < kQueenRotations; ++k) {
            r1 = _mm_shuffle_epi8(r1, rot);
            r2 = _mm_shuffle_epi8(r2, rot);
            const __m128i eq = _mm_or_si128(_mm_cmpeq_epi8(r1, d1), _mm_cmpeq_epi8(r2, d2));
            acc |= static_cast<unsigned>(_mm_movemask_epi8(eq));
        }
        if ((acc & ((1u << kQueenN) - 1u)) == 0u) {
            ++pass.solutions;
            alignas(16) uint8_t lanes[kQueenLanes];
            _mm_store_si128(reinterpret_cast<__m128i*>(lanes), v);
            pass.solsum += queen_solution_hash(lanes);
        }

        // Next permutation (Heap's algorithm): one lane swap per step.
        int i = 1;
        while (i < kQueenN && c[i] == i) {
            c[i] = 0;
            ++i;
        }
        if (i == kQueenN) break;
        const int j = (i & 1) ? c[i] : 0;
        ++c[i];
        v = _mm_shuffle_epi8(v, _mm_load_si128(
                reinterpret_cast<const __m128i*>(masks[static_cast<size_t>(i * 10 + j)].data())));
    }
    return pass;
}

// Independent reference for the self test: enumerate all 10! permutations
// with std::next_permutation and a scalar O(N^2) conflict test, summing the
// same solution fingerprint.  Returns the sum and reports the count through
// `count`.
uint64_t queen_scan_all_reference(uint64_t& count) {
    uint8_t p[kQueenN];
    for (int i = 0; i < kQueenN; ++i) p[i] = static_cast<uint8_t>(i);
    count = 0;
    uint64_t sum = 0;
    for (;;) {
        bool ok = true;
        for (int a = 0; a < kQueenN && ok; ++a) {
            for (int b = a + 1; b < kQueenN; ++b) {
                const int d = static_cast<int>(p[a]) - static_cast<int>(p[b]);
                const int r = b - a;
                if (d == r || d == -r) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            ++count;
            sum += queen_solution_hash(p);
        }
        if (!std::next_permutation(p, p + kQueenN)) break;
    }
    return sum;
}

class QueenBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "queen";
        i.label = "CPU Queen";
        i.unit = "kPos/s";
        i.group = "CPU";
        i.unit_scale = 1e-3;   // positions/s -> kPos/s
        i.description =
            "N-Queens on a 10x10 board: all 3,628,800 column permutations "
            "(no pruning) screened by a byte-lane SIMD diagonal test, as "
            "AIDA64's CPU Queen kernel does";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        workers_.clear();
        workers_.resize(static_cast<size_t>(threads));
    }

    // One call == one pass over the full 10! permutation space ("a board").
    // The work returned is the number of candidate boards checked, which is
    // exactly the quantity AIDA64 counts as "checked positions".
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        (void)min_seconds;
        if (workers_.empty()) prepare(1);
        QueenWorker& w = workers_[clamp_slot(worker, workers_.size())];
        const QueenPass p = queen_simd_pass();
        w.positions += p.positions;
        w.solutions += p.solutions;
        w.digest_value = p.solsum;   // 724 solutions, same sum on every pass
        return static_cast<double>(p.positions);
    }

    // Fingerprint of the last pass's solution set.  Every pass finds the same
    // 724 boards, so this is identical in every repetition and across workers;
    // the harness re-checks it after every round.
    uint64_t digest() const override {
        uint64_t h = 0x717565656eull;   // "queen"
        for (const QueenWorker& w : workers_) {
            h = hash_bytes(&w.digest_value, sizeof(w.digest_value), h);
        }
        return h;
    }
    bool has_digest() const override { return true; }

    // The whole point of this workload is the solution set, so worker 0's
    // fingerprint is the reference value: a correct 10x10 pass finds exactly
    // 724 solutions with a fixed order-independent sum.
    uint64_t full_digest() const override {
        if (workers_.empty()) return  0;
        return  corrupt_if_testing(hash_bytes(&workers_[0].digest_value, sizeof(workers_[0].digest_value),
                          0x717565656eull));
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        return "10x10 board, all 10! = " + std::to_string(kQueenPerms10) +
               " column permutations scanned with a byte-lane SIMD diagonal "
               "check (cyclic lane rotation, " + std::to_string(kQueenRotations) +
               " rotations), 1 work unit = 1 candidate board checked (an "
               "AIDA64 \"position\"), " + std::to_string(kQueenSolutions10) +
               " solutions per pass";
    }

private:
    std::vector<QueenWorker> workers_;
};

// ===========================================================================
// 5. CPU CheckMate - 16x16 N-Queens, AIDA64 v7.35+ NGQueen flavour
// ===========================================================================
//
// AIDA64 v7.35 replaced CPU Queen with CPU CheckMate: the same NGQueen class
// driven with NGQueenSize = 16, where the auto method resolves to the DLL's
// AVX512_VPOPCNTDQ kernel on this host (docs section 16).  That kernel is not
// a pruned solver - it enumerates the complete 16! permutation space in
// chunks.  The top N-3 ranks are an ordered selection of distinct columns
// (the factoradic tree); every such prefix forms one chunk that evaluates the
// 3! = 6 completions of the last three ranks, and the "checked position"
// counter advances by 6 per chunk - one unit per permutation.
//
// The accounting is directly observable: driven with NGQueenSize = 8 the DLL
// completes a board every 8! = 40,320 checked positions (6,720 chunks) with
// exactly 92 solutions, and a 16x16 run can never complete a board (16! / 6 =
// 3,491,317,114,880 chunks), which is why the DLL reports
// NGQueenCompletedBoards = 0 for this test.  The GUI prints this rate in MP/s
// (12.2-12.7 GPos/s on this host at 16 threads).
//
// This kernel reproduces the chunk semantics: an iteratively advanced
// factoradic prefix, the two prefix diagonal masks carried incrementally, and
// a branch-free evaluation of the six completions per chunk.
constexpr uint64_t chase_fact(int n) {
    uint64_t f = 1;
    for (int i = 2; i <= n; ++i) f *= static_cast<uint64_t>(i);
    return f;
}

// Index of the lowest set bit; the masks below are never zero at the call sites.
inline int chase_ctz(uint32_t v) {
    unsigned long idx = 0;
    _BitScanForward(&idx, v);
    return static_cast<int>(idx);
}

template <int N>
class CheckMateChase {
public:
    static_assert(N >= 6 && N <= 16, "32-bit column and diagonal masks");

    static constexpr int kPrefix = N - 3;              // ranks the chunk tree iterates
    static constexpr int kLast = N - 1;                // d2 bias: col - rank + N - 1
    static constexpr uint64_t kChunks = chase_fact(N) / 6u;   // 6 positions per chunk

    // Chunks below one level-`level` choice of the factoradic tree.
    static constexpr uint64_t subtree_chunks(int level) {
        return chase_fact(N - level - 1) / 6u;
    }

    CheckMateChase() { reset(); }

    void reset() { reset_state(); ordinal_ = 0; }

    void reset_state() {
        used_ = (1u << N) - 1u;
        d1_ = 0;
        d2_ = 0;
        bad_ = 0;
        std::memset(d1c_, 0, sizeof(d1c_));
        std::memset(d2c_, 0, sizeof(d2c_));
        level_ = 0;
        for (int i = 0; i < kPrefix; ++i) sel_[i] = -1;
        nest_active_ = false;
        nest_ca_ = -1;
        nest_a1_ = 0;
        nest_a2_ = 0;
        nest_free_a_ = 0;
        nest_free_b_ = 0;
    }

    // Jump to a chunk ordinal (wraps).  The worker partition hands every
    // thread a contiguous slice of the chunk space, like the DLL's partition
    // of the NGQueen index space.
    void seek(uint64_t ordinal) {
        reset();
        ordinal %= kChunks;
        ordinal_ = ordinal;
        while (level_ < kPrefix) {
            const uint64_t sub = subtree_chunks(level_);
            const uint64_t digit = ordinal / sub;
            ordinal %= sub;
            uint32_t rem = used_;
            int c = chase_ctz(rem);
            for (uint64_t k = 0; k < digit; ++k) {
                rem &= rem - 1;
                c = chase_ctz(rem);
            }
            commit(level_, c);
            ++level_;
        }
    }

    // The two deepest prefix ranks are unrolled into a tight nest.  Once the
    // outer ranks are placed, the last two branch factors are four and five,
    // and the generic odometer pays a full scan/backtrack round trip per chunk
    // there - measured to dominate the runtime, with the six-completion
    // evaluation itself almost free beside it.  The nest keeps its cursors in
    // members so a batch budget can expire mid-subtree and the next call
    // resumes exactly where the previous one stopped.
    uint64_t nest_chunks(uint64_t limit, uint64_t& sols, uint64_t& done) {
        uint64_t positions = 0;
        if (!nest_active_) {
            // A seek can land in the middle of a rank N-4 subtree; the ranks
            // before the seek ordinal were another worker's slice, so the nest
            // starts after the column the seek committed.
            const int last_a = sel_[kPrefix - 2];
            nest_free_a_ = used_ & ~((last_a >= 0) ? ((1u << (last_a + 1)) - 1u) : 0u);
            nest_ca_ = -1;
            nest_free_b_ = 0;
            nest_active_ = true;
        }
        while (true) {
            if (nest_ca_ < 0) {
                if (nest_free_a_ == 0 || done >= limit) break;
                const int ca = chase_ctz(nest_free_a_);
                nest_free_a_ &= nest_free_a_ - 1;
                used_ &= ~(1u << ca);
                nest_ca_ = ca;
                nest_a1_ = 1u << (ca + kPrefix - 2);
                nest_a2_ = 1u << (ca - (kPrefix - 2) + kLast);
                nest_free_b_ = used_;
            }
            while (nest_free_b_ != 0 && done < limit) {
                const int cb = chase_ctz(nest_free_b_);
                nest_free_b_ &= nest_free_b_ - 1;
                const uint32_t b1 = 1u << (cb + kPrefix - 1);
                const uint32_t b2 = 1u << (cb - (kPrefix - 1) + kLast);
                const uint32_t ab1 = nest_a1_ | b1;
                const uint32_t ab2 = nest_a2_ | b2;
                // The two nest ranks never touch the shared masks, the
                // diagonal counters or bad_: their bits are merged into the
                // prefix view handed to the six-completion evaluation, and
                // the checks they still need are done here - against the
                // outer prefix, and the two ranks against each other (they
                // are adjacent, so a diagonal conflict is a column distance
                // of exactly one).
                const int d = nest_ca_ > cb ? nest_ca_ - cb : cb - nest_ca_;
                used_ &= ~(1u << cb);
                uint32_t s = eval_chunk(d1_ | ab1, d2_ | ab2);
                const uint32_t used_tail = used_;
                used_ |= 1u << cb;
                // The nest ranks bypass the counters, so their own conflicts
                // (against the outer prefix and each other - they are
                // adjacent, hence a column distance of exactly one) are
                // masked off here instead of inside the evaluation.
                if (d == 1 || (d1_ & ab1) != 0 || (d2_ & ab2) != 0) s = 0;
                const uint64_t mix = ordinal_ ^ (s * 0x9e3779b97f4a7c15ull) ^
                                     (static_cast<uint64_t>(used_tail) * 0x100000001b3ull);
                sols += s;
                positions += 6;
                if (track_digest_) digest_ += mix;
                if (!first_done_) {
                    first_done_ = true;
                    first_sols_ = s;
                    first_digest_ = mix;
                }
                if (++ordinal_ == kChunks) ordinal_ = 0;
                ++done;
            }
            if (nest_free_b_ != 0) break;   // budget expired mid-rank
            used_ |= 1u << nest_ca_;
            nest_a1_ = 0;
            nest_a2_ = 0;
            nest_ca_ = -1;
        }
        if (nest_ca_ < 0 && nest_free_a_ == 0) nest_active_ = false;
        return positions;
    }

    // Evaluate up to `max_chunks` chunks; returns the checked positions (six
    // per chunk) and accumulates the solutions found.
    uint64_t run_chunks(uint64_t max_chunks, uint64_t& sols) {
        uint64_t positions = 0;
        for (uint64_t done = 0; done < max_chunks;) {
            if (level_ == kPrefix - 2) {
                positions += nest_chunks(max_chunks, sols, done);
                if (done >= max_chunks) break;
                // The rank N-4 x rank N-3 subtree below this level-N-5 choice
                // is finished: release the choice like the generic backtrack
                // would and let its parent advance.
                sel_[level_] = -1;
                --level_;
                uncommit(level_);
                continue;
            }
            if (level_ < kPrefix) {
                const int next = sel_[level_] + 1;
                const uint32_t avail = used_ & ~((next >= 32) ? 0u : ((1u << next) - 1u));
                if (avail != 0) {
                    commit(level_, chase_ctz(avail));
                    ++level_;
                } else {
                    sel_[level_] = -1;
                    if (--level_ < 0) {
                        // A whole space was walked (tiny boards only): wrap.
                        reset_state();
                    } else {
                        uncommit(level_);
                    }
                }
                continue;
            }

            // Chunks reached through the generic path only happen right after
            // a seek, before the nest takes over; their prefix went through
            // the counter commits, so bad_ covers levels 11-12 as well.
            uint32_t s = eval_chunk(d1_, d2_);
            const uint64_t mix = ordinal_ ^ (s * 0x9e3779b97f4a7c15ull) ^
                                 (static_cast<uint64_t>(used_) * 0x100000001b3ull);
            sols += s;
            positions += 6;
            if (track_digest_) digest_ += mix;
            if (!first_done_) {
                first_done_ = true;
                first_sols_ = s;
                first_digest_ = mix;
            }
            if (++ordinal_ == kChunks) ordinal_ = 0;

            // Backtrack one rank: the deepest prefix level is released again.
            --level_;
            uncommit(level_);
            ++done;
        }
        return positions;
    }

    void enable_digest() { track_digest_ = true; digest_ = 0; }
    uint64_t digest() const { return digest_; }
    uint64_t first_digest() const { return first_digest_; }
    uint64_t first_sols() const { return first_sols_; }

private:
    inline void commit(int level, int col) {
        sel_[level] = static_cast<int8_t>(col);
        used_ &= ~(1u << col);
        const uint32_t k1 = static_cast<uint32_t>(col + level);
        const uint32_t k2 = static_cast<uint32_t>(col - level + kLast);
        // Diagonal occupancy is counted per diagonal: two prefix queens on the
        // same diagonal make that diagonal "bad" (and the chunk scores zero).
        bad_ += (d1c_[k1] == 1) + (d2c_[k2] == 1);
        ++d1c_[k1];
        ++d2c_[k2];
        d1_ |= 1u << k1;
        d2_ |= 1u << k2;
    }

    inline void uncommit(int level) {
        const int col = sel_[level];
        used_ |= 1u << col;
        const uint32_t k1 = static_cast<uint32_t>(col + level);
        const uint32_t k2 = static_cast<uint32_t>(col - level + kLast);
        --d1c_[k1];
        --d2c_[k2];
        bad_ -= (d1c_[k1] == 1) + (d2c_[k2] == 1);
        // The shared OR mask only clears when the last queen leaves the diagonal.
        if (d1c_[k1] == 0) d1_ &= ~(1u << k1);
        if (d2c_[k2] == 0) d2_ &= ~(1u << k2);
    }

    // The six completions of the three free columns across the last three
    // ranks.  d1 carries the occupied (col + rank) diagonals of the prefix,
    // d2 the (col - rank) ones; conflicts between tail queens are covered by
    // the pair distances.  Returns how many completions leave the board clean.
    // The generic prefix conflicts (bad_) gate the whole evaluation; conflicts
    // between the two nest ranks are masked by the caller on the result.
    inline uint32_t eval_chunk(uint32_t d1, uint32_t d2) const {
        if (bad_ != 0) return 0;          // the outer prefix already collides
        uint32_t rem = used_;
        const int x = chase_ctz(rem); rem &= rem - 1;
        const int y = chase_ctz(rem); rem &= rem - 1;
        const int z = chase_ctz(rem);

        // d1 bits: col + rank for ranks N-3, N-2, N-1.
        const uint32_t xa1 = 1u << (x + kPrefix);
        const uint32_t xb1 = 1u << (x + kPrefix + 1);
        const uint32_t xc1 = 1u << (x + kLast);
        const uint32_t ya1 = 1u << (y + kPrefix);
        const uint32_t yb1 = 1u << (y + kPrefix + 1);
        const uint32_t yc1 = 1u << (y + kLast);
        const uint32_t za1 = 1u << (z + kPrefix);
        const uint32_t zb1 = 1u << (z + kPrefix + 1);
        const uint32_t zc1 = 1u << (z + kLast);
        // d2 bits: col - rank + N - 1.
        const uint32_t xa2 = 1u << (x - kPrefix + kLast);
        const uint32_t xb2 = 1u << (x - kPrefix - 1 + kLast);
        const uint32_t xc2 = 1u << x;
        const uint32_t ya2 = 1u << (y - kPrefix + kLast);
        const uint32_t yb2 = 1u << (y - kPrefix - 1 + kLast);
        const uint32_t yc2 = 1u << y;
        const uint32_t za2 = 1u << (z - kPrefix + kLast);
        const uint32_t zb2 = 1u << (z - kPrefix - 1 + kLast);
        const uint32_t zc2 = 1u << z;

        // Pair distances: the three free columns are two, three or more apart.
        const uint32_t dxy = static_cast<uint32_t>(x > y ? x - y : y - x);
        const uint32_t dyz = static_cast<uint32_t>(y > z ? y - z : z - y);
        const uint32_t dxz = static_cast<uint32_t>(x > z ? x - z : z - x);
        const uint32_t xy1 = (dxy == 1), xy2 = (dxy == 2);
        const uint32_t yz1 = (dyz == 1), yz2 = (dyz == 2);
        const uint32_t xz1 = (dxz == 1), xz2 = (dxz == 2);

        uint32_t sols = 0;
        // (x, y, z) at ranks (N-3, N-2, N-1)
        sols += !(d1 & (xa1 | yb1 | zc1)) & !(d2 & (xa2 | yb2 | zc2)) & !xy1 & !yz1 & !xz2;
        // (x, z, y)
        sols += !(d1 & (xa1 | zb1 | yc1)) & !(d2 & (xa2 | zb2 | yc2)) & !xz1 & !yz1 & !xy2;
        // (y, x, z)
        sols += !(d1 & (ya1 | xb1 | zc1)) & !(d2 & (ya2 | xb2 | zc2)) & !xy1 & !xz1 & !yz2;
        // (y, z, x)
        sols += !(d1 & (ya1 | zb1 | xc1)) & !(d2 & (ya2 | zb2 | xc2)) & !yz1 & !xz1 & !xy2;
        // (z, x, y)
        sols += !(d1 & (za1 | xb1 | yc1)) & !(d2 & (za2 | xb2 | yc2)) & !xz1 & !xy1 & !yz2;
        // (z, y, x)
        sols += !(d1 & (za1 | yb1 | xc1)) & !(d2 & (za2 | yb2 | xc2)) & !yz1 & !xy1 & !xz2;
        return sols;
    }

    // Field order matters: everything a chunk touches lives in the first 128
    // bytes, and the benchmark pads each worker to 128 bytes, so no two
    // workers ever share a cache line that is written per chunk (adjacent
    // workers ping-ponging on a boundary line cost more than half the
    // multi-thread throughput; see the nest cursor fields).
    uint32_t used_ = 0, d1_ = 0, d2_ = 0;
    uint8_t d1c_[2 * N - 1], d2c_[2 * N - 1];
    uint32_t bad_ = 0;
    int8_t sel_[kPrefix];
    int level_ = 0;
    uint64_t ordinal_ = 0;
    bool nest_active_ = false;
    int nest_ca_ = -1;
    uint32_t nest_a1_ = 0, nest_a2_ = 0;   // rank N-4 diagonal bits
    uint32_t nest_free_a_ = 0, nest_free_b_ = 0;
    bool track_digest_ = false;
    uint64_t digest_ = 0;
    bool first_done_ = false;
    uint64_t first_digest_ = 0, first_sols_ = 0;
};

struct alignas(128) CheckMateWorker {
    CheckMateChase<16> chase;
    uint64_t solutions = 0;
    unsigned char pad_[64] = {};   // keep neighbouring workers off this line
};

// Independent scalar walk of the first `chunks` chunks: enumerates the same
// ordered prefixes in the same increasing-column order and checks the six
// completions with a naive pairwise conflict test.  Used by the self test to
// validate the fast kernel's chunk semantics (including at N = 16, where the
// full space cannot be walked).
template <int N>
uint64_t chase_ref_walk(int rank, uint32_t used, int* cols, uint64_t& chunks_left) {
    if (chunks_left == 0) return 0;
    if (rank == N - 3) {
        --chunks_left;
        int rem[3];
        int m = 0;
        for (int c = 0; c < N; ++c) {
            if (used & (1u << c)) rem[m++] = c;
        }
        static const int kPerm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                        {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        uint64_t sols = 0;
        for (int p = 0; p < 6; ++p) {
            int pos[N];
            for (int r = 0; r < N - 3; ++r) pos[r] = cols[r];
            pos[N - 3] = rem[kPerm[p][0]];
            pos[N - 2] = rem[kPerm[p][1]];
            pos[N - 1] = rem[kPerm[p][2]];
            bool ok = true;
            for (int a = 0; a < N && ok; ++a) {
                for (int b = a + 1; b < N && ok; ++b) {
                    const int d = pos[a] > pos[b] ? pos[a] - pos[b] : pos[b] - pos[a];
                    if (d == b - a) ok = false;
                }
            }
            if (ok) ++sols;
        }
        return sols;
    }
    uint64_t total = 0;
    for (int c = 0; c < N; ++c) {
        if (!(used & (1u << c))) continue;
        cols[rank] = c;
        total += chase_ref_walk<N>(rank + 1, used & ~(1u << c), cols, chunks_left);
        if (chunks_left == 0) break;
    }
    return total;
}

template <int N>
uint64_t chase_ref(uint64_t chunks) {
    int cols[N - 3];
    uint64_t left = chunks;
    return chase_ref_walk<N>(0, (1u << N) - 1u, cols, left);
}

// Lexicographically first solution of the n x n board (rows ascending, then
// columns ascending): the classic pruned bitmask DFS visits leaves in that
// order, so the first leaf reached is the requested board.
inline bool queens_first_solution_walk(int n, int row, unsigned cols, unsigned d1m,
                                       unsigned d2m, int* out) {
    if (row == n) return true;
    unsigned avail = ((1u << n) - 1u) & ~(cols | d1m | d2m);
    while (avail != 0) {
        const int c = chase_ctz(avail);
        avail &= avail - 1;
        out[row] = c;
        if (queens_first_solution_walk(n, row + 1, cols | (1u << c), (d1m | (1u << c)) << 1,
                                       (d2m | (1u << c)) >> 1, out)) {
            return true;
        }
    }
    return false;
}

inline bool queens_first_solution(int n, int* out) {
    return queens_first_solution_walk(n, 0, 0u, 0u, 0u, out);
}

class CheckMateBench final : public Benchmark {
public:
    using Chase = CheckMateChase<16>;

    BenchInfo info() const override {
        BenchInfo i;
        i.test = "checkmate";
        i.label = "CPU CheckMate";
        i.unit = "MP/s";
        i.group = "CPU";
        i.higher_is_better = true;
        i.unit_scale = 1e-6;                  // checked positions -> MP/s
        i.description =
            "16x16 N-Queens: complete 16! permutation space in 3-rank chunks, "
            "AIDA64 v7.35 NGQueen AVX512_VPOPCNTDQ semantics";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        workers_.clear();
        workers_.resize(static_cast<size_t>(threads));
        const uint64_t slice = Chase::kChunks / static_cast<uint64_t>(workers_.size());
        for (size_t k = 0; k < workers_.size(); ++k) {
            workers_[k].chase.seek(slice * k);
        }
    }

    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        if (workers_.empty()) prepare(1);
        CheckMateWorker& w = workers_[clamp_slot(worker, workers_.size())];
        const double t0 = now_seconds();
        uint64_t work = 0;
        do {
            uint64_t sols = 0;
            work += w.chase.run_chunks(1u << 13, sols);
            w.solutions += sols;
        } while (now_seconds() - t0 < min_seconds);
        return static_cast<double>(work);
    }

    // Each worker's first chunk: a fixed property of its slice of the chunk
    // space, so the digest does not depend on how long the run lasted.
    uint64_t digest() const override {
        uint64_t h = 0x636865636b6d6174ull;   // "checkmat"
        for (const CheckMateWorker& w : workers_) {
            const uint64_t pair[2] = {w.chase.first_digest(), w.chase.first_sols()};
            h = hash_bytes(pair, sizeof(pair), h);
        }
        return h;
    }
    bool has_digest() const override { return true; }

    // The first 2^18 chunks from the start of the space, single-threaded.
    uint64_t full_digest() const override {
        Chase c;
        c.seek(0);
        c.enable_digest();
        uint64_t sols = 0;
        c.run_chunks(1u << 18, sols);
        const uint64_t pair[2] = {c.digest(), sols};
        return corrupt_if_testing(hash_bytes(pair, sizeof(pair), 0x636865636b6d6174ull));
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        return "16x16 board, factoradic enumeration of the complete 16! permutation "
               "space, chunk = 3 ranks (3! = 6 completions), 1 checked position = "
               "1 permutation, " + std::to_string(Chase::kChunks) +
               " chunks per complete board, module=Base_NGQueen_AVX512_VPOPCNTDQ";
    }

private:
    std::vector<CheckMateWorker> workers_;
};

// ===========================================================================
// 6. Self test
// ===========================================================================
// Independently written reference transforms for the photo kernels.  They only
// use the mappings documented in section 11 of the reverse-engineering notes.
using PwImage = std::vector<uint32_t>;

PwImage make_test_image(int w, int h, uint64_t seed) {
    PwImage img(static_cast<size_t>(w) * static_cast<size_t>(h), 0u);
    uint64_t s = seed;
    for (size_t i = 0; i < img.size(); ++i) {
        const uint64_t r = splitmix64(s);
        img[i] = static_cast<uint32_t>(r) & 0x00FFFFFFu;   // BGRA, alpha zero
    }
    return img;
}

// dst[x][h-1-y] = src[y][x]; dst is h wide and w tall.
PwImage ref_rot90(const PwImage& src, int w, int h) {
    PwImage dst(static_cast<size_t>(w) * static_cast<size_t>(h), 0u);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            dst[static_cast<size_t>(x) * h + (h - 1 - y)] = src[static_cast<size_t>(y) * w + x];
        }
    }
    return dst;
}

PwImage ref_rot180(const PwImage& src) {
    PwImage dst(src.size(), 0u);
    for (size_t i = 0; i < src.size(); ++i) dst[i] = src[src.size() - 1 - i];
    return dst;
}

PwImage ref_diff(const PwImage& a, const PwImage& b) {
    PwImage dst(a.size(), 0u);
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t out = 0;
        for (int shift = 0; shift < 24; shift += 8) {
            const unsigned p = (a[i] >> shift) & 0xFFu;
            const unsigned q = (b[i] >> shift) & 0xFFu;
            out |= (p > q ? p - q : q - p) << shift;
        }
        dst[i] = out;
    }
    return dst;
}

// Double precision BT.601 studio-swing model of the fixed-point conversion.
struct YuvRefPixel {
    double y;
    double u;
    double v;
};

YuvRefPixel yuv_ref(double r, double g, double b) {
    YuvRefPixel c;
    c.y = 16.0 + (8421.0 * r + 16515.0 * g + 3211.0 * b) / 32768.0;
    c.u = 128.0 + (-4850.0 * r - 9536.0 * g + 14385.0 * b) / 32768.0;
    c.v = 128.0 + (14385.0 * r - 12059.0 * g - 2327.0 * b) / 32768.0;
    return c;
}

int fail(int code, const char* what) {
    std::printf("image selftest: FAIL(%d) at %s\n", code, what);
    return code;
}

} // namespace

int verify_image_kernels() {
    // --- N-Queens -----------------------------------------------------------
    {
        const uint64_t q8 = Queens<8>::count_all();
        std::printf("  queen  8x8  solutions: %llu\n", static_cast<unsigned long long>(q8));
        if (q8 != kQueenSolutions8) return fail(1, "8x8 bitmask count != 92");
        if (queens_ref(8) != kQueenSolutions8) return fail(2, "8x8 reference count != 92");

        const uint64_t q10 = Queens<kQueenN>::count_all();
        std::printf("  queen 10x10 solutions: %llu\n", static_cast<unsigned long long>(q10));
        if (q10 != kQueenSolutions10) return fail(3, "10x10 bitmask count != 724");
        if (queens_ref(10) != kQueenSolutions10) return fail(4, "10x10 reference count != 724");

        // The AIDA64-aligned CPU Queen kernel: one pass must check the whole
        // 10! permutation space, find exactly 724 solutions, and fingerprint
        // them identically to an independent scan of all permutations.
        const QueenPass qpass = queen_simd_pass();
        std::printf("  queen 10x10 SIMD     : %llu positions, %llu solutions\n",
                    static_cast<unsigned long long>(qpass.positions),
                    static_cast<unsigned long long>(qpass.solutions));
        if (qpass.positions != kQueenPerms10) {
            return fail(33, "queen SIMD pass did not check 10! positions");
        }
        if (qpass.solutions != kQueenSolutions10) {
            return fail(34, "queen SIMD pass found != 724 solutions");
        }
        uint64_t ref_solutions = 0;
        const uint64_t ref_sum = queen_scan_all_reference(ref_solutions);
        if (ref_solutions != kQueenSolutions10) {
            return fail(35, "all-permutations reference found != 724 solutions");
        }
        if (qpass.solsum != ref_sum) {
            return fail(36, "queen SIMD solution fingerprint != reference scan");
        }

        // The v7.35 CheckMate enumeration runs the complete N! permutation space
        // in 3-rank chunks.  Small boards enumerate exhaustively and must match
        // the known counts; the 16x16 kernel is probed chunk-wise against an
        // independent scalar walk (a full 16! space cannot be walked here).
        {
            uint64_t s8 = 0;
            CheckMateChase<8> c8;
            const uint64_t p8 = c8.run_chunks(CheckMateChase<8>::kChunks, s8);
            std::printf("  checkmate  8x8      : %llu positions, %llu solutions\n",
                        static_cast<unsigned long long>(p8),
                        static_cast<unsigned long long>(s8));
            if (p8 != chase_fact(8) || s8 != kQueenSolutions8) {
                return fail(10, "8x8 chunk enumeration != 8! positions / 92 solutions");
            }

            uint64_t s9 = 0;
            CheckMateChase<9> c9;
            const uint64_t p9 = c9.run_chunks(CheckMateChase<9>::kChunks, s9);
            if (p9 != chase_fact(9) || s9 != chase_ref<9>(CheckMateChase<9>::kChunks)) {
                return fail(11, "9x9 chunk enumeration disagrees with the reference walk");
            }

            uint64_t s10 = 0;
            CheckMateChase<10> c10;
            const uint64_t p10 = c10.run_chunks(CheckMateChase<10>::kChunks, s10);
            std::printf("  checkmate 10x10     : %llu positions, %llu solutions\n",
                        static_cast<unsigned long long>(p10),
                        static_cast<unsigned long long>(s10));
            if (p10 != chase_fact(10) || s10 != kQueenSolutions10) {
                return fail(12, "10x10 chunk enumeration != 10! positions / 724 solutions");
            }

            // Worker slices must partition the chunk space exactly.
            uint64_t tp = 0, ts = 0;
            const uint64_t slice = CheckMateChase<10>::kChunks / 4;
            for (int k = 0; k < 4; ++k) {
                CheckMateChase<10> c;
                c.seek(slice * static_cast<uint64_t>(k));
                uint64_t s = 0;
                tp += c.run_chunks(slice, s);
                ts += s;
            }
            if (tp != chase_fact(10) || ts != kQueenSolutions10) {
                return fail(13, "10x10 worker slices do not partition the chunk space");
            }

            // 16x16: the first 2^16 chunks against the independent walk.
            uint64_t s16 = 0;
            CheckMateChase<16> c16;
            const uint64_t probe = 1u << 16;
            const uint64_t p16 = c16.run_chunks(probe, s16);
            const uint64_t r16 = chase_ref<16>(probe);
            std::printf("  checkmate 16x16 probe: %llu positions, %llu solutions "
                        "(reference: %llu)\n",
                        static_cast<unsigned long long>(p16),
                        static_cast<unsigned long long>(s16),
                        static_cast<unsigned long long>(r16));
            if (p16 != probe * 6 || s16 != r16) {
                return fail(14, "16x16 chunk probe disagrees with the reference walk");
            }

            // 16x16 solution probe: locate the chunk of the lexicographically
            // first solution through an independently written factoradic rank,
            // then the fast kernel must find exactly the reference count there.
            int sol[16];
            if (!queens_first_solution(16, sol)) return fail(15, "no 16x16 solution found");
            uint64_t ordinal = 0;
            uint32_t free_cols = (1u << 16) - 1u;
            for (int L = 0; L < 13; ++L) {
                const int c = sol[L];
                if (!(free_cols & (1u << c))) return fail(16, "16x16 solution is not a permutation");
                uint64_t digit = 0;
                for (uint32_t m = free_cols & ((1u << c) - 1u); m != 0; m &= m - 1) ++digit;
                ordinal += digit * (chase_fact(16 - L - 1) / 6u);
                free_cols &= ~(1u << c);
            }
            uint64_t ref_chunk = 0;
            {
                int rem[3];
                int m = 0;
                for (int c = 0; c < 16; ++c) {
                    if (free_cols & (1u << c)) rem[m++] = c;
                }
                static const int kPerm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                                {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
                for (int p = 0; p < 6; ++p) {
                    int pos[16];
                    for (int r = 0; r < 13; ++r) pos[r] = sol[r];
                    pos[13] = rem[kPerm[p][0]];
                    pos[14] = rem[kPerm[p][1]];
                    pos[15] = rem[kPerm[p][2]];
                    bool ok = true;
                    for (int a = 0; a < 16 && ok; ++a) {
                        for (int b = a + 1; b < 16 && ok; ++b) {
                            const int d = pos[a] > pos[b] ? pos[a] - pos[b] : pos[b] - pos[a];
                            if (d == b - a) ok = false;
                        }
                    }
                    if (ok) ++ref_chunk;
                }
            }
            CheckMateChase<16> c16s;
            c16s.seek(ordinal);
            uint64_t s_fast = 0;
            const uint64_t p_fast = c16s.run_chunks(1, s_fast);
            std::printf("  checkmate 16x16 solution chunk: rank %llu, %llu solutions "
                        "(reference: %llu)\n",
                        static_cast<unsigned long long>(ordinal),
                        static_cast<unsigned long long>(s_fast),
                        static_cast<unsigned long long>(ref_chunk));
            if (p_fast != 6 || ref_chunk == 0 || s_fast != ref_chunk) {
                return fail(17, "16x16 solution chunk not found at its factoradic rank");
            }
        }

        // The pruned bitmask solver stays in the tree as the independent
        // reference for the known 16x16 solution total.
        const std::vector<std::pair<int, int>>& prefixes = checkmate_prefixes();
        std::printf("  checkmate prefixes  : %zu (expect 210)\n", prefixes.size());
        if (prefixes.size() != 210) return fail(5, "16x16 two-row prefix table is not 210 entries");
        for (size_t i = 0; i < prefixes.size(); ++i) {
            for (size_t j = i + 1; j < prefixes.size(); ++j) {
                if (prefixes[i] == prefixes[j]) return fail(6, "duplicate 16x16 prefix");
            }
        }
        int ref_probes = 0;
        for (size_t i = 0; i < prefixes.size() && ref_probes < 3; i += 83, ++ref_probes) {
            const std::pair<int, int>& p = prefixes[i];
            const uint64_t fast = Queens<kCheckMateN>::count_prefix(p.first, p.second);
            const uint64_t slow = queens_ref_prefix(kCheckMateN, p.first, p.second);
            if (fast != slow) return fail(7, "16x16 subtree disagrees with the reference solver");
        }

        int threads = 8;
        if (cpu_info().logical_cores > 0) threads = std::min(threads, cpu_info().logical_cores);
        const uint64_t full_a = checkmate_count_full(threads);
        std::printf("  checkmate 16x16 pruned solver: %llu solutions\n",
                    static_cast<unsigned long long>(full_a));
        if (full_a != kCheckMateSolutions16) return fail(8, "16x16 total != 14772512");
        const uint64_t full_b = checkmate_count_full(threads);
        std::printf("  checkmate 16x16 repeated : %llu\n", static_cast<unsigned long long>(full_b));
        if (full_a != full_b) return fail(9, "16x16 count is not deterministic across repetitions");
    }

    // --- PhotoWorxx stage kernels ------------------------------------------
    {
        // Pattern: the gradient the DLL writes for every pixel.
        {
            const int w = 64, h = 12;
            PwImage img(static_cast<size_t>(w) * h, 0u);
            pw_pattern(img.data(), w, 0, h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const uint32_t want = (static_cast<uint32_t>(y) << 16) |
                                          (static_cast<uint32_t>(x) << 8) |
                                          static_cast<uint32_t>((x + y) & 0xFF);
                    if (img[static_cast<size_t>(y) * w + x] != want) {
                        return fail(10, "pattern gradient value");
                    }
                }
            }
        }
        // Rot90: exact mapping on a tiled size, and four rotations are the
        // identity.
        {
            const int w = 24, h = 16;
            const PwImage src = make_test_image(w, h, 0xABCDEF01ull);
            PwImage dst(static_cast<size_t>(w) * h, 0u);
            pw_rot90(dst.data(), src.data(), w, h, 0, h);
            if (dst != ref_rot90(src, w, h)) return fail(11, "rot90 mapping");
            PwImage r2(dst.size(), 0u), r3(dst.size(), 0u), r4(dst.size(), 0u);
            pw_rot90(r2.data(), dst.data(), h, w, 0, w);
            pw_rot90(r3.data(), r2.data(), w, h, 0, h);
            pw_rot90(r4.data(), r3.data(), h, w, 0, w);
            if (r4 != src) return fail(12, "rot90 applied four times is not the identity");
        }
        // Rot180: reversal (with a tail that is not a multiple of four, which
        // exercises the scalar path) and involution.
        {
            const size_t n = 257;
            const PwImage src = make_test_image(1, static_cast<int>(n), 0x12345678ull);
            PwImage dst(n, 0u);
            pw_rot180(dst.data(), src.data(), n, 0, n);
            if (dst != ref_rot180(src)) return fail(13, "rot180 mapping");
            PwImage back(n, 0u);
            pw_rot180(back.data(), dst.data(), n, 0, n);
            if (back != src) return fail(14, "rot180 applied twice is not the identity");
        }
        // Diff: per channel absolute difference with the alpha byte cleared.
        {
            const PwImage a = make_test_image(8, 16, 0x0F0F0F0Full);
            const PwImage b = make_test_image(8, 16, 0x22222222ull);
            PwImage d(a.size(), 0xFFFFFFFFu);
            pw_diff(d.data(), a.data(), b.data(), 0, a.size());
            if (d != ref_diff(a, b)) return fail(15, "diff mapping");
            for (uint32_t p : d) {
                if ((p >> 24) != 0u) return fail(16, "diff alpha is not cleared");
            }
        }
        // RGB2YUV: luma and 4:2:0 chroma against the double precision BT.601
        // studio model.  The fixed-point kernel rounds each channel with the
        // 0x84000 / 0x404000 (0.5 * 2^15) bias, i.e. round-half-up, and then
        // averages the chroma with the (x+1)>>1 chain - both are modelled here
        // so the comparison can be exact.
        {
            const int w = 32, h = 8;
            const PwImage img = make_test_image(w, h, 0x5A5A1234ull);
            const int hw = w / 2, hh = h / 2;
            std::vector<uint8_t> yplane(static_cast<size_t>(w) * h, 0);
            std::vector<uint8_t> uplane(static_cast<size_t>(hw) * hh, 0);
            std::vector<uint8_t> vplane(uplane.size(), 0);
            pw_rgb2yuv(img.data(), yplane.data(), uplane.data(), vplane.data(), w, h, 0, h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const uint32_t p = img[static_cast<size_t>(y) * w + x];
                    const YuvRefPixel want = yuv_ref(pw_r(p), pw_g(p), pw_b(p));
                    const int got = yplane[static_cast<size_t>(y) * w + x];
                    const int expect = static_cast<int>(want.y + 0.5);   // floor, all positive
                    if (got != expect) return fail(17, "yuv luma");
                }
            }
            for (int by = 0; by < hh; ++by) {
                for (int bx = 0; bx < hw; ++bx) {
                    int us[2] = {0, 0};
                    int vs[2] = {0, 0};
                    for (int l = 0; l < 2; ++l) {
                        for (int k = 0; k < 2; ++k) {
                            const uint32_t p = img[static_cast<size_t>(2 * by + l) * w + 2 * bx + k];
                            const YuvRefPixel c = yuv_ref(pw_r(p), pw_g(p), pw_b(p));
                            // column averages: rows 2*by+0 and 2*by+1 share a column
                            us[k] += static_cast<int>(c.u + 0.5);
                            vs[k] += static_cast<int>(c.v + 0.5);
                        }
                    }
                    const int u_expect = ((us[0] + 1) / 2 + (us[1] + 1) / 2 + 1) / 2;
                    const int v_expect = ((vs[0] + 1) / 2 + (vs[1] + 1) / 2 + 1) / 2;
                    const size_t at = static_cast<size_t>(by) * hw + bx;
                    if (uplane[at] != u_expect) return fail(18, "yuv chroma u");
                    if (vplane[at] != v_expect) return fail(19, "yuv chroma v");
                }
            }
            // Solid red must have a high V and low U, solid blue the reverse.
            {
                const uint32_t colours[2] = {0x00FF0000u, 0x000000FFu};   // red, blue
                for (uint32_t c : colours) {
                    PwImage flat(static_cast<size_t>(2) * 2, c);
                    std::vector<uint8_t> yp(4, 0), up(1, 0), vp(1, 0);
                    pw_rgb2yuv(flat.data(), yp.data(), up.data(), vp.data(), 2, 2, 0, 2);
                    if (c == 0x00FF0000u) {
                        if (!(vp[0] > 200 && up[0] < 120)) return fail(20, "yuv red");
                    } else {
                        if (!(up[0] > 200 && vp[0] < 120)) return fail(21, "yuv blue");
                    }
                }
            }
        }
        // Fill / Move / Access round trip.
        {
            PwImage a = make_test_image(64, 32, 0x777777ull);
            const PwImage orig = a;
            pw_fill(a.data(), 100, 500);
            for (size_t i = 100; i < 500; ++i) {
                if (a[i] != 0u) return fail(22, "fill did not zero its range");
            }
            if (!std::equal(orig.begin(), orig.begin() + 100, a.begin())) {
                return fail(23, "fill touched pixels outside its range");
            }
            PwImage b(a.size(), 0u);
            pw_move(b.data(), a.data(), 0, a.size());
            if (b != a) return fail(24, "move corrupted data");
            uint64_t expect = 0;
            for (uint32_t p : a) expect ^= p;
            if (pw_access(a.data(), 0, a.size(), 0) != expect) {
                return fail(25, "access sink does not match the data");
            }
        }
        // Deterministic digest of the full ten-stage pipeline: two
        // single-threaded passes must produce the same fingerprint.
        {
            PwImages imgs;
            imgs.a.assign(kPwMaxPixels, 0u);
            imgs.b.assign(kPwMaxPixels, 0u);
            imgs.c.assign(2048ull * 2048ull, 0u);
            imgs.cold_ring.assign(kPwRingFactor * kPwMaxPixels, 0u);
            imgs.move_ring.assign(kPwRingFactor * kPwMidPixels, 0u);
            imgs.y1.assign(kPwMidPixels, 0);
            imgs.y2.assign(kPwMidPixels, 0);
            imgs.u.assign(1536ull * 1024ull, 0);
            imgs.v.assign(1536ull * 1024ull, 0);
            pw_swrandom(imgs.cold_ring.data(), 0, imgs.cold_ring.size(), kPwSeed ^ 0xC01DCAFEull);
            pw_swrandom(imgs.move_ring.data(), 0, imgs.move_ring.size(), kPwSeed ^ 0x0C0FFEE0ull);
            double t0 = now_seconds();
            const uint64_t h1 = pw_pipeline_digest(imgs);
            const double pass_seconds = now_seconds() - t0;
            const uint64_t h2 = pw_pipeline_digest(imgs);
            if (h1 != h2) return fail(26, "photoworxx digest is not reproducible");
            std::printf("  photoworxx 10 stages: %.2f ms (%dx%d images), digest %016llx\n",
                        pass_seconds * 1e3, 4096, 3072,
                        static_cast<unsigned long long>(h1));
            // Single-threaded per-stage rates (one full pass each, warm
            // buffers).  These are the numbers to compare against the DLL's
            // per-stage results; the benchmark's figure is the sum of the
            // stage rates at the requested thread count.
            PwWorkerState ws;
            for (int i = 0; i < kPwStageCount; ++i) {
                const double s0 = now_seconds();
                const size_t px = pw_stage_slice(i, 0, 1, imgs, ws);
                const double s1 = now_seconds();
                std::printf("    stage %-8s %8.2f ms  %9.1f Mpx/s (1 thread)\n",
                            kPwStages[i].name, (s1 - s0) * 1e3,
                            static_cast<double>(px) / (s1 - s0) / 1e6);
            }
        }
    }

    // --- Benchmark objects ---------------------------------------------------
    {
        Benchmark* benches[3] = {make_bench_photoworxx(), make_bench_queen(),
                                 make_bench_checkmate()};
        const char* expected[3] = {"photoworxx", "queen", "checkmate"};
        for (int i = 0; i < 3; ++i) {
            Benchmark* b = benches[i];
            const BenchInfo info = b->info();
            if (info.test != expected[i]) return fail(24, "benchmark id mismatch");
            if (info.group != "CPU") return fail(25, "benchmark group mismatch");
            if (info.unit_scale <= 0.0) return fail(26, "unit_scale must be positive");
            b->prepare(2);
            // The harness hands a budget of up to 0.25 s per call.
            const double work0 = b->run_once(0, 2, 0.25);
            const double work1 = b->run_once(1, 2, 0.25);
            if (!(work0 > 0.0) || !(work1 > 0.0)) return fail(27, "run_once returned no work");
            if (!b->has_digest()) return fail(28, "benchmark reports no digest");
            const uint64_t d0 = b->digest();
            if (d0 == 0) return fail(29, "benchmark digest is zero");
            // The digest must not change as further units complete.
            b->run_once(0, 2, 0.25);
            if (b->digest() != d0) return fail(30, "benchmark digest is not repetition stable");
            std::printf("  [%s] unit work = %.0f / %.0f units, digest %016llx\n", info.test.c_str(),
                        work0, work1, static_cast<unsigned long long>(d0));
        }
        // Queen checks the whole 10! permutation space per call; a checkmate
        // call returns checked positions - always a multiple of the six
        // completions evaluated per 3-rank chunk.
        if (benches[1]->run_once(0, 2, 0.25) != static_cast<double>(kQueenPerms10)) {
            return fail(31, "queen work unit != 10!");
        }
        const double cm = benches[2]->run_once(0, 2, 0.25);
        const uint64_t cm_pos = static_cast<uint64_t>(cm);
        if (cm_pos == 0 || cm_pos % 6 != 0) return fail(32, "checkmate work is not a multiple of 6 positions");
    }

    return 0;
}

Benchmark* make_bench_photoworxx() {
    static PhotoWorxxBench b;
    return &b;
}

Benchmark* make_bench_queen() {
    static QueenBench b;
    return &b;
}

Benchmark* make_bench_checkmate() {
    static CheckMateBench b;
    return &b;
}

} // namespace cst