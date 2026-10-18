// Memory-subsystem benchmark module: Memory Read / Write / Copy / Latency.
//
// The three bandwidth tests are aligned with the kernels AIDA64 actually runs
// for its "Memory Read / Write / Copy" page.  Those came from the MTMBW bench
// class, whose "Mem"-level code dumps (CodeDump_MTMBW_X64_<dir>_Mem_001.bin,
// docs/reverse-engineering.md section 9) show, on an AVX-512 CPU, the ZMM_DQ64
// variant as the selected one:
//
//   Memory Read    linear sweep of non-temporal 64-byte loads (vmovntdqa) over
//                  a 96 MiB, 64 KiB-aligned buffer per thread (AIDA64 reports
//                  SizeInKiBytes = 98304 for this level).
//   Memory Write   linear sweep of non-temporal 64-byte stores (vmovntdq) over
//                  a 96 MiB buffer per thread.  The store register is all
//                  zeros, exactly as in AIDA64's kernel (its dump zeroes the
//                  vector registers with vpxord before the loop) - on Zen5 that
//                  is not cosmetic, see "Buffer contents" below.
//   Memory Copy    non-temporal 64-byte loads + 64-byte NT stores over a 48 MiB
//                  source and a 48 MiB destination per thread (AIDA64 reports
//                  49152), one load->register->store per 64 B, never memcpy/
//                  rep movs.  The source holds zeros, as AIDA64's does.
//   Memory Latency single-threaded dependent-load pointer chase over a chained
//                  structure of >= 16 MiB built out of 4 KiB pages, with a node
//                  stride of >= 128 bytes so the L2 adjacent-line prefetcher
//                  cannot drag the next node's line in with the current one.
//
// AIDA64's kernels unroll one 8 KiB swing per iteration for read/write and one
// 4 KiB swing for copy, and put a clflushopt sweep of the whole region (plus
// mfence) between repetitions; the flush is excluded from its timed window.
// These kernels iterate in 512 B units - the DRAM access pattern is the same -
// and skip the flush, because the harness times every run_once() call with a
// wall clock (a flush can cost more than the timed sweep itself) and the
// per-thread working set is far larger than the LLC anyway.
//
// The kernels select the AVX-512 sweep when cpu_info().has_avx512f is set and
// fall back to byte-identical 256-bit AVX2 sweeps otherwise (mem_kernels.h).
//
// ---------------------------------------------------------------------------
// Buffer contents and how they are verified
// ---------------------------------------------------------------------------
// Read keeps a deterministic pattern - a stream of little-endian 64-bit words
// w[j] = j - and its kernel folds every vector it loads, so its digest covers
// the whole buffer and verify_memory_kernels() can re-derive the pattern
// byte-for-byte.
//
// Write and copy stream all-zero 64-byte blocks, which is what AIDA64's kernels
// do: MTMBW zeroes its vector registers with vpxord and stores them, and its
// copy source holds zeros as well (the 60.8 GB/s copy "Mem" number this host
// reports is only reachable on the all-zero store path).  The data pattern is
// not cosmetic on Zen5: measured on this host with the same 96 MiB sweep,
// 64-byte non-temporal stores run at ~86 GB/s when the data is all-zero but
// drop to ~30 GB/s for any non-zero constant or ramp, independent of the buffer
// size (96 MiB and 512 MiB alike), i.e. it is a store-path property and not
// cache residency.  Storing a "w[j] = j" ramp would therefore report roughly a
// third of the AIDA64 number this test exists to compare against.
//
// All-zero data carries nothing to fold, so those two kernels end every pass
// with a sampled read-back: one contiguous 4 KiB page every 1 MiB (0.4% of the
// buffer) is read back after the store fence and folded into the digest.  That
// keeps the digest a function of the bytes that actually reached memory at a
// bounded cost, and verify_memory_kernels() additionally reads the whole buffer
// back once and checks it is all zeros.
//
// ---------------------------------------------------------------------------
// Repetition granularity: what one run_once() returns
// ---------------------------------------------------------------------------
// `cst::Benchmark::run_once(worker, thread_count, min_seconds)` executes exactly ONE
// repetition; the harness (src/main.cpp) calls it in a loop until the requested
// duration elapses, so the kernels are stateless with respect to time and never
// sleep.  One repetition is:
//
//   memread   one full sequential pass over the worker's 96 MiB buffer
//             -> returns BYTES READ            (100663296)
//   memwrite  one full sequential pass over the worker's 96 MiB buffer
//             -> returns BYTES WRITTEN         (100663296)
//   memcopy   one full sequential pass over the worker's 48 MiB source
//             -> returns BYTES COPIED          (50331648)
//   memlat    one full traversal of the pointer chain (single-threaded)
//             -> returns DEPENDENT LOADS       (4194304)
//
// Unit conversion (BenchInfo::unit_scale, used by the harness as
// `value = sum(work) * unit_scale / elapsed`):
//
//   memread / memwrite / memcopy : 1e-6   bytes -> MB   => MB/s
//   memlat                       : 1.0    raw loads
//
// The harness's generic throughput formula yields LOADS PER SECOND for memlat
// (it cannot express an inverse), so ns/access must be derived as
//
//     ns_per_access = elapsed_seconds * 1e9 / sum(work)
//
// `higher_is_better == false` marks the inversion.  verify_memory_kernels()
// prints exactly this number.
//
// memlat is documented by AIDA64 as a single-core test: only worker 0 does
// meaningful work, every other worker returns 0 immediately.  The DLL's
// NGMemLat times one pass of 2^22 inner repetitions x 64 dependent loads
// (2^28 loads, ~20 s on this host) over a 512 MiB page-random chain built from
// 128-byte-strided nodes; the port walks the same chain in whole-traversal
// passes and lets the harness average over as many passes as the duration
// allows - the per-load average is the same either way (see docs section 14).
//
// ---------------------------------------------------------------------------
// Memory footprint / laziness
// ---------------------------------------------------------------------------
// prepare(threads) only sizes the per-worker slot table; the actual 64 KiB-
// aligned blocks are allocated (and filled) on first use by the owning worker
// thread, and each worker only ever touches its own slots.  Worst case on the
// 8-core/16-thread host: 16 x 96 MiB = 1.5 GiB for the read test, 1.5 GiB for
// the write test, 16 x (48+48) MiB = 1.5 GiB for the copy test, and a single
// 512 MiB structure for the latency test.  A 1-thread run touches only one
// worker's blocks; the memory is released in the destructor.
//
// The working sets follow the sizes AIDA64 reports for the MTMBW "Mem" level
// (SizeInKiBytes = 98304 read/write, 49152 copy) and are deliberately larger
// than this host's ~98 MB L3, so the sweeps measure the memory system rather
// than a cache (see the all-zero note above for the one store-path caveat); the
// non-temporal hints additionally keep the streams from evicting the cache
// contents.

#include "bench_memory.h"

#include "common.h"
#include "mem_kernels.h"

#include <immintrin.h>
#include <malloc.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cst {
namespace {

// ---------------------------------------------------------------------------
// Sizes, alignment, unroll factors
// ---------------------------------------------------------------------------
constexpr size_t kMiB = 1024u * 1024u;

constexpr size_t kAlign      = 64u * 1024u;   // 64 KiB: every buffer start
constexpr size_t kVec        = 32u;           // bytes per AVX2 register
constexpr size_t kBlockBytes = 256u;          // bytes per fully unrolled iteration (8 x 32 B)
constexpr size_t kSwingBytes = 512u;          // bytes per bandwidth-kernel iteration (8 x 64 B)

// Working sets of the AIDA64 MTMBW "Mem" level (SizeInKiBytes 98304/98304/49152).
constexpr size_t kReadBytes  = 96u * kMiB;    // read buffer per thread
constexpr size_t kWriteBytes = 96u * kMiB;    // write buffer per thread
constexpr size_t kCopyBytes  = 48u * kMiB;    // one of the two copy buffers per thread

constexpr size_t kLatBytes   = 512u * kMiB;   // AIDA64 NGMemLat band 0: UsedSizeInBytes
constexpr size_t kLatStride  = 128u;          // AIDA64 UsedStrideInBytes (band 0)
constexpr uint64_t kLatSeed  = 0xA1DA64F00Dull;
constexpr size_t kPageBytes  = 4096u;         // TLB window unit: 4 KiB pages
constexpr size_t kLatWindowBytes = 8u * kMiB; // AIDA64 UsedBlockSizeInBytes: page shuffle window

// Pin one worker per logical processor on first use.  Remove/disable if the
// harness prefers the OS scheduler to place the threads.
constexpr bool kPinWorkers = true;

static_assert(kReadBytes  % kSwingBytes == 0, "read pass must be a whole number of swings");
static_assert(kWriteBytes % kSwingBytes == 0, "write pass must be a whole number of swings");
static_assert(kCopyBytes  % kSwingBytes == 0, "copy pass must be a whole number of swings");
static_assert(kReadBytes  % kBlockBytes == 0, "read pass must be a whole number of blocks");
static_assert(kWriteBytes % kBlockBytes == 0, "write pass must be a whole number of blocks");
static_assert(kCopyBytes  % kBlockBytes == 0, "copy pass must be a whole number of blocks");
static_assert(kSamplePageBytes % kVec == 0, "a sample must be a whole number of vectors");
static_assert(kSampleStride % kSamplePageBytes == 0, "samples must be page-contiguous");
static_assert(kWriteBytes % kSampleStride == 0, "write buffer must hold whole samples");
static_assert(kCopyBytes  % kSampleStride == 0, "copy buffer must hold whole samples");
static_assert(kAlign % kVec == 0, "64 KiB alignment implies 32 B alignment");
static_assert(kLatBytes % kLatStride == 0, "structure must be a whole number of nodes");
static_assert(kLatBytes % kPageBytes == 0, "structure must be a whole number of 4 KiB pages");
static_assert(kLatBytes % kLatWindowBytes == 0, "structure must be a whole number of shuffle windows");
static_assert(kLatWindowBytes % kPageBytes == 0, "a shuffle window must be a whole number of pages");
static_assert(kLatStride >= 128, "AIDA64: stride must be at least 128 bytes");
static_assert(kLatStride % 64u == 0, "nodes must be cache-line aligned");
static_assert(kLatBytes >= 16u * kMiB, "AIDA64: structure must be at least 16 MiB");
static_assert(kLatStride % 8u == 0, "AIDA64: stride must be a multiple of 8 bytes");

constexpr size_t kLatNodes = kLatBytes / kLatStride;

// ---------------------------------------------------------------------------
// Aligned, owned memory block
// ---------------------------------------------------------------------------
struct AlignedBlock {
    void* raw = nullptr;
    unsigned char* data = nullptr;   // >= 64 KiB aligned, `bytes` usable
    size_t bytes = 0;

    AlignedBlock() = default;
    AlignedBlock(const AlignedBlock&) = delete;
    AlignedBlock& operator=(const AlignedBlock&) = delete;
    AlignedBlock(AlignedBlock&& o) noexcept : raw(o.raw), data(o.data), bytes(o.bytes) {
        o.raw = nullptr;
        o.data = nullptr;
        o.bytes = 0;
    }
    AlignedBlock& operator=(AlignedBlock&& o) noexcept {
        if (this != &o) {
            release();
            raw = o.raw;
            data = o.data;
            bytes = o.bytes;
            o.raw = nullptr;
            o.data = nullptr;
            o.bytes = 0;
        }
        return *this;
    }
    ~AlignedBlock() { release(); }

    void release() noexcept {
        if (raw) {
            _aligned_free(raw);
            raw = nullptr;
        }
        data = nullptr;
        bytes = 0;
    }

    // Over-allocate by one alignment unit and align up, so the returned pointer
    // is a multiple of 64 KiB regardless of what _aligned_malloc returned.
    bool alloc(size_t n) noexcept {
        release();
        void* p = _aligned_malloc(n + kAlign, kAlign);
        if (!p) return false;
        raw = p;
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        data = reinterpret_cast<unsigned char*>((a + (kAlign - 1)) &
                                               ~static_cast<uintptr_t>(kAlign - 1));
        bytes = n;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Deterministic pattern: little-endian 64-bit words w[j] = j.
// ---------------------------------------------------------------------------
void fill_ramp(unsigned char* p, size_t bytes) noexcept {
    const __m256i inc = _mm256_set1_epi64x(4);
    __m256i v = _mm256_set_epi64x(3, 2, 1, 0);   // lanes 0..3 -> words 0..3
    size_t i = 0;
    for (; i + kVec <= bytes; i += kVec) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p + i), v);
        v = _mm256_add_epi64(v, inc);
    }
    for (; i + 8 <= bytes; i += 8) {
        const uint64_t w = static_cast<uint64_t>(i / 8);
        std::memcpy(p + i, &w, sizeof(w));
    }
}

// Byte-exact re-derivation of fill_ramp(); returns false on the first mismatch.
bool check_ramp(const unsigned char* p, size_t bytes) noexcept {
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, sizeof(w));
        if (w != static_cast<uint64_t>(i / 8)) return false;
    }
    return i == bytes;
}

// True when every 8-byte word of the buffer is zero - the content the write and
// copy kernels stream.
bool check_zeros(const unsigned char* p, size_t bytes) noexcept {
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, sizeof(w));
        if (w != 0) return false;
    }
    return i == bytes;
}

// ---------------------------------------------------------------------------
// Digest helpers.  A kernel and the matching independent fold in this file fold
// the *same* bytes into the same two accumulators, so the per-pass digest and
// the recomputed digest agree bit-for-bit: read folds every vector it loads,
// write/copy fold the sampled read-back (fold_samples_avx2 below).  Two
// accumulators are used because an XOR-only fold cancels itself on structured
// data: the word ramp w[j] = j XORs down to the same constant for *any* number
// of vectors, which would make the "digest unchanged between repetitions" check
// meaningless.  The vector sum is order-insensitive but not cancel-prone, and
// both cost one ALU op per vector - far below the DRAM bottleneck.
// mix_accums() combines them; mem_kernels_avx512.cpp carries a byte-for-byte
// copy of it for its 512-bit accumulators.
// ---------------------------------------------------------------------------
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

uint64_t fold_bytes_avx2(const unsigned char* p, size_t bytes) noexcept {
    __m256i x0 = _mm256_setzero_si256();
    __m256i x1 = _mm256_setzero_si256();
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + kBlockBytes <= bytes; i += kBlockBytes) {
        const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 0));
        const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 32));
        const __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 64));
        const __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 96));
        const __m256i v4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 128));
        const __m256i v5 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 160));
        const __m256i v6 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 192));
        const __m256i v7 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 224));
        x0 = _mm256_xor_si256(x0, _mm256_xor_si256(v0, v1));
        x1 = _mm256_xor_si256(x1, _mm256_xor_si256(v2, v3));
        s0 = _mm256_add_epi64(_mm256_add_epi64(s0, v0), v1);
        s1 = _mm256_add_epi64(_mm256_add_epi64(s1, v2), v3);
        x0 = _mm256_xor_si256(x0, _mm256_xor_si256(v4, v5));
        x1 = _mm256_xor_si256(x1, _mm256_xor_si256(v6, v7));
        s0 = _mm256_add_epi64(_mm256_add_epi64(s0, v4), v5);
        s1 = _mm256_add_epi64(_mm256_add_epi64(s1, v6), v7);
    }
    for (; i + kVec <= bytes; i += kVec) {
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i));
        x0 = _mm256_xor_si256(x0, v);
        s0 = _mm256_add_epi64(s0, v);
    }
    return mix_accums(_mm256_xor_si256(x0, x1), _mm256_add_epi64(s0, s1));
}

// Independent fold of the sampled read-back regions of `p`, using the same
// geometry the kernels use: one contiguous kSamplePageBytes page every
// kSampleStride bytes.  This is what the write/copy digests must equal.
uint64_t fold_samples_avx2(const unsigned char* p, size_t bytes) noexcept {
    __m256i x0 = _mm256_setzero_si256();
    __m256i s0 = _mm256_setzero_si256();
    for (size_t off = 0; off + kSamplePageBytes <= bytes; off += kSampleStride) {
        for (size_t i = 0; i < kSamplePageBytes; i += kVec) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(p + off + i));
            x0 = _mm256_xor_si256(x0, v);
            s0 = _mm256_add_epi64(s0, v);
        }
    }
    return mix_accums(x0, s0);
}

// Two independent 64-bit folds -> one digest; a kernel that folds both a source
// stream and a sampled read-back uses this (mirrors mem_kernels_avx512.cpp).
inline uint64_t combine2(uint64_t a, uint64_t b) noexcept {
    return a ^ (b * 0x9E3779B97F4A7C15ull);
}

// ---------------------------------------------------------------------------
// Bandwidth sweeps.  The AVX-512 variants (8 x 64 B per 512 B swing) live in
// mem_kernels_avx512.cpp; the fallbacks below move the same bytes with the same
// non-temporal hints using 256-bit registers (16 x 32 B per swing) and return
// the same fold: both XOR/sum every vector of a sweep exactly once, and the
// 512-bit accumulator pair of the AVX-512 kernels reduces to the 256-bit pair
// built here.  Read folds every loaded vector; write and copy fold their
// sampled read-back (see the header comment).
// ---------------------------------------------------------------------------
inline bool have_avx512_sweeps() noexcept { return cpu_info().has_avx512f; }

inline const char* sweep_isa_name() noexcept {
    return have_avx512_sweeps() ? "AVX-512 (8x64B swing)" : "AVX2 (16x32B swing)";
}

// 16 x 32 B non-temporal loads per 512 B iteration, linear sweep.
uint64_t read_sweep_avx2(const unsigned char* p, size_t bytes) noexcept {
    const __m256i* q = reinterpret_cast<const __m256i*>(p);
    __m256i x0 = _mm256_setzero_si256();
    __m256i s0 = _mm256_setzero_si256();
    const size_t n = bytes / kVec;
    for (size_t k = 0; k < n; k += 16) {
        for (size_t j = 0; j < 16; ++j) {
            const __m256i v = _mm256_stream_load_si256(q + k + j);
            x0 = _mm256_xor_si256(x0, v);
            s0 = _mm256_add_epi64(s0, v);
        }
    }
    return mix_accums(x0, s0);
}

// 16 x 32 B non-temporal stores of an all-zero register per 512 B iteration,
// then the sampled read-back fold (see the header comment).
uint64_t write_sweep_avx2(unsigned char* p, size_t bytes) noexcept {
    const __m256i z = _mm256_setzero_si256();
    for (size_t off = 0; off < bytes; off += kSwingBytes) {
        for (size_t j = 0; j < 16; ++j) {
            _mm256_stream_si256(reinterpret_cast<__m256i*>(p + off + j * kVec), z);
        }
    }
    _mm_sfence();
    return fold_samples_avx2(p, bytes);
}

// 16 x (32 B non-temporal load + 32 B non-temporal store) per 512 B iteration,
// folding the loaded vectors, then the sampled read-back fold of the
// destination.
uint64_t copy_sweep_avx2(unsigned char* dst, const unsigned char* src, size_t bytes) noexcept {
    __m256i x0 = _mm256_setzero_si256();
    __m256i s0 = _mm256_setzero_si256();
    for (size_t off = 0; off < bytes; off += kSwingBytes) {
        for (size_t j = 0; j < 16; ++j) {
            const __m256i v =
                _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(src + off + j * kVec));
            _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + off + j * kVec), v);
            x0 = _mm256_xor_si256(x0, v);
            s0 = _mm256_add_epi64(s0, v);
        }
    }
    _mm_sfence();
    return combine2(mix_accums(x0, s0), fold_samples_avx2(dst, bytes));
}

// 32-bit node accessors.  memcpy keeps the raw-byte buffer free of strict-
// aliasing violations and still compiles to one plain load/store.
inline uint32_t load_u32(const unsigned char* p) noexcept {
    uint32_t w = 0;
    std::memcpy(&w, p, sizeof(w));
    return w;
}
inline void store_u32(unsigned char* p, uint32_t w) noexcept {
    std::memcpy(p, &w, sizeof(w));
}

// ---------------------------------------------------------------------------
// Per-worker block table.  Slots are default-constructed here (no memory
// touched); the owning worker materialises its own blocks on first run_once().
// ---------------------------------------------------------------------------
class BlockSet {
public:
    void reset(int threads, int nblocks, size_t bytes_per_block) {
        threads_ = threads < 1 ? 1 : threads;
        nblocks_ = nblocks < 1 ? 1 : nblocks;
        bytes_ = bytes_per_block;
        blocks_.clear();
        blocks_.resize(static_cast<size_t>(threads_) * static_cast<size_t>(nblocks_));
    }

    AlignedBlock* block(int worker_index, int slot) noexcept {
        if (worker_index < 0 || worker_index >= threads_) return nullptr;
        if (slot < 0 || slot >= nblocks_) return nullptr;
        return &blocks_[static_cast<size_t>(worker_index) * static_cast<size_t>(nblocks_) +
                        static_cast<size_t>(slot)];
    }

    int threads() const noexcept { return threads_; }

private:
    std::vector<AlignedBlock> blocks_;
    int threads_ = 0;
    int nblocks_ = 1;
    size_t bytes_ = 0;
};

// Per-worker flags/values.  Every slot has exactly one writer (the worker whose
// index it is), so no synchronisation is needed; digest() is read by the harness
// after the workers have been joined.
using WorkerFlags = std::vector<unsigned char>;
using WorkerValues = std::vector<uint64_t>;

inline void pin_worker_once(int worker, WorkerFlags& pinned) {
    if (!kPinWorkers) return;
    if (worker < 0 || worker >= static_cast<int>(pinned.size())) return;
    if (pinned[static_cast<size_t>(worker)]) return;
    pinned[static_cast<size_t>(worker)] = 1;
    pin_thread_to_processor(worker);
}

// cst::Benchmark::digest() takes no worker index, so the per-worker digests are
// folded into one value (FNV-1a over the slots).  `fold_one` is consequently the
// digest of a single-worker prepare, which is what the selftest compares against
// an independent content fold.
inline uint64_t fold_one(uint64_t v) noexcept {
    return (0xcbf29ce484222325ull ^ v) * 0x100000001b3ull;
}
inline uint64_t fold_slots(const WorkerValues& v) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t x : v) h = (h ^ x) * 0x100000001b3ull;
    return h;
}

// ---------------------------------------------------------------------------
// Memory Read
// ---------------------------------------------------------------------------
class MemReadBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "memread";
        i.label = "Memory Read";
        i.unit = "MB/s";
        i.group = "Memory";
        i.higher_is_better = true;
        i.unit_scale = 1e-6;   // raw units are bytes -> MB
        i.description =
            "One non-temporal (vmovntdqa) sequential pass over a 96 MiB, 64 KiB-aligned "
            "buffer per thread";
        return i;
    }

    void prepare(int threads) override {
        set_.reset(threads, 1, kReadBytes);
        digests_.assign(static_cast<size_t>(set_.threads()), 0);
        pinned_.assign(static_cast<size_t>(set_.threads()), 0);
    }

    // One repetition: a full sequential pass over this worker's buffer.
    // Returns BYTES READ (info().unit_scale = 1e-6 -> MB/s).
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        (void)min_seconds;   // one bounded pass is far shorter than the budget
        AlignedBlock* b = ensure(worker);
        if (!b) return 0.0;
        pin_worker_once(worker, pinned_);

        // The fold escapes into digests_, so no load can be optimised away.
        const uint64_t fold = have_avx512_sweeps()
            ? mem_read_sweep_avx512(b->data, kReadBytes)
            : read_sweep_avx2(b->data, kReadBytes);
        if (worker >= 0 && worker < static_cast<int>(digests_.size())) {
            digests_[static_cast<size_t>(worker)] = fold;
        }
        return static_cast<double>(kReadBytes);
    }

    // Worker-independent: the harness verifies one digest per benchmark, so fold
    // every worker slot together.  Each slot has a single writer, so this is safe
    // once the workers have been joined.
    uint64_t digest() const override { return fold_slots(digests_); }
    bool has_digest() const override { return true; }

    std::string params() const override {
        return "buffer=" + std::to_string(kReadBytes / kMiB) + " MiB/thread, align=" +
               std::to_string(kAlign / 1024) + " KiB, access=non-temporal (vmovntdqa), "
               "data=w[j]=j ramp, fold=xor64+sum64, isa=" +
               sweep_isa_name() + (kPinWorkers ? ", pinned=1 thread/logical CPU" : "");
    }

    // --- harness/selftest hook -------------------------------------------------
    // Materialises worker `wi`'s buffer on first use (lazy allocation).
    AlignedBlock* ensure(int wi) {
        AlignedBlock* b = set_.block(wi, 0);
        if (!b) return nullptr;
        if (!b->data) {
            if (!b->alloc(kReadBytes)) return nullptr;
            fill_ramp(b->data, b->bytes);
        }
        return b;
    }

private:
    BlockSet set_;
    WorkerValues digests_;
    WorkerFlags pinned_;
};

// ---------------------------------------------------------------------------
// Memory Write
// ---------------------------------------------------------------------------
class MemWriteBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "memwrite";
        i.label = "Memory Write";
        i.unit = "MB/s";
        i.group = "Memory";
        i.higher_is_better = true;
        i.unit_scale = 1e-6;   // raw units are bytes -> MB
        i.description =
            "One pass of non-temporal 64-byte stores (vmovntdq) of an all-zero register over a "
            "96 MiB, 64 KiB-aligned buffer per thread (the data AIDA64's write kernel streams)";
        return i;
    }

    void prepare(int threads) override {
        set_.reset(threads, 1, kWriteBytes);
        digests_.assign(static_cast<size_t>(set_.threads()), 0);
        pinned_.assign(static_cast<size_t>(set_.threads()), 0);
    }

    // One repetition: a full sequential pass over this worker's buffer.
    // Returns BYTES WRITTEN (info().unit_scale = 1e-6 -> MB/s).
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        (void)min_seconds;   // one bounded pass is far shorter than the budget
        AlignedBlock* b = ensure(worker);
        if (!b) return 0.0;
        pin_worker_once(worker, pinned_);

        // The sweep fences the non-temporal stores before returning, then folds
        // the sampled read-back of the pages it just wrote (see the header
        // comment): all-zero store data cannot be folded as it is produced.
        const uint64_t fold = have_avx512_sweeps()
            ? mem_write_sweep_avx512(b->data, kWriteBytes)
            : write_sweep_avx2(b->data, kWriteBytes);
        if (worker >= 0 && worker < static_cast<int>(digests_.size())) {
            digests_[static_cast<size_t>(worker)] = fold;
        }
        return static_cast<double>(kWriteBytes);
    }

    // Worker-independent: the harness verifies one digest per benchmark, so fold
    // every worker slot together.  Each slot has a single writer, so this is safe
    // once the workers have been joined.
    uint64_t digest() const override { return fold_slots(digests_); }
    bool has_digest() const override { return true; }

    std::string params() const override {
        return "buffer=" + std::to_string(kWriteBytes / kMiB) + " MiB/thread, align=" +
               std::to_string(kAlign / 1024) +
               " KiB, stores=non-temporal (vmovntdq), data=zero register (AIDA64 vpxord shape), "
               "fold=sampled 4 KiB/MiB read-back (xor64+sum64), isa=" + sweep_isa_name() +
               (kPinWorkers ? ", pinned=1 thread/logical CPU" : "");
    }

    // Materialises worker `wi`'s buffer on first use (lazy allocation).  It
    // starts out all zero, matching the data the kernel streams.
    AlignedBlock* ensure(int wi) {
        AlignedBlock* b = set_.block(wi, 0);
        if (!b) return nullptr;
        if (!b->data) {
            if (!b->alloc(kWriteBytes)) return nullptr;
            std::memset(b->data, 0, b->bytes);
        }
        return b;
    }

private:
    BlockSet set_;
    WorkerValues digests_;
    WorkerFlags pinned_;
};

// ---------------------------------------------------------------------------
// Memory Copy
// ---------------------------------------------------------------------------
class MemCopyBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "memcopy";
        i.label = "Memory Copy";
        i.unit = "MB/s";
        i.group = "Memory";
        i.higher_is_better = true;
        i.unit_scale = 1e-6;   // raw units are bytes -> MB
        i.description =
            "One non-temporal (vmovntdqa/vmovntdq) copy pass of 48 MiB of zero blocks into "
            "another 48 MiB, 64 KiB-aligned buffer per thread";
        return i;
    }

    void prepare(int threads) override {
        set_.reset(threads, 2, kCopyBytes);
        digests_.assign(static_cast<size_t>(set_.threads()), 0);
        pinned_.assign(static_cast<size_t>(set_.threads()), 0);
    }

    // One repetition: a full sequential pass over this worker's source buffer.
    // Returns BYTES COPIED (source bytes moved, not source+destination).
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        (void)min_seconds;   // one bounded pass is far shorter than the budget
        AlignedBlock* s = ensure(worker);
        if (!s) return 0.0;
        pin_worker_once(worker, pinned_);

        const unsigned char* src = s->data;
        unsigned char* dst = (s + 1)->data;

        // The sweep fences the non-temporal stores before returning; the fold
        // covers the loaded vectors plus the sampled read-back of the
        // destination (see the header comment).
        const uint64_t fold = have_avx512_sweeps()
            ? mem_copy_sweep_avx512(dst, src, kCopyBytes)
            : copy_sweep_avx2(dst, src, kCopyBytes);
        if (worker >= 0 && worker < static_cast<int>(digests_.size())) {
            digests_[static_cast<size_t>(worker)] = fold;
        }
        return static_cast<double>(kCopyBytes);
    }

    // Worker-independent: the harness verifies one digest per benchmark, so fold
    // every worker slot together.  Each slot has a single writer, so this is safe
    // once the workers have been joined.
    uint64_t digest() const override { return fold_slots(digests_); }
    bool has_digest() const override { return true; }

    std::string params() const override {
        return "src=" + std::to_string(kCopyBytes / kMiB) + " MiB (zeros, as AIDA64's), dst=" +
               std::to_string(kCopyBytes / kMiB) + " MiB, align=" + std::to_string(kAlign / 1024) +
               " KiB, load=non-temporal (vmovntdqa), store=non-temporal (vmovntdq), isa=" +
               sweep_isa_name() + ", fold=loaded+sampled 4 KiB/MiB (xor64+sum64)" +
               (kPinWorkers ? ", pinned=1 thread/logical CPU" : "");
    }

    // Materialises worker `wi`'s source+destination pair on first use.  The
    // source is all zero (AIDA64's copy source holds zeros; on Zen5 that is
    // also what makes the destination stores take the all-zero fast path).
    AlignedBlock* ensure(int wi) {
        AlignedBlock* a = set_.block(wi, 0);
        AlignedBlock* b = set_.block(wi, 1);
        if (!a || !b) return nullptr;
        if (!a->data) {
            if (!a->alloc(kCopyBytes) || !b->alloc(kCopyBytes)) {
                a->release();
                b->release();
                return nullptr;
            }
            std::memset(a->data, 0, a->bytes);
            std::memset(b->data, 0x5A, b->bytes);   // poison: any byte not copied shows up
        }
        return b->data ? a : nullptr;
    }

private:
    BlockSet set_;
    WorkerValues digests_;
    WorkerFlags pinned_;
};

// ---------------------------------------------------------------------------
// Memory Latency
// ---------------------------------------------------------------------------
class MemLatencyBench final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "memlat";
        i.label = "Memory Latency";
        i.unit = "ns";
        i.group = "Memory";
        i.higher_is_better = false;   // lower ns/access is better
        i.unit_scale = 1e9;           // seconds -> nanoseconds
        i.per_unit = true;            // report ns per dependent load
        i.default_threads = 1;        // AIDA64 documents this as a single-core test
        i.description =
            "Single-threaded dependent-load pointer chase over a 512 MiB page-random chain "
            "(4 KiB pages, 128-byte node stride, 8 MiB shuffle windows, one line per page per pass; "
            "AIDA64 NGMemLat band 0)";
        return i;
    }

    // Documented as a single-core test: only one chain exists.
    void prepare(int threads) override {
        (void)threads;
        set_.reset(1, 1, kLatBytes);
        digests_.assign(1, 0);
        pinned_.assign(1, 0);
        sink_.assign(1, 0);
        // Build the structure here, before the harness starts its clock: the
        // build, the digest scan and the warm-up traversal must not land in
        // the first timed repetition.
        (void)ensure();
    }

    // One repetition: one full traversal of the chain, which is a single cycle
    // over every node, so it performs exactly kLatNodes dependent loads and ends
    // back on the entry node.
    //
    // Returns the NUMBER OF DEPENDENT LOADS.  The reported unit is ns, which the
    // harness's forward throughput formula cannot produce, so per-access latency
    // is derived as:  ns_per_access = elapsed_seconds * 1e9 / sum(work).
    //
    // Worker 0 does the work regardless of thread_count (thread_count > 1 means
    // the harness oversubscribed a single-core test); every other worker returns
    // 0 immediately without touching memory.
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        (void)min_seconds;   // one bounded traversal (~0.3 s) is below the budget
        if (worker != 0) return 0.0;
        AlignedBlock* b = ensure();
        if (!b) return 0.0;
        pin_worker_once(0, pinned_);

        // A full cycle returns to the entry node: observable proof the chase
        // ran.  The structure digest is computed once at build time, not here:
        // it would scan the whole 512 MiB and pollute the timed repetition.
        sink_[0] = chase(b->data);
        return static_cast<double>(kLatNodes);
    }

    // Digest of the *structure* (FNV-1a over every node's successor), not of the
    // traversal: the chain itself is deterministic, and repeating the traversal
    // reproduces exactly this value.
    // Worker-independent: the harness verifies one digest per benchmark, so fold
    // every worker slot together.  Each slot has a single writer, so this is safe
    // once the workers have been joined.
    uint64_t digest() const override { return fold_slots(digests_); }
    bool has_digest() const override { return true; }
    // This is the only memory test that gets a golden reference.  The latency
    // test is single-threaded and walks a fixed pointer chain, so its
    // fingerprint is a genuine function of the workload.  The read/write/copy
    // bandwidth tests deliberately have none: throughput is a timing
    // measurement with no canonical "correct" value, and hashing N per-worker
    // buffers would make the fingerprint depend on the thread count.  Their
    // correctness is covered by selftest_memory.exe instead.
    uint64_t full_digest() const override { return  corrupt_if_testing(fold_slots(digests_)); }
    bool has_full_digest() const override { return  true; }

    std::string params() const override {
        return  "chain=" + std::to_string(kLatBytes / kMiB) + " MiB, nodes=" +
               std::to_string(kLatNodes) + ", stride=" + std::to_string(kLatStride) +
               " B, page=" + std::to_string(kPageBytes / 1024) + " KiB, window=" +
               std::to_string(kLatWindowBytes / kMiB) + " MiB (page-random order), align=" +
               std::to_string(kAlign / 1024) +
               " KiB, threads=1 (single-core test), ns/access = seconds*1e9/loads, "
               "AIDA64: 2^22 inner reps * 64 dependent loads per timed pass" +
               (kPinWorkers ? ", pinned=1 thread/logical CPU" : "");
    }

    // Builds (once, lazily) the chained structure and returns it.
    AlignedBlock* ensure() {
        AlignedBlock* b = set_.block(0, 0);
        if (!b) return nullptr;
        if (!b->data) {
            if (!b->alloc(kLatBytes)) return nullptr;
            build_chain(b->data);
            if (!digests_.empty()) digests_[0] = chain_digest(b->data);
            // One untimed traversal so the first measured repetition already
            // runs with the TLB and page-walk caches warm, the way AIDA64's
            // single 2^28-load pass spends nearly all of its time.
            (void)chase(b->data);
        }
        return b;
    }

    size_t nodes() const noexcept { return kLatNodes; }
    size_t stride() const noexcept { return kLatStride; }
    // Final index of the last traversal: 0 means the cycle closed on its entry
    // node, i.e. the chase really covered the whole structure.
    uint32_t last_node() const noexcept {
        return sink_.empty() ? 0xFFFFFFFFu : static_cast<uint32_t>(sink_[0]);
    }

private:
    // A single Hamiltonian cycle over every node, laid out to reproduce the
    // DLL's band-0 "Page_Random" method as measured on this host: the walk
    // moves through 8 MiB windows of shuffled pages and visits exactly one
    // node (one cache line) per page per pass.  Taking a single line per page
    // keeps the L2 region prefetcher from covering the rest of the page,
    // while the window keeps the TLB warm; consecutive loads therefore jump
    // to a random page of the current window at DRAM latency, which is the
    // ~79 ns the DLL reports for Page_Random (see docs section 14).  Each page's 32
    // nodes are spread over 32 passes, and the last node closes the cycle.
    static void build_chain(unsigned char* base) {
        constexpr size_t kNodesPerPage = kPageBytes / kLatStride;       // 32
        constexpr size_t kPages       = kLatBytes / kPageBytes;         // 131072
        constexpr size_t kWindowPages = kLatWindowBytes / kPageBytes;   // 2048
        constexpr size_t kWindows     = kPages / kWindowPages;          // 64

        std::vector<uint32_t> pages(kWindowPages);
        uint64_t st = kLatSeed;
        uint32_t prev = 0xFFFFFFFFu;
        uint32_t first = 0;
        for (size_t w = 0; w < kWindows; ++w) {
            const size_t page_base = w * kWindowPages;
            for (size_t pass = 0; pass < kNodesPerPage; ++pass) {
                for (size_t i = 0; i < kWindowPages; ++i) pages[i] = static_cast<uint32_t>(i);
                for (size_t i = kWindowPages - 1; i > 0; --i) {
                    const uint64_t r = splitmix64(st);
                    const size_t j = static_cast<size_t>(r % static_cast<uint64_t>(i + 1));
                    const uint32_t t = pages[i];
                    pages[i] = pages[j];
                    pages[j] = t;
                }
                for (size_t i = 0; i < kWindowPages; ++i) {
                    const uint32_t node =
                        static_cast<uint32_t>((page_base + pages[i]) * kNodesPerPage) +
                        static_cast<uint32_t>(pass);
                    if (prev == 0xFFFFFFFFu) {
                        first = node;
                    } else {
                        store_u32(base + static_cast<size_t>(prev) * kLatStride, node);
                    }
                    prev = node;
                }
            }
        }
        store_u32(base + static_cast<size_t>(prev) * kLatStride, first);
    }

    // One traversal of the whole cycle; always ends on the entry node 0.  The
    // next address is the value just loaded - a genuine dependent load chain
    // the out-of-order engine cannot overlap or speculate past - and the fence
    // stops the *compiler* from reordering, merging or hoisting the loads.
    static uint32_t chase(const unsigned char* base) noexcept {
        uint32_t cur = 0;
        for (size_t k = 0; k < kLatNodes; ++k) {
            const uint32_t next = load_u32(base + static_cast<size_t>(cur) * kLatStride);
            std::atomic_signal_fence(std::memory_order_seq_cst);
            cur = next;
        }
        return cur;
    }

    static uint64_t chain_digest(const unsigned char* base) noexcept {
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < kLatNodes; ++i) {
            h ^= static_cast<uint64_t>(load_u32(base + i * kLatStride));
            h *= 0x100000001B3ull;
        }
        return h;
    }

    BlockSet set_;
    WorkerValues digests_;
    WorkerFlags pinned_;
    WorkerValues sink_;   // final chain index of the last traversal (0 == closed)
};

// ---------------------------------------------------------------------------
// Self-test helpers
// ---------------------------------------------------------------------------
constexpr double kShortSeconds = 0.05;   // ~50 ms per kernel

// Failure codes returned by verify_memory_kernels().
enum : int {
    kFailAlignment = 1 << 0,
    kFailRead      = 1 << 1,
    kFailWrite     = 1 << 2,
    kFailCopy      = 1 << 3,
    kFailLatency   = 1 << 4,
    kFailTiming    = 1 << 5,
    kFailThreads   = 1 << 6,
};

bool check_align(const char* name, int worker, const AlignedBlock* b, size_t expect) {
    if (!b || !b->data) {
        std::printf("  [FAIL] %-8s worker %d: allocation failed\n", name, worker);
        return false;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t off = addr % static_cast<uintptr_t>(kAlign);
    const bool ok = (off == 0) && (b->bytes >= expect) && (b->bytes % kAlign == 0);
    std::printf("  [%s] %-8s worker %d: addr%%64KiB=%llu  size=%llu B (need >= %llu)\n",
                ok ? "PASS" : "FAIL", name, worker,
                static_cast<unsigned long long>(off),
                static_cast<unsigned long long>(b->bytes),
                static_cast<unsigned long long>(expect));
    return ok;
}

// Walk the chain and prove it is one single cycle covering every node, with a
// node stride of at least 128 bytes and no node repeated before the cycle closes.
bool verify_chain(const AlignedBlock& b, size_t nodes, size_t stride, std::string& why) {
    if (stride < 128) { why = "stride < 128 bytes"; return false; }
    if (b.bytes < 16u * kMiB) { why = "structure smaller than 16 MiB"; return false; }
    if (nodes != b.bytes / stride) { why = "node count != bytes / stride"; return false; }
    if (nodes > 0xFFFFFFFFull) { why = "node count exceeds the 32-bit index space"; return false; }

    std::vector<unsigned char> seen(nodes, 0);
    uint32_t cur = 0;
    uint32_t prev = 0;
    size_t visited = 0;
    bool first = true;

    for (;;) {
        if (cur >= nodes) { why = "chain index out of range"; return false; }
        if (seen[cur]) break;                  // closed a cycle (possibly early)
        seen[cur] = 1;
        ++visited;
        if (!first) {
            const size_t delta = (cur > prev) ? static_cast<size_t>(cur - prev)
                                              : static_cast<size_t>(prev - cur);
            if (delta * stride < 128) { why = "step between nodes < 128 bytes"; return false; }
        }
        first = false;
        prev = cur;
        cur = load_u32(b.data + static_cast<size_t>(cur) * stride);
    }

    if (cur != 0) { why = "cycle does not close back on the entry node"; return false; }
    if (visited != nodes) { why = "cycle covers only part of the structure"; return false; }
    for (size_t i = 0; i < nodes; ++i) {
        if (!seen[i]) { why = "node never reached"; return false; }
    }
    return true;
}

struct TimeResult {
    double work = 0.0;
    double seconds = 0.0;
    uint64_t reps = 0;
};

// Drives a benchmark exactly like src/main.cpp::measure() does: every worker
// repeats run_once() until the wall-clock window closes, then the per-worker work
// is summed.
template <typename RepFn>
TimeResult time_kernel(RepFn rep, int threads, double seconds) {
    TimeResult tr;
    std::mutex mu;
    const double t_end = now_seconds() + seconds;
    auto body = [&](int wi) {
        double local = 0.0;
        uint64_t n = 0;
        do {
            local += rep(wi);
            ++n;
        } while (now_seconds() < t_end);
        std::lock_guard<std::mutex> lock(mu);
        tr.work += local;
        tr.reps += n;
    };
    const double t0 = now_seconds();
    run_workers(threads, body);
    tr.seconds = now_seconds() - t0;
    return tr;
}

} // namespace

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------
Benchmark* make_bench_mem_read() { return new MemReadBench(); }
Benchmark* make_bench_mem_write() { return new MemWriteBench(); }
Benchmark* make_bench_mem_copy() { return new MemCopyBench(); }
Benchmark* make_bench_mem_latency() { return new MemLatencyBench(); }

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------
int verify_memory_kernels() {
    int rc = 0;

    std::printf("== memory kernel self-test ==\n");
    std::printf("   data      : read = w[j] = j ramp; write/copy = all-zero blocks\n");
    std::printf("   digest    : read folds every load; write/copy fold a sampled read-back\n");
    std::printf("   sweeps    : %s\n", sweep_isa_name());
    std::printf("   one rep   : one full pass over the buffer (1 MiB = 2^20 B, MB = 1e6 B)\n\n");

    // -- 1. alignment of every buffer --------------------------------------
    std::printf("-- alignment (64 KiB) --\n");
    {
        MemReadBench r;
        MemWriteBench w;
        MemCopyBench c;
        MemLatencyBench l;
        r.prepare(2);
        w.prepare(2);
        c.prepare(2);
        l.prepare(1);

        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            ok &= check_align("memread", i, r.ensure(i), kReadBytes);
            ok &= check_align("memwrite", i, w.ensure(i), kWriteBytes);
            AlignedBlock* src = c.ensure(i);
            ok &= check_align("memcopy", i, src, kCopyBytes);
            ok &= check_align("memcopy.d", i, src ? (src + 1) : nullptr, kCopyBytes);
        }
        ok &= check_align("memlat", 0, l.ensure(), kLatBytes);
        if (!ok) rc |= kFailAlignment;
        std::printf("  -> %s\n\n", ok ? "alignment PASS" : "alignment FAIL");
    }

    // -- 2. read round-trip -------------------------------------------------
    std::printf("-- read pattern round-trip --\n");
    {
        MemReadBench r;
        r.prepare(1);
        AlignedBlock* b = r.ensure(0);
        const bool pre = b && check_ramp(b->data, b->bytes);
        const double work = b ? r.run_once(0, 1, 0.25) : 0.0;
        const bool post = b && check_ramp(b->data, b->bytes);
        // The kernel XOR-folds every 32-byte vector of the pass; an independent
        // content fold of the same buffer must therefore agree.  prepare(1) has a
        // single worker slot, so the folded digest of one slot is fold_one().
        const bool digest_ok =
            b && r.digest() != 0 && r.digest() == fold_one(fold_bytes_avx2(b->data, b->bytes));
        const bool ok = pre && post && digest_ok && work == static_cast<double>(kReadBytes);
        std::printf("  [%s] content before=%s after=%s digest_matches_buffer=%s bytes=%.0f\n",
                    ok ? "PASS" : "FAIL", pre ? "exact" : "corrupt", post ? "exact" : "corrupt",
                    digest_ok ? "yes" : "no", work);
        if (!ok) rc |= kFailRead;
    }

    // -- 3. write round-trip (all-zero stream) ------------------------------
    std::printf("-- write (all-zero) round-trip --\n");
    {
        MemWriteBench w;
        w.prepare(1);
        AlignedBlock* b = w.ensure(0);
        const double work = b ? w.run_once(0, 1, 0.25) : 0.0;
        const bool exact = b && check_zeros(b->data, b->bytes);
        // The kernel folds the sampled read-back of the pages it just wrote, so
        // its digest must equal an independent fold of those same pages.
        const bool digest_ok =
            exact && w.digest() != 0 &&
            w.digest() == fold_one(fold_samples_avx2(b->data, b->bytes));
        const bool ok = exact && digest_ok && work == static_cast<double>(kWriteBytes);
        std::printf("  [%s] buffer all zeros after streaming stores=%s digest_matches_readback=%s "
                    "bytes=%.0f\n",
                    ok ? "PASS" : "FAIL", exact ? "yes" : "no", digest_ok ? "yes" : "no", work);
        if (!ok) rc |= kFailWrite;
    }

    // -- 4. copy identity ----------------------------------------------------
    std::printf("-- copy identity --\n");
    {
        MemCopyBench c;
        c.prepare(1);
        AlignedBlock* s = c.ensure(0);
        const double work = s ? c.run_once(0, 1, 0.25) : 0.0;
        bool identical = (s != nullptr);
        size_t bad_at = 0;
        if (s) {
            const unsigned char* src = s->data;
            const unsigned char* dst = (s + 1)->data;
            for (size_t i = 0; i < kCopyBytes; ++i) {
                if (src[i] != dst[i]) {
                    identical = false;
                    bad_at = i;
                    break;
                }
            }
        }
        char where[48];
        if (identical) {
            std::snprintf(where, sizeof(where), "n/a");
        } else {
            std::snprintf(where, sizeof(where), "byte %llu",
                          static_cast<unsigned long long>(bad_at));
        }
        const bool zeros = s && check_zeros(s->data, kCopyBytes);
        const bool ok = s && zeros && identical && work == static_cast<double>(kCopyBytes);
        // The digest folds the loaded vectors plus the sampled read-back of the
        // destination, so it must equal that same independent combination.
        const bool digest_ok =
            identical && zeros && c.digest() != 0 &&
            c.digest() == fold_one(combine2(fold_bytes_avx2(s->data, kCopyBytes),
                                            fold_samples_avx2((s + 1)->data, kCopyBytes)));
        std::printf("  [%s] src all zeros=%s dst byte-identical to src=%s (first mismatch at %s) "
                    "digest_matches=%s bytes=%.0f\n",
                    (ok && digest_ok) ? "PASS" : "FAIL", zeros ? "yes" : "no",
                    identical ? "yes" : "no", where, digest_ok ? "yes" : "no", work);
        if (!ok || !digest_ok) rc |= kFailCopy;
    }

    // -- 5. latency pointer chain shape -------------------------------------
    std::printf("-- latency pointer chain --\n");
    {
        MemLatencyBench l;
        l.prepare(1);
        AlignedBlock* b = l.ensure();
        std::string why;
        const bool chain_ok = b && verify_chain(*b, l.nodes(), l.stride(), why);
        std::printf("  [%s] nodes=%llu stride=%llu B structure=%llu MiB cycles=1 %s\n",
                    chain_ok ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(l.nodes()),
                    static_cast<unsigned long long>(l.stride()),
                    static_cast<unsigned long long>(kLatBytes / kMiB),
                    chain_ok ? "(single cycle, every node visited once)" : why.c_str());

        // One repetition is one full traversal and must end on the entry node.
        const double loads = l.run_once(0, 1, 0.25);
        const uint32_t last = l.last_node();
        const bool closed = (last == 0) && (loads == static_cast<double>(l.nodes()));
        // The documented single-core rule: workers != 0 return 0 immediately.
        const double other = l.run_once(3, 4, 0.25);
        const bool single = (other == 0.0);

        std::printf("  [%s] one rep = %llu dependent loads, chain closes on entry node (last=%u)\n",
                    closed ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(l.nodes()), static_cast<unsigned>(last));
        std::printf("  [%s] worker 3 of 4 returns 0 immediately (single-threaded test)\n",
                    single ? "PASS" : "FAIL");
        if (!chain_ok || !closed || !single) rc |= kFailLatency;
    }
    std::printf("\n");

    // -- 6. ~50 ms timing of all four kernels (single-threaded) -------------
    std::printf("-- ~50 ms timing, 1 thread (the harness sums workers the same way) --\n");
    {
        MemReadBench r;
        const BenchInfo ri = r.info();
        r.prepare(1);
        const TimeResult t = time_kernel([&r](int wi) { return r.run_once(wi, 1, 0.25); }, 1, kShortSeconds);
        const double value = t.seconds > 0 ? t.work * ri.unit_scale / t.seconds : 0.0;
        std::printf("  memread  : %9.1f %-5s (%.4f s, %llu reps, %.0f bytes, digest=%016llx)\n",
                    value, ri.unit.c_str(), t.seconds, static_cast<unsigned long long>(t.reps),
                    t.work, static_cast<unsigned long long>(r.digest()));
        if (!(t.work > 0.0) || !(t.seconds > 0.0)) rc |= kFailTiming;
    }
    {
        MemWriteBench w;
        const BenchInfo wi = w.info();
        w.prepare(1);
        const TimeResult t = time_kernel([&w](int k) { return w.run_once(k, 1, 0.25); }, 1, kShortSeconds);
        const double value = t.seconds > 0 ? t.work * wi.unit_scale / t.seconds : 0.0;
        std::printf("  memwrite : %9.1f %-5s (%.4f s, %llu reps, %.0f bytes, digest=%016llx)\n",
                    value, wi.unit.c_str(), t.seconds, static_cast<unsigned long long>(t.reps),
                    t.work, static_cast<unsigned long long>(w.digest()));
        if (!(t.work > 0.0) || !(t.seconds > 0.0)) rc |= kFailTiming;
    }
    {
        MemCopyBench c;
        const BenchInfo ci = c.info();
        c.prepare(1);
        const TimeResult t = time_kernel([&c](int k) { return c.run_once(k, 1, 0.25); }, 1, kShortSeconds);
        const double value = t.seconds > 0 ? t.work * ci.unit_scale / t.seconds : 0.0;
        std::printf("  memcopy  : %9.1f %-5s (%.4f s, %llu reps, %.0f bytes, digest=%016llx)\n",
                    value, ci.unit.c_str(), t.seconds, static_cast<unsigned long long>(t.reps),
                    t.work, static_cast<unsigned long long>(c.digest()));
        if (!(t.work > 0.0) || !(t.seconds > 0.0)) rc |= kFailTiming;
    }
    {
        MemLatencyBench l;
        const BenchInfo li = l.info();
        l.prepare(1);
        const TimeResult t = time_kernel([&l](int k) { return l.run_once(k, 1, 0.25); }, 1, kShortSeconds);
        // The harness prints work*unit_scale/seconds, which for an inverse metric
        // is loads/s; the reported latency is elapsed/loads.
        const double ns = t.work > 0.0 ? t.seconds * 1e9 / t.work : 0.0;
        std::printf("  memlat   : %9.2f %-5s (%.4f s, %llu zig-zag traversals, %.0f dependent "
                    "loads, digest=%016llx)\n",
                    ns, li.unit.c_str(), t.seconds, static_cast<unsigned long long>(t.reps), t.work,
                    static_cast<unsigned long long>(l.digest()));
        std::printf("             ns/access = elapsed_seconds * 1e9 / sum(work)  "
                    "[harness formula gives %.4g loads/s]\n",
                    t.seconds > 0 ? t.work * li.unit_scale / t.seconds : 0.0);
        if (!(t.work > 0.0) || !(t.seconds > 0.0)) rc |= kFailTiming;
    }

    // -- 7. multi-worker correctness ---------------------------------------
    std::printf("\n-- 2-worker check (read) --\n");
    {
        MemReadBench r;
        r.prepare(2);
        const double per_worker[2] = {r.run_once(0, 2, 0.25), r.run_once(1, 2, 0.25)};
        // digest() folds both worker slots, so it must equal the fold of each
        // worker's own buffer content: that proves both workers produced their
        // share of the pattern and that the fold is deterministic.
        const WorkerValues expect{fold_bytes_avx2(r.ensure(0)->data, kReadBytes),
                                  fold_bytes_avx2(r.ensure(1)->data, kReadBytes)};
        const uint64_t d0 = r.digest();
        const bool ok = per_worker[0] == static_cast<double>(kReadBytes) &&
                        per_worker[1] == static_cast<double>(kReadBytes) && d0 != 0 &&
                        d0 == fold_slots(expect) && d0 == r.digest();
        std::printf("  [%s] worker work = %.0f / %.0f bytes, folded digest matches both buffers "
                    "(%016llx)\n",
                    ok ? "PASS" : "FAIL", per_worker[0], per_worker[1],
                    static_cast<unsigned long long>(d0));
        if (!ok) rc |= kFailThreads;
    }

    std::printf("\n== memory kernel self-test: %s (rc=%d) ==\n", rc == 0 ? "PASS" : "FAIL", rc);
    return rc;
}

} // namespace cst