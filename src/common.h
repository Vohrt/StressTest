// Threading, affinity and timing helpers.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cst {

// Monotonic seconds since process start (high resolution).
double now_seconds();

// Process-wide CPU brand string + core/thread counts.
struct CpuInfo {
    std::string brand;
    int physical_cores = 0;
    int logical_cores = 0;
    uint64_t l2_bytes = 0;
    uint64_t l3_bytes = 0;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_avx512vl = false;
    bool has_aes = false;
    bool has_vaes = false;
    bool has_sha_ni = false;
    std::string isa_summary() const;
};
const CpuInfo& cpu_info();

// Run `body(worker_index)` on `threads` worker threads and join them all.
// `threads <= 1` runs inline on the calling thread.
void run_workers(int threads, const std::function<void(int)>& body);

// Pin the calling thread to logical processor `logical_index` (best effort).
void pin_thread_to_processor(int logical_index);

// Deterministic PRNG (splitmix64) so every run uses identical data.
uint64_t splitmix64(uint64_t& state);

// Fill `count` doubles with deterministic values in [-1, 1].
void fill_doubles(double* dst, size_t count, uint64_t seed);

// Digest helpers.
uint64_t hash_bytes(const void* data, size_t bytes, uint64_t seed = 0xcbf29ce484222325ull);

// Test hook: when the environment variable CST_TEST_CORRUPT is set to a
// non-zero value, reference fingerprints are perturbed by one bit.
//
// That is exactly what a miscomputing CPU produces - a single wrong bit in the
// result - so it lets the reference-verification path be exercised end to end
// and lets the "score 0 on mismatch" behaviour be demonstrated on demand.  It
// has no effect unless the variable is set, so normal runs are untouched.
bool test_corruption_enabled();

// Applies the hook above to a reference fingerprint.
uint64_t corrupt_if_testing(uint64_t digest);

} // namespace cst
