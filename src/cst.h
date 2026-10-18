// Core types for the CpuStressTest benchmark suite.
//
// This project re-implements the *workloads* of the AIDA64 CPU/FPU/Memory
// benchmark page from public documentation plus static analysis of
// aida_bench64.dll, so they can be run once or sustained for a fixed
// wall-clock duration.  It contains no FinalWire code.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cst {

// ---------------------------------------------------------------------------
// Result of one measurement of one benchmark.
// ---------------------------------------------------------------------------
struct Result {
    std::string test;        // machine readable id, e.g. "f64ray"
    std::string label;       // display name, e.g. "FP64 Ray-Trace"
    std::string unit;        // reported unit, e.g. "KRay/s", "MB/s", "ns"
    std::string group;       // "FPU" / "CPU" / "Memory"

    double value = 0.0;      // measured quantity, expressed in `unit`
    bool higher_is_better = true;

    double seconds = 0.0;    // total measured wall clock across all workers
    double work = 0.0;       // harness-defined work units completed
    uint64_t reps = 0;       // completed repetitions
    int threads = 1;         // worker threads used

    std::string method;      // fastest kernel selected for this CPU
    std::string params;      // workload parameters, for reproducibility

    // AIDA64-comparable index.  Without a calibration file this equals `value`.
    double index = 0.0;
    bool calibrated = false;

    // Deterministic-output check across repetitions.
    uint64_t digest = 0;
    bool digest_verified = false;
};

// ---------------------------------------------------------------------------
// Description of a benchmark.
// ---------------------------------------------------------------------------
struct BenchInfo {
    std::string test;         // stable id, e.g. "f64ray"
    std::string label;        // display name
    std::string unit;         // reported unit
    std::string group;        // "FPU" / "CPU" / "Memory"
    bool higher_is_better = true;
    // Raw run_rep() work units are multiplied by this to obtain `unit`.
    // e.g. rays -> KRay/s uses 1e-3.
    double unit_scale = 1.0;
    // When true the reported figure is *per work unit* rather than per second:
    //   value = elapsed_seconds * unit_scale / total_work
    // which is what a latency measurement needs (nanoseconds per access).
    bool per_unit = false;
    // Worker count used when the command line does not override it.  0 means
    // "every logical core"; a single-core test like Memory Latency sets 1 so
    // the harness does not spin idle workers (and their SMT siblings) while
    // the one real chase runs.
    int default_threads = 0;
    std::string description;
};

// ---------------------------------------------------------------------------
// Benchmark interface.
//
// The harness calls prepare() once, then repeatedly calls run_once() on every
// worker until the requested duration has elapsed.  run_once() must respect
// min_seconds as an *upper bound* on how long it may run - it should do one
// bounded unit of work and return, letting the harness own the clock.  The
// harness then computes
//   value = sum(work over all calls and workers) / unit_scale / elapsed_seconds
// so an unusually fast or slow call cannot skew the measurement.
// ---------------------------------------------------------------------------
class Benchmark {
public:
    virtual ~Benchmark() = default;

    virtual BenchInfo info() const = 0;

    // Allocate per-worker state and choose kernels for the host CPU.
    virtual void prepare(int threads) = 0;

    // Do one bounded unit of work and return its work units.  Called
    // concurrently on all workers; implementations may only touch per-worker
    // state.  `worker` is in [0, thread_count).
    virtual double run_once(int worker, int thread_count, double min_seconds) = 0;

    // Digest of the most recent output, used to prove the workload really ran
    // and produced identical results in every repetition.
    virtual uint64_t digest() const { return 0; }
    virtual bool has_digest() const { return false; }

    // Fingerprint of a fixed, reproducible reference result for this workload.
    // This is what gets compared against the recorded golden reference, so it
    // must be deterministic for a given workload regardless of how many threads
    // were used: an unstable CPU recomputes it to something else, and that is
    // the signal the score-0-on-mismatch behaviour depends on.
    virtual uint64_t full_digest() const { return digest(); }
    virtual bool has_full_digest() const { return false; }

    // Workload parameter string for reporting.
    virtual std::string params() const { return {}; }
};

// Registry.
const std::vector<Benchmark*>& all_benchmarks();
Benchmark* find_benchmark(const std::string& id);

} // namespace cst
