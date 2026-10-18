// Shared measurement session.
//
// This is the one implementation of "measure these workloads with these
// options, verify them against the recorded references and report the results".
// Both front ends use it:
//
//   * the cst command line (src/main.cpp) passes default hooks, which print to
//     stdout/stderr exactly as before;
//   * the WinUI GUI (gui/) passes hooks that forward the same text to its log
//     pane and hand back structured per-test results.
//
// Keeping a single session layer means a GUI run and a CLI run measure and
// verify identically - the GUI never parses console output.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cst {

struct RunConfig {
    std::string target = "all";   // "all", "f64ray", "f64ray,memread", ...
    bool sustained = false;       // false = one --seconds measurement
    double seconds = 15.0;
    double minutes = 20.0;
    int threads = 0;              // 0 = workload default / all logical CPUs
    bool quiet = false;
    bool accept_reference = false;
    bool no_verify = false;
    std::string csv_path;
    std::string calibration_path = "calibration.txt";
    std::string reference_path = "reference.txt";
};

// What one finished workload produced.  `rejected` means the score was forced
// to 0 because the fingerprint did not reproduce.
struct TestOutcome {
    std::string id;
    std::string label;
    std::string unit;
    std::string params;
    int threads = 0;
    double elapsed = 0.0;
    uint64_t reps = 0;
    double value = 0.0;           // throughput in `unit`, 0 when rejected
    double index = 0.0;           // AIDA64 index (calibrated or identity)
    bool calibrated = false;
    bool rejected = false;

    bool has_full_digest = false;
    uint64_t digest = 0;
    uint64_t reference_digest = 0;
    bool reference_missing = false;
    bool reference_mismatch = false;
    bool reference_params_changed = false;
    bool reference_build_differs = false;
    bool digest_unstable = false;
    std::string verification;     // human readable status, as printed by cst

    std::vector<double> samples;  // per-round throughput (sustained mode)
};

// Callbacks into the front end.  `log`/`warn`/`result`/`progress`/`round` are
// invoked from the thread that called run_session(); `cancel` is invoked from
// the benchmark worker threads, so it must be thread-safe.
struct SessionHooks {
    std::function<void(const std::string&)> log;       // text the CLI prints
    std::function<void(const std::string&)> warn;      // text the CLI prints to stderr
    std::function<void(const std::string& id, const std::string& label)> begin;  // workload started
    std::function<void(const TestOutcome&)> result;    // one call per finished workload
    std::function<void(int done, int total)> progress; // workloads finished
    std::function<void(int round, int rounds)> round;  // rings during a workload
    std::function<bool()> cancel;                      // true = stop as soon as possible
};

// A plain number with a few significant digits - no SI prefix.  Every unit this
// tool reports already carries its magnitude ("KRay/s", "MiB/s", ...), so
// prefixing the value as well read as a second magnitude on screen
// ("201.9 k KRay/s" for 201.9 million rays/s).  A bare number next to the unit
// is unambiguous and matches how AIDA64 prints its rows.  Shared with the GUI
// so both front ends render numbers the same way.  (The `unit` argument is
// kept so call sites read the same.)
std::string format_value(double v, const std::string& unit = {});

// Per-round statistics of a sustained run.  These are what the report block
// prints under "--- sustained ---"; exposing them lets the GUI put the verdict
// in the result table instead of leaving it in the log text only.
enum class SustainVerdict {
    Stable,
    MildDecay,     // -5% < drift <= -2%
    Throttling,    // drift <= -5%
    Increasing,    // drift >= +5%
};

struct SustainSummary {
    bool enough = false;      // false when there are fewer than 3 usable samples
    size_t count = 0;
    double mean = 0.0;
    double sd = 0.0;
    double sd_percent = 0.0;  // sd / mean, in percent
    double min = 0.0;
    double median = 0.0;
    double p5 = 0.0;
    double p95 = 0.0;
    double head = 0.0;        // mean of the first third of the run
    double tail = 0.0;        // mean of the last third
    double drift_percent = 0.0;
    SustainVerdict verdict = SustainVerdict::Stable;
};

SustainSummary summarize_samples(std::vector<double> const& samples);

// The wording the CLI prints for a verdict (English).  The GUI has its own
// Chinese text.
const char* sustain_verdict_text(SustainVerdict verdict);

// Runs the session.  Returns a process-style exit code:
//   0  ok
//   2  unknown workload in cfg.target
//   3  at least one workload failed verification and scored 0
//   4  cancelled through SessionHooks::cancel
//   5  the session threw (allocation failure, ...) - only the CLI produces it,
//      the GUI catches the exception itself so it can show the message
int run_session(const RunConfig& cfg, const SessionHooks& hooks);

class Benchmark;

// Resolves a target spec ("all", "f64ray", "f64ray,memread") to the matching
// benchmarks; empty when the spec names an unknown workload.
std::vector<Benchmark*> resolve_targets(const std::string& target);

} // namespace cst
