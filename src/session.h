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

// Three significant digits, switching to SI prefixes so a 65 GB/s memory result
// reads as "65.0 G" rather than "6.503e+10".  Shared with the GUI so both front
// ends render numbers the same way.
std::string format_value(double v, const std::string& unit = {});

// Runs the session.  Returns a process-style exit code:
//   0  ok
//   2  unknown workload in cfg.target
//   3  at least one workload failed verification and scored 0
//   4  cancelled through SessionHooks::cancel
int run_session(const RunConfig& cfg, const SessionHooks& hooks);

class Benchmark;

// Resolves a target spec ("all", "f64ray", "f64ray,memread") to the matching
// benchmarks; empty when the spec names an unknown workload.
std::vector<Benchmark*> resolve_targets(const std::string& target);

} // namespace cst
