#include "session.h"

#include "calibration.h"
#include "common.h"
#include "cst.h"
#include "reference.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cst {
namespace {

// The hooks of the session that is currently running.  Worker threads only read
// it to poll the cancel callback, which is documented as thread-safe.
const SessionHooks* g_hooks = nullptr;

void emit(const std::string& s) {
    if (g_hooks && g_hooks->log) {
        g_hooks->log(s);
    } else {
        std::fputs(s.c_str(), stdout);
    }
}

std::string vformat(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    std::string s;
    if (n > 0) {
        s.resize(static_cast<size_t>(n) + 1);
        std::vsnprintf(&s[0], s.size(), fmt, ap2);
        s.resize(static_cast<size_t>(n));
    }
    va_end(ap2);
    return s;
}

void emitf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    emit(vformat(fmt, ap));
    va_end(ap);
}

void warnf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const std::string s = vformat(fmt, ap);
    va_end(ap);
    if (g_hooks && g_hooks->warn) {
        g_hooks->warn(s);
    } else {
        std::fputs(s.c_str(), stderr);
    }
}

void flush_log() {
    if (!g_hooks || !g_hooks->log) std::fflush(stdout);
}

bool cancelled() {
    return g_hooks && g_hooks->cancel && g_hooks->cancel();
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------
struct RoundSample {
    int index = 0;
    double t_end = 0.0;
    double work = 0.0;
    uint64_t reps = 0;
    double value = 0.0;    // instantaneous throughput in the benchmark's unit
};

struct MeasureResult {
    std::vector<RoundSample> rounds;
    double total_work = 0.0;
    uint64_t total_reps = 0;
    double elapsed = 0.0;
    bool digest_ok = true;
    uint64_t digest = 0;
    bool cancelled = false;
    int threads = 1;
};

MeasureResult measure(Benchmark* bench, const RunConfig& cfg) {
    const BenchInfo info = bench->info();
    MeasureResult result;
    int threads = cfg.threads > 0 ? cfg.threads
                                  : (info.default_threads > 0 ? info.default_threads
                                                              : cpu_info().logical_cores);
    // Never start more workers than the machine has logical processors.  The
    // per-thread working sets multiply by the thread count (96 MiB per worker
    // for the memory tests), the workload model gives every worker its own
    // slice, and AIDA64's own selection stops at the logical core count - so a
    // larger number buys nothing but can exhaust memory.  run_session() warns
    // when this clamps a value the caller asked for.
    threads = std::min(threads, std::max(1, cpu_info().logical_cores));
    threads = std::max(1, threads);
    result.threads = threads;

    const double duration = cfg.sustained ? cfg.minutes * 60.0 : cfg.seconds;
    const double round_len = cfg.sustained ? std::min(1.0, duration) : duration;
    const int round_count = std::max(1, static_cast<int>(std::ceil(duration / round_len)));

    bench->prepare(threads);

    // std::atomic<double> has no fetch_add, and each round is only written by
    // the workers once at its end, so a small mutex per round is contention-free
    // in practice while keeping every worker's contribution exact.
    struct RoundAccum {
        double work = 0.0;
        uint64_t reps = 0;
        std::mutex mu;
    };
    std::vector<RoundAccum> accum(static_cast<size_t>(round_count));

    const double scale = info.unit_scale;
    const double t_start = now_seconds();

    // Upper bound handed to each benchmark call.  run_once() does one bounded
    // unit of work; the harness owns the clock and loops until the round ends.
    const double rep_budget = std::min(0.25, round_len);

    std::atomic<bool> stop{false};
    std::atomic<bool> cancel_seen{false};

    auto worker = [&](int worker_index) {
        for (int r = 0; r < round_count && !stop.load(); ++r) {
            const double round_end = t_start + (r + 1) * round_len;
            double local_work = 0.0;
            uint64_t local_reps = 0;
            for (;;) {
                if (stop.load()) break;
                const double now = now_seconds();
                if (now >= round_end) break;
                const double budget = std::min(rep_budget, round_end - now);
                local_work += bench->run_once(worker_index, threads, budget);
                ++local_reps;
                if (cancelled()) {
                    cancel_seen.store(true);
                    stop.store(true);
                    break;
                }
            }
            // Check the digest after every round: if the workload stops doing
            // real work this catches it within a second instead of at the end.
            if (bench->has_digest() && worker_index == 0) {
                const uint64_t d = bench->digest();
                if (r == 0) {
                    result.digest = d;
                    result.digest_ok = (d != 0);
                } else if (d != result.digest) {
                    result.digest_ok = false;
                }
            }
            RoundAccum& a = accum[static_cast<size_t>(r)];
            {
                std::lock_guard<std::mutex> lock(a.mu);
                a.work += local_work;
                a.reps += local_reps;
            }
            if (worker_index == 0 && g_hooks && g_hooks->round) {
                g_hooks->round(r + 1, round_count);
            }
        }
    };

    run_workers(threads, worker);
    const double elapsed = now_seconds() - t_start;

    // Attribute each round's work to the time that round actually took.  The
    // final round ends when the last worker finished rather than at its nominal
    // boundary, and the first `round_count` rounds never sum to more than the
    // previous ones, so this stays monotonic.
    result.rounds.clear();
    double prev_end = 0.0;
    for (int r = 0; r < round_count; ++r) {
        const RoundAccum& a = accum[static_cast<size_t>(r)];
        const double nominal = std::min((r + 1) * round_len, elapsed);
        const double t_end = std::max(prev_end, nominal);
        const double len = std::max(1e-9, t_end - prev_end);
        RoundSample s;
        s.index = r;
        s.t_end = t_end;
        s.work = a.work;
        s.reps = a.reps;
        s.value = info.per_unit ? (len * scale / (a.work > 0.0 ? a.work : 1.0))
                                             : (a.work * scale / len);
        result.rounds.push_back(s);
        result.total_work += a.work;
        result.total_reps += s.reps;
        prev_end = t_end;
    }
    result.elapsed = elapsed;
    result.cancelled = cancel_seen.load();

    // A cancelled run is discarded: the observed rounds cover less than the
    // requested duration, so the figure would not be comparable.
    return result;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * (v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

// The reported throughput and index of one finished measurement.  Shared by the
// report text and the structured outcome the GUI consumes, so both agree.
struct Score {
    double value = 0.0;
    double index = 0.0;
    bool calibrated = false;
    bool rejected = false;
};

Score score_for(const BenchInfo& info, const MeasureResult& m, const Calibration& cal,
                bool digest_ok, const ReferenceCheck& ref) {
    Score s;
    s.value = info.per_unit
        ? (m.elapsed * info.unit_scale / std::max(1.0, m.total_work))
        : (m.total_work * info.unit_scale / std::max(1e-9, m.elapsed));
    const auto [scale, found] = cal.scale_for(info.test);
    s.calibrated = found;
    s.index = s.value * scale;

    // A failed verification means the hardware computed a different result than
    // the recorded reference, so the throughput figure is not a measurement of
    // the intended workload.  Report 0 instead of a plausible-looking number,
    // the way AIDA64 does when a scene fails its error-margin check.
    const bool failed = (ref.status == ReferenceCheck::Status::Mismatch) || !digest_ok;
    if (failed) {
        s.value = 0.0;
        s.index = 0.0;
    }
    s.rejected = failed;
    return s;
}

void report(Benchmark* bench, const RunConfig& cfg, const MeasureResult& m,
            const Calibration& cal, bool sustained, const ReferenceCheck& ref) {
    const BenchInfo info = bench->info();
    const Score score = score_for(info, m, cal, m.digest_ok, ref);
    const double value = score.value;
    const double index = score.index;
    const auto [scale, found] = cal.scale_for(info.test);

    emitf("\n=== %s (%s) ===\n", info.label.c_str(), info.test.c_str());
    emitf("  workload      : %s\n", info.description.c_str());
    emitf("  parameters    : %s\n", bench->params().c_str());
    emitf("  threads       : %d\n", m.threads);
    emitf("  elapsed       : %.3f s over %llu repetitions\n", m.elapsed,
          static_cast<unsigned long long>(m.total_reps));
    emitf("  throughput    : %s %s%s\n", format_value(value, info.unit).c_str(), info.unit.c_str(),
          info.higher_is_better ? "" : " (lower is better)");
    if (found) {
        emitf("  AIDA64 index  : %.1f  [calibrated, scale %.6g, ref %.6g %s]\n", index,
              scale, cal.entries.at(info.test).reference, info.unit.c_str());
    } else {
        emitf("  AIDA64 index  : %.1f  [uncalibrated]\n", index);
    }
    if (bench->has_full_digest()) {
        emitf("  verification  : %s\n", ref.describe());
        emitf("                  reference %016llx / computed %016llx\n",
              static_cast<unsigned long long>(ref.expected),
              static_cast<unsigned long long>(ref.actual));
        if (ref.status == ReferenceCheck::Status::Missing) {
            emit("                  record a good run with --accept-reference so later\n"
                 "                  runs can be verified against it\n");
        } else if (ref.status == ReferenceCheck::Status::Mismatch) {
            emit("                  >>> RESULT REJECTED: score forced to 0.  The CPU\n"
                 "                  produced a different answer than the recorded\n"
                 "                  reference, which is what an unstable overclock,\n"
                 "                  bad memory or a failing core looks like.\n");
            if (ref.build_differs) {
                emit("                  (note: the reference was recorded by a different\n"
                     "                   build, so a changed binary is an alternative\n"
                     "                   explanation.  Re-record with --accept-reference\n"
                     "                   only if you know the code changed.)\n");
            }
        } else if (ref.status == ReferenceCheck::Status::ParamsChanged) {
            emit("                  (not comparable; re-record with --accept-reference)\n");
        }
    }
    if (bench->has_digest() && !bench->has_full_digest()) {
        emitf("  output digest : %016llx  [%s]\n",
              static_cast<unsigned long long>(m.digest),
              m.digest_ok ? "identical in every repetition" : "CHANGED - result unstable!");
    }

    if (sustained) {
        std::vector<double> vals;
        vals.reserve(m.rounds.size());
        for (const auto& r : m.rounds) vals.push_back(r.value);
        const SustainSummary s = summarize_samples(vals);
        if (s.enough) {
            emit("  --- sustained ---\n");
            emitf("  samples       : %zu\n", s.count);
            emitf("  min / median  : %s / %s %s\n",
                  format_value(s.min, info.unit).c_str(),
                  format_value(s.median, info.unit).c_str(), info.unit.c_str());
            emitf("  mean +- sd    : %s +- %s %s (%.2f%%)\n", format_value(s.mean, info.unit).c_str(),
                  format_value(s.sd, info.unit).c_str(), info.unit.c_str(), s.sd_percent);
            emitf("  p5 / p95      : %s / %s\n",
                  format_value(s.p5, info.unit).c_str(),
                  format_value(s.p95, info.unit).c_str());
            emitf("  first/last 1/3: %s -> %s (%+.2f%% drift)\n", format_value(s.head, info.unit).c_str(),
                  format_value(s.tail, info.unit).c_str(), s.drift_percent);
            emitf("  verdict       : %s\n", sustain_verdict_text(s.verdict));
        }
    }
}

void write_csv(const std::string& path, Benchmark* bench, const MeasureResult& m) {
    std::ofstream out(path);
    if (!out) {
        warnf("warning: cannot write %s\n", path.c_str());
        return;
    }
    const BenchInfo info = bench->info();
    out << "test,round,t_seconds,reps,work,throughput,unit\n";
    for (const auto& r : m.rounds) {
        out << info.test << ',' << r.index << ',' << r.t_end << ',' << r.reps << ','
            << r.work << ',' << r.value << ',' << info.unit << '\n';
    }
}

// The text line the CLI prints for a reference comparison; also carried in the
// structured outcome so the GUI can show the same wording.
std::string describe_reference(const ReferenceCheck& ref) {
    return ref.describe();
}

} // namespace

// A plain number with a few significant digits - no SI prefix.  Every unit this
// tool reports already carries its magnitude ("KRay/s", "MiB/s", ...), so
// prefixing the value as well read as a second magnitude on screen
// ("201.9 k KRay/s" for 201.9 million rays/s).  A bare number next to the unit
// is unambiguous and matches how AIDA64 prints its rows.  Shared with the GUI
// so both front ends render numbers the same way.
std::string format_value(double v, const std::string& unit) {
    (void)unit;   // kept in the signature so every call site reads the same
    if (!std::isfinite(v)) return "n/a";
    char buf[64];
    const double a = std::fabs(v);
    if (a >= 1000.0) std::snprintf(buf, sizeof(buf), "%.0f", v);
    else if (a >= 10.0) std::snprintf(buf, sizeof(buf), "%.1f", v);
    else if (a >= 1.0) std::snprintf(buf, sizeof(buf), "%.2f", v);
    else if (a > 0.0) std::snprintf(buf, sizeof(buf), "%.4g", v);
    else std::snprintf(buf, sizeof(buf), "0");
    return buf;
}

SustainSummary summarize_samples(std::vector<double> const& samples) {
    SustainSummary s;
    if (samples.size() < 3) return s;

    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(samples.size());
    if (!(mean > 0.0)) return s;

    double var = 0.0;
    for (double v : samples) var += (v - mean) * (v - mean);
    var /= static_cast<double>(samples.size());

    // Drift: compare the first and last third of the run.
    const size_t third = std::max<size_t>(1, samples.size() / 3);
    double head = 0.0, tail = 0.0;
    for (size_t i = 0; i < third; ++i) head += samples[i];
    for (size_t i = samples.size() - third; i < samples.size(); ++i) tail += samples[i];
    head /= static_cast<double>(third);
    tail /= static_cast<double>(third);

    s.enough = true;
    s.count = samples.size();
    s.mean = mean;
    s.sd = std::sqrt(var);
    s.sd_percent = s.sd / mean * 100.0;
    s.min = percentile(samples, 0.0);
    s.median = percentile(samples, 0.5);
    s.p5 = percentile(samples, 0.05);
    s.p95 = percentile(samples, 0.95);
    s.head = head;
    s.tail = tail;
    s.drift_percent = head > 0.0 ? (tail - head) / head * 100.0 : 0.0;
    if (s.drift_percent <= -5.0) s.verdict = SustainVerdict::Throttling;
    else if (s.drift_percent <= -2.0) s.verdict = SustainVerdict::MildDecay;
    else if (s.drift_percent >= 5.0) s.verdict = SustainVerdict::Increasing;
    return s;
}

const char* sustain_verdict_text(SustainVerdict verdict) {
    switch (verdict) {
    case SustainVerdict::Throttling: return "THROTTLING (throughput falls over the run)";
    case SustainVerdict::MildDecay: return "mild decay";
    case SustainVerdict::Increasing: return "increasing (unusual: check background load)";
    default: return "stable";
    }
}

std::vector<Benchmark*> resolve_targets(const std::string& target) {
    std::vector<Benchmark*> targets;
    if (target == "all" || target.empty()) return all_benchmarks();
    // Allow comma separated lists: --target f64ray,memread
    std::string list = target;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const std::string id =
            list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!id.empty()) {
            Benchmark* b = find_benchmark(id);
            if (!b) return {};
            targets.push_back(b);
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return targets;
}

int run_session(const RunConfig& cfg, const SessionHooks& hooks) {
    g_hooks = &hooks;

    std::vector<Benchmark*> targets = resolve_targets(cfg.target);
    if (targets.empty()) {
        warnf("error: unknown test '%s' (try `cst list`)\n", cfg.target.c_str());
        g_hooks = nullptr;
        return 2;
    }

    Calibration cal;
    if (!cal.load(cfg.calibration_path) && cfg.calibration_path != "calibration.txt") {
        warnf("warning: cannot read calibration file %s\n", cfg.calibration_path.c_str());
    }

    // Golden references: the fingerprints of a known-good run.  A measurement
    // that does not reproduce them is rejected with a score of 0.
    ReferenceStore refs;
    if (!refs.load(cfg.reference_path) && cfg.reference_path != "reference.txt") {
        warnf("warning: cannot read reference file %s\n", cfg.reference_path.c_str());
    }

    const bool sustained = cfg.sustained;
    const CpuInfo& ci = cpu_info();
    if (cfg.threads > ci.logical_cores) {
        warnf("warning: --threads %d exceeds the %d logical CPUs of this machine; using %d\n",
              cfg.threads, ci.logical_cores, ci.logical_cores);
    }
    emitf("CpuStressTest  |  %s\n", ci.brand.c_str());
    emitf("  %d physical / %d logical cores  |  ISA: %s\n", ci.physical_cores,
          ci.logical_cores, ci.isa_summary().c_str());
    emitf("  mode: %s", sustained ? "sustained" : "single");
    if (sustained) emitf(" (%.1f minutes)", cfg.minutes);
    else emitf(" (%.1f seconds)", cfg.seconds);
    emitf("  |  threads: %d\n", cfg.threads > 0 ? std::min(cfg.threads, ci.logical_cores)
                                               : ci.logical_cores);
    if (!cfg.no_verify) {
        emitf("  verification: on (%zu recorded reference%s from %s)\n", refs.size(),
              refs.size() == 1 ? "" : "s", cfg.reference_path.c_str());
    }

    int rejected = 0;
    int done = 0;
    const int total = static_cast<int>(targets.size());
    bool stopped = false;
    for (Benchmark* b : targets) {
        if (cancelled()) {
            stopped = true;
            break;
        }
        const BenchInfo info = b->info();
        if (!cfg.quiet) {
            emitf("\n>> running %s ...\n", info.label.c_str());
            flush_log();
        }
        if (hooks.begin) hooks.begin(info.test, info.label);
        const MeasureResult m = measure(b, cfg);
        if (m.cancelled) {
            stopped = true;
            break;
        }

        // Compare the workload's reference fingerprint with the recorded one.
        ReferenceCheck ref;
        if (b->has_full_digest()) {
            const uint64_t d = b->full_digest();
            ref = refs.check(info.test, b->params(), d);
            if (cfg.no_verify) {
                ref.waived = ref.status != ReferenceCheck::Status::Ok;
                ref.status = ReferenceCheck::Status::Ok;
            }
            if (cfg.accept_reference) {
                ReferenceEntry e;
                e.params = b->params();
                e.build = build_id();
                e.digest = d;
                refs.record(info.test, e);
                ref.expected = d;
                ref.actual = d;
                ref.status = ReferenceCheck::Status::Ok;
            }
        }

        report(b, cfg, m, cal, sustained, ref);
        const Score score = score_for(info, m, cal, m.digest_ok, ref);
        if (score.rejected) ++rejected;
        if (!cfg.csv_path.empty()) {
            std::string path = cfg.csv_path;
            if (targets.size() > 1) path = info.test + "-" + cfg.csv_path;
            write_csv(path, b, m);
        }
        flush_log();

        if (hooks.result) {
            TestOutcome o;
            o.id = info.test;
            o.label = info.label;
            o.unit = info.unit;
            o.params = b->params();
            o.threads = m.threads;
            o.elapsed = m.elapsed;
            o.reps = m.total_reps;
            o.value = score.value;
            o.index = score.index;
            o.calibrated = score.calibrated;
            o.rejected = score.rejected;
            o.has_full_digest = b->has_full_digest();
            o.digest = o.has_full_digest ? b->full_digest() : m.digest;
            o.reference_digest = ref.expected;
            o.reference_missing = (ref.status == ReferenceCheck::Status::Missing);
            o.reference_mismatch = (ref.status == ReferenceCheck::Status::Mismatch);
            o.reference_params_changed = (ref.status == ReferenceCheck::Status::ParamsChanged);
            o.reference_build_differs = ref.build_differs;
            o.digest_unstable = !m.digest_ok;
            o.verification = describe_reference(ref);
            for (const auto& r : m.rounds) o.samples.push_back(r.value);
            hooks.result(o);
        }
        ++done;
        if (hooks.progress) hooks.progress(done, total);
    }

    if (cfg.accept_reference) {
        if (refs.save(cfg.reference_path)) {
            emitf("\nrecorded %zu reference%s in %s\n", refs.size(),
                  refs.size() == 1 ? "" : "s", cfg.reference_path.c_str());
        } else {
            warnf("error: cannot write %s\n", cfg.reference_path.c_str());
        }
    }

    if (stopped) {
        emit("\n>> stopped\n");
        g_hooks = nullptr;
        return 4;
    }
    if (rejected > 0) {
        emitf("\n*** %d of %zu measurement%s FAILED verification and scored 0.\n"
              "*** The CPU produced a different result than the recorded reference.\n",
              rejected, targets.size(), targets.size() == 1 ? "" : "s");
        g_hooks = nullptr;
        return 3;
    }
    g_hooks = nullptr;
    return 0;
}

} // namespace cst
