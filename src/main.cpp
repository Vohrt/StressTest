// Command line front end of CpuStressTest.
//
// The executable that carries this file is also the GUI (gui/): started with
// arguments it behaves exactly like the classic `cst` command line, started
// without arguments (or with `gui`) it opens the WinUI window.  The entry point
// that decides between the two lives in gui/App.xaml.cpp.
//
//   cst list [--tsv]
//   cst run <test|all> [--mode single|sustained] [--minutes 20] [--seconds 15]
//                       [--threads N] [--calibration FILE] [--reference FILE]
//                       [--accept-reference] [--no-verify] [--csv FILE]
//                       [--emit-calibration FILE]
#include "cli.h"

#include "calibration.h"
#include "common.h"
#include "cst.h"
#include "reference.h"
#include "session.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace cst;

namespace {

void printf_isa();

struct Options {
    std::string command;
    std::string target = "all";
    std::string mode = "single";
    double seconds = 15.0;
    double minutes = 20.0;
    int threads = 0;              // 0 = all logical processors
    std::string calibration_path = "calibration.txt";
    std::string reference_path = "reference.txt";
    std::string csv_path;
    std::string emit_calibration;
    bool accept_reference = false;
    bool no_verify = false;
    bool quiet = false;
    bool list_tsv = false;        // `cst list --tsv`: machine-readable listing
};

void usage() {
    std::printf(
        "CpuStressTest - AIDA64 benchmark workloads, single-shot or sustained\n\n"
        "Usage:\n"
        "  cst list [--tsv]\n"
        "  cst run <test|all> [options]\n"
        "  cst gui\n"
        "  cst                    (no arguments opens the GUI)\n\n"
        "Tests: queen checkmate zlib aes sha3 photoworxx julia mandel sinjulia\n"
        "       f32ray f64ray memread memwrite memcopy memlat\n\n"
        "Options:\n"
        "  --mode single        one measurement of --seconds (default)\n"
        "  --mode sustained     repeat and sample until --minutes elapse\n"
        "  --seconds N          measurement length for single mode (default 15)\n"
        "  --minutes N          duration for sustained mode (default 20)\n"
        "  --threads N          worker threads (default: all logical CPUs; values\n"
        "                       above the logical core count are capped to it)\n"
        "  --calibration FILE   index calibration file (default calibration.txt)\n"
        "  --csv FILE           write per-round samples to FILE\n"
        "  --emit-calibration F write a calibration template for this machine\n"
        "  --reference FILE     golden reference file (default reference.txt)\n"
        "  --accept-reference   record this run's output as the golden reference\n"
        "  --no-verify          skip reference verification (score is never forced to 0)\n"
        "  --quiet              suppress progress output\n"
        "\n"
        "Verification: each workload's output is fingerprinted and compared with the\n"
        "recorded reference.  On a mismatch the score is reported as 0 - the same\n"
        "behaviour AIDA64 uses when a ray-traced scene fails its error-margin check -\n"
        "which is how an unstable overclock shows up.\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    if (argc < 2) return false;
    o.command = argv[1];
    if (o.command == "list") {
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--tsv") o.list_tsv = true;
        }
        return true;
    }
    if (o.command != "run") return false;
    if (argc >= 3 && argv[2][0] != '-') {
        o.target = argv[2];
    }
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mode") o.mode = next("--mode");
        else if (a == "--seconds") o.seconds = std::atof(next("--seconds"));
        else if (a == "--minutes") o.minutes = std::atof(next("--minutes"));
        else if (a == "--threads") o.threads = std::atoi(next("--threads"));
        else if (a == "--calibration") o.calibration_path = next("--calibration");
        else if (a == "--reference") o.reference_path = next("--reference");
        else if (a == "--accept-reference") o.accept_reference = true;
        else if (a == "--no-verify") o.no_verify = true;
        else if (a == "--csv") o.csv_path = next("--csv");
        else if (a == "--emit-calibration") o.emit_calibration = next("--emit-calibration");
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--target") o.target = next("--target");
        else if (a[0] == '-') {
            std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
            return false;
        } else {
            o.target = a;
        }
    }
    return true;
}

void print_list(bool tsv) {
    if (tsv) {
        // Machine-readable listing: one row per benchmark, tab separated, no
        // banner and no ISA footer so the whole output parses.
        for (Benchmark* b : all_benchmarks()) {
            const BenchInfo i = b->info();
            std::printf("%s\t%s\t%s\t%s\t%d\t%d\t%s\n", i.test.c_str(), i.label.c_str(),
                        i.unit.c_str(), i.group.c_str(),
                        i.default_threads > 0 ? i.default_threads : 0,
                        i.higher_is_better ? 1 : 0, i.description.c_str());
        }
        return;
    }
    std::printf("%-12s %-18s %-10s %-7s %s\n", "id", "label", "unit", "group", "workload");
    std::printf("%s\n", std::string(112, '-').c_str());
    for (Benchmark* b : all_benchmarks()) {
        const BenchInfo i = b->info();
        std::string desc = i.description;
        if (desc.size() > 56) desc = desc.substr(0, 53) + "...";
        std::printf("%-12s %-18s %-10s %-7s %s\n", i.test.c_str(), i.label.c_str(),
                    i.unit.c_str(), i.group.c_str(), desc.c_str());
    }
    printf_isa();
}

void printf_isa() {
    const CpuInfo& ci = cpu_info();
    std::printf("\nCPU: %s\n", ci.brand.c_str());
    std::printf("     %d physical / %d logical cores, L2 %llu KiB, L3 %llu KiB\n",
                ci.physical_cores, ci.logical_cores,
                static_cast<unsigned long long>(ci.l2_bytes / 1024),
                static_cast<unsigned long long>(ci.l3_bytes / 1024));
    std::printf("     ISA in use: %s\n", ci.isa_summary().c_str());
}

void emit_calibration(const std::string& path, const std::vector<Benchmark*>& benches) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        return;
    }
    out << "# CpuStressTest calibration file v1\n"
        << "# Fill in the `scale` column: run `cst run <test>` to see this machine's\n"
        << "# measured throughput, run the same test in AIDA64 to get its score, then\n"
        << "#   scale = aida64_score / cst_measured_value\n"
        << "# with `reference` set to the AIDA64 score.\n"
        << "#\n"
        << "# test          scale        reference    unit\n";
    char line[256];
    for (Benchmark* b : benches) {
        const BenchInfo i = b->info();
        std::snprintf(line, sizeof(line), "%-14s %-12s %-12s %s\n", i.test.c_str(), "1.0",
                      "0.0", i.unit.c_str());
        out << line;
    }
    std::printf("wrote calibration template to %s\n", path.c_str());
}

std::vector<Benchmark*> calibration_targets(const std::string& target) {
    return resolve_targets(target);
}

} // namespace

namespace cst {

int cli_main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return 1;
    }
    if (opt.command == "list") {
        print_list(opt.list_tsv);
        return 0;
    }

    if (!opt.emit_calibration.empty()) {
        emit_calibration(opt.emit_calibration, calibration_targets(opt.target));
    }

    RunConfig cfg;
    cfg.target = opt.target;
    cfg.sustained = (opt.mode == "sustained");
    cfg.seconds = opt.seconds;
    cfg.minutes = opt.minutes;
    cfg.threads = opt.threads;
    cfg.quiet = opt.quiet;
    cfg.accept_reference = opt.accept_reference;
    cfg.no_verify = opt.no_verify;
    cfg.csv_path = opt.csv_path;
    cfg.calibration_path = opt.calibration_path;
    cfg.reference_path = opt.reference_path;

    // Default hooks: the session prints to stdout/stderr exactly like the
    // classic command line build.  A failure inside the session (an allocation
    // that cannot be satisfied, for example) must end in a message and an exit
    // code, not std::terminate.
    SessionHooks hooks;
    try {
        return run_session(cfg, hooks);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: run failed: %s\n", e.what());
        return 5;
    } catch (...) {
        std::fprintf(stderr, "error: run failed: unknown exception\n");
        return 5;
    }
}

} // namespace cst
