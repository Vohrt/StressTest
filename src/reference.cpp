#include "reference.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace cst {

const char* ReferenceCheck::describe() const {
    if (waived) return "verification skipped (--no-verify)";
    switch (status) {
        case Status::Ok:            return "matches reference";
        case Status::Mismatch:      return "DOES NOT MATCH REFERENCE - hardware computed a different result";
        case Status::ParamsChanged: return "reference recorded for different workload parameters";
        case Status::Missing:       return "no reference recorded yet";
    }
    return "unknown";
}

bool ReferenceStore::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    source_ = path;
    entries_.clear();

    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        if (line.empty()) continue;

        // Fields are tab separated so a parameter string may contain spaces.
        std::istringstream ls(line);
        std::string test, params, build, dig;
        if (!std::getline(ls, test, '\t')) continue;
        if (!std::getline(ls, params, '\t')) continue;
        if (!std::getline(ls, build, '\t')) continue;
        if (!std::getline(ls, dig)) continue;
        if (test.empty() || dig.empty()) continue;

        try {
            ReferenceEntry e;
            e.params = params;
            e.build = build;
            e.digest = std::stoull(dig, nullptr, 16);
            entries_[test] = e;
        } catch (...) {
            // Ignore malformed lines rather than failing the whole load.
        }
    }
    return true;
}

bool ReferenceStore::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    out << "# CpuStressTest golden references\n"
        << "#\n"
        << "# Each line records the output fingerprint of a known-good run:\n"
        << "#   <test>\\t<params>\\t<build>\\t<digest>\n"
        << "#\n"
        << "# The benchmark recomputes the fingerprint every measurement.  If it does\n"
        << "# not match, the hardware produced a different result - the signature of an\n"
        << "# unstable overclock, bad memory or a failing core - and the score is\n"
        << "# reported as 0 instead of a plausible-looking number.\n"
        << "#\n"
        << "# Regenerate with --accept-reference after an intentional change to the\n"
        << "# build, the workload parameters or the thread count.\n"
        << "#\n"
        << "# test\tparams\tbuild\tdigest\n";
    for (const auto& kv : entries_) {
        out << kv.first << '\t' << kv.second.params << '\t' << kv.second.build << '\t'
            << std::hex << kv.second.digest << std::dec << '\n';
    }
    return true;
}

ReferenceCheck ReferenceStore::check(const std::string& test, const std::string& params,
                                     uint64_t digest) const {
    ReferenceCheck r;
    r.actual = digest;
    auto it = entries_.find(test);
    if (it == entries_.end()) {
        r.status = ReferenceCheck::Status::Missing;
        return r;
    }
    r.expected = it->second.digest;

    // A matching digest is conclusive on its own, so it verifies regardless of
    // the recorded build or parameter string: a rebuild that produces the same
    // answer is a valid reference, and blocking on __DATE__/__TIME__ would mean
    // re-recording after every compile for no benefit.
    if (digest == it->second.digest) {
        r.status = ReferenceCheck::Status::Ok;
        return r;
    }

    // The digest differs.  Report that as a mismatch: it is the signal the
    // score-0 behaviour exists for, and gating it on the recorded build id would
    // let a genuinely unstable CPU print a valid-looking score instead of 0 -
    // exactly the failure this mechanism is for.
    //
    // Parameters are still checked, because a different workload legitimately
    // hashes differently and calling that instability would be a false alarm.
    if (!params.empty() && !it->second.params.empty() && it->second.params != params) {
        r.status = ReferenceCheck::Status::ParamsChanged;
        return r;
    }
    // A differing build id is worth saying out loud, because "the code changed"
    // and "the CPU computed something else" look identical from here.  It does
    // not change the verdict: the user decides by re-recording.
    r.build_differs = !it->second.build.empty() && it->second.build != build_id();
    r.status = ReferenceCheck::Status::Mismatch;
    return r;
}

const std::string& build_id() {
    // __DATE__/__TIME__ change whenever the translation unit is recompiled, which
    // is exactly the signal wanted: a digest is only comparable within one build.
    static const std::string id = std::string(__DATE__) + " " + __TIME__;
    return id;
}

} // namespace cst
