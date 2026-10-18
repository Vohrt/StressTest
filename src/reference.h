// Golden-reference verification.
//
// AIDA64's ray-trace test validates each computed scene against its reference
// within a tolerance (`RayTraceErrMargin` / `RayTraceFunctionErrMargin%02d%02d`)
// and reports a score of 0 when the result does not match.  That is what makes
// the test useful for catching an unstable overclock: the CPU silently
// miscomputes, the frame no longer matches, and the run scores 0.
//
// This module implements the equivalent for every workload here:
//
//   * a benchmark's `digest()` reduces its output to a 64-bit fingerprint;
//   * the fingerprint of a known-good run is recorded once in a reference file
//     (`reference.txt` beside the executable, or --reference FILE);
//   * every subsequent measurement recomputes the fingerprint and compares it
//     with the recorded one;
//   * a mismatch means the hardware computed something different, so the
//     reported score becomes 0 rather than a plausible-looking number.
//
// The reference is keyed by test id *and* by the workload parameters, so a
// config change (resolution, iteration count, thread count) is detected and
// reported as "no reference yet" instead of as an instability.
#pragma once

#include <map>
#include <string>

namespace cst {

struct ReferenceEntry {
    std::string params;      // workload parameter string the digest belongs to
    std::string build;       // build id (binary timestamp) that produced it
    uint64_t digest = 0;
};

struct ReferenceCheck {
    enum class Status {
        Ok,            // digest matched the recorded reference
        Mismatch,      // digest differed -> hardware computed something else
        ParamsChanged, // recorded under different parameters: not comparable
        Missing,       // no reference recorded for this test yet
    };
    Status status = Status::Missing;
    bool waived = false;       // --no-verify: a failing check is reported, not enforced
    // Set when a mismatch happened and the reference came from a different build.
    // It does NOT change the verdict - a mismatch is still a mismatch, otherwise
    // a genuinely unstable CPU could print a valid-looking score - but the report
    // says so, because "the code changed" and "the CPU computed something else"
    // look identical from here.
    bool build_differs = false;
    uint64_t expected = 0;
    uint64_t actual = 0;

    bool verified() const { return status == Status::Ok; }
    const char* describe() const;
};

class ReferenceStore {
public:
    // Loads "test paramsTABbuildTABdigest" lines; '#' starts a comment.
    bool load(const std::string& path);
    // Writes the whole store back out.
    bool save(const std::string& path) const;

    void record(const std::string& test, const ReferenceEntry& e) { entries_[test] = e; }
    ReferenceCheck check(const std::string& test, const std::string& params,
                         uint64_t digest) const;

    size_t size() const { return entries_.size(); }
    const std::string& source() const { return source_; }

private:
    std::map<std::string, ReferenceEntry> entries_;
    std::string source_;
};

// Identifier for this binary, so a reference recorded by a different build is
// reported as stale rather than as instability.
const std::string& build_id();

} // namespace cst
