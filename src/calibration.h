// Calibration: map this tool's measured throughput onto AIDA64-comparable
// index numbers using reference scores recorded from AIDA64 itself.
//
// Format (also written by `cst run ... --emit-calibration FILE`):
//
//   # CpuStressTest calibration file v1
//   # test          scale        reference   unit
//   f64ray          0.0001234    45678.0     KRay/s
//
// index = measured_value * scale.  Scale is chosen so that on the reference
// machine the index equals the AIDA64 score that was measured for that machine.
//
// A row with no reference (0.0) is a template line and is ignored: it would
// otherwise report the raw throughput under the name "AIDA64 index".  The
// front ends then show the index as uncalibrated instead, so it is never
// mistaken for a score AIDA64 would produce.
#pragma once

#include <map>
#include <string>

namespace cst {

struct CalibrationEntry {
    double scale = 1.0;
    double reference = 0.0;   // AIDA64 score measured on the reference machine
    std::string unit;
    std::string note;
};

struct Calibration {
    std::map<std::string, CalibrationEntry> entries;
    std::string source;

    // Returns (scale, found).
    std::pair<double, bool> scale_for(const std::string& test) const;

    // Loads "key scale [reference] [unit...]" lines; '#' starts a comment.
    // Returns false when the file cannot be opened.
    bool load(const std::string& path);
};

} // namespace cst
