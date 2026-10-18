#include "calibration.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cst {

std::pair<double, bool> Calibration::scale_for(const std::string& test) const {
    auto it = entries.find(test);
    if (it == entries.end()) return {1.0, false};
    return {it->second.scale, true};
}

bool Calibration::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    source = path;
    entries.clear();

    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        CalibrationEntry e;
        if (!(ls >> e.scale)) continue;
        if (!(ls >> e.reference)) e.reference = 0.0;
        std::string rest;
        std::getline(ls, rest);
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.erase(rest.begin());
        e.unit = rest;
        if (e.scale <= 0.0) continue;
        entries[key] = e;
    }
    return true;
}

} // namespace cst
