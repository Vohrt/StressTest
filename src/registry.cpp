#include "bench_fp.h"
#include "bench_crypto.h"
#include "bench_image.h"
#include "bench_memory.h"
#include "cst.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace cst {

namespace {

struct Entry {
    const char* test;
    Benchmark* (*make)();
};

// Order mirrors the AIDA64 benchmark page: CPU first, then FPU, then Memory.
const Entry kEntries[] = {
    {"queen",      &make_bench_queen},
    {"checkmate",  &make_bench_checkmate},
    {"zlib",       &make_bench_zlib},
    {"aes",        &make_bench_aes},
    {"sha3",       &make_bench_sha3},
    {"photoworxx", &make_bench_photoworxx},
    {"julia",      &make_bench_julia},
    {"mandel",     &make_bench_mandel},
    {"sinjulia",   &make_bench_sinjulia},
    {"f32ray",     &make_bench_f32ray},
    {"f64ray",     &make_bench_f64ray},
    {"memread",    &make_bench_mem_read},
    {"memwrite",   &make_bench_mem_write},
    {"memcopy",    &make_bench_mem_copy},
    {"memlat",     &make_bench_mem_latency},
};

std::vector<Benchmark*>& registry() {
    static std::vector<Benchmark*> reg;
    if (reg.empty()) {
        for (const auto& e : kEntries) reg.push_back(e.make());
    }
    return reg;
}

} // namespace

const std::vector<Benchmark*>& all_benchmarks() { return registry(); }

Benchmark* find_benchmark(const std::string& id) {
    for (Benchmark* b : registry()) {
        if (b->info().test == id) return b;
    }
    return nullptr;
}

} // namespace cst
