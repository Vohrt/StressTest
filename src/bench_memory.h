// Memory-subsystem benchmark module: Memory Read / Write / Copy / Latency.
//
// The four kernels are implemented in bench_memory.cpp and are AVX2 (256-bit);
// the module is built with /arch:AVX2 and every buffer is 64 KiB aligned.
#pragma once

#include "cst.h"

namespace cst {

Benchmark* make_bench_mem_read();
Benchmark* make_bench_mem_write();
Benchmark* make_bench_mem_copy();
Benchmark* make_bench_mem_latency();

// Self-consistency check of the four memory kernels: verifies the 64 KiB
// alignment of every buffer, the byte-exact pattern round-trip of the read and
// write kernels, the byte-identity of the copy destination, and the shape of
// the latency pointer chain (one single cycle covering every node, stride
// >= 128 bytes).  A short (~50 ms) timing of all four kernels is printed to
// stdout so the measured MB/s and ns/access numbers are visible.
//
// Returns 0 when every self-consistency check passes, non-zero = failure code
// (a bitmask of the individual checks that failed).
int verify_memory_kernels();

} // namespace cst
