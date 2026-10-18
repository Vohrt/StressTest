// Hash / cipher / compression benchmarks: AES, SHA3, ZLib.
#pragma once

#include "cst.h"

namespace cst {

Benchmark* make_bench_aes();
Benchmark* make_bench_sha3();
Benchmark* make_bench_zlib();

// Returns 0 when every self-consistency check passes.
int verify_crypto_kernels();

} // namespace cst
