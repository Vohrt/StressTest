// CPU image/decision benchmarks: PhotoWorxx, Queen, CheckMate.
#pragma once

#include "cst.h"

namespace cst {

Benchmark* make_bench_photoworxx();
Benchmark* make_bench_queen();
Benchmark* make_bench_checkmate();

// Returns 0 when every self-consistency check passes.
int verify_image_kernels();

} // namespace cst
