// FPU benchmarks: FP32/FP64 Ray-Trace and the Julia / Mandel / SinJulia fractals.
#pragma once

#include "cst.h"

namespace cst {

Benchmark* make_bench_f64ray();
Benchmark* make_bench_f32ray();
Benchmark* make_bench_julia();
Benchmark* make_bench_mandel();
Benchmark* make_bench_sinjulia();

} // namespace cst
