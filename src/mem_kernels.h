#pragma once
// ===========================================================================
// Memory-bandwidth sweeps (Memory Read / Write / Copy).
//
// The AVX-512 entries live in mem_kernels_avx512.cpp, which is compiled with
// /arch:AVX512 (MSVC only keeps __m512 registers whole under that switch) and
// is entered only after a CPUID check - the same arrangement as
// raytrace_avx512.cpp.
//
// Shapes mirror AIDA64's MTMBW "Mem"-level kernels (the ZMM_DQ64 variant the
// bench selects on AVX-512 CPUs - see docs/reverse-engineering.md):
//
//   read : linear sweep of non-temporal 64-byte loads   (vmovntdqa)
//   write: linear sweep of non-temporal 64-byte stores of an all-zero register
//          (vmovntdq; AIDA64 zeroes its registers with vpxord first)
//   copy : 64-byte non-temporal loads + non-temporal 64-byte stores over an
//          all-zero source (AIDA64's copy source holds zeros)
//
// Digest contract (bench_memory.cpp): read folds every vector it loads.  Write
// and copy store all-zero data, which carries nothing to fold, so they end each
// pass with a sampled read-back - one contiguous kSamplePageBytes page every
// kSampleStride bytes - and fold that instead.  Both translation units use the
// same sample geometry and the same accumulator math, so a kernel digest equals
// the independent fold computed in bench_memory.cpp and checked by
// selftest_memory.exe.
// ===========================================================================
#include <cstddef>
#include <cstdint>

namespace cst {

// Sampled read-back geometry of the write/copy digests: one contiguous page
// every stride bytes (0.4% of a 96 MiB buffer).
inline constexpr size_t kSamplePageBytes = 4096;              // 4 KiB per sample
inline constexpr size_t kSampleStride    = 1024u * 1024u;     // one sample per MiB

uint64_t mem_read_sweep_avx512(const void* p, size_t bytes);
uint64_t mem_write_sweep_avx512(void* p, size_t bytes);
uint64_t mem_copy_sweep_avx512(void* dst, const void* src, size_t bytes);

} // namespace cst
