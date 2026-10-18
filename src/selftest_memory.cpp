// Standalone entry point for the memory-module self-consistency check.
//
// Build (MSVC, x64) -- note there must be NO trailing backslash on the last
// comment line, because MSVC continues a // comment across a backslash-newline:
//
//   cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3
//      /Fe:build\selftest_memory.exe
//      src\selftest_memory.cpp src\bench_memory.cpp src\common.cpp
//      build\obj\mem_kernels_avx512.obj /Fo:build\
//
#include <cstdio>

namespace cst {
int verify_memory_kernels();
}

int main() {
    const int rc = cst::verify_memory_kernels();
    std::printf("memory selftest: %s (%d)\n", rc == 0 ? "PASS" : "FAIL", rc);
    return rc;
}
