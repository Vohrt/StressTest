// Standalone self test for the CPU PhotoWorxx / CPU Queen / CPU CheckMate
// kernels.  Exits with 0 when every check passes.
//
// Build (MSVC), from the repository root, after `mkdir build`:
//   cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3
//      /Fe:build/selftest_image.exe src/selftest_image.cpp src/bench_image.cpp
//      src/common.cpp /Fo:build/
#include <cstdio>

namespace cst { int verify_image_kernels(); }

int main() {
    const int rc = cst::verify_image_kernels();
    std::printf("image selftest: %s (%d)\n", rc == 0 ? "PASS" : "FAIL", rc);
    return rc;
}
