// Conformance test for the DEFLATE decoder: decode reference streams produced by
// zlib 1.2.x and check the plaintext byte for byte.
#define main cst_bench_crypto_main_disabled
#include "../src/bench_crypto.cpp"
#undef main

#include "zlib_ref.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    int failures = 0;
    for (size_t i = 0; i < kZlibRefCount; ++i) {
        const ZlibRefCase& c = kZlibRefCases[i];
        std::vector<uint8_t> out;
        const bool ok = cst::deflate::inflate(c.data, c.n, out,
                                              c.plain_n ? c.plain_n : 1);
        const bool size_ok = out.size() == c.plain_n;
        const bool data_ok = size_ok && (c.plain_n == 0 ||
                                         std::memcmp(out.data(), c.plain, c.plain_n) == 0);
        std::printf("%-10s comp=%6zu plain=%7zu -> ok=%d size=%s data=%s\n",
                    c.name, c.n, c.plain_n, ok ? 1 : 0,
                    size_ok ? "ok" : "BAD", data_ok ? "ok" : "BAD");
        if (!ok || !size_ok || !data_ok) ++failures;
    }
    std::printf("\n%d/%zu cases failed\n", failures, kZlibRefCount);
    return failures ? 1 : 0;
}
