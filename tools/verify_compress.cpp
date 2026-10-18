// Compress a single input size (from CST_N) and print only the compressed bytes
// as hex, so an external decoder can validate the stream.
#define main cst_bench_crypto_main_disabled
#include "../src/bench_crypto.cpp"
#undef main

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const size_t n = std::getenv("CST_N") ? static_cast<size_t>(std::atol(std::getenv("CST_N"))) : 3;
    std::vector<uint8_t> buf(n ? n : 1);
    cst::deflate::fill_text(buf.data(), buf.size(), 0x9e37ull);

    cst::deflate::Deflator d;
    std::vector<uint8_t> comp;
    d.compress(buf.data(), n, comp);

    std::printf("N=%zu LEN=%zu HEX=", n, comp.size());
    for (uint8_t c : comp) std::printf("%02x", c);
    std::printf("\n");
    return 0;
}
