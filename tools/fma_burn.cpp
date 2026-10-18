// Reference "power virus": a pure AVX-512 double-precision FMA loop.
//
// Used to establish what a genuinely FPU-saturating load looks like on this
// machine (effective clock, utilisation), so the ray-trace benchmark can be
// compared against something that is known to reach the thermal limit.
//
//   fma_burn.exe [threads] [seconds]
#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 16;
    const double seconds = argc > 2 ? std::atof(argv[2]) : 60.0;

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total{0};

    auto body = [&](int) {
        // Eight independent chains so the two FMA pipes stay fed.
        __m512d a0 = _mm512_set1_pd(1.0000001);
        __m512d a1 = _mm512_set1_pd(1.0000002);
        __m512d a2 = _mm512_set1_pd(1.0000003);
        __m512d a3 = _mm512_set1_pd(1.0000004);
        __m512d a4 = _mm512_set1_pd(1.0000005);
        __m512d a5 = _mm512_set1_pd(1.0000006);
        __m512d a6 = _mm512_set1_pd(1.0000007);
        __m512d a7 = _mm512_set1_pd(1.0000008);
        const __m512d k = _mm512_set1_pd(1.00000006);
        const __m512d c = _mm512_set1_pd(-1e-9);
        uint64_t iters = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            // 16 rounds unrolled: 128 FMAs = 2048 FLOP per outer iteration.
            for (int u = 0; u < 16; ++u) {
                a0 = _mm512_fmadd_pd(a0, k, c);
                a1 = _mm512_fmadd_pd(a1, k, c);
                a2 = _mm512_fmadd_pd(a2, k, c);
                a3 = _mm512_fmadd_pd(a3, k, c);
                a4 = _mm512_fmadd_pd(a4, k, c);
                a5 = _mm512_fmadd_pd(a5, k, c);
                a6 = _mm512_fmadd_pd(a6, k, c);
                a7 = _mm512_fmadd_pd(a7, k, c);
            }
            iters += 16 * 8;
        }
        // Keep the accumulators observable so nothing is optimised away.
        const __m512d s = _mm512_add_pd(
            _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3)),
            _mm512_add_pd(_mm512_add_pd(a4, a5), _mm512_add_pd(a6, a7)));
        double out[8];
        _mm512_storeu_pd(out, s);
        double sum = 0.0;
        for (double v : out) sum += v;
        if (sum == 12345.6789) std::printf("");   // never true
        total.fetch_add(iters, std::memory_order_relaxed);
    };

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(body, i);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : pool) t.join();
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double fmas = static_cast<double>(total.load());
    const double gflops = fmas * 8.0 * 2.0 / el / 1e9;   // 8 lanes, 2 FLOP per FMA
    std::printf("threads %d, %.1f s: %.1f GFLOP/s (%.2f GFLOP/s/thread)\n",
                threads, el, gflops, gflops / threads);
    return 0;
}
