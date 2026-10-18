#include "common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#endif

namespace cst {

double now_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

namespace {

#ifdef _WIN32
std::string cpuid_brand() {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown x86-64";
    char brand[49] = {0};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand + 16 * i, regs, 16);
    }
    std::string s(brand);
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s.empty() ? "unknown x86-64" : s;
}
#endif

CpuInfo detect() {
    CpuInfo ci;
#ifdef _WIN32
    ci.brand = cpuid_brand();

    SYSTEM_INFO si{};
    GetNativeSystemInfo(&si);
    ci.logical_cores = static_cast<int>(si.dwNumberOfProcessors);

    // Physical core count: one logical processor per (package, core) pair.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len) {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                                             &len)) {
            int cores = 0;
            for (DWORD off = 0; off < len;) {
                auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
                if (e->Relationship == RelationProcessorCore) ++cores;
                off += e->Size;
            }
            ci.physical_cores = cores;
        }
    }
    if (ci.physical_cores <= 0) ci.physical_cores = ci.logical_cores;

    // Cache sizes.
    len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len) {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationCache,
                                             reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                                             &len)) {
            for (DWORD off = 0; off < len;) {
                auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
                if (e->Relationship == RelationCache) {
                    const auto& c = e->Cache;
                    if (c.Level == 2) ci.l2_bytes = std::max<uint64_t>(ci.l2_bytes, c.CacheSize);
                    if (c.Level == 3) ci.l3_bytes = std::max<uint64_t>(ci.l3_bytes, c.CacheSize);
                }
                off += e->Size;
            }
        }
    }

    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_basic = regs[0];
    __cpuid(regs, 1);
    ci.has_aes = (regs[2] & (1 << 25)) != 0;      // AES-NI
    ci.has_sha_ni = (regs[2] & (1 << 29)) != 0;   // SHA extensions
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx_hw = (regs[2] & (1 << 28)) != 0;

    uint64_t xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);
    const bool ymm_ok = osxsave && ((xcr0 & 0x6) == 0x6);
    const bool zmm_ok = osxsave && ((xcr0 & 0xE6) == 0xE6);

    if (max_basic >= 7) {
        __cpuidex(regs, 7, 0);
        const bool avx2_hw = (regs[1] & (1 << 5)) != 0;
        ci.has_avx2 = avx_hw && avx2_hw && ymm_ok;
        ci.has_fma = avx_hw && ((regs[2] & (1 << 12)) != 0) && ymm_ok;
        const bool f512 = (regs[1] & (1 << 16)) != 0;
        ci.has_avx512f = f512 && zmm_ok;
        ci.has_avx512dq = ci.has_avx512f && ((regs[1] & (1 << 17)) != 0);
        ci.has_avx512vl = ci.has_avx512f && ((regs[2] & (1 << 31)) != 0);
        ci.has_vaes = ci.has_avx2 && ((regs[2] & (1 << 9)) != 0);
    }
#else
    ci.brand = "unknown";
    ci.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    ci.physical_cores = ci.logical_cores;
#endif
    return ci;
}

} // namespace

const CpuInfo& cpu_info() {
    static const CpuInfo ci = detect();
    return ci;
}

std::string CpuInfo::isa_summary() const {
    std::string s;
    auto add = [&s](const char* n) {
        if (!s.empty()) s += "+";
        s += n;
    };
    if (has_avx512f) add("AVX-512F");
    if (has_avx512dq) add("AVX-512DQ");
    if (has_avx512vl) add("AVX-512VL");
    if (has_vaes) add("VAES");
    if (has_avx2) add("AVX2");
    if (has_fma) add("FMA3");
    if (has_aes) add("AES-NI");
    if (has_sha_ni) add("SHA-NI");
    add("SSE4.2");
    return s;
}

void run_workers(int threads, const std::function<void(int)>& body) {
    if (threads <= 1) {
        body(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    // An exception escaping a worker thread would call std::terminate and kill
    // the process without a word.  Capture the first one and rethrow it on the
    // calling thread, where the session can report it.
    std::mutex error_mutex;
    std::exception_ptr error;
    for (int i = 0; i < threads; ++i) {
        pool.emplace_back([&body, i, &error, &error_mutex] {
            try {
                body(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        });
    }
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

void pin_thread_to_processor(int logical_index) {
#ifdef _WIN32
    if (logical_index < 0 || logical_index >= 64) return; // single-group case
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << logical_index);
#else
    (void)logical_index;
#endif
}

uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_doubles(double* dst, size_t count, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < count; ++i) {
        const uint64_t r = splitmix64(s);
        // uniform in [-1, 1)
        dst[i] = (static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
    }
}

uint64_t hash_bytes(const void* data, size_t bytes, uint64_t seed) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

bool test_corruption_enabled() {
#ifdef _WIN32
    char buf[16] = {0};
    const DWORD n = GetEnvironmentVariableA("CST_TEST_CORRUPT", buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return false;
    return buf[0] != '0';
#else
    const char* v = std::getenv("CST_TEST_CORRUPT");
    return v && v[0] != '0';
#endif
}

uint64_t corrupt_if_testing(uint64_t digest) {
    return test_corruption_enabled() ? (digest ^ 1ull) : digest;
}

} // namespace cst
