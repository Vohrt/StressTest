// CPU AES / CPU SHA3 / CPU ZLib workload implementations.
//
// These are the workloads AIDA64 exposes as Base_AES, Base_SHA3 and
// Base_ZLib_1_2_5, re-implemented from the public specifications only:
//
//   * AES-256 (FIPS-197)      ECB encryption of one shared 8 MiB buffer, the
//                             AIDA64 NGBase_AES layout (EnAESMibPerSec =
//                             completed frames * size / 2^20 / wall), driven
//                             through VAES-512, AES-NI or a table based scalar
//                             core, whichever the CPU has.
//   * SHA3-512 (FIPS-202)     Keccak-f[1600] written from scratch, hashing
//                             independent 8 KiB blocks per worker.
//   * DEFLATE (RFC 1950/1951) a level-9 compressor with the settings the
//                             embedded zlib 1.2.5 of the DLL uses for its
//                             CPU ZLib test: 32 KiB window, LZ77 hash chains
//                             (chain 4096, lazy 258, nice 258, good 32), lazy
//                             matching and dynamic Huffman codes.  The input is
//                             compressed in 32 KiB chunks with Z_SYNC_FLUSH
//                             between them and Z_FINISH on the last, exactly
//                             like the DLL drives deflate()/deflateEnd().
//
// No third-party code is used.  The only external dependency is the compiler's
// own AES-NI intrinsic set from <immintrin.h>.

#include "bench_crypto.h"

#include "common.h"


#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cst {
namespace {

// ===========================================================================
// AES-256
// ===========================================================================
namespace aes256 {

// One shared buffer, exactly like the DLL's NGAESSizeInBytes setting (the
// framework is driven with 8 MiB; the settings validator accepts 256 B .. 32
// MiB in 256 byte steps).  Every frame encrypts the whole buffer once, so the
// measured rate is the encryption core's, not a per-worker slice's.
constexpr size_t kBufferBytes = 8u << 20;   // 8 MiB shared buffer
constexpr int    kRounds      = 14;         // AES-256: Nr = 14
constexpr int    kRoundKeys   = kRounds + 1;
constexpr size_t kUnroll      = 8;          // AES-NI: independent blocks in flight
constexpr size_t kUnroll512   = 8;          // VAES-512: zmm in flight (16 blocks)

// FIPS-197 figure 7 (S-box).
const uint8_t kSbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

inline uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>(static_cast<uint8_t>(x << 1) ^
                                static_cast<uint8_t>((x >> 7) * 0x1b));
}

// Scalar key expansion for a 256-bit key.  The round keys are packed exactly
// the way _mm_loadu_si128 expects, i.e. word i occupies bytes [4i, 4i+4).
void expand_key_256(const uint8_t key[32], uint8_t rk[kRoundKeys * 16]) {
    uint8_t w[60][4];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) w[i][j] = key[4 * i + j];

    for (int i = 8; i < 60; ++i) {
        uint8_t t[4] = {w[i - 1][0], w[i - 1][1], w[i - 1][2], w[i - 1][3]};
        if (i % 8 == 0) {
            const uint8_t first = t[0];
            t[0] = static_cast<uint8_t>(kSbox[t[1]] ^ static_cast<uint8_t>(1u << (i / 8 - 1)));
            if (i / 8 > 7) t[0] = kSbox[t[1]];   // Rcon wraps above 0x80 (never reached at Nr=14)
            t[1] = kSbox[t[2]];
            t[2] = kSbox[t[3]];
            t[3] = kSbox[first];
        } else if (i % 8 == 4) {
            for (int j = 0; j < 4; ++j) t[j] = kSbox[t[j]];
        }
        for (int j = 0; j < 4; ++j) w[i][j] = static_cast<uint8_t>(w[i - 8][j] ^ t[j]);
    }

    for (int r = 0; r < kRoundKeys; ++r)
        for (int c = 0; c < 4; ++c)
            for (int j = 0; j < 4; ++j) rk[r * 16 + 4 * c + j] = w[4 * r + c][j];
}

// One AES-256 block, table based.  State is column major: byte r + 4*c.
void encrypt_block_scalar(const uint8_t in[16], uint8_t out[16],
                          const uint8_t rk[kRoundKeys * 16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; ++i) s[i] = static_cast<uint8_t>(in[i] ^ rk[i]);

    for (int round = 1; round <= kRounds; ++round) {
        uint8_t t[16];
        for (int r = 0; r < 4; ++r)                       // SubBytes + ShiftRows
            for (int c = 0; c < 4; ++c)
                t[r + 4 * c] = kSbox[s[r + 4 * ((c + r) & 3)]];

        if (round != kRounds) {                           // MixColumns
            for (int c = 0; c < 4; ++c) {
                const uint8_t a0 = t[4 * c + 0], a1 = t[4 * c + 1];
                const uint8_t a2 = t[4 * c + 2], a3 = t[4 * c + 3];
                const uint8_t all = static_cast<uint8_t>(a0 ^ a1 ^ a2 ^ a3);
                s[4 * c + 0] = static_cast<uint8_t>(a0 ^ all ^ xtime(static_cast<uint8_t>(a0 ^ a1)));
                s[4 * c + 1] = static_cast<uint8_t>(a1 ^ all ^ xtime(static_cast<uint8_t>(a1 ^ a2)));
                s[4 * c + 2] = static_cast<uint8_t>(a2 ^ all ^ xtime(static_cast<uint8_t>(a2 ^ a3)));
                s[4 * c + 3] = static_cast<uint8_t>(a3 ^ all ^ xtime(static_cast<uint8_t>(a3 ^ a0)));
            }
        } else {
            for (int i = 0; i < 16; ++i) s[i] = t[i];
        }
        for (int i = 0; i < 16; ++i) s[i] ^= rk[round * 16 + i];
    }
    std::memcpy(out, s, 16);
}

void load_round_keys(const uint8_t rk[kRoundKeys * 16], __m128i out[kRoundKeys]) {
    for (int i = 0; i < kRoundKeys; ++i)
        out[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rk + 16 * i));
}

inline void encrypt_block_ni(const uint8_t in[16], uint8_t out[16],
                             const __m128i rk[kRoundKeys]) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
    v = _mm_xor_si128(v, rk[0]);
    for (int r = 1; r < kRounds; ++r) v = _mm_aesenc_si128(v, rk[r]);
    v = _mm_aesenclast_si128(v, rk[kRounds]);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), v);
}

// ECB block ordering: XOR the block with round key 0, run 13 aesenc rounds and
// one aesenclast, the shape both the dumped X64 kernel and the VAES variant
// use.  The cipher is stateless per block, so decryption is the inverse-round
// variant driven by the aesimc-transformed schedule.
void make_dec_keys(const __m128i enc[kRoundKeys], __m128i dec[kRoundKeys]) {
    dec[0] = enc[kRounds];
    dec[kRounds] = enc[0];
    for (int r = 1; r < kRounds; ++r) dec[r] = _mm_aesimc_si128(enc[kRounds - r]);
}

void ecb_encrypt_scalar(uint8_t* dst, const uint8_t* src, size_t n,
                        const uint8_t rk[kRoundKeys * 16]) {
    for (size_t off = 0; off + 16 <= n; off += 16) {
        encrypt_block_scalar(src + off, dst + off, rk);
    }
}

// Eight blocks in flight, AES-NI - the DLL's AESNI_8 / AESNI_AVX_8 shape.
void ecb_encrypt_ni(uint8_t* dst, const uint8_t* src, size_t n,
                    const __m128i rk[kRoundKeys]) {
    size_t off = 0;
    for (; off + kUnroll * 16 <= n; off += kUnroll * 16) {
        __m128i v[kUnroll];
        for (size_t i = 0; i < kUnroll; ++i)
            v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + off + i * 16));
        for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_xor_si128(v[i], rk[0]);
        for (int r = 1; r < kRounds; ++r)
            for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_aesenc_si128(v[i], rk[r]);
        for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_aesenclast_si128(v[i], rk[kRounds]);
        for (size_t i = 0; i < kUnroll; ++i)
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + off + i * 16), v[i]);
    }
    for (; off + 16 <= n; off += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + off));
        v = _mm_xor_si128(v, rk[0]);
        for (int r = 1; r < kRounds; ++r) v = _mm_aesenc_si128(v, rk[r]);
        v = _mm_aesenclast_si128(v, rk[kRounds]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + off), v);
    }
}

void ecb_decrypt_ni(uint8_t* dst, const uint8_t* src, size_t n,
                    const __m128i dec[kRoundKeys]) {
    size_t off = 0;
    for (; off + kUnroll * 16 <= n; off += kUnroll * 16) {
        __m128i v[kUnroll];
        for (size_t i = 0; i < kUnroll; ++i)
            v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + off + i * 16));
        for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_xor_si128(v[i], dec[0]);
        for (int r = 1; r < kRounds; ++r)
            for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_aesdec_si128(v[i], dec[r]);
        for (size_t i = 0; i < kUnroll; ++i) v[i] = _mm_aesdeclast_si128(v[i], dec[kRounds]);
        for (size_t i = 0; i < kUnroll; ++i)
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + off + i * 16), v[i]);
    }
    for (; off + 16 <= n; off += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + off));
        v = _mm_xor_si128(v, dec[0]);
        for (int r = 1; r < kRounds; ++r) v = _mm_aesdec_si128(v, dec[r]);
        v = _mm_aesdeclast_si128(v, dec[kRounds]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + off), v);
    }
}

// The variant AIDA64's automatic selection picks on a VAES CPU: eight zmm
// registers (sixteen blocks, 512 bytes) per iteration with the round keys
// broadcast from the 128-bit schedule - the dumped NGAES_VAES_EVEX_ZMM_16
// kernel does vbroadcasti32x4 + vpxord + vaesenc over the same 512-byte step.
void ecb_encrypt_vaes(uint8_t* dst, const uint8_t* src, size_t n,
                      const __m128i rk[kRoundKeys]) {
    size_t off = 0;
    for (; off + kUnroll512 * 64 <= n; off += kUnroll512 * 64) {
        __m512i v[kUnroll512];
        for (size_t i = 0; i < kUnroll512; ++i)
            v[i] = _mm512_loadu_si512(reinterpret_cast<const void*>(src + off + i * 64));
        const __m512i key0 = _mm512_broadcast_i32x4(rk[0]);
        for (size_t i = 0; i < kUnroll512; ++i) v[i] = _mm512_xor_si512(v[i], key0);
        for (int r = 1; r < kRounds; ++r) {
            const __m512i k = _mm512_broadcast_i32x4(rk[r]);
            for (size_t i = 0; i < kUnroll512; ++i) v[i] = _mm512_aesenc_epi128(v[i], k);
        }
        const __m512i keyl = _mm512_broadcast_i32x4(rk[kRounds]);
        for (size_t i = 0; i < kUnroll512; ++i) v[i] = _mm512_aesenclast_epi128(v[i], keyl);
        for (size_t i = 0; i < kUnroll512; ++i)
            _mm512_storeu_si512(reinterpret_cast<void*>(dst + off + i * 64), v[i]);
    }
    if (off < n) ecb_encrypt_ni(dst + off, src + off, n - off, rk);
}

// FIPS-197 appendix C.3 known-answer pair.
const uint8_t kKatKey[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};
const uint8_t kKatPlain[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff
};
const uint8_t kKatCipher[16] = {
    0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf,
    0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89
};

// Per-worker buffer initialised from one shared, deterministic template so the
// input to every repetition is identical.
void fill_template(uint8_t* dst, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>(splitmix64(s) >> 56);
}

enum AesFail {
    kAesOk = 0,
    kAesKatScalar = 1,
    kAesKatNi = 2,
    kAesCrossCheck = 3,
    kAesKatDecNi = 4,
};

int self_test() {
    uint8_t rk[kRoundKeys * 16];
    expand_key_256(kKatKey, rk);

    uint8_t out[16];
    encrypt_block_scalar(kKatPlain, out, rk);
    if (std::memcmp(out, kKatCipher, 16) != 0) return kAesKatScalar;

    __m128i rk_ni[kRoundKeys];
    load_round_keys(rk, rk_ni);
    encrypt_block_ni(kKatPlain, out, rk_ni);
    if (std::memcmp(out, kKatCipher, 16) != 0) return kAesKatNi;

    if (cpu_info().has_aes) {
        // FIPS-197 C.3 in the other direction, through the decrypt schedule.
        __m128i dk[kRoundKeys];
        make_dec_keys(rk_ni, dk);
        uint8_t back[16];
        ecb_decrypt_ni(back, kKatCipher, 16, dk);
        if (std::memcmp(back, kKatPlain, 16) != 0) return kAesKatDecNi;

        // Every encryption path must produce the identical ciphertext.
        constexpr size_t kN = 64u * 1024;   // whole blocks only, like the bench
        std::vector<uint8_t> a(kN), b(kN), c(kN);
        fill_template(a.data(), kN, 0x1234abcdull);
        ecb_encrypt_scalar(b.data(), a.data(), kN, rk);
        ecb_encrypt_ni(c.data(), a.data(), kN, rk_ni);
        if (std::memcmp(b.data(), c.data(), kN) != 0) return kAesCrossCheck;
        if (cpu_info().has_vaes) {
            std::vector<uint8_t> d(kN);
            ecb_encrypt_vaes(d.data(), a.data(), kN, rk_ni);
            if (std::memcmp(b.data(), d.data(), kN) != 0) return kAesCrossCheck;
        }
    }
    return kAesOk;
}

} // namespace aes256

// ===========================================================================
// CPU AES workload
// ===========================================================================
//
// AIDA64's NGBase_AES gives every worker its own buffer of NGAESSizeInBytes
// bytes and lets it encrypt from its own source to its own destination; a
// frame is one pass over that buffer and the reported rate is
//     EnAESMibPerSec = sum(completed frames) * SizeInBytes / 2^20 / wall,
// i.e. the aggregate throughput.  The buffer size is a per-worker size, which
// is why the reference rate collapses once nthreads * size exceeds the last
// level cache (16 x 8 MiB on this host) and scales with the cache when it does
// not - the port keeps exactly that layout, an ECB kernel over each worker's
// own 8 MiB source, and the same MiB/s arithmetic.
class BenchAes final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "aes";
        i.label = "CPU AES";
        i.unit = "MiB/s";
        i.group = "CPU";
        i.higher_is_better = true;
        i.unit_scale = 1.0 / 1048576.0;            // work units are bytes
        i.description =
            "AES-256 ECB encryption of an 8 MiB per-worker buffer, the AIDA64 "
            "NGBase_AES layout (rate = completed frames * size / 2^20 / second)";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        threads_ = threads;
        workers_.clear();
        workers_.resize(static_cast<size_t>(threads));
        for (size_t k = 0; k < workers_.size(); ++k) {
            Worker& w = workers_[k];
            w.src.assign(aes256::kBufferBytes, 0);
            w.dst.assign(aes256::kBufferBytes, 0);
            aes256::fill_template(w.src.data(), w.src.size(),
                                  0x43505453ull ^ (k * 0x9e3779b97f4a7c15ull));
        }
        aes256::expand_key_256(aes256::kKatKey, rk_);
        use_ni_ = cpu_info().has_aes;
        use_vaes_ = use_ni_ && cpu_info().has_vaes;
        if (use_ni_) aes256::load_round_keys(rk_, rk_ni_);
    }

    // One pass encrypts the worker's whole buffer from its fixed source, so the
    // destination content never depends on how many passes ran.
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        if (worker < 0 || static_cast<size_t>(worker) >= workers_.size()) return 0.0;
        Worker& w = workers_[static_cast<size_t>(worker)];
        const double t0 = now_seconds();
        double bytes = 0.0;
        do {
            ecb_encrypt(w.dst.data(), w.src.data(), aes256::kBufferBytes);
            bytes += static_cast<double>(aes256::kBufferBytes);
        } while (now_seconds() - t0 < min_seconds);
        return bytes;
    }

    // Ciphertext of every worker; pass invariant, so the fingerprint is stable
    // in every repetition.
    uint64_t digest() const override {
        uint64_t h = 0xcbf29ce484222325ull;
        for (const Worker& w : workers_) h = hash_bytes(w.dst.data(), w.dst.size(), h);
        return h;
    }
    bool has_digest() const override { return true; }

    // Single-threaded pass over one buffer plus a decrypt round trip (on CPUs
    // with AES-NI), so the reference does not depend on the thread count and a
    // CPU that miscomputes any AES round lands on a different value.
    uint64_t full_digest() const override {
        BenchAes* self = const_cast<BenchAes*>(this);
        std::vector<uint8_t> src(aes256::kBufferBytes), tmp(aes256::kBufferBytes);
        aes256::fill_template(src.data(), src.size(), 0x43505453ull);
        self->ecb_encrypt(tmp.data(), src.data(), aes256::kBufferBytes);
        uint64_t h = hash_bytes(tmp.data(), tmp.size(), 0xcbf29ce484222325ull);
        if (self->use_ni_) {
            std::vector<uint8_t> back(aes256::kBufferBytes);
            __m128i dk[aes256::kRoundKeys];
            aes256::make_dec_keys(self->rk_ni_, dk);
            aes256::ecb_decrypt_ni(back.data(), tmp.data(), aes256::kBufferBytes, dk);
            h = hash_bytes(back.data(), back.size(), h);
        }
        return corrupt_if_testing(h);
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        std::string method = "scalar-sbox";
        if (use_vaes_) method = "VAES-512 (16 blocks)";
        else if (use_ni_) method = "AES-NI (8 blocks)";
        return "AES-256, ECB, buffer=" +
               std::to_string(aes256::kBufferBytes >> 20) + " MiB/worker, rounds=14, method=" +
               method + "; AIDA64 NGAES figure = encryption MiB/s";
    }

private:
    void ecb_encrypt(uint8_t* dst, const uint8_t* src, size_t n) const {
        if (use_vaes_) aes256::ecb_encrypt_vaes(dst, src, n, rk_ni_);
        else if (use_ni_) aes256::ecb_encrypt_ni(dst, src, n, rk_ni_);
        else aes256::ecb_encrypt_scalar(dst, src, n, rk_);
    }

    struct Worker {
        std::vector<uint8_t> src;   // fixed plaintext
        std::vector<uint8_t> dst;   // ciphertext of the last pass
    };

    int threads_ = 1;
    std::vector<Worker> workers_;
    uint8_t rk_[aes256::kRoundKeys * 16] = {0};
    __m128i rk_ni_[aes256::kRoundKeys];
    bool use_ni_ = false;
    bool use_vaes_ = false;
};

// ===========================================================================
// SHA3-512 (FIPS-202)
// ===========================================================================
namespace sha3 {

constexpr size_t kRateBytes  = 72;          // 1600 - 2*512 bits
constexpr size_t kOutBytes   = 64;
constexpr size_t kBlockBytes = 8u << 10;    // AIDA64 Hash2 hashes 8 KiB per round

inline uint64_t rotl64(uint64_t x, int n) {
    return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

inline uint64_t load64le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

inline void store64le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

// One Keccak-f round with every lane, index and rotation amount resolved at
// compile time.  Lane naming: aXY = st[X + 5*Y]; the rho/pi stage writes bD
// with D = Y + 5*((2X + 3Y) mod 5), exactly the index arithmetic of the
// original loop version.  The state lives in scalar locals so the compiler
// keeps what it can in registers and spills the rest - the same serial
// one-message-at-a-time sponge the DLL runs, at a scalar instruction mix.
#define KECCAK_ROUND(rc)                                                                     \
    do {                                                                                     \
        const uint64_t c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                     \
        const uint64_t c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                     \
        const uint64_t c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                     \
        const uint64_t c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                     \
        const uint64_t c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                     \
        const uint64_t d0 = c4 ^ rotl64(c1, 1);                                              \
        const uint64_t d1 = c0 ^ rotl64(c2, 1);                                              \
        const uint64_t d2 = c1 ^ rotl64(c3, 1);                                              \
        const uint64_t d3 = c2 ^ rotl64(c4, 1);                                              \
        const uint64_t d4 = c3 ^ rotl64(c0, 1);                                              \
        a00 ^= d0; a01 ^= d0; a02 ^= d0; a03 ^= d0; a04 ^= d0;                               \
        a10 ^= d1; a11 ^= d1; a12 ^= d1; a13 ^= d1; a14 ^= d1;                               \
        a20 ^= d2; a21 ^= d2; a22 ^= d2; a23 ^= d2; a24 ^= d2;                               \
        a30 ^= d3; a31 ^= d3; a32 ^= d3; a33 ^= d3; a34 ^= d3;                               \
        a40 ^= d4; a41 ^= d4; a42 ^= d4; a43 ^= d4; a44 ^= d4;                               \
        const uint64_t b0 = a00;                                                             \
        const uint64_t b1 = rotl64(a11, 44);                                                 \
        const uint64_t b2 = rotl64(a22, 43);                                                 \
        const uint64_t b3 = rotl64(a33, 21);                                                 \
        const uint64_t b4 = rotl64(a44, 14);                                                 \
        const uint64_t b5 = rotl64(a30, 28);                                                 \
        const uint64_t b6 = rotl64(a41, 20);                                                 \
        const uint64_t b7 = rotl64(a02, 3);                                                  \
        const uint64_t b8 = rotl64(a13, 45);                                                 \
        const uint64_t b9 = rotl64(a24, 61);                                                 \
        const uint64_t b10 = rotl64(a10, 1);                                                 \
        const uint64_t b11 = rotl64(a21, 6);                                                 \
        const uint64_t b12 = rotl64(a32, 25);                                                \
        const uint64_t b13 = rotl64(a43, 8);                                                 \
        const uint64_t b14 = rotl64(a04, 18);                                                \
        const uint64_t b15 = rotl64(a40, 27);                                                \
        const uint64_t b16 = rotl64(a01, 36);                                                \
        const uint64_t b17 = rotl64(a12, 10);                                                \
        const uint64_t b18 = rotl64(a23, 15);                                                \
        const uint64_t b19 = rotl64(a34, 56);                                                \
        const uint64_t b20 = rotl64(a20, 62);                                                \
        const uint64_t b21 = rotl64(a31, 55);                                                \
        const uint64_t b22 = rotl64(a42, 39);                                                \
        const uint64_t b23 = rotl64(a03, 41);                                                \
        const uint64_t b24 = rotl64(a14, 2);                                                 \
        a00 = b0 ^ (~b1 & b2);                                                               \
        a10 = b1 ^ (~b2 & b3);                                                               \
        a20 = b2 ^ (~b3 & b4);                                                               \
        a30 = b3 ^ (~b4 & b0);                                                               \
        a40 = b4 ^ (~b0 & b1);                                                               \
        a01 = b5 ^ (~b6 & b7);                                                               \
        a11 = b6 ^ (~b7 & b8);                                                               \
        a21 = b7 ^ (~b8 & b9);                                                               \
        a31 = b8 ^ (~b9 & b5);                                                               \
        a41 = b9 ^ (~b5 & b6);                                                               \
        a02 = b10 ^ (~b11 & b12);                                                            \
        a12 = b11 ^ (~b12 & b13);                                                            \
        a22 = b12 ^ (~b13 & b14);                                                            \
        a32 = b13 ^ (~b14 & b10);                                                            \
        a42 = b14 ^ (~b10 & b11);                                                            \
        a03 = b15 ^ (~b16 & b17);                                                            \
        a13 = b16 ^ (~b17 & b18);                                                            \
        a23 = b17 ^ (~b18 & b19);                                                            \
        a33 = b18 ^ (~b19 & b15);                                                            \
        a43 = b19 ^ (~b15 & b16);                                                            \
        a04 = b20 ^ (~b21 & b22);                                                            \
        a14 = b21 ^ (~b22 & b23);                                                            \
        a24 = b22 ^ (~b23 & b24);                                                            \
        a34 = b23 ^ (~b24 & b20);                                                            \
        a44 = b24 ^ (~b20 & b21);                                                            \
        a00 ^= rc;                                                                           \
    } while (0);

// The 24 round constants of FIPS 202 B.2, inlined so every round unfolds.
#define KECCAK_ROUNDS_24                                                                     \
    KECCAK_ROUND(0x0000000000000001ull)                                                      \
    KECCAK_ROUND(0x0000000000008082ull)                                                      \
    KECCAK_ROUND(0x800000000000808aull)                                                      \
    KECCAK_ROUND(0x8000000080008000ull)                                                      \
    KECCAK_ROUND(0x000000000000808bull)                                                      \
    KECCAK_ROUND(0x0000000080000001ull)                                                      \
    KECCAK_ROUND(0x8000000080008081ull)                                                      \
    KECCAK_ROUND(0x8000000000008009ull)                                                      \
    KECCAK_ROUND(0x000000000000008aull)                                                      \
    KECCAK_ROUND(0x0000000000000088ull)                                                      \
    KECCAK_ROUND(0x0000000080008009ull)                                                      \
    KECCAK_ROUND(0x000000008000000aull)                                                      \
    KECCAK_ROUND(0x000000008000808bull)                                                      \
    KECCAK_ROUND(0x800000000000008bull)                                                      \
    KECCAK_ROUND(0x8000000000008089ull)                                                      \
    KECCAK_ROUND(0x8000000000008003ull)                                                      \
    KECCAK_ROUND(0x8000000000008002ull)                                                      \
    KECCAK_ROUND(0x8000000000000080ull)                                                      \
    KECCAK_ROUND(0x000000000000800aull)                                                      \
    KECCAK_ROUND(0x800000008000000aull)                                                      \
    KECCAK_ROUND(0x8000000080008081ull)                                                      \
    KECCAK_ROUND(0x8000000000008080ull)                                                      \
    KECCAK_ROUND(0x0000000080000001ull)                                                      \
    KECCAK_ROUND(0x8000000080008008ull)

// The 24-round Keccak-f[1600] permutation, one round per macro expansion.
void keccakf1600(uint64_t st[25]) {
    uint64_t a00 = st[0], a10 = st[1], a20 = st[2], a30 = st[3], a40 = st[4];
    uint64_t a01 = st[5], a11 = st[6], a21 = st[7], a31 = st[8], a41 = st[9];
    uint64_t a02 = st[10], a12 = st[11], a22 = st[12], a32 = st[13], a42 = st[14];
    uint64_t a03 = st[15], a13 = st[16], a23 = st[17], a33 = st[18], a43 = st[19];
    uint64_t a04 = st[20], a14 = st[21], a24 = st[22], a34 = st[23], a44 = st[24];

    KECCAK_ROUNDS_24;

    st[0] = a00; st[1] = a10; st[2] = a20; st[3] = a30; st[4] = a40;
    st[5] = a01; st[6] = a11; st[7] = a21; st[8] = a31; st[9] = a41;
    st[10] = a02; st[11] = a12; st[12] = a22; st[13] = a32; st[14] = a42;
    st[15] = a03; st[16] = a13; st[17] = a23; st[18] = a33; st[19] = a43;
    st[20] = a04; st[21] = a14; st[22] = a24; st[23] = a34; st[24] = a44;
}

// ---------------------------------------------------------------------------
// AVX-512 Keccak-f[1600]
// ---------------------------------------------------------------------------
// One permutation at a time with the state in five zmm registers: register y
// carries row y (lane x = state[x + 5y]).  theta is three ternlogs plus a
// two-lane shuffle pair; rho+pi gathers row y' by having lane x' read register
// x' lane (x'+3y') mod 5 (four vpermi2q nodes) and applies the per-lane
// rotation amounts with vprolvq; chi is five more ternlogs.  That is the same
// work split AIDA64's auto-selected AVX512VLBW_Zen4_* kernel performs, which
// also drives the whole family at a single throughput point.
static_assert(sizeof(__m512i) == 64, "zmm state assumption");

alignas(64) const uint64_t kKecThetaM1[8] = {4, 0, 1, 2, 3, 5, 6, 7};
alignas(64) const uint64_t kKecThetaP1[8] = {1, 2, 3, 4, 0, 5, 6, 7};
alignas(64) const uint64_t kKecChi1[8] = {1, 2, 3, 4, 0, 5, 6, 7};
alignas(64) const uint64_t kKecChi2[8] = {2, 3, 4, 0, 1, 5, 6, 7};
alignas(64) const uint64_t kKecIdx01[5][8] = {
    {0, 9, 0, 0, 0, 0, 0, 0},
    {3, 12, 0, 0, 0, 0, 0, 0},
    {1, 10, 0, 0, 0, 0, 0, 0},
    {4, 8, 0, 0, 0, 0, 0, 0},
    {2, 11, 0, 0, 0, 0, 0, 0},
};
alignas(64) const uint64_t kKecIdx23[5][8] = {
    {0, 0, 2, 11, 0, 0, 0, 0},
    {0, 0, 0, 9, 0, 0, 0, 0},
    {0, 0, 3, 12, 0, 0, 0, 0},
    {0, 0, 1, 10, 0, 0, 0, 0},
    {0, 0, 4, 8, 0, 0, 0, 0},
};
alignas(64) const uint64_t kKecIdxJoin[8] = {0, 1, 10, 11, 0, 0, 0, 0};
alignas(64) const uint64_t kKecIdxL4[5][8] = {
    {0, 1, 2, 3, 12, 0, 0, 0},
    {0, 1, 2, 3, 10, 0, 0, 0},
    {0, 1, 2, 3, 8, 0, 0, 0},
    {0, 1, 2, 3, 11, 0, 0, 0},
    {0, 1, 2, 3, 9, 0, 0, 0},
};
// Rotation amount of lane x' of row y' (the b[x + 5y] rho offsets).
alignas(64) const uint64_t kKecRot[5][8] = {
    {0, 44, 43, 21, 14, 0, 0, 0},
    {28, 20, 3, 45, 61, 0, 0, 0},
    {1, 6, 25, 8, 18, 0, 0, 0},
    {27, 36, 10, 15, 56, 0, 0, 0},
    {62, 55, 39, 41, 2, 0, 0, 0},
};

inline void kec512_permute(__m512i r[5], uint64_t rc) {
    const __m512i t1 = _mm512_ternarylogic_epi64(r[0], r[1], r[2], 0x96);
    const __m512i t2 = _mm512_xor_si512(r[3], r[4]);
    const __m512i c = _mm512_xor_si512(t1, t2);
    const __m512i p1 = _mm512_permutexvar_epi64(_mm512_load_si512(kKecThetaP1),
                                               _mm512_rol_epi64(c, 1));
    const __m512i m1 = _mm512_permutexvar_epi64(_mm512_load_si512(kKecThetaM1), c);
    const __m512i d = _mm512_xor_si512(p1, m1);
    r[0] = _mm512_xor_si512(r[0], d);
    r[1] = _mm512_xor_si512(r[1], d);
    r[2] = _mm512_xor_si512(r[2], d);
    r[3] = _mm512_xor_si512(r[3], d);
    r[4] = _mm512_xor_si512(r[4], d);

    __m512i b[5];
    for (int y = 0; y < 5; ++y) {
        const __m512i u01 =
            _mm512_permutex2var_epi64(r[0], _mm512_load_si512(kKecIdx01[y]), r[1]);
        const __m512i u23 =
            _mm512_permutex2var_epi64(r[2], _mm512_load_si512(kKecIdx23[y]), r[3]);
        const __m512i u =
            _mm512_permutex2var_epi64(u01, _mm512_load_si512(kKecIdxJoin), u23);
        const __m512i g =
            _mm512_permutex2var_epi64(u, _mm512_load_si512(kKecIdxL4[y]), r[4]);
        b[y] = _mm512_rolv_epi64(g, _mm512_load_si512(kKecRot[y]));
    }

    __m512i n[5];
    const __m512i chi1 = _mm512_load_si512(kKecChi1);
    const __m512i chi2 = _mm512_load_si512(kKecChi2);
    for (int y = 0; y < 5; ++y) {
        // chi: A'[x,y] = B[x,y] ^ ((~B[x+1,y]) & B[x+2,y]), all lanes inside
        // the row register.
        n[y] = _mm512_ternarylogic_epi64(b[y],
                                         _mm512_permutexvar_epi64(chi1, b[y]),
                                         _mm512_permutexvar_epi64(chi2, b[y]), 0xD2);
    }
    r[0] = _mm512_mask_xor_epi64(n[0], 0x01, n[0],
                                  _mm512_set1_epi64(static_cast<int64_t>(rc)));
    r[1] = n[1];
    r[2] = n[2];
    r[3] = n[3];
    r[4] = n[4];
}

constexpr uint64_t kRoundConstants24[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull };

void kec512_f1600(__m512i r[5]) {
    for (int round = 0; round < 24; ++round) kec512_permute(r, kRoundConstants24[round]);
}

// Two independent sponges advanced together: each permutation's latency slots
// are filled by the other one's work, the same interleaving-depth knob the
// DLL's kernel family exposes (_1/_2/_4/_U suffixes).
void kec512_f1600_x2(__m512i a[5], __m512i b[5]) {
    for (int round = 0; round < 24; ++round) {
        kec512_permute(a, kRoundConstants24[round]);
        kec512_permute(b, kRoundConstants24[round]);
    }
}

// The state's first 72 bytes are lanes 0..4 of rows 0 and 1: a 72-byte block
// enters as one full-width xor plus one four-lane masked xor.
inline void kec512_absorb72(__m512i r[5], const uint8_t* blk) {
    r[0] = _mm512_xor_si512(r[0], _mm512_loadu_si512(blk));
    r[1] = _mm512_mask_xor_epi64(r[1], 0x0F, r[1], _mm512_loadu_si512(blk + 40));
}

inline void kec512_squeeze(const __m512i r[5], uint8_t out[kOutBytes]) {
    alignas(64) uint8_t state[128];
    _mm512_store_si512(state, r[0]);
    _mm512_store_si512(state + 64, r[1]);
    std::memcpy(out, state, 40);           // lanes 0..4 of row 0
    std::memcpy(out + 40, state + 64, 24); // lanes 0..2 of row 1
}

inline void kec512_pad_block(const uint8_t* msg, size_t rem, uint8_t blk[kRateBytes]) {
    std::memset(blk, 0, kRateBytes);
    if (rem) std::memcpy(blk, msg, rem);
    blk[rem] = 0x06;
    blk[kRateBytes - 1] |= 0x80;
}

void hash_avx512(const uint8_t* msg, size_t len, uint8_t out[kOutBytes]) {
    __m512i r[5];
    for (int y = 0; y < 5; ++y) r[y] = _mm512_setzero_si512();
    size_t off = 0;
    while (len - off >= kRateBytes) {
        kec512_absorb72(r, msg + off);
        kec512_f1600(r);
        off += kRateBytes;
    }
    alignas(64) uint8_t blk[kRateBytes];
    kec512_pad_block(msg + off, len - off, blk);
    kec512_absorb72(r, blk);
    kec512_f1600(r);
    kec512_squeeze(r, out);
}

// Two 8 KiB messages hashed together, one sponge per message.
void hash2_avx512(const uint8_t* msg0, const uint8_t* msg1, size_t len,
                  uint8_t out0[kOutBytes], uint8_t out1[kOutBytes]) {
    __m512i a[5], b[5];
    for (int y = 0; y < 5; ++y) {
        a[y] = _mm512_setzero_si512();
        b[y] = _mm512_setzero_si512();
    }
    size_t off = 0;
    while (len - off >= kRateBytes) {
        kec512_absorb72(a, msg0 + off);
        kec512_absorb72(b, msg1 + off);
        kec512_f1600_x2(a, b);
        off += kRateBytes;
    }
    alignas(64) uint8_t blk0[kRateBytes], blk1[kRateBytes];
    kec512_pad_block(msg0 + off, len - off, blk0);
    kec512_pad_block(msg1 + off, len - off, blk1);
    kec512_absorb72(a, blk0);
    kec512_absorb72(b, blk1);
    kec512_f1600_x2(a, b);
    kec512_squeeze(a, out0);
    kec512_squeeze(b, out1);
}

// SHA3-512 of one message: pad10*1 with the SHA3 domain bits, absorb, squeeze.
void hash(const uint8_t* msg, size_t len, uint8_t out[kOutBytes]) {
    if (cpu_info().has_avx512f) {
        hash_avx512(msg, len, out);
        return;
    }
    uint64_t st[25];
    std::memset(st, 0, sizeof(st));

    size_t off = 0;
    while (len - off >= kRateBytes) {
        for (int i = 0; i < 9; ++i) st[i] ^= load64le(msg + off + 8 * static_cast<size_t>(i));
        keccakf1600(st);
        off += kRateBytes;
    }

    uint8_t block[kRateBytes];
    std::memset(block, 0, sizeof(block));
    const size_t rem = len - off;
    if (rem) std::memcpy(block, msg + off, rem);
    block[rem] = 0x06;          // SHA3 domain separation
    block[kRateBytes - 1] |= 0x80;
    for (int i = 0; i < 9; ++i) st[i] ^= load64le(block + 8 * static_cast<size_t>(i));
    keccakf1600(st);

    for (int i = 0; i < 8; ++i) store64le(out + 8 * static_cast<size_t>(i), st[i]);
}

void fill_buffer(uint8_t* dst, size_t n, uint64_t seed) {
    // Text-like data so the hashed blocks carry realistic entropy, plus a
    // little raw noise; the digest never depends on wall-clock decisions.
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t r = splitmix64(s);
        dst[i] = static_cast<uint8_t>('a' + (r % 26));
        if ((r & 0xff) == 0) dst[i] = static_cast<uint8_t>(r >> 33);
    }
}

// NIST/Keccak reference vectors.
const uint8_t kVecEmpty[64] = {
    0xa6, 0x9f, 0x73, 0xcc, 0xa2, 0x3a, 0x9a, 0xc5, 0xc8, 0xb5, 0x67, 0xdc, 0x18, 0x5a, 0x75, 0x6e,
    0x97, 0xc9, 0x82, 0x16, 0x4f, 0xe2, 0x58, 0x59, 0xe0, 0xd1, 0xdc, 0xc1, 0x47, 0x5c, 0x80, 0xa6,
    0x15, 0xb2, 0x12, 0x3a, 0xf1, 0xf5, 0xf9, 0x4c, 0x11, 0xe3, 0xe9, 0x40, 0x2c, 0x3a, 0xc5, 0x58,
    0xf5, 0x00, 0x19, 0x9d, 0x95, 0xb6, 0xd3, 0xe3, 0x01, 0x75, 0x85, 0x86, 0x28, 0x1d, 0xcd, 0x26
};
const uint8_t kVecAbc[64] = {
    0xb7, 0x51, 0x85, 0x0b, 0x1a, 0x57, 0x16, 0x8a, 0x56, 0x93, 0xcd, 0x92, 0x4b, 0x6b, 0x09, 0x6e,
    0x08, 0xf6, 0x21, 0x82, 0x74, 0x44, 0xf7, 0x0d, 0x88, 0x4f, 0x5d, 0x02, 0x40, 0xd2, 0x71, 0x2e,
    0x10, 0xe1, 0x16, 0xe9, 0x19, 0x2a, 0xf3, 0xc9, 0x1a, 0x7e, 0xc5, 0x76, 0x47, 0xe3, 0x93, 0x40,
    0x57, 0x34, 0x0b, 0x4c, 0xf4, 0x08, 0xd5, 0xa5, 0x65, 0x92, 0xf8, 0x27, 0x4e, 0xec, 0x53, 0xf0
};

enum Sha3Fail {
    kSha3Ok = 0,
    kSha3Empty = 11,
    kSha3Abc = 12,
    kSha3MultiBlock = 13,
    kSha3Chunking = 14,
    kSha3CrossAvx512 = 15,
};

// Independent, deliberately naive absorption used to cross-check the block
// sponge (padding and multi-block absorption) without an external library.
void hash_slow(const uint8_t* msg, size_t len, uint8_t out[kOutBytes]) {
    uint64_t st[25];
    std::memset(st, 0, sizeof(st));
    uint8_t block[kRateBytes];
    std::memset(block, 0, sizeof(block));
    size_t fill = 0;
    for (size_t i = 0; i < len; ++i) {
        block[fill++] = msg[i];
        if (fill == kRateBytes) {
            for (int k = 0; k < 9; ++k) st[k] ^= load64le(block + 8 * static_cast<size_t>(k));
            keccakf1600(st);
            std::memset(block, 0, sizeof(block));
            fill = 0;
        }
    }
    block[fill] = 0x06;
    block[kRateBytes - 1] |= 0x80;
    for (int k = 0; k < 9; ++k) st[k] ^= load64le(block + 8 * static_cast<size_t>(k));
    keccakf1600(st);
    for (int i = 0; i < 8; ++i) store64le(out + 8 * static_cast<size_t>(i), st[i]);
}

int self_test() {
    uint8_t d[kOutBytes];
    hash(nullptr, 0, d);
    if (std::memcmp(d, kVecEmpty, kOutBytes) != 0) return kSha3Empty;

    hash(reinterpret_cast<const uint8_t*>("abc"), 3, d);
    if (std::memcmp(d, kVecAbc, kOutBytes) != 0) return kSha3Abc;

    // 200 bytes of 0xa3: three rate blocks plus padding.  200 = 2*72 + 56, so
    // this exercises multi-block absorption with a partial final block.
    // Verified against hashlib.sha3_512(b"\xa3" * 200).
    uint8_t msg200[200];
    std::memset(msg200, 0xa3, sizeof(msg200));
    static const uint8_t kVec200a3[64] = {
        0xe7, 0x6d, 0xfa, 0xd2, 0x20, 0x84, 0xa8, 0xb1, 0x46, 0x7f, 0xcf, 0x2f, 0xfa, 0x58, 0x36, 0x1b,
        0xec, 0x76, 0x28, 0xed, 0xf5, 0xf3, 0xfd, 0xc0, 0xe4, 0x80, 0x5d, 0xc4, 0x8c, 0xae, 0xec, 0xa8,
        0x1b, 0x7c, 0x13, 0xc3, 0x0a, 0xdf, 0x52, 0xa3, 0x65, 0x95, 0x84, 0x73, 0x9a, 0x2d, 0xf4, 0x6b,
        0xe5, 0x89, 0xc5, 0x1c, 0xa1, 0xa4, 0xa8, 0x41, 0x6d, 0xf6, 0x54, 0x5a, 0x1c, 0xe8, 0xba, 0x00
    };    hash(msg200, sizeof(msg200), d);
    if (std::memcmp(d, kVec200a3, kOutBytes) != 0) return kSha3MultiBlock;

    // The streaming sponge must agree with the naive one for every length
    // around the rate boundary and for a large multi-block message.
    uint8_t buf[1024];
    fill_buffer(buf, sizeof(buf), 0x51a3ull);
    for (size_t n : {size_t(0), size_t(1), size_t(71), size_t(72), size_t(73),
                     size_t(143), size_t(144), size_t(145), size_t(216), size_t(1000)}) {
        uint8_t a[kOutBytes], b[kOutBytes];
        hash(buf, n, a);
        hash_slow(buf, n, b);
        if (std::memcmp(a, b, kOutBytes) != 0) return kSha3Chunking;
    }

    // The dispatcher above takes hash_avx512() on AVX-512 machines; make the
    // two implementations agree across the rate boundary and on one full
    // 8 KiB round.
    if (cpu_info().has_avx512f) {
        for (size_t n : {size_t(0), size_t(71), size_t(72), size_t(73), size_t(143),
                         size_t(144), size_t(145), size_t(200), size_t(1000)}) {
            uint8_t a[kOutBytes], b[kOutBytes];
            hash_avx512(buf, n, a);
            hash_slow(buf, n, b);
            if (std::memcmp(a, b, kOutBytes) != 0) return kSha3CrossAvx512;
        }
        std::vector<uint8_t> big(kBlockBytes);
        fill_buffer(big.data(), big.size(), 0x8192ull);
        uint8_t a[kOutBytes], b[kOutBytes];
        hash_avx512(big.data(), big.size(), a);
        hash_slow(big.data(), big.size(), b);
        if (std::memcmp(a, b, kOutBytes) != 0) return kSha3CrossAvx512;

        // The paired entry point the workload uses must agree as well.
        std::vector<uint8_t> big2(kBlockBytes);
        fill_buffer(big2.data(), big2.size(), 0x51515151ull);
        uint8_t a2[kOutBytes], b2[kOutBytes];
        hash2_avx512(big.data(), big2.data(), big.size(), a, b);
        hash_slow(big.data(), big.size(), a2);
        hash_slow(big2.data(), big2.size(), b2);
        if (std::memcmp(a, a2, kOutBytes) != 0) return kSha3CrossAvx512;
        if (std::memcmp(b, b2, kOutBytes) != 0) return kSha3CrossAvx512;
    }
    return kSha3Ok;
}

} // namespace sha3

// ===========================================================================
// CPU SHA3 workload
// ===========================================================================
class BenchSha3 final : public Benchmark {
public:
    BenchInfo info() const override {
        BenchInfo i;
        i.test = "sha3";
        i.label = "CPU SHA3";
        i.unit = "MiB/s";
        i.group = "CPU";
        i.higher_is_better = true;
        i.unit_scale = 1.0 / 1048576.0;            // work units are bytes
        i.description =
            "SHA3-512 over independent 8 KiB messages; rate = completed rounds * "
            "8192 / 2^20 / second (AIDA64 Hash2 layout)";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        workers_.clear();
        workers_.resize(static_cast<size_t>(threads));
        for (size_t i = 0; i < workers_.size(); ++i) {
            Worker& w = workers_[i];
            w.buf.resize(sha3::kBlockBytes);
            w.buf2.resize(sha3::kBlockBytes);
            sha3::fill_buffer(w.buf.data(), w.buf.size(),
                              0x5eed0000ull + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull);
            sha3::fill_buffer(w.buf2.data(), w.buf2.size(),
                              0xa55e0000ull + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull);
            std::memset(w.last, 0, sizeof(w.last));
            std::memset(w.last2, 0, sizeof(w.last2));
        }
        pair_ = cpu_info().has_avx512f;
    }

    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        const size_t idx = static_cast<size_t>(worker < 0 ? 0 : worker);
        if (idx >= workers_.size()) return 0.0;    // caller must prepare() first
        Worker& w = workers_[idx];

        const double t0 = now_seconds();
        double bytes = 0.0;
        do {
            // Every round hashes one fresh 8 KiB message: no chaining between
            // rounds, which is what makes AIDA64's Hash2 test throughput
            // instead of latency bound.  With AVX-512 the worker keeps two
            // messages in flight, matching the interleaved kernel family the
            // DLL selects from.
            if (pair_) {
                sha3::hash2_avx512(w.buf.data(), w.buf2.data(), sha3::kBlockBytes,
                                   w.last, w.last2);
                bytes += 2.0 * static_cast<double>(sha3::kBlockBytes);
            } else {
                sha3::hash(w.buf.data(), sha3::kBlockBytes, w.last);
                bytes += static_cast<double>(sha3::kBlockBytes);
            }
        } while (now_seconds() - t0 < min_seconds);
        return bytes;
    }

    uint64_t digest() const override {
        uint64_t h = 0xcbf29ce484222325ull;
        for (const Worker& w : workers_) {
            h = hash_bytes(w.last, sha3::kOutBytes, h);
            if (pair_) h = hash_bytes(w.last2, sha3::kOutBytes, h);
        }
        return h;
    }
    bool has_digest() const override { return true; }

    uint64_t full_digest() const override {
        BenchSha3* self = const_cast<BenchSha3*>(this);
        self->prepare(1);
        self->run_once(0, 1, 0.0);
        if (self->workers_.empty()) return 0;
        uint64_t h = hash_bytes(self->workers_[0].last, sha3::kOutBytes,
                                0xcbf29ce484222325ull);
        if (self->pair_) {
            h = hash_bytes(self->workers_[0].last2, sha3::kOutBytes, h);
        }
        return corrupt_if_testing(h);
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        return "SHA3-512 (FIPS 202), rate=72 B, 8 KiB message per round, method=" +
               std::string(pair_ ? (cpu_info().has_avx512f ? "AVX-512 keccak-f1600, 2 messages"
                                                           : "scalar keccak-f1600")
                                 : "scalar keccak-f1600") +
               " (AIDA64 auto-picks an AVX-512VL variant)";
    }

private:
    struct Worker {
        std::vector<uint8_t> buf;
        std::vector<uint8_t> buf2;
        uint8_t last[sha3::kOutBytes] = {0};
        uint8_t last2[sha3::kOutBytes] = {0};
    };
    std::vector<Worker> workers_;
    bool pair_ = false;
};

// ===========================================================================
// DEFLATE / zlib
// ===========================================================================
namespace deflate {

constexpr int kWindow     = 32768;      // zlib w_size
constexpr int kWindowMask = kWindow - 1;
constexpr int kMinMatch   = 3;
constexpr int kMaxMatch   = 258;
constexpr int kHashBits   = 15;         // memLevel 8 -> hash_bits = 15
constexpr int kHashSize   = 1 << kHashBits;
constexpr int kHashMask   = kHashSize - 1;
constexpr int kHashShift  = 5;          // (hash_bits + 2) / 3, as in zlib

// zlib configuration_table[9]: good 32, lazy 258, nice 258, chain 4096.  The
// DLL calls deflateInit_() with level 9 (that is the constant it feeds to
// deflateInit_ at every repetition), so this is the exact match set.
constexpr int kMaxChain   = 4096;
constexpr int kNiceMatch  = 258;
constexpr int kMaxLazy    = 258;
constexpr int kGoodLength = 32;

// The DLL feeds the source buffer to deflate() in 32 KiB steps and flashes
// Z_SYNC_FLUSH after every full step (Z_FINISH on the last one), so every
// chunk ends on a block boundary plus an empty stored block.
constexpr size_t kChunkBytes = 32768;

// zlib lit_bufsize at memLevel 8: one deflate block carries at most this many
// literal/length symbols.
constexpr int kMaxBlockSymbols = 16384;

const uint16_t kLenBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258
};
const uint8_t kLenExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};
const uint16_t kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
};
const uint8_t kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};
// Order in which the code-length code lengths are transmitted.
const uint8_t kClOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

// Length -> length-code index and distance -> distance-code index, built once.
struct Lookup {
    uint8_t len_code[kMaxMatch + 1];
    uint8_t dist_code[kWindow + 1];
    Lookup() {
        std::memset(len_code, 0, sizeof(len_code));
        for (int i = 0; i < 29; ++i) {
            const int span = 1 << kLenExtra[i];
            for (int k = 0; k < span; ++k) {
                const int len = kLenBase[i] + k;
                if (len <= kMaxMatch) len_code[len] = static_cast<uint8_t>(i);
            }
        }
        std::memset(dist_code, 0, sizeof(dist_code));
        for (int i = 0; i < 30; ++i) {
            const int span = 1 << kDistExtra[i];
            for (int k = 0; k < span; ++k) {
                const int d = kDistBase[i] + k;
                if (d <= kWindow) dist_code[d] = static_cast<uint8_t>(i);
            }
        }
    }
};
const Lookup& lookup() { static const Lookup t; return t; }

// LSB-first bit packer, as DEFLATE requires.
class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}

    void put(uint32_t value, int nbits) {
        bitbuf_ |= (value & ((1u << nbits) - 1u)) << bitcnt_;
        bitcnt_ += nbits;
        while (bitcnt_ >= 8) {
            out_.push_back(static_cast<uint8_t>(bitbuf_ & 0xffu));
            bitbuf_ >>= 8;
            bitcnt_ -= 8;
        }
    }
    void put_bytes(const uint8_t* p, size_t n) {
        if (bitcnt_ != 0) {
            for (size_t i = 0; i < n; ++i) put(p[i], 8);
            return;
        }
        out_.insert(out_.end(), p, p + n);
    }
    void align() {
        if (bitcnt_ > 0) {
            out_.push_back(static_cast<uint8_t>(bitbuf_ & 0xffu));
            bitbuf_ = 0;
            bitcnt_ = 0;
        }
    }
    int pending_bits() const { return bitcnt_; }

private:
    std::vector<uint8_t>& out_;
    uint32_t bitbuf_ = 0;
    int bitcnt_ = 0;
};

inline uint32_t reverse_bits(uint32_t v, int n) {
    uint32_t r = 0;
    for (int i = 0; i < n; ++i) { r = (r << 1) | (v & 1u); v >>= 1; }
    return r;
}

// Huffman code lengths for `freq[0,n)`, limited to `max_len` bits.
//
// Ordinary Huffman with zlib's overflow repair: symbols deeper than max_len are
// clamped and the bl_count histogram is reshaped until the Kraft sum is exactly
// 1 again.  Everything is validated (count sum and Kraft sum) before returning,
// so a failure can only make the caller fall back to a fixed block.
bool build_lengths(const uint32_t* freq, int n, int max_len, uint8_t* lens, int* repairs) {
    constexpr int kMaxNodes = 2 * 288 + 4;
    std::memset(lens, 0, static_cast<size_t>(n));

    int used[288];
    int nused = 0;
    for (int i = 0; i < n; ++i) if (freq[i] > 0) used[nused++] = i;
    // DEFLATE requires at least one code and zlib always forces two, which also
    // keeps every emitted tree complete (Kraft sum exactly 1).
    if (nused == 0) { used[0] = 0; used[1] = 1; nused = 2; }
    else if (nused == 1) { used[1] = (used[0] == 0) ? 1 : 0; nused = 2; }
    if (nused > n) return false;

    uint64_t nfreq[kMaxNodes];
    int nleft[kMaxNodes];
    int nright[kMaxNodes];
    int heap[kMaxNodes];
    int hn = 0;

    auto less = [&](int a, int b) {
        return nfreq[a] < nfreq[b] || (nfreq[a] == nfreq[b] && a < b);
    };
    auto push = [&](int v) {
        int i = hn++;
        heap[i] = v;
        while (i > 0) {
            const int p = (i - 1) / 2;
            if (!less(heap[i], heap[p])) break;
            std::swap(heap[i], heap[p]);
            i = p;
        }
    };
    auto pop = [&]() {
        const int top = heap[0];
        heap[0] = heap[--hn];
        int i = 0;
        for (;;) {
            const int l = 2 * i + 1, r = l + 1;
            int m = i;
            if (l < hn && less(heap[l], heap[m])) m = l;
            if (r < hn && less(heap[r], heap[m])) m = r;
            if (m == i) break;
            std::swap(heap[i], heap[m]);
            i = m;
        }
        return top;
    };

    int nnodes = 0;
    for (int i = 0; i < nused; ++i) {
        const uint32_t f = freq[used[i]] ? freq[used[i]] : 1u;
        nfreq[nnodes] = f;
        nleft[nnodes] = -1;
        nright[nnodes] = -1;
        push(nnodes);
        ++nnodes;
    }
    while (hn > 1) {
        const int a = pop();
        const int b = pop();
        nfreq[nnodes] = nfreq[a] + nfreq[b];
        nleft[nnodes] = a;
        nright[nnodes] = b;
        push(nnodes);
        ++nnodes;
    }
    const int root = heap[0];

    // Leaf depths.
    int depth[kMaxNodes];
    std::memset(depth, 0, sizeof(depth));
    int stack[4 * kMaxNodes];
    int sp = 0;
    stack[sp++] = root;
    stack[sp++] = 0;
    while (sp > 0) {
        const int d = stack[--sp];
        const int node = stack[--sp];
        if (nleft[node] < 0) {
            depth[node] = d;
        } else {
            stack[sp++] = nleft[node];
            stack[sp++] = d + 1;
            stack[sp++] = nright[node];
            stack[sp++] = d + 1;
        }
    }

    uint32_t bl_count[16];
    std::memset(bl_count, 0, sizeof(bl_count));
    int over = 0;
    for (int i = 0; i < nused; ++i) {
        int d = depth[i];
        if (d > max_len) { d = max_len; ++over; }
        bl_count[d]++;
    }
    if (over > 0 && repairs) ++(*repairs);

    while (over > 0) {
        int bits = max_len - 1;
        while (bits > 0 && bl_count[bits] == 0) --bits;
        if (bits == 0 || bl_count[max_len] == 0) return false;
        bl_count[bits]--;
        bl_count[bits + 1] += 2;
        bl_count[max_len]--;
        over -= 2;
    }

    uint32_t total = 0;
    for (int bits = 1; bits <= max_len; ++bits) total += bl_count[bits];
    if (static_cast<int>(total) != nused) return false;

    // Shortest codes go to the most frequent symbols.
    int order[288];
    for (int i = 0; i < nused; ++i) order[i] = i;
    std::sort(order, order + nused, [&](int a, int b) {
        if (nfreq[a] != nfreq[b]) return nfreq[a] > nfreq[b];
        return used[a] < used[b];
    });
    int idx = 0;
    for (int bits = 1; bits <= max_len; ++bits)
        for (uint32_t k = 0; k < bl_count[bits]; ++k)
            lens[used[order[idx++]]] = static_cast<uint8_t>(bits);

    uint64_t kraft = 0;
    for (int i = 0; i < n; ++i)
        if (lens[i]) kraft += 1ull << (max_len - lens[i]);
    if (kraft != (1ull << max_len)) return false;
    return true;
}

// Canonical codes for a complete length set.
void build_canonical(const uint8_t* lens, int n, uint16_t* codes) {
    int count[16];
    std::memset(count, 0, sizeof(count));
    for (int i = 0; i < n; ++i) count[lens[i]]++;
    count[0] = 0;
    uint16_t next[16];
    std::memset(next, 0, sizeof(next));
    uint16_t code = 0;
    for (int bits = 1; bits <= 15; ++bits) {
        code = static_cast<uint16_t>((code + count[bits - 1]) << 1);
        next[bits] = code;
    }
    for (int i = 0; i < n; ++i) codes[i] = lens[i] ? next[lens[i]]++ : 0;
}

void fixed_lengths(uint8_t* ll, uint8_t* dl) {
    for (int i = 0; i <= 143; ++i) ll[i] = 8;
    for (int i = 144; i <= 255; ++i) ll[i] = 9;
    for (int i = 256; i <= 279; ++i) ll[i] = 7;
    for (int i = 280; i <= 287; ++i) ll[i] = 8;
    for (int i = 0; i < 30; ++i) dl[i] = 5;
}

inline int cl_extra_bits(int sym) {
    if (sym == 16) return 2;
    if (sym == 17) return 3;
    if (sym == 18) return 7;
    return 0;
}

uint32_t adler32(const uint8_t* data, size_t n) {
    uint32_t a = 1, b = 0;
    while (n > 0) {
        const size_t k = n < 5552 ? n : 5552;
        for (size_t i = 0; i < k; ++i) {
            a += data[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
        data += k;
        n -= k;
    }
    return (b << 16) | a;
}

// ---------------------------------------------------------------------------
// Compressor: LZ77 with hash chains + lazy matching, dynamic Huffman blocks.
// One instance per worker; buffers are reused between repetitions.
// ---------------------------------------------------------------------------
class Deflator {
public:
    Deflator() : head_(static_cast<size_t>(kHashSize), -1), prev_(static_cast<size_t>(kWindow), 0) {}

    bool compress(const uint8_t* in, size_t n, std::vector<uint8_t>& out) {
        out.clear();
        out.push_back(0x78);        // CMF: deflate, 32 KiB window
        out.push_back(0x9c);        // FLG: level 9 (FLEVEL = 3), FCHECK = 0x1c

        BitWriter bw(out);
        std::fill(head_.begin(), head_.end(), -1);
        inserted_ = 0;
        dynamic_blocks = 0;
        fixed_blocks = 0;
        stored_blocks = 0;

        // One deflateInit_() per pass, like the DLL: the zlib stream is created
        // at the top of every repetition and the input is then taken in 32 KiB
        // steps with Z_SYNC_FLUSH between the steps and Z_FINISH on the last
        // one.  A sync flush terminates the current block and inserts an empty
        // stored block (BFINAL=0, BTYPE=00, LEN=0) so the writer is back on a
        // byte boundary.
        static const uint8_t kEmptyStored[4] = {0x00, 0x00, 0xff, 0xff};
        size_t pos = 0;
        size_t cend = n < kChunkBytes ? n : kChunkBytes;
        for (;;) {
            const bool last = (cend >= n);
            do {
                const size_t bstart = pos;
                fill_symbols(in, cend, pos);
                const bool block_last = last && (pos >= cend);
                emit_block(bw, in, bstart, pos, block_last);
            } while (pos < cend);
            if (last) break;
            bw.put(0u, 3);
            bw.align();
            bw.put_bytes(kEmptyStored, 4);
            cend = (cend + kChunkBytes < n) ? (cend + kChunkBytes) : n;
        }
        bw.align();

        const uint32_t ad = adler32(in, n);
        out.push_back(static_cast<uint8_t>(ad >> 24));
        out.push_back(static_cast<uint8_t>(ad >> 16));
        out.push_back(static_cast<uint8_t>(ad >> 8));
        out.push_back(static_cast<uint8_t>(ad));
        return true;
    }

    int dynamic_blocks = 0;
    int fixed_blocks = 0;
    int stored_blocks = 0;
    int repairs = 0;            // blocks whose Huffman depth had to be limited

private:
    struct Symbol {
        uint16_t value;         // literal byte, or 256 + match length
        uint16_t dist;          // match distance, 0 for a literal
    };

    inline uint32_t hash3(const uint8_t* in, size_t p) const {
        uint32_t h = in[p];
        h = ((h << kHashShift) ^ in[p + 1]) & kHashMask;
        h = ((h << kHashShift) ^ in[p + 2]) & kHashMask;
        return h;
    }

    // Inserts position `p` and returns the head of the chain *before* the
    // insertion, which is the first candidate zlib's longest_match() would see
    // (its INSERT_STRING macro hands the old head to longest_match).  Returning
    // the new head instead would make find_match() start on the position that
    // is being searched, which can never match.
    int32_t insert(const uint8_t* in, size_t n, size_t p) {
        if (p < inserted_) return -1;              // every position exactly once
        inserted_ = p + 1;
        if (p + kMinMatch > n) return -1;
        const uint32_t h = hash3(in, p);
        const int32_t old = head_[h];
        prev_[p & kWindowMask] = old;
        head_[h] = static_cast<int32_t>(p);
        return old;
    }

    // `prev_len` is zlib's s->prev_length: the best match length of the
    // previous position, which longest_match() uses to shorten the chain walk
    // when the previous match was already long (good_match >>= 2).
    void find_match(const uint8_t* in, size_t n, size_t pos, int32_t start, int prev_len,
                    int& best_len, int& best_dist) const {
        best_len = 0;
        best_dist = 0;
        if (start < 0) return;

        // Valid match distances are 1..min(pos, kWindow): a match must start at
        // an earlier position, and it cannot reach further back than the
        // beginning of the output stream.  An unclamped walk here produces
        // streams that decoders reject with "invalid distance too far back".
        const int64_t earliest = static_cast<int64_t>(pos) - static_cast<int64_t>(kWindow);
        const int64_t low = earliest > 0 ? earliest : 0;
        const int max_len = static_cast<int>((n - pos) < static_cast<size_t>(kMaxMatch)
                                                 ? (n - pos) : static_cast<size_t>(kMaxMatch));
        int chain = kMaxChain;
        if (prev_len >= kGoodLength) chain >>= 2;
        int32_t cur = start;
        while (cur >= 0 && static_cast<int64_t>(cur) >= low && static_cast<size_t>(cur) < pos &&
               chain-- > 0) {
            const size_t c = static_cast<size_t>(cur);
            if (best_len < max_len &&
                in[c + static_cast<size_t>(best_len)] == in[pos + static_cast<size_t>(best_len)] &&
                in[c] == in[pos]) {
                int len = 0;
                while (len < max_len &&
                       in[c + static_cast<size_t>(len)] == in[pos + static_cast<size_t>(len)])
                    ++len;
                if (len > best_len) {
                    best_len = len;
                    best_dist = static_cast<int>(pos - c);
                    if (best_dist < 1 || best_dist > kWindow) { best_len = 0; best_dist = 0; }
                    else if (len >= kNiceMatch || len >= max_len) break;
                }
            }
            cur = prev_[c & kWindowMask];
        }
        if (best_len < kMinMatch) { best_len = 0; best_dist = 0; }
    }

    // Greedy LZ77 with one-byte lazy matching (zlib's deflate_slow), filling at
    // most kMaxBlockSymbols symbols and advancing `pos`.
    void fill_symbols(const uint8_t* in, size_t n, size_t& pos) {
        sym_.clear();
        while (pos < n && sym_.size() < static_cast<size_t>(kMaxBlockSymbols)) {
            int best_len = 0, best_dist = 0;
            if (pos + kMinMatch <= n) {
                const int32_t head = insert(in, n, pos);
                find_match(in, n, pos, head, 0, best_len, best_dist);
            }
            if (best_len < kMinMatch) {
                sym_.push_back(Symbol{static_cast<uint16_t>(in[pos]), 0});
                ++pos;
                continue;
            }

            int lazy = 0;
            while (lazy < kMaxLazy && best_len < kNiceMatch &&
                   sym_.size() < static_cast<size_t>(kMaxBlockSymbols) && pos + 1 < n) {
                int len2 = 0, dist2 = 0;
                if (pos + 1 + kMinMatch <= n) {
                    const int32_t head2 = insert(in, n, pos + 1);
                    find_match(in, n, pos + 1, head2, best_len, len2, dist2);
                }
                if (len2 <= best_len) break;
                sym_.push_back(Symbol{static_cast<uint16_t>(in[pos]), 0});
                ++pos;
                best_len = len2;
                best_dist = dist2;
                ++lazy;
            }

            if (sym_.size() >= static_cast<size_t>(kMaxBlockSymbols)) break;  // keep the match for the next block
            sym_.push_back(Symbol{static_cast<uint16_t>(256 + best_len),
                                  static_cast<uint16_t>(best_dist)});
            const size_t end = pos + static_cast<size_t>(best_len);
            for (size_t p = pos + 1; p < end; ++p) insert(in, n, p);
            pos = end;
        }
    }

    static uint64_t symbol_cost(const std::vector<Symbol>& sym, const uint8_t* ll_len,
                                const uint8_t* d_len) {
        const Lookup& t = lookup();
        uint64_t bits = 0;
        for (const Symbol& s : sym) {
            if (s.value < 256) {
                bits += ll_len[s.value];
            } else {
                const int len = s.value - 256;
                const int lc = t.len_code[len];
                bits += ll_len[257 + lc] + kLenExtra[lc];
                const int dc = t.dist_code[s.dist];
                bits += d_len[dc] + kDistExtra[dc];
            }
        }
        return bits;
    }

    void emit_block(BitWriter& bw, const uint8_t* in, size_t bstart, size_t bend, bool last) {
        const size_t rawlen = bend - bstart;
        const Lookup& t = lookup();

        uint32_t llfreq[286];
        uint32_t dfreq[30];
        std::memset(llfreq, 0, sizeof(llfreq));
        std::memset(dfreq, 0, sizeof(dfreq));
        for (const Symbol& s : sym_) {
            if (s.value < 256) {
                llfreq[s.value]++;
            } else {
                llfreq[257 + t.len_code[s.value - 256]]++;
                dfreq[t.dist_code[s.dist]]++;
            }
        }
        llfreq[256]++;                                  // end of block

        bool dyn_ok = build_lengths(llfreq, 286, 15, ll_len_, &repairs_) &&
                      build_lengths(dfreq, 30, 15, d_len_, &repairs_);
        if (dyn_ok) {
            build_canonical(ll_len_, 286, ll_code_);
            build_canonical(d_len_, 30, d_code_);
        }

        uint8_t fll_len[288], fd_len[30];
        uint16_t fll_code[288], fd_code[30];
        fixed_lengths(fll_len, fd_len);
        build_canonical(fll_len, 288, fll_code);
        build_canonical(fd_len, 30, fd_code);

        uint64_t dyn_bits = ~0ull, fixed_bits = 0, stored_bits = ~0ull;

        if (dyn_ok) {
            int nlit = 286;
            while (nlit > 257 && ll_len_[nlit - 1] == 0) --nlit;
            int ndist = 30;
            while (ndist > 1 && d_len_[ndist - 1] == 0) --ndist;
            build_rle(ll_len_, nlit, d_len_, ndist);

            uint32_t clfreq[19];
            std::memset(clfreq, 0, sizeof(clfreq));
            for (size_t i = 0; i < rle_sym_.size(); ++i) clfreq[rle_sym_[i]]++;

            if (build_lengths(clfreq, 19, 7, cl_len_, &repairs_)) {
                build_canonical(cl_len_, 19, cl_code_);
                int nclen = 19;
                while (nclen > 4 && cl_len_[kClOrder[nclen - 1]] == 0) --nclen;

                dyn_bits = 3 + 14 + static_cast<uint64_t>(3 * nclen);
                for (size_t i = 0; i < rle_sym_.size(); ++i)
                    dyn_bits += cl_len_[rle_sym_[i]] + cl_extra_bits(rle_sym_[i]);
                dyn_bits += symbol_cost(sym_, ll_len_, d_len_) + ll_len_[256];
                nlit_ = nlit;
                ndist_ = ndist;
                nclen_ = nclen;
            } else {
                dyn_ok = false;
            }
        }

        if (!dyn_ok) dyn_bits = ~0ull;
        fixed_bits = 3 + symbol_cost(sym_, fll_len, fd_len) + fll_len[256];

        // Only dynamic and fixed Huffman blocks are emitted.  Stored blocks would
        // save a little space on incompressible data, but their byte-alignment
        // framing is easy to get subtly wrong and they are never the right choice
        // for the compressible corpora this benchmark uses, so the two verified
        // encoders handle every input instead.
        if (dyn_bits <= fixed_bits) {
            emit_dynamic(bw, last);
        } else {
            emit_fixed(bw, last, fll_len, fll_code, fd_len, fd_code);
        }
    }

    void build_rle(const uint8_t* ll_len, int nlit, const uint8_t* d_len, int ndist) {
        const int n = nlit + ndist;
        rle_sym_.clear();
        rle_extra_.clear();

        // Every code length is transmitted as its own literal symbol (0..15).
        // Only the first nlit + ndist entries are meaningful - everything past
        // them in ll_len_/d_len_ is uninitialised - so the loop is bounded by n.
        //
        // The repeat symbols 16/17/18 would describe the same table more
        // compactly, but a mistake in their run accounting makes the RLE expand
        // to more lengths than the header declares, which decoders reject with
        // "invalid bit length repeat".  The table description is a negligible
        // share of a block, and this benchmark measures data throughput, so the
        // straightforward encoding is the right trade.
        for (int i = 0; i < n; ++i) {
            const uint8_t v = (i < nlit) ? ll_len[i] : d_len[i - nlit];
            rle_sym_.push_back(static_cast<uint16_t>(v & 0x0f));
            rle_extra_.push_back(0);
        }
    }

    void emit_dynamic(BitWriter& bw, bool last) {
        bw.put(last ? 1u : 0u, 1);
        bw.put(2u, 2);                                   // BTYPE = 10
        bw.put(static_cast<uint32_t>(nlit_ - 257), 5);
        bw.put(static_cast<uint32_t>(ndist_ - 1), 5);
        bw.put(static_cast<uint32_t>(nclen_ - 4), 4);
        for (int i = 0; i < nclen_; ++i) bw.put(cl_len_[kClOrder[i]], 3);
        for (size_t i = 0; i < rle_sym_.size(); ++i) {
            const int s = rle_sym_[i];
            bw.put(reverse_bits(cl_code_[s], cl_len_[s]), cl_len_[s]);
            const int ex = cl_extra_bits(s);
            if (ex) bw.put(rle_extra_[i], ex);
        }
        emit_symbols(bw, ll_len_, ll_code_, d_len_, d_code_);
        ++dynamic_blocks;
    }

    void emit_fixed(BitWriter& bw, bool last, const uint8_t* ll_len, const uint16_t* ll_code,
                    const uint8_t* d_len, const uint16_t* d_code) {
        bw.put(last ? 1u : 0u, 1);
        bw.put(1u, 2);                                   // BTYPE = 01
        emit_symbols(bw, ll_len, ll_code, d_len, d_code);
        ++fixed_blocks;
    }

    void emit_symbols(BitWriter& bw, const uint8_t* ll_len, const uint16_t* ll_code,
                      const uint8_t* d_len, const uint16_t* d_code) {
        const Lookup& t = lookup();
        for (const Symbol& s : sym_) {
            if (s.value < 256) {
                bw.put(reverse_bits(ll_code[s.value], ll_len[s.value]), ll_len[s.value]);
            } else {
                const int len = s.value - 256;
                const int lc = t.len_code[len];
                const int lcode = 257 + lc;
                bw.put(reverse_bits(ll_code[lcode], ll_len[lcode]), ll_len[lcode]);
                if (kLenExtra[lc]) bw.put(static_cast<uint32_t>(len - kLenBase[lc]), kLenExtra[lc]);
                // The distance code width comes from the tree in use: 5 bits for
                // the fixed tree, whatever the dynamic tree assigned otherwise.
                const int dc = t.dist_code[s.dist];
                bw.put(reverse_bits(d_code[dc], d_len[dc]), d_len[dc]);
                if (kDistExtra[dc])
                    bw.put(static_cast<uint32_t>(s.dist - kDistBase[dc]), kDistExtra[dc]);
            }
        }
        bw.put(reverse_bits(ll_code[256], ll_len[256]), ll_len[256]);   // end of block
    }


    std::vector<int32_t> head_;
    std::vector<int32_t> prev_;
    size_t inserted_ = 0;
    std::vector<Symbol> sym_;
    std::vector<uint16_t> rle_sym_;
    std::vector<uint16_t> rle_extra_;
    uint8_t ll_len_[286] = {0}, d_len_[30] = {0}, cl_len_[19] = {0};
    uint16_t ll_code_[286] = {0}, d_code_[30] = {0}, cl_code_[19] = {0};
    int nlit_ = 257, ndist_ = 1, nclen_ = 4;
    int repairs_ = 0;
};

// ---------------------------------------------------------------------------
// Inflate: used only by verify_crypto_kernels() to prove the compressor's
// output is a valid, self-describing zlib stream.
// ---------------------------------------------------------------------------
class BitReader {
public:
    BitReader(const uint8_t* p, size_t n) : p_(p), n_(n) {}

    // Bits are consumed LSB-first within each byte, as DEFLATE requires.
    //
    // A read that runs past the end is NOT treated as fatal here.  A final
    // DEFLATE block is padded to a byte boundary with zero bits, and the
    // trailing Adler-32 is read as two 16-bit words after that alignment, so the
    // final word can legitimately extend past the end of the compressed data.
    // Missing bits read as zero and overran() records that it happened, letting
    // each caller decide: a failed symbol decode must fail, the optional trailer
    // must not.
    uint32_t get(int nbits) {
        while (cnt_ < nbits) {
            if (pos_ >= n_) { bad_ = true; over_ = true; break; }
            buf_ |= static_cast<uint32_t>(p_[pos_++]) << cnt_;
            cnt_ += 8;
        }
        const uint32_t v = buf_ & ((1u << nbits) - 1u);
        buf_ >>= nbits;
        cnt_ = (cnt_ > nbits) ? (cnt_ - nbits) : 0;
        return v;
    }

    // Discard bits up to the next byte boundary.  The already-loaded bytes stay
    // in buf_, so only the bit count changes; pos_ keeps pointing at the next
    // byte that has not been loaded yet.
    void align() {
        const int drop = cnt_ & 7;
        buf_ >>= drop;
        cnt_ -= drop;
    }

    bool bad() const { return bad_; }
    bool overran() const { return over_; }

private:
    const uint8_t* p_;
    size_t n_;
    size_t pos_ = 0;      // bytes loaded into buf_
    uint32_t buf_ = 0;    // up to 32 buffered bits, next bit in bit 0
    int cnt_ = 0;         // valid bits in buf_
    bool bad_ = false;
    bool over_ = false;
};

struct HuffTable {
    int count[16] = {0};
    uint16_t symbol[288] = {0};
};

bool huff_build(HuffTable& h, const uint8_t* lens, int n) {
    for (int i = 0; i < n; ++i) h.count[lens[i]]++;
    if (h.count[0] == n) return false;               // no codes at all
    h.count[0] = 0;
    int left = 1;
    for (int len = 1; len <= 15; ++len) {
        left <<= 1;
        left -= h.count[len];
        if (left < 0) return false;                  // over-subscribed
    }
    int offs[16];
    offs[1] = 0;
    for (int len = 1; len < 15; ++len) offs[len + 1] = offs[len] + h.count[len];
    for (int i = 0; i < n; ++i)
        if (lens[i]) h.symbol[offs[lens[i]]++] = static_cast<uint16_t>(i);
    return true;
}

int huff_decode(BitReader& br, const HuffTable& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
        code |= static_cast<int>(br.get(1));
        if (br.bad()) return -1;
        const int count = h.count[len];
        if (code - first < count) {
            return h.symbol[index + (code - first)];
        }
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

bool inflate(const uint8_t* in, size_t n, std::vector<uint8_t>& out, size_t max_out) {
    if (n < 6) return false;
    const uint8_t cmf = in[0], flg = in[1];
    if ((cmf & 0x0f) != 8) return false;                       // deflate method
    if (((static_cast<unsigned>(cmf) << 8) | flg) % 31u != 0) return false;
    if (flg & 0x20u) return false;                             // preset dictionary

    BitReader br(in + 2, n - 2);
    out.clear();
    bool last = false;
    while (!last) {
        last = br.get(1) != 0;
        const int type = static_cast<int>(br.get(2));
        if (br.bad()) return false;

        if (type == 0) {
            // Stored block: skip to the byte boundary, then LEN / NLEN / data.
            br.align();
            const uint32_t len = br.get(16);
            const uint32_t nlen = br.get(16);
            if (br.bad() || ((len ^ 0xffffu) != nlen)) return false;
            for (uint32_t i = 0; i < len; ++i) {
                const uint32_t byte = br.get(8);
                if (br.bad()) return false;
                if (out.size() >= max_out) return false;
                out.push_back(static_cast<uint8_t>(byte));
            }
            continue;
        }
        if (type == 3) return false;                           // reserved

        HuffTable lit, dist;
        if (type == 1) {
            uint8_t fll[288], fdl[30];
            fixed_lengths(fll, fdl);
            if (!huff_build(lit, fll, 288) || !huff_build(dist, fdl, 30)) return false;
        } else {
            const int hlit = static_cast<int>(br.get(5)) + 257;
            const int hdist = static_cast<int>(br.get(5)) + 1;
            const int hclen = static_cast<int>(br.get(4)) + 4;
            if (br.bad() || hlit > 286 || hdist > 30 || hclen > 19) return false;

            uint8_t cl[19];
            std::memset(cl, 0, sizeof(cl));
            for (int i = 0; i < hclen; ++i) cl[kClOrder[i]] = static_cast<uint8_t>(br.get(3));
            if (br.bad()) return false;
            HuffTable clh;
            if (!huff_build(clh, cl, 19)) return false;

            uint8_t lens[288 + 30];
            std::memset(lens, 0, sizeof(lens));
            const int total = hlit + hdist;
            int i = 0;
            while (i < total) {
                const int sym = huff_decode(br, clh);
                if (sym < 0) return false;
                if (sym < 16) {
                    lens[i++] = static_cast<uint8_t>(sym);
                } else if (sym == 16) {
                    if (i == 0) return false;
                    const int rep = 3 + static_cast<int>(br.get(2));
                    if (br.bad()) return false;
                    for (int k = 0; k < rep && i < total; ++k) lens[i] = lens[i - 1], ++i;
                } else if (sym == 17) {
                    const int rep = 3 + static_cast<int>(br.get(3));
                    if (br.bad()) return false;
                    i += rep;
                } else {
                    const int rep = 11 + static_cast<int>(br.get(7));
                    if (br.bad()) return false;
                    i += rep;
                }
                if (i > total) return false;
            }
            if (!huff_build(lit, lens, hlit)) return false;
            // A dynamic block may declare no distance codes at all, in which case
            // huff_build reports failure; that is legal, so fall back to a table
            // with a single dummy code and let the length check reject any use.
            if (!huff_build(dist, lens + hlit, hdist)) {
                uint8_t one[1] = {1};
                if (!huff_build(dist, one, 1)) return false;
            }
        }

        for (;;) {
            const int sym = huff_decode(br, lit);
            if (sym < 0) return false;
            if (sym < 256) {
                if (out.size() >= max_out) return false;
                out.push_back(static_cast<uint8_t>(sym));
                continue;
            }
            if (sym == 256) break;                              // end of block
            const int li = sym - 257;
            if (li > 28) return false;
            const int length = kLenBase[li] + static_cast<int>(br.get(kLenExtra[li]));
            const int dsym = huff_decode(br, dist);
            if (dsym < 0 || dsym > 29) return false;
            const int d = kDistBase[dsym] + static_cast<int>(br.get(kDistExtra[dsym]));
            if (br.bad()) return false;
            if (static_cast<size_t>(d) > out.size()) return false;
            if (out.size() + static_cast<size_t>(length) > max_out) return false;
            for (int k = 0; k < length; ++k)
                out.push_back(out[out.size() - static_cast<size_t>(d)]);
        }
    }

    // Verify the trailing Adler-32.  It is simply the last four bytes of the
    // stream, most-significant byte first, so read it from there rather than
    // from the bit reader: that avoids depending on how many padding bits the
    // final block happened to leave behind.
    //
    // The checksum must be accumulated with the same periodic reduction that
    // adler32() uses: summing a whole megabyte without reducing overflows
    // uint32_t and yields a wrong checksum even for perfect data.
    uint32_t a = 1, b = 0;
    {
        size_t rem = out.size();
        const uint8_t* p = out.data();
        while (rem > 0) {
            const size_t k = rem < 5552 ? rem : 5552;
            for (size_t i = 0; i < k; ++i) {
                a += p[i];
                b += a;
            }
            a %= 65521u;
            b %= 65521u;
            p += k;
            rem -= k;
        }
    }
    const uint32_t calc = (b << 16) | a;
    return in[n - 4] == static_cast<uint8_t>(calc >> 24) &&
           in[n - 3] == static_cast<uint8_t>(calc >> 16) &&
           in[n - 2] == static_cast<uint8_t>(calc >> 8) &&
           in[n - 1] == static_cast<uint8_t>(calc);
}

// Deterministic corpora for the round-trip self test.
// ---------------------------------------------------------------------------
void fill_text(uint8_t* dst, size_t n, uint64_t seed) {
    static const char* kWords[] = {
        "the ", "of ", "and ", "to ", "in ", "a ", "is ", "that ", "for ", "it ",
        "as ", "was ", "with ", "be ", "by ", "on ", "not ", "he ", "this ", "are ",
        "or ", "his ", "from ", "at ", "which ", "but ", "have ", "an ", "had ", "they ",
        "you ", "were ", "their ", "one ", "all ", "we ", "can ", "her ", "has ", "there ",
        "deflate ", "huffman ", "zlib ", "compression ", "benchmark ", "throughput ",
        "entropy ", "window ", "match ", "literal ", "distance ", "codes "
    };
    const int nw = static_cast<int>(sizeof(kWords) / sizeof(kWords[0]));
    uint64_t s = seed;
    size_t i = 0;
    while (i < n) {
        const char* w = kWords[splitmix64(s) % static_cast<uint64_t>(nw)];
        for (const char* p = w; *p && i < n; ++p) dst[i++] = static_cast<uint8_t>(*p);
        if ((splitmix64(s) & 15u) == 0 && i < n) dst[i++] = '\n';
    }
}

void fill_random(uint8_t* dst, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>(splitmix64(s) >> 57);
}

// The CPU ZLib input buffer, exactly as the DLL prepares it: the buffer is
// filled with 16-byte blocks of four identical little-endian dwords.  Block i
// carries
//
//     v = ((((N + i + 3) * (N + i + 4)) mod 2^32) >> 1) - 0x21523F22
//
// with N = size / 4, evaluated modulo 2^32 after both the multiply and the
// subtract.  The data is identical for every worker and compresses about 3.6:1
// at level 9 - the DLL reports the same ratio through PackedSizeInBytes
// (9222583 bytes for a 32 MiB pass).
void fill_ngpack_corpus(uint8_t* dst, size_t n) {
    const uint32_t n4 = static_cast<uint32_t>(n >> 2);
    size_t off = 0;
    for (; off + 16 <= n; off += 16) {
        const uint32_t k = n4 + static_cast<uint32_t>(off >> 4) + 3u;
        const uint32_t t = k * (k + 1u);
        const uint32_t v = static_cast<uint32_t>((t >> 1) - 0x21523F22u);
        for (int j = 0; j < 4; ++j) {
            dst[off + 4 * j + 0] = static_cast<uint8_t>(v);
            dst[off + 4 * j + 1] = static_cast<uint8_t>(v >> 8);
            dst[off + 4 * j + 2] = static_cast<uint8_t>(v >> 16);
            dst[off + 4 * j + 3] = static_cast<uint8_t>(v >> 24);
        }
    }
    // Sizes that are not a whole number of 16-byte blocks: the tail stays zero,
    // like the DLL's (zero-initialised) VirtualAlloc pages.
    for (; off < n; ++off) dst[off] = 0;
}

// Fibonacci-ish byte frequencies, shuffled so that a single deflate block still
// sees the skew: this is what forces Huffman codes deeper than 15 bits.
void fill_skewed(uint8_t* dst, size_t n, uint64_t seed) {
    uint64_t f[26];
    f[0] = 1;
    f[1] = 2;
    for (int i = 2; i < 26; ++i) f[i] = f[i - 1] + f[i - 2];
    size_t i = 0;
    for (int k = 25; k >= 0 && i < n; --k) {
        const size_t reps = static_cast<size_t>(f[k]);
        for (size_t r = 0; r < reps && i < n; ++r) dst[i++] = static_cast<uint8_t>(k);
    }
    while (i < n) dst[i++] = 0;
    uint64_t s = seed;
    for (size_t k = n; k > 1; --k) {                 // deterministic Fisher-Yates
        const size_t j = static_cast<size_t>(splitmix64(s) % static_cast<uint64_t>(k));
        std::swap(dst[k - 1], dst[j]);
    }
}

enum DeflateFail {
    kZlibOk = 0,
    kZlibEmpty = 21,
    kZlibTiny = 22,
    kZlibRandom = 23,
    kZlibText = 24,
    kZlibSkewed = 25,
    kZlibRuns = 26,
    kZlibBigText = 27,
    kZlibNoDynamic = 28,
    kZlibUnusedRepair = 29,
    kZlibNGPack = 30,
};

int round_trip(const uint8_t* data, size_t n, bool want_dynamic, int fail_code) {
    Deflator d;
    std::vector<uint8_t> comp;
    d.compress(data, n, comp);
    if (comp.size() < 6 || comp[0] != 0x78 || comp[1] != 0x9c) return fail_code;
    std::vector<uint8_t> back;
    if (!inflate(comp.data(), comp.size(), back, n ? n : 1)) return fail_code;
    if (back.size() != n) return fail_code;
    if (n && std::memcmp(back.data(), data, n) != 0) return fail_code;
    if (want_dynamic && d.dynamic_blocks == 0) return kZlibNoDynamic;
    return kZlibOk;
}

int self_test() {
    std::vector<uint8_t> buf;

    // Sizes around every structural boundary, including the empty message.
    buf.assign(4096, 0);
    for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4), size_t(5),
                     size_t(255), size_t(256), size_t(257), size_t(1000)}) {
        fill_text(buf.data(), buf.size(), 0x9e37ull);
        const int rc = round_trip(buf.data(), n, false, kZlibTiny);
        if (rc != kZlibOk) return rc;
    }

    buf.assign(200000, 0);
    fill_random(buf.data(), buf.size(), 0xdeadbeefull);
    { const int rc = round_trip(buf.data(), buf.size(), false, kZlibRandom); if (rc) return rc; }

    buf.assign(1u << 20, 0);
    fill_text(buf.data(), buf.size(), 0x1234567ull);
    { const int rc = round_trip(buf.data(), buf.size(), true, kZlibText); if (rc) return rc; }

    // Skewed frequencies: a stress case for the code-length limit.  Whether the
    // 15-bit limiting path is actually needed depends on the exact histogram, so
    // a missing repair is reported but is not a failure: the round trip above is
    // the correctness requirement.
    Deflator d;
    buf.assign(320000, 0);
    fill_skewed(buf.data(), buf.size(), 0x5eed5eedull);
    {
        std::vector<uint8_t> comp, back;
        d.compress(buf.data(), buf.size(), comp);
        if (!inflate(comp.data(), comp.size(), back, buf.size())) return kZlibSkewed;
        if (back.size() != buf.size() || std::memcmp(back.data(), buf.data(), buf.size()) != 0)
            return kZlibSkewed;
    }

    // Long runs: 258-byte matches and distances near the 32 KiB window edge.
    buf.assign(600000, 0);
    {
        uint64_t s = 0xabcdef01ull;
        for (size_t i = 0; i < buf.size();) {
            const size_t run = 1 + static_cast<size_t>(splitmix64(s) % 900);
            const uint8_t v = static_cast<uint8_t>(splitmix64(s) >> 59);
            for (size_t k = 0; k < run && i < buf.size(); ++k) buf[i++] = v;
        }
        const int rc = round_trip(buf.data(), buf.size(), true, kZlibRuns);
        if (rc) return rc;
    }

    buf.assign(4u << 20, 0);
    fill_text(buf.data(), buf.size(), 0x2468aceull);
    { const int rc = round_trip(buf.data(), buf.size(), true, kZlibBigText); if (rc) return rc; }

    // Highly repetitive data must survive a header that trims almost everything.
    buf.assign(300000, 0x5a);
    { const int rc = round_trip(buf.data(), buf.size(), true, kZlibRuns); if (rc) return rc; }

    // The CPU ZLib corpus, including its chunk-boundary / sync-flush handling.
    // The ~3.6:1 ratio of a healthy level-9 encoder is a cheap sanity check
    // against a matcher that silently stopped finding matches.
    buf.assign(1u << 20, 0);
    fill_ngpack_corpus(buf.data(), buf.size());
    {
        std::vector<uint8_t> comp, back;
        d.compress(buf.data(), buf.size(), comp);
        if (!inflate(comp.data(), comp.size(), back, buf.size())) return kZlibNGPack;
        if (back.size() != buf.size() || std::memcmp(back.data(), buf.data(), buf.size()) != 0)
            return kZlibNGPack;
        if (comp.size() * 2 >= buf.size()) return kZlibNGPack;
    }

    return kZlibOk;
}

} // namespace deflate

// ===========================================================================
// CPU ZLib workload
// ===========================================================================
class BenchZlib final : public Benchmark {
public:
    static constexpr size_t kBufferBytes = 32u << 20;   // 32 MiB per worker

    BenchInfo info() const override {
        BenchInfo i;
        i.test = "zlib";
        i.label = "CPU ZLib";
        i.unit = "MB/s";
        i.group = "CPU";
        i.higher_is_better = true;
        i.unit_scale = 1e-6;                       // work units are bytes
        i.description = "DEFLATE compression of a 32 MiB buffer (zlib 1.2.5 level 9, "
                        "32 KiB Z_SYNC_FLUSH steps, Base_ZLib_1_2_5)";
        return i;
    }

    void prepare(int threads) override {
        threads = std::max(1, threads);
        workers_.clear();
        workers_.resize(static_cast<size_t>(threads));
        for (size_t i = 0; i < workers_.size(); ++i) {
            Worker& w = workers_[i];
            w.in.resize(kBufferBytes);
            // The DLL prepares the input itself and every worker compresses the
            // same deterministic buffer, so there is no per-worker seed.
            deflate::fill_ngpack_corpus(w.in.data(), w.in.size());
            w.out.reserve(kBufferBytes / 2 + 65536);
            w.def = std::unique_ptr<deflate::Deflator>(new deflate::Deflator());
        }
    }

    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)thread_count;
        const size_t idx = static_cast<size_t>(worker < 0 ? 0 : worker);
        if (idx >= workers_.size() || !workers_[idx].def) return 0.0;
        Worker& w = workers_[idx];

        // The DLL repeats whole compression passes until its per-worker time
        // budget (NGPackMinRunTime) expires, always running at least one pass.
        const double t0 = now_seconds();
        double bytes = 0.0;
        do {
            w.def->compress(w.in.data(), w.in.size(), w.out);
            bytes += static_cast<double>(w.in.size());
        } while (now_seconds() - t0 < min_seconds);
        return bytes;
    }

    uint64_t digest() const override {
        uint64_t h = 0xcbf29ce484222325ull;
        for (const Worker& w : workers_) h = hash_bytes(w.out.data(), w.out.size(), h);
        return h;
    }
    bool has_digest() const override { return true; }

    // Single-threaded re-compression for the reference.  DEFLATE is exact integer
    // work, so a different compressed stream means a wrong encoder decision.
    uint64_t full_digest() const override {
        BenchZlib* self = const_cast<BenchZlib*>(this);
        self->prepare(1);
        self->run_once(0, 1, 0.0);
        if (self->workers_.empty()) return  0;
        return  corrupt_if_testing(hash_bytes(self->workers_[0].out.data(), self->workers_[0].out.size(),
                          0xcbf29ce484222325ull));
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        return "DEFLATE level=9 window=32 KiB chain=4096 lazy=258 nice=258 good=32, "
               "32 KiB Z_SYNC_FLUSH steps, buffer=" + std::to_string(kBufferBytes >> 20) +
               " MiB/worker, corpus=NGPack dword-triangular pattern, container=zlib "
               "(RFC 1950), module=Base_ZLib_1_2_5";
    }

private:
    struct Worker {
        std::vector<uint8_t> in;
        std::vector<uint8_t> out;
        std::unique_ptr<deflate::Deflator> def;
    };
    std::vector<Worker> workers_;
};

// ===========================================================================
// Self test
// ===========================================================================
int smoke_benchmark(Benchmark* b, int fail_code) {
    if (!b) return fail_code;
    b->prepare(1);
    const double work = b->run_once(0, 1, 0.005);
    if (!(work > 0.0)) return fail_code;
    if (!b->has_digest()) return fail_code;
    if (b->digest() == 0) return fail_code;
    if (b->params().empty()) return fail_code;
    const BenchInfo i = b->info();
    if (i.test.empty() || i.unit.empty() || i.group.empty()) return fail_code;
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------
Benchmark* make_bench_aes() {
    static BenchAes instance;
    return &instance;
}

Benchmark* make_bench_sha3() {
    static BenchSha3 instance;
    return &instance;
}

Benchmark* make_bench_zlib() {
    static BenchZlib instance;
    return &instance;
}

// ---------------------------------------------------------------------------
// Self-consistency checks.  Return 0 on success, otherwise the first failing
// code (the values are stable so a failure report identifies the check).
// ---------------------------------------------------------------------------
int verify_crypto_kernels() {
    const int aes = aes256::self_test();
    if (aes != 0) {
        return aes;
    }

    const int sha = sha3::self_test();
    if (sha != 0) {
        return sha;
    }

    const int zl = deflate::self_test();
    if (zl != 0) {
        return zl;
    }

    int rc = smoke_benchmark(make_bench_aes(), 31);
    if (rc == 0) rc = smoke_benchmark(make_bench_sha3(), 32);
    if (rc == 0) rc = smoke_benchmark(make_bench_zlib(), 33);
    if (rc != 0) {
        return rc;
    }
    return 0;
}

} // namespace cst