// Render a frame with this project's kernel and write it as 8-bit BGRA, so it
// can be compared with the framebuffer captured from AIDA64.
//
//   verify_compare.exe <aida.bin> [f32|f64] [strip_bias]
//
// With strip_bias != 0 the per-channel mean is removed before differencing,
// which separates "wrong scene" from "right scene, wrong transfer function".
#include "../src/raytrace.h"
#include "../src/common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: verify_compare <aida.bin> [f32|f64] [strip_bias]\n");
        return 2;
    }
    const bool f32 = (argc >= 3 && std::strcmp(argv[2], "f32") == 0);
    const bool strip = (argc >= 4 && std::atoi(argv[3]) != 0);

    const int W = 1024, H = 1024;

    cst::rt::Config cfg;
    cfg.width = W;
    cfg.height = H;
    cfg.ss = 1;
    cfg.depth = 15;
    cfg.scene = 0;
    const cst::rt::Scene sc = cst::rt::make_scene(cfg.scene, cfg.seed);

    std::vector<double> d8buf;
    std::vector<float> f16buf;
    const size_t chan = static_cast<size_t>(W) * H * 3;
    if (f32) {
        f16buf.assign(chan, 0.0f);
        for (int y = 0; y < H; ++y) {
            cst::rt::trace_frame_jittered_f32(sc, cfg, f16buf.data(), y, y + 1, 0.0f);
        }
    } else {
        d8buf.assign(chan, 0.0);
        for (int y = 0; y < H; ++y) {
            cst::rt::trace_frame_jittered_f64(sc, cfg, d8buf.data(), y, y + 1, 0.0);
        }
    }

    // Load the AIDA64 capture (BGRA8).
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::vector<unsigned char> ref(static_cast<size_t>(W) * H * 4);
    const size_t got = std::fread(ref.data(), 1, ref.size(), f);
    std::fclose(f);
    if (got != ref.size()) {
        std::fprintf(stderr, "short read: %zu\n", got);
        return 1;
    }

    // Optional: dump my render as P6 PPM for offline correlation analysis.
    if (argc >= 5) {
        std::FILE* o = std::fopen(argv[4], "wb");
        if (o) {
            std::fprintf(o, "P6\n%d %d\n255\n", W, H);
            for (size_t p = 0; p < static_cast<size_t>(W) * H; ++p) {
                unsigned char px[3];
                // Match dbg_render/image.bin: byte order B, G, R.
                for (int c = 0; c < 3; ++c) {
                    const double v = f32 ? static_cast<double>(f16buf[p * 3 + c])
                                         : d8buf[p * 3 + c];
                    double s = v * 255.0 + 0.5;
                    if (s < 0) s = 0;
                    if (s > 255) s = 255;
                    // c = 0(R),1(G),2(B) -> PPM wants R,G,B, but write BGR order
                    // so the file matches the .bin layout when inspected.
                    px[2 - c] = static_cast<unsigned char>(s);
                }
                std::fwrite(px, 1, 3, o);
            }
            std::fclose(o);
            std::printf("wrote %s\n", argv[4]);
        }
    }

    // The AIDA64 framebuffer is stored bottom-up (its renderer writes row 0 of
    // the image at the far end of the buffer), while our render is top-down, so
    // the capture has to be flipped before differencing.  Without this the
    // comparison scores ~49/255 mean absolute error instead of ~1.35.
    {
        std::vector<unsigned char> flipped(ref.size());
        const size_t stride = static_cast<size_t>(W) * 4;
        for (int y = 0; y < H; ++y) {
            std::memcpy(&flipped[y * stride], &ref[(H - 1 - y) * stride], stride);
        }
        ref.swap(flipped);
    }

    // Compare my RGB against the capture interpreted as BGRA, then as RGBA.
    const char* names[2] = {"BGRA", "RGBA"};
    for (int ord = 0; ord < 2; ++ord) {
        const int ri = ord == 0 ? 2 : 0;
        const int gi = 1;
        const int bi = ord == 0 ? 0 : 2;

        double sum_abs[3] = {0, 0, 0};
        double sum_ref[3] = {0, 0, 0};
        double sum_mine[3] = {0, 0, 0};
        size_t within1 = 0, within2 = 0, within4 = 0, within8 = 0, within16 = 0;

        for (size_t p = 0; p < static_cast<size_t>(W) * H; ++p) {
            const unsigned char R = ref[p * 4 + ri];
            const unsigned char G = ref[p * 4 + gi];
            const unsigned char B = ref[p * 4 + bi];
            int mine[3];
            if (f32) {
                mine[0] = static_cast<int>(f16buf[p * 3 + 0] * 255.0f + 0.5f);
                mine[1] = static_cast<int>(f16buf[p * 3 + 1] * 255.0f + 0.5f);
                mine[2] = static_cast<int>(f16buf[p * 3 + 2] * 255.0f + 0.5f);
            } else {
                mine[0] = static_cast<int>(d8buf[p * 3 + 0] * 255.0 + 0.5);
                mine[1] = static_cast<int>(d8buf[p * 3 + 1] * 255.0 + 0.5);
                mine[2] = static_cast<int>(d8buf[p * 3 + 2] * 255.0 + 0.5);
            }
            const int rv[3] = {R, G, B};
            int maxdiff = 0;
            for (int c = 0; c < 3; ++c) {
                const int d = std::abs(mine[c] - rv[c]);
                sum_abs[c] += d;
                sum_ref[c] += rv[c];
                sum_mine[c] += mine[c];
                if (d > maxdiff) maxdiff = d;
            }
            if (maxdiff <= 1) ++within1;
            if (maxdiff <= 2) ++within2;
            if (maxdiff <= 4) ++within4;
            if (maxdiff <= 8) ++within8;
            if (maxdiff <= 16) ++within16;
        }

        const double n = static_cast<double>(W) * H;
        std::printf("--- capture read as %s (mine is RGB) ---\n", names[ord]);
        for (int c = 0; c < 3; ++c) {
            const char* cn = "RGB";
            (void)cn;
            std::printf("  ch%d: ref mean %7.3f  mine mean %7.3f  mean|d| %7.3f\n",
                        c, sum_ref[c] / n, sum_mine[c] / n, sum_abs[c] / n);
        }
        std::printf("  pixel |d|max <=1: %.1f%%  <=2: %.1f%%  <=4: %.1f%%  <=8: %.1f%%  <=16: %.1f%%\n",
                    100.0 * within1 / n, 100.0 * within2 / n, 100.0 * within4 / n,
                    100.0 * within8 / n, 100.0 * within16 / n);
        if (strip) {
            // Same again with the per-channel mean removed from both sides.
            double sa[3] = {0, 0, 0};
            size_t w2 = 0, w8 = 0;
            for (size_t p = 0; p < static_cast<size_t>(W) * H; ++p) {
                int mine[3];
                if (f32) {
                    mine[0] = static_cast<int>(f16buf[p * 3 + 0] * 255.0f + 0.5f);
                    mine[1] = static_cast<int>(f16buf[p * 3 + 1] * 255.0f + 0.5f);
                    mine[2] = static_cast<int>(f16buf[p * 3 + 2] * 255.0f + 0.5f);
                } else {
                    mine[0] = static_cast<int>(d8buf[p * 3 + 0] * 255.0 + 0.5);
                    mine[1] = static_cast<int>(d8buf[p * 3 + 1] * 255.0 + 0.5);
                    mine[2] = static_cast<int>(d8buf[p * 3 + 2] * 255.0 + 0.5);
                }
                const int rv[3] = {ref[p * 4 + ri], ref[p * 4 + gi], ref[p * 4 + bi]};
                int md = 0;
                for (int c = 0; c < 3; ++c) {
                    const double dm = mine[c] - sum_mine[c] / n;
                    const double dr = rv[c] - sum_ref[c] / n;
                    sa[c] += std::fabs(dm - dr);
                    const int d = static_cast<int>(std::fabs(dm - dr) + 0.5);
                    if (d > md) md = d;
                }
                if (md <= 2) ++w2;
                if (md <= 8) ++w8;
            }
            std::printf("  mean-stripped mean|d| = %.3f %.3f %.3f   <=2: %.1f%%  <=8: %.1f%%\n",
                        sa[0] / n, sa[1] / n, sa[2] / n,
                        100.0 * w2 / n, 100.0 * w8 / n);
        }
    }
    return 0;
}
