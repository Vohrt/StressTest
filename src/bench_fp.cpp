#include "bench_fp.h"
#include "common.h"
#include "cst.h"
#include "fractal_kernel.h"
#include "raytrace.h"
#include "simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace cst {
namespace {

// ===========================================================================
// Ray-Trace
//
// One work unit == one traced primary ray.  AIDA64 reports this test in
// kilo-rays/second (its internal key is literally "RayTraceKiloRayPerSec"),
// so unit_scale converts rays/s to KRay/s.
// ===========================================================================
class RayTraceBench : public Benchmark {
public:
    explicit RayTraceBench(bool f64) : f64_(f64) {}

    BenchInfo info() const override {
        BenchInfo i;
        i.test = f64_ ? "f64ray" : "f32ray";
        i.label = f64_ ? "FP64 Ray-Trace" : "FP32 Ray-Trace";
        i.unit = "KRay/s";
        i.group = "FPU";
        i.unit_scale = 1e-3;
        const bool avx512 = rt::avx512_selected();
        i.description = f64_
            ? (avx512 ? "Double-precision analytic ray tracer, 8 rays/iteration (AVX-512)"
                      : "Double-precision analytic ray tracer, 4 rays/iteration (AVX2+FMA)")
            : (avx512 ? "Single-precision analytic ray tracer, 16 rays/iteration (AVX-512)"
                      : "Single-precision analytic ray tracer, 8 rays/iteration (AVX2+FMA)");
        return i;
    }

    void prepare(int threads) override {
        threads_ = std::max(1, threads);
        cfg_.width = 1024;
        cfg_.height = 1024;
        cfg_.ss = 1;
        cfg_.depth = 15;     // reflection budget of the reference kernel
        cfg_.scene = 0;      // reference layout: floor + three glossy spheres
        cfg_.seed = 0x5eed1234ull;

        // A small ring of pre-built, immutable scenes.  Rebuilding the scene
        // inside run_once() would race with the other workers, so we build them
        // all up front and just rotate which one a repetition uses.
        scenes_.clear();
        scenes_.reserve(kSceneRing);
        for (int i = 0; i < kSceneRing; ++i) {
            scenes_.push_back(rt::make_scene(cfg_.scene, cfg_.seed + 0x9E3779B9ull * i));
        }

        const size_t pixels = static_cast<size_t>(cfg_.width) * cfg_.height * 3;
        if (f64_) {
            f64buf_.assign(static_cast<size_t>(threads_) * pixels, 0.0);
            f64canon_.assign(pixels, 0.0);
        } else {
            f32buf_.assign(static_cast<size_t>(threads_) * pixels, 0.0f);
            f32canon_.assign(pixels, 0.0f);
        }
        rep_.assign(static_cast<size_t>(threads_), 0);
    }

    // Repetition-to-repetition variation comes from a deterministic rotation
    // over the scene ring plus a per-worker sub-pixel jitter.  Both are derived
    // from the repetition counter, so from the workload's point of view every
    // repetition does exactly the same amount of work, yet no repetition can be
    // memoised.  Each worker only touches its own buffer and its own counter.
    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)min_seconds;
        const int W = cfg_.width, H = cfg_.height;
        uint64_t rays = 0;
        const uint64_t rep = rep_[static_cast<size_t>(worker)];
        const rt::Scene& scene = scenes_[static_cast<size_t>(rep % kSceneRing)];
        const double jitter = static_cast<double>(rep % 16) * 0.001953125;

        // Trace a slice of lines, interleaved across workers.  Chunking the
        // scanlines keeps every worker busy on a similar mix of expensive and
        // cheap rows, mirroring AIDA64's "completed lines" progress metric.
        const int chunk = 4;
        for (int y0 = worker * chunk; y0 < H; y0 += thread_count * chunk) {
            const int y1 = std::min(y0 + chunk, H);
            if (f64_) {
                double* buf = f64buf_.data() + static_cast<size_t>(worker) * W * H * 3;
                rays += rt::trace_frame_jittered_f64(scene, cfg_, buf, y0, y1, jitter);
            } else {
                float* buf = f32buf_.data() + static_cast<size_t>(worker) * W * H * 3;
                rays += rt::trace_frame_jittered_f32(scene, cfg_, buf, y0, y1,
                                                     static_cast<float>(jitter));
            }
        }

        // Repeat repetition 0 into the canonical buffer, which is what digest()
        // reports, so the digest is reproducible across runs.
        if (rep == 0) {
            if (f64_) {
                for (int y0 = worker * chunk; y0 < H; y0 += thread_count * chunk) {
                    const int y1 = std::min(y0 + chunk, H);
                    rt::trace_frame_jittered_f64(scene, cfg_, f64canon_.data(), y0, y1, 0.0);
                }
            } else {
                for (int y0 = worker * chunk; y0 < H; y0 += thread_count * chunk) {
                    const int y1 = std::min(y0 + chunk, H);
                    rt::trace_frame_jittered_f32(scene, cfg_, f32canon_.data(), y0, y1, 0.0f);
                }
            }
        }

        rep_[static_cast<size_t>(worker)] = rep + 1;
        return static_cast<double>(rays);
    }

    uint64_t digest() const override {
        const size_t pixels = static_cast<size_t>(cfg_.width) * cfg_.height * 3;
        if (f64_) return hash_bytes(f64canon_.data(), pixels * sizeof(double));
        return hash_bytes(f32canon_.data(), pixels * sizeof(float));
    }
    bool has_digest() const override { return true; }

    // Fingerprint of a complete frame traced from scratch, on one thread, from
    // the canonical scene.  Unlike digest() - which hashes whatever the workers
    // happened to produce - this is independent of the thread count, so it can be
    // compared with a reference recorded by an earlier run.
    uint64_t full_digest() const override {
        RayTraceBench* self = const_cast<RayTraceBench*>(this);
        const size_t pixels = static_cast<size_t>(cfg_.width) * cfg_.height * 3;
        if (f64_) {
            self->f64verify_.assign(pixels, 0.0);
            for (int y = 0; y < cfg_.height; ++y) {
                rt::trace_frame_jittered_f64(self->scenes_[0], cfg_, self->f64verify_.data(),
                                             y, y + 1, 0.0);
            }
            return corrupt_if_testing(
                hash_bytes(self->f64verify_.data(), pixels * sizeof(double)));
        }
        self->f32verify_.assign(pixels, 0.0f);
        for (int y = 0; y < cfg_.height; ++y) {
            rt::trace_frame_jittered_f32(self->scenes_[0], cfg_, self->f32verify_.data(),
                                         y, y + 1, 0.0f);
        }
        return corrupt_if_testing(
            hash_bytes(self->f32verify_.data(), pixels * sizeof(float)));
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        return "res=" + std::to_string(cfg_.width) + "x" + std::to_string(cfg_.height) +
               " ss=" + std::to_string(cfg_.ss) +
               " depth=" + std::to_string(cfg_.depth) +
               " spheres=" + std::to_string(scenes_.empty() ? 0 : scenes_[0].sphere_count) +
               " method=" + (f64_ ? rt::f64_method_name() : rt::f32_method_name());
    }

private:
    static constexpr int kSceneRing = 8;
    bool f64_;
    int threads_ = 1;
    rt::Config cfg_;
    std::vector<rt::Scene> scenes_;
    std::vector<uint64_t> rep_;
    std::vector<double> f64buf_;
    std::vector<float> f32buf_;
    std::vector<double> f64canon_;
    std::vector<float> f32canon_;
    std::vector<double> f64verify_;
    std::vector<float> f32verify_;
};

// ===========================================================================
// ===========================================================================
// Fractals (Julia / Mandel / SinJulia)
//
// Algorithmically aligned with AIDA64's NGFractal kernels (see
// docs/reverse-engineering.md and src/fractal_kernel.h): image size, iteration
// budget, precision, escape tests and the animated view all match.
//   julia    - f32,  1024x1024, 999 iterations, z' = z^2 + c, z0 = pixel,
//              c = -0.770885 + 0.1i
//   mandel   - f64,  1024x1024, 999 iterations, z' = z^2 + c, z0 = c = pixel
//   sinjulia - x87-extended (emulated in double), 256x256, 70 iterations,
//              z' = c * sin(z), c = 0.98059 + 0.45075i
// ===========================================================================

class FractalBench : public Benchmark {
public:
    explicit FractalBench(fractal::Kind kind) : p_(fractal::make_params(kind)) {}

    BenchInfo info() const override {
        BenchInfo i;
        i.unit = "MPixel/s";
        i.group = "FPU";
        i.unit_scale = 1e-6;   // pixels -> megapixels
        switch (p_.kind) {
            case fractal::Kind::Julia: i.test = "julia"; i.label = "FPU Julia"; break;
            case fractal::Kind::Mandel: i.test = "mandel"; i.label = "FPU Mandel"; break;
            case fractal::Kind::SinJulia: i.test = "sinjulia"; i.label = "FPU SinJulia"; break;
        }
        i.description = (p_.kind == fractal::Kind::SinJulia)
            ? "z' = c*sin(z) Julia variant, extended-precision transcendentals"
            : (p_.kind == fractal::Kind::Julia
                   ? "Escape-time iteration z' = z^2 + c, f32 pipeline"
                   : "Escape-time iteration z' = z^2 + c, f64 pipeline, z0 = c = pixel");
        return i;
    }

    void prepare(int threads) override {
        threads_ = std::max(1, threads);
        buf_.assign(static_cast<size_t>(threads_) * p_.w * p_.h, 0.0f);
        rep_.assign(static_cast<size_t>(threads_), 0);
    }

    double run_once(int worker, int thread_count, double min_seconds) override {
        (void)min_seconds;
        // One full frame per call, like AIDA64's worker loop; the view is spun
        // by pi/400 per frame and the per-pixel work never changes.
        const int frame = rep_[worker]++;
        const fractal::View view = fractal::make_view(p_, fractal::frame_angle(frame));
        float* img = buf_.data() + static_cast<size_t>(worker) * p_.w * p_.h;
        for (int y = worker; y < p_.h; y += thread_count)
            fractal::render_row(p_, view, y, img + static_cast<size_t>(y) * p_.w);
        return static_cast<double>(p_.w) * static_cast<double>(p_.h);
    }

    // The reported digest must be reproducible for a given workload, so it comes
    // from a fixed slice rendered just for the check rather than from whichever
    // rows the workers happened to touch.  That is outside the measured loop, so
    // throughput is unaffected.
    uint64_t digest() const override {
        const int rows = 8 < p_.h ? 8 : p_.h;
        std::vector<float> slice(static_cast<size_t>(rows) * p_.w);
        const fractal::View view = fractal::make_view(p_, 0.0);
        fractal::render_rows(p_, view, 0, rows, slice.data(), p_.w);
        return hash_bytes(slice.data(), slice.size() * sizeof(float));
    }
    bool has_digest() const override { return true; }

    // Re-renders a fixed reference image and returns its fingerprint.  Much
    // larger than the per-repeat slice above, so an unstable CPU that corrupts
    // only some regions is still caught; the cost is well under one second and
    // is charged to verification, not to the measured loop.
    uint64_t full_digest() const override {
        std::vector<float> img(static_cast<size_t>(p_.w) * p_.h);
        const fractal::View view = fractal::make_view(p_, 0.0);
        fractal::render_rows(p_, view, 0, p_.h, img.data(), p_.w);
        return corrupt_if_testing(hash_bytes(img.data(), img.size() * sizeof(float)));
    }
    bool has_full_digest() const override { return true; }

    std::string params() const override {
        std::string s = "res=" + std::to_string(p_.w) + "x" + std::to_string(p_.h) +
                        " iter=" + std::to_string(p_.max_iter) +
                        " rot=pi/400/frame";
        if (p_.kind == fractal::Kind::Julia) {
            s += " precision=fp32 c=-0.770885+0.1i";
        } else if (p_.kind == fractal::Kind::Mandel) {
            s += " precision=fp64 z0=c=pixel";
        } else {
            s += " precision=x87-extended-emulated z'=c*sin(z) c=0.980591+0.450747i";
        }
        return s;
    }

private:
    fractal::Params p_;
    int threads_ = 1;
    std::vector<float> buf_;
    std::vector<int> rep_;
};

} // namespace

Benchmark* make_bench_f64ray() { static RayTraceBench b(true); return &b; }
Benchmark* make_bench_f32ray() { static RayTraceBench b(false); return &b; }
Benchmark* make_bench_julia() { static FractalBench b(fractal::Kind::Julia); return &b; }
Benchmark* make_bench_mandel() { static FractalBench b(fractal::Kind::Mandel); return &b; }
Benchmark* make_bench_sinjulia() { static FractalBench b(fractal::Kind::SinJulia); return &b; }

} // namespace cst