// ===========================================================================
// Dumps one NGFractal frame using the exact kernels the benchmark runs, so the
// output can be cross-checked against AIDA64's own frame dumps (see
// docs/reverse-engineering.md - "NGFractal alignment").
//
//   fractal_dump <julia|mandel|sinjulia> <frame> <out-prefix>
//
// Outputs:
//   <prefix>.bin  raw little-endian uint32 iteration counts, w*h entries
//                 (row-major, row 0 on top, same orientation as the BMPs)
//   <prefix>.pgm  8-bit P5 preview, counts scaled by 255/max_iter
// ===========================================================================
#include "../src/fractal_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool write_all(const std::string& path, const void* data, size_t size) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: fractal_dump <julia|mandel|sinjulia> <frame> <out-prefix>\n");
        return 2;
    }

    const std::string name = argv[1];
    cst::fractal::Kind kind;
    if (name == "julia") kind = cst::fractal::Kind::Julia;
    else if (name == "mandel") kind = cst::fractal::Kind::Mandel;
    else if (name == "sinjulia") kind = cst::fractal::Kind::SinJulia;
    else { std::fprintf(stderr, "unknown fractal '%s'\n", name.c_str()); return 2; }

    const int frame = std::atoi(argv[2]);
    const cst::fractal::Params p = cst::fractal::make_params(kind);
    std::vector<float> img(static_cast<size_t>(p.w) * p.h);
    cst::fractal::render_frame(p, frame, img.data(), p.w);

    std::vector<uint32_t> counts(img.size());
    std::vector<uint8_t> pgm(img.size());
    const float maxc = static_cast<float>(p.max_iter);
    for (size_t i = 0; i < img.size(); ++i) {
        const float v = img[i] * maxc;
        counts[i] = static_cast<uint32_t>(v + 0.5f);
        pgm[i] = static_cast<uint8_t>(img[i] * 255.0f + 0.5f);
    }

    const std::string bin_path = std::string(argv[3]) + ".bin";
    if (!write_all(bin_path, counts.data(), counts.size() * sizeof(uint32_t))) {
        std::fprintf(stderr, "cannot write %s\n", bin_path.c_str());
        return 1;
    }

    char header[64];
    const int hn = std::snprintf(header, sizeof header, "P5\n%d %d\n255\n", p.w, p.h);
    const std::string pgm_path = std::string(argv[3]) + ".pgm";
    std::FILE* f = std::fopen(pgm_path.c_str(), "wb");
    if (!f || std::fwrite(header, 1, static_cast<size_t>(hn), f) != static_cast<size_t>(hn) ||
        std::fwrite(pgm.data(), 1, pgm.size(), f) != pgm.size()) {
        if (f) std::fclose(f);
        std::fprintf(stderr, "cannot write %s\n", pgm_path.c_str());
        return 1;
    }
    std::fclose(f);

    std::printf("%s: %dx%d iter=%d frame=%d angle=%.9f rad -> %s / %s\n",
                name.c_str(), p.w, p.h, p.max_iter, frame,
                cst::fractal::frame_angle(frame), bin_path.c_str(), pgm_path.c_str());
    return 0;
}
