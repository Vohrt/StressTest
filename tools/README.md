# Regression checks

Two checks here go beyond the built-in self-tests by validating against an
*independent* implementation. Both need the build directory to exist first
(`.\build.ps1`), and the `cl` snippets below write into `build\obj\`:

```powershell
New-Item -ItemType Directory -Force build\obj | Out-Null
```

## 1. DEFLATE against CPython's zlib

`gen_zlib_ref.py` writes `zlib_ref.h` with reference streams produced by
CPython's `zlib`; `../../build` then decodes them with the module's own
`inflate`. This is the check that caught the compressor's stored-block framing
and its distance-too-far-back bug.

```powershell
$py = '<path to python>'
& $py tools\gen_zlib_ref.py
$vc = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "call `"$vc`" >nul && cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3 /I src /I tools /Fe:build\verify_inflate.exe tools\verify_inflate.cpp src\common.cpp /Fo:build\obj\\"
.\build\verify_inflate.exe     # expect "0/9 cases failed"
```

`validate_zlib.py` runs the same idea in the other direction: it compresses a
range of input sizes with the module's compressor (via `verify_compress.exe`) and asks
CPython's `zlib` to decode them.

```powershell
cmd /c "call `"$vc`" >nul && cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3 /D_CRT_SECURE_NO_WARNINGS /I src /Fe:build\verify_compress.exe tools\verify_compress.cpp src\common.cpp /Fo:build\obj\\"
& $py tools\validate_zlib.py   # expect "failures: 0" across 30 sizes
```

## 2. Fractal kernel against a scalar reference

`verify_fractal.cpp` re-implements the AIDA64-aligned kernels in
`src/fractal_kernel.h` scalar-ly (same formulas, same precision, same evaluation
order) and compares them pixel by pixel with the SIMD versions, for all three
methods and two animation frames.

```powershell
cmd /c "call `"$vc`" >nul && cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3 /I src /Fe:build\verify_fractal.exe tools\verify_fractal.cpp /Fo:build\obj\\"
.\build\verify_fractal.exe     # expect mismatches=0 on every row
```

## 2b. Fractal frames against AIDA64's own dumps

`fractal_dump.cpp` renders one frame with the benchmark's kernels and writes raw
uint32 iteration counts plus a PGM preview:

```powershell
.\build\fractal_dump.exe julia 0 out\julia_f0      # also: mandel, sinjulia
```

Compare the `.bin` against the BMP frames AIDA64 saves from the same method
(remember to flip one of them - the BMPs are stored bottom-up). Measured
agreement is recorded in `docs/reverse-engineering.md` §8: the Mandel interior
and count-0 masks match pixel-exactly, Julia matches to within a 2-pixel
boundary band, SinJulia is a uniform frame on both sides.

## 3. Ray-tracer image sanity

`verify_rt.cpp` traces a frame and reports colour statistics, catching a
degenerate all-zero or all-NaN image:

```powershell
cmd /c "call `"$vc`" >nul && cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /W3 /I src /Fe:build\verify_rt.exe tools\verify_rt.cpp src\raytrace.cpp src\common.cpp /Fo:build\obj\\"
.\build\verify_rt.exe
```

Expect a non-zero `mean` (roughly 0.42 with the default settings), a varied
centre row rather than a constant, and `nonzero` covering nearly every sample.

## 4. Kernel-width micro-benchmark

`bench_kernels.cpp` times each ray-tracer width directly - both widths in one
translation unit compiled with `/arch:AVX512`, so nothing is split or spilled by
the build flags. This is what settled the dispatch question: on this machine
(Zen 5, MSVC 14.51) the wide kernels lose to the narrow ones while producing
byte-identical output, which is why the AVX2 kernels are the default (see the
"Resolved" note in the main README).

```powershell
cmd /c "call `"$vc`" >nul && cl /nologo /std:c++17 /O2 /arch:AVX512 /EHsc /W3 /D_CRT_SECURE_NO_WARNINGS /I src /Fe:build\bench_kernels.exe tools\bench_kernels.cpp src\common.cpp src\raytrace.cpp src\raytrace_avx512.cpp /Fo:build\obj\\"
.\build\bench_kernels.exe 3 f64
.\build\bench_kernels.exe 3 f32
```
