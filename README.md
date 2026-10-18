# CpuStressTest

A Windows benchmarking/stress tool that re-implements the workload **families** of
the AIDA64 CPU / FPU / Memory benchmark page, so they can be run **once** or
**sustained for a fixed wall-clock duration** (e.g. 20 minutes).

AIDA64's own benchmark page has no repeat/loop mode — every measurement is a
fixed few-second burst that ends by printing a score. This tool fills exactly
that gap: the same class of workloads, but with a sustained mode that reports
throughput over time, stability, drift and a throttling verdict.

**Priority test: FP64 Ray-Trace** (and its FP32 twin), implemented as a real
SIMD ray tracer rather than a synthetic floating-point loop.

---

## 1. Background: what was recovered from AIDA64

The workloads were derived from two sources: FinalWire's public documentation
(workload descriptions, buffer sizes, units) and static analysis of the locally
installed benchmark engine, `aida_bench64.dll` ("AIDA64 Benchmark Module 4.7",
13 MB, MSVC C++). Full notes are in [docs/reverse-engineering.md](docs/reverse-engineering.md).

| Recovered fact | Source string in `aida_bench64.dll` |
|---|---|
| The metric of the ray-trace test is **kilo-rays/second** | `RayTraceKiloRayPerSec` |
| Progress is counted in **scanlines**, not frames | `RayTraceCompletedLines` |
| Two methods: double and single precision | `RayTraceDouble`, `RayTraceSingle` |
| Workload knobs | `RayTraceX`, `RayTraceY`, `RayTraceDepth`, `RayTraceScene`, `RayTraceErrMargin`, `RayTraceStartMethod`, `RayTraceMinRunTime` |
| Per-method shape | `RayTraceFunctionXSize%02d`, `...YSize%02d`, `...MaxDepth/MinDepth/DefDepth%02d`, `...MinScene/MaxScene/DefScene%02d`, `...ErrMargin%02d%02d` |
| SIMD implementations are chosen per CPU | `Base128_SSE2`, `Base128_SSE41`, `Base256_AVX_FMA3`, `Base256_AVX2_FMA3`, `Base512_AVX512F`, `Base512_AVX512BWDQ`, `TP*` (two-pipe) variants |
| Julia / Mandel / SinJulia are three *methods* of one fractal test | `Julia`, `Mandel`, `SinJulia`, `NGFractalMethod[XY]Size%02d`, `NGFractalMaxIter%02d` |
| ZLib test is pinned to zlib 1.2.5 | `Base_ZLib_1_2_5` |
| A debug switch exists to dump the JIT-assembled kernels | `RayTraceDumpCode`, `CodeDump_RayTrace_%s_%05d.bin` |
| AIDA64 assembles its kernels at runtime | RTTI classes `BenchAsm_Inst_*`, `BenchAsm_REX/VEX/EVEX` |

Three conclusions shaped the design:

1. **AIDA64 assembles its benchmark kernels at runtime** (the `BenchAsm_*`
   code-generator classes), so there is no static `FP64 Ray-Trace` function to
   disassemble. The algorithm is expressed in C++ generator code.
2. Therefore this project does **not** copy or re-derive AIDA64 machine code. It
   re-implements equivalent workloads in original C++17 using hand-written
   SIMD, and offers a **calibration mechanism** to place its numbers on the same
   scale as AIDA64's (see §6).
3. The score formula and normalisation for the FP/Queen tests are **not publicly
   documented**, which is why calibration is a first-class feature here.

## 2. What is implemented

| id | label | unit | workload |
|---|---|---|---|
| `f64ray` | FP64 Ray-Trace | KRay/s | double-precision analytic ray tracer, 4 rays/iteration AVX2+FMA (8 on the AVX-512 path, `CST_FORCE_AVX512=1`) |
| `f32ray` | FP32 Ray-Trace | KRay/s | single-precision analytic ray tracer, 8 rays/iteration AVX2+FMA (16 on the AVX-512 path) |
| `julia` | FPU Julia | MPixel/s | AIDA64-aligned Julia escape-time fractal, f32, 1024×1024 × 999 iterations/frame |
| `mandel` | FPU Mandel | MPixel/s | AIDA64-aligned Mandelbrot, f64, 1024×1024 × 999 iterations/frame |
| `sinjulia` | FPU SinJulia | MPixel/s | AIDA64-aligned z' = c·sin(z) variant (x87-extended emulated), 256×256 × 70 iterations/frame |
| `aes` | CPU AES | MiB/s | AIDA64 NGBase_AES layout: AES-256 ECB encryption, 8 MiB buffer per worker, VAES-512 (16 blocks) / AES-NI (8) / scalar S-box |
| `sha3` | CPU SHA3 | MiB/s | AIDA64 Hash2 layout: SHA3-512 of independent 8 KiB messages, rate = rounds * 8192 / 2^20 / s, AVX-512 (2 messages in flight) / scalar Keccak-f[1600] |
| `zlib` | CPU ZLib | MB/s | AIDA64 NGPack `Base_ZLib_1_2_5`: level-9 DEFLATE of the 32 MiB triangular-dword corpus in 32 KiB Z_SYNC_FLUSH steps, rate = bytes * 1e-6 / s |
| `photoworxx` | CPU PhotoWorxx | MPixel/s | AIDA64 NGImage2D pipeline: ten stages (SWRandom 4096×3072, Rot90, Rot180, RGB2YUV, VP8YUV 3072×2048, Diff 2048×2048, Pattern, Access, Fill 4096×3072, Move 3072×2048), score = sum of the stage rates |
| `queen` | CPU Queen | kPos/s | 10×10 N-Queens: all 10! = 3,628,800 column permutations (no pruning) screened by a byte-lane SIMD diagonal test, 724 solutions per pass |
| `checkmate` | CPU CheckMate | MP/s | AIDA64 v7.35 NGQueen (`AVX512_VPOPCNTDQ` flavour): complete 16! permutation space in 3-rank chunks — a 13-column factoradic prefix plus the 3! = 6 completions of the last three ranks, one checked position per permutation |
| `memread` | Memory Read | MB/s | 96 MiB, 64 KiB-aligned buffer read per worker (non-temporal 64-byte loads) |
| `memwrite` | Memory Write | MB/s | 96 MiB, 64 KiB-aligned buffer streamed with non-temporal stores of all-zero blocks (AIDA64's write kernel stores a `vpxord`-zeroed register) |
| `memcopy` | Memory Copy | MB/s | 48 MiB → 48 MiB non-temporal copy per worker (all-zero source, as AIDA64's) |
| `memlat` | Memory Latency | ns | AIDA64 NGMemLat band 0: 512 MiB page-random chain, 128-byte stride, 8 MiB windows, one line/page/pass, single-threaded |

Notes on fidelity:

* One work unit is one traced **primary ray**, so the ray-trace figure is
  kilo-rays/second — the quantity AIDA64 uses.
* `checkmate` reports **MP/s** (millions of checked positions per second), the
  unit AIDA64's v7.35 CPU CheckMate prints.  One position is one board of the
  complete 16! permutation space — nothing is pruned — and the kernel walks it
  in 3-rank chunks (a factoradic prefix of 13 columns plus the 3! = 6
  completions of the last three ranks, recounted as +6 positions per chunk),
  the same accounting the DLL's `AVX512_VPOPCNTDQ` kernel uses: driven with an
  8 x 8 board the DLL completes it every 8! = 40,320 positions with 92
  solutions.  See `docs/reverse-engineering.md`, section 16.
* `queen` reports **kPos/s**, AIDA64's "checked positions per second": one
  position is one complete candidate board — the kernel scans *all* 10! column
  permutations (no pruning, unlike the CheckMate kernel) and screens each one
  with the same byte-lane SIMD diagonal test shape as AIDA64's NGQueen kernel,
  so the two figures are directly comparable (see
  `docs/reverse-engineering.md`, section 10).
* The sin/cos/exp-heavy path uses polynomial approximations evaluated in double
  precision rather than `libm`, matching the "x87 transcendental" character
  AIDA64 documents for SinJulia while staying deterministic and fast.
* The memory bandwidth kernels stream the same bytes AIDA64's do, all-zero
  blocks included: on Zen5 the all-zero non-temporal store path runs about 3x
  faster than any non-zero pattern, so `memwrite` / `memcopy` are only
  comparable with AIDA64's numbers when the stored data is all zero (see
  `docs/reverse-engineering.md`, section 9).

## 3. Build

Requires the Visual Studio C++ workload (tested with Visual Studio 18 / MSVC
14.51). The only packages are the Windows App SDK and C++/WinRT from NuGet;
restore happens automatically (`build.ps1` passes `/restore`).

```powershell
.\build.ps1                       # Release build of the whole solution
.\build.ps1 -Configuration Debug  # Debug build into build\debug\
.\build.ps1 -SelfTests            # only the module self-tests
```

`build.ps1` drives MSBuild over `CpuStressTest.sln` - the same solution you open
in Visual Studio. Release artifacts:

| artifact | what it is |
|---|---|
| `build\cst.exe` | the app: double-click opens the WinUI GUI, arguments run the command line |
| `build\lib\cst_engine.lib` | the measurement engine as a static library, linked into `cst.exe` |
| `build\selftest_crypto.exe` / `_image` / `_memory` | module self-tests |
| `build\fractal_dump.exe` | AIDA64 frame-alignment dumper |

The app is *unpackaged* and uses a **self-contained** Windows App SDK
(`WindowsAppSDKSelfContained`), so the build copies the runtime next to
`cst.exe` - a large set of `Microsoft.*.dll` / `*.pri` files plus the language
folders you can see in `build\`. That is what lets the exe run on a machine
without the Windows App Runtime installed, and it is needed for the command line
front end too: keep the `build\` folder together when you copy the app
somewhere. (`build\lib\`, `build\vs\` and `build\debug\` are build intermediates
and can be deleted at any time.)

The engine's translation units keep the flags the recorded reference
fingerprints were made with - C++17, `/O2`, `/arch:AVX2`, `/fp:precise`, `/MD`,
no precompiled header - while the WinUI shell is C++/WinRT (C++20, `/utf-8`,
precompiled header). `src/raytrace_avx512.cpp` and `src/mem_kernels_avx512.cpp`
override the ISA to `/arch:AVX512`: MSVC only keeps 512-bit intrinsics in real
registers when AVX-512 is part of the file's baseline, and those kernels are
roughly 3× faster for it. The rest of the program stays on `/arch:AVX2` and
enters them only behind a CPUID check, so the binary still runs on CPUs without
AVX-512.

### Visual Studio solution

`CpuStressTest.sln` opens the whole project in Visual Studio (x64 only, Debug
and Release):

| project | output | notes |
|---|---|---|
| `cst` | `build\cst.exe` | the WinUI 3 app **and** the command line front end |
| `cst_engine` | `build\lib\cst_engine.lib` | workloads, verification, calibration, measurement session |
| `selftest_crypto` / `selftest_image` / `selftest_memory` | `build\selftest_*.exe` | module self-tests |
| `fractal_dump` | `build\fractal_dump.exe` | AIDA64 frame-alignment dumper |

`cst` references `cst_engine`, so pressing F5 builds the engine and then the GUI
(F5 launches the GUI - the app starts the command line only when it receives
arguments). Release builds land in the same `build\` folder `build.ps1` writes,
so the app, the tools and VS always agree on where `cst.exe` is built. Debug
builds go to `build\debug\` (`build\lib\debug\` for the library) so they can
never overwrite the Release binary the recorded reference fingerprints were made
with - a Debug build does not reproduce the optimized floating-point code
generation, and the GUI says so in its log when you start it. NuGet restore is a
one-time step; Visual Studio runs it on first open, on the command line it is
`MSBuild.exe CpuStressTest.sln /restore`.

## 4. Usage

```
cst                              (no arguments: open the GUI)
cst gui
cst list [--tsv]
cst run <test|all> [options]
```

One executable carries both front ends: started without arguments (or with
`gui`) it opens the WinUI window, started with arguments it behaves like the
classic command line tool. In command line mode it attaches to the console it
was started from, so `cst list`, `cst run ...` and so on print as usual.

**Redirection.** `cmd` redirection and PowerShell pipes both work:

```powershell
cmd /c "build\cst.exe run all > run.txt"        # works
build\cst.exe run all | Select-String "FP64"    # works (waits for the run)
build\cst.exe run all --seconds 5 --csv run.csv # per-round samples
```

Windows PowerShell's own `>` operator, however, **cannot** capture a
GUI-subsystem program's output: PowerShell 5.1 does not read the child's pipe
for such a process, returns at once and leaves an empty file - `cst run all >
run.txt` produces a 0-byte `run.txt` no matter what the tool does. Use `cmd`
redirection, a pipe, `--csv`, or `Start-Process -RedirectStandardOutput ...
-Wait` when scripting from PowerShell.

`cst list --tsv` prints one tab-separated row per test (id, label, unit, group,
default threads, higher-is-better, description) with no banner - a
machine-readable listing for scripts.

| Option | Meaning |
|---|---|
| `--mode single` | one measurement of `--seconds` (default) |
| `--mode sustained` | repeat and sample until `--minutes` elapse |
| `--seconds N` | measurement length for single mode (default 15) |
| `--minutes N` | duration for sustained mode (default 20) |
| `--threads N` | worker threads (default: all logical CPUs; capped at the logical core count) |
| `--calibration FILE` | index calibration file (default `calibration.txt`) |
| `--reference FILE` | golden reference file (default `reference.txt`) |
| `--accept-reference` | record this run's fingerprints as the golden reference |
| `--no-verify` | skip reference verification (never forces a score to 0) |
| `--csv FILE` | write per-round samples |
| `--emit-calibration F` | write a calibration template |
| `--quiet` | suppress progress lines |

`<test>` also accepts a comma-separated list.

### GUI front end (WinUI 3)

The GUI is part of `build\cst.exe` (project `cst`, sources in `gui\`) and calls
the measurement session **in-process** - the same code the command line runs, so
a GUI run and a CLI run measure and verify identically, and nothing is parsed
from console output.

| control | what it does |
|---|---|
| test table | one row per workload; click anywhere on a row (or its check box) to include it; **全选** / **全不选** toggle everything |
| **运行选中** / **全部运行** | run the checked workloads in sequence, or every workload |
| **停止** | cancel the running session (the round in flight finishes first) |
| 单项 + 秒 | `--seconds N`: one measurement per workload |
| 持续 + 分钟 | `--minutes N`: sample once a second; the status column then shows the spread, the drift and the verdict (稳定 / 轻微衰减 / 降频 / 上升（异常）), with mean and p5/p95 in a tool tip |
| 线程 | `--threads N` (0 = all logical CPUs; per-test defaults then apply, e.g. Memory Latency runs single-threaded). The box is capped at the logical core count: the per-thread working sets multiply by it, so a value like 1024 used to be able to exhaust memory and kill the process |
| **记录为参考…** | `--accept-reference` for one run: asks for confirmation, then re-records the golden fingerprints of the checked workloads for that run only. It is a button, not a check box, and it is never written to the settings file - a recorded-once baseline cannot silently switch verification off in later sessions |
| 跳过校验 | `--no-verify`; starts unchecked on every launch and is not persisted either |
| 写出 CSV | `--csv <path>`; with several workloads the test id is prefixed to the file name |

Every row fills in throughput, unit, AIDA64 index and the verification state
(✓ 校验通过 / · 无参考记录 / ✗ 校验失败（得分按 0 计）). The **AIDA64 指数**
column shows 未校准 (and `[uncalibrated]` in the CLI) until `calibration.txt`
carries a real AIDA64 reference for that test, so the raw throughput is never
dressed up as a score. The log pane shows exactly the report blocks the command
line prints. Settings (duration, mode, threads, selection, CSV flag, window
size) persist to `%LOCALAPPDATA%\CpuStressTestGui\settings.ini`.

The window enforces a minimum size measured from the control rows and converted
through the display scale factor, so the toolbars cannot be clipped on
high-DPI monitors; a run in progress also survives closing the window (the
session stops at the next round and the detached measurement thread never
touches the window object again).

Measurements run with the repository root as the working directory (found by
walking up from `build\cst.exe` to the folder that holds `reference.txt`), so
references, calibration and relative CSV paths resolve exactly as they do on the
command line.

A UI-automation smoke test drives the parts that are easy to break - clearing
the selection, running a single workload, sustained run + stop, and the
record-reference dialog (取消 must not start anything, 运行并记录 must run and
re-record; `reference.txt` is backed up and restored around the test):

```powershell
powershell -ExecutionPolicy Bypass -File tools\gui_smoke.ps1
```

It launches the GUI, asserts those scenarios and prints `GUI SMOKE TEST PASSED`.

### The FP64 Ray-Trace workload

```powershell
# one measurement
.\build\cst.exe run f64ray --seconds 15

# sustained: 20 minutes, sampled once per second, with throttle analysis
.\build\cst.exe run f64ray --mode sustained --minutes 20 --csv f64ray.csv
```

One work unit is one traced **primary ray**, so the reported number is
kilo-rays/second — the same quantity AIDA64 uses. The default frame is 1024×1024
with 1×1 supersampling and reflection depth 15, rendering the four-sphere
reference scene of AIDA64's RayTrace test — the same geometry, camera, lights,
shading model (Lambert diffuse plus a `pow()` Phong highlight), `[5e-6, 1]`
segment window and one recursively traced reflection per bounce that the
JIT-generated `Base512_AVX512F` kernel uses (see
[docs/reverse-engineering.md](docs/reverse-engineering.md)). One repetition
traces 1024 × 1024 primary rays plus the shadow and reflection rays they spawn.

The scene is an immutable ring of 8 pre-built variants. Each repetition rotates
to the next variant and applies a deterministic sub-pixel jitter, so no
repetition can be memoised, while every repetition performs the same amount of
work. Repetition 0 also renders into a canonical buffer whose hash is checked
after every round: if the output ever stops being bit-identical the run reports
`CHANGED - result unstable!` instead of silently reporting a bogus throughput.

### Sustained-mode output

```
=== FP64 Ray-Trace (f64ray) ===
  workload      : Double-precision analytic ray tracer, 4 rays/iteration (AVX2+FMA)
  parameters    : res=1024x1024 ss=1 depth=15 spheres=4 method=Base256_AVX2_FMA3_F64
  threads       : 16
  elapsed       : 60.021 s over 183453 repetitions
  throughput    : 200309 KRay/s
  AIDA64 index  : 200309.2  [uncalibrated]
  verification  : matches reference
                  reference 0effdbc6000812f8 / computed 0effdbc6000812f8
  --- sustained ---
  samples       : 60
  min / median  : 185598 / 201884 KRay/s
  mean +- sd    : 200380 +- 3563 KRay/s (1.78%)
  p5 / p95      : 195173 / 203620
  first/last 1/3: 202546 -> 196251 (-3.11% drift)
  verdict       : mild decay
```

Values are printed as plain numbers next to the unit, so `200309 KRay/s` is
exactly AIDA64's quantity - no second prefix stacked onto a unit that already
carries one. `[uncalibrated]` (the GUI prints 未校准) means `calibration.txt`
has no AIDA64 reference for this test, so the index column names that instead of
repeating the raw throughput under a misleading label.

The verdict line is the point of sustained mode: `THROTTLING` when throughput
decays by more than 5 % from the first third of the run to the last third,
`mild decay` between -2 % and -5 %, `stable` inside that band. The run above,
taken next to everyday desktop load, sits in the mild-decay band; a quiet
20-minute run on this host reports `stable` with ~±0.5 % spread.

### Example results (Ryzen 7 9800X3D, single mode, 8 threads unless noted)

| test | measured |
|---|---|
| `f64ray` | ~202000 KRay/s on 16 threads (the default AVX2+FMA path); ~94500 with `CST_FORCE_AVX512=1` |
| `f32ray` | ~313000 KRay/s on 16 threads |
| `julia` | ~5140 MPixel/s on 16 threads |
| `mandel` | ~2520 MPixel/s |
| `sinjulia` | ~146 MPixel/s |
| `aes` | ~65000 MiB/s on 16 threads; ~87000 at 8 (AIDA64's CPU AES, same host: 46700 at 16 threads, 69900 at 8 — see docs §12) |
| `sha3` | ~4700 MiB/s on 16 threads; ~3000 at 8, ~430 single (AIDA64's CPU SHA3, same host: ~5400 MiB/s at 16 threads — see docs §13) |
| `zlib` | ~1.53–1.55 GB/s on 16 threads, ~1.0 GB/s at 8, ~150 MB/s single (AIDA64's CPU ZLib, same host: 1.44 GB/s at 16 threads, 728.7 MB/s at 8, 119.6 MB/s single — see docs §15) |
| `queen` | ~2310000 kPos/s; ~2690000 at 16 threads (AIDA64's CPU Queen at the same 16 threads: 1890000 kPos/s, its fastest NGQueen method) |
| `photoworxx` | ~94000 MPixel/s at 8 threads; ~107000 MPixel/s at 16 threads (AIDA64's ten NGImage2D stage rates sum to 95800 on the same host at 16 threads — per-stage table in docs §11) |
| `checkmate` | ~13400–14000 MP/s on 16 threads, ~8900–9600 at 8, ~1280–1330 single (AIDA64's CPU CheckMate, same host: 12710 MP/s at 16 threads, 6380 at 8, 1180 single — see docs §16) |
| `memread` | ~63 GB/s (AIDA64's Mem row: 63797.9 MiB/s at the same 8 threads) |
| `memwrite` | ~75 GB/s; ~85 GB/s at 2 threads, the count AIDA64's write row uses (78208.2 MiB/s) |
| `memcopy` | ~41 GB/s; ~60 GB/s at 2 threads, the count AIDA64's copy row uses (60838.1 MiB/s) |
| `memlat` | ~78–82 ns (AIDA64's NGMemLat band 0 `Page_Random` row: 79.2 ns on the same host, single CPU — per-band table in docs §14) |

> **Resolved: the ray-trace dispatch now prefers the AVX2 kernels.**
> The AVX-512 instantiation of this kernel is 2–3× slower than the AVX2 one on
> this machine (Ryzen 7 9800X3D, MSVC 14.51), so `src/raytrace.cpp` selects the
> AVX2+FMA kernels by default and leaves the wide path reachable behind
> `CST_FORCE_AVX512=1`. Both produce **identical fingerprints**, so a reference
> recorded on either path verifies on the other. At 16 threads the default
> measures ~202000 KRay/s against ~94500 for the wide path — and because it does
> more real work per second it also draws more power, which is why the AVX2 run
> (like AIDA64's own test) is the one that actually heats the CPU.
>
> `tools/bench_kernels.cpp` times each kernel width directly, both in one
> translation unit compiled with `/arch:AVX512` so no 512-bit value can be
> split or spilled by the build flags:
>
> | precision | narrow | wide | |
> |---|---|---|---|
> | double | D4 4-wide: 20,180 k KRay/s | D8 8-wide: 6,651 k KRay/s | 3.0× slower |
> | float | F8 8-wide: 42,693 k KRay/s | F16 16-wide: 23,370 k KRay/s | 1.8× slower |
>
> The wide path is not losing to the hardware: a micro-benchmark of the raw
> instructions shows this CPU runs 512-bit div/sqrt/add at full width (zmm
> `vdivpd`/`vsqrtpd` are as fast as or faster than their 256-bit forms).
> Disassembly of the D8 kernel shows what actually happens — MSVC's register
> allocator splits 512-bit live ranges into 256-bit halves and rebuilds them
> through the stack (`vmovups ymm, [rbp]` + `[rbp+32]` pairs, `vextractf64x4`,
> 127+ `vmovups` in one function against 24 for the D4 twin), and the mask
> handling round-trips through general-purpose registers (`kmovb` … GPR `and`
> … `kmovb`). Replacing the trait's mask logic with native k-mask intrinsics
> removes the GPR round-trips (`kandb`/`korb`/`kandnb`) and changes nothing
> measurable, so the dominant cost is the 512-bit live-range splitting — a
> codegen limitation of this compiler for a kernel this size. The AVX2 traits
> keep all masks in full vector registers and never split, which is why they
> win.
>
> Task manager shows "100 % CPU" in both cases; the difference is package
> power. Measured with the platform's RAPL energy counter, the default path now
> draws ~1.35× the power of the old wide-path default — the same ratio the
> temperature gap against AIDA64 implied.

## 5. Verification: score 0 when the CPU miscomputes

AIDA64's ray-trace test validates each computed scene against its reference
within a tolerance (`RayTraceErrMargin`) and reports **score 0** when it does not
match. That is what makes the test useful for finding an unstable overclock: the
CPU silently miscomputes, the frame no longer matches, and you get a zero instead
of a plausible-looking number.

This tool implements the equivalent, for every workload that has a well-defined
result:

1. each benchmark reduces its output to a 64-bit **fingerprint**;
2. the fingerprint of a known-good run is recorded once in `reference.txt`:
   ```powershell
   .\build\cst.exe run f64ray --accept-reference    # record the baseline
   ```
3. every later measurement recomputes the fingerprint and compares it with the
   recorded one;
4. **on mismatch the throughput and index are forced to 0**, the reason is
   printed, and the process exits with code 3.

```
  throughput    : 0 KRay/s
  AIDA64 index  : 0.0  [uncalibrated]
  verification  : DOES NOT MATCH REFERENCE - hardware computed a different result
                  reference 783f4ddaa8009206 / computed e264d8f23bf198e3
                  >>> RESULT REJECTED: score forced to 0.  The CPU
                  produced a different answer than the recorded
                  reference, which is what an unstable overclock,
                  bad memory or a failing core looks like.
```

Design points that matter:

* **The fingerprint is thread-count independent.** It is computed from a fixed
  single-threaded re-run of the workload, not from whichever worker buffers
  happened to be touched, so a baseline recorded at 16 threads still verifies at
  4. Every test's reference is checked this way.
* **A changed *value* is the signal, not a changed *time*.** The same code on the
  same input must produce bit-identical output; a wrong bit means the hardware
  computed something else. Throughput, by contrast, legitimately varies with
  clocks and load, so it is never used as the correctness test.
* **Parameter changes are not failures.** A reference recorded at a different
  resolution, iteration count or test id is reported as "not comparable" rather
  than as instability, so you can change settings without false alarms.
* **A matching fingerprint always verifies**, even if the binary was rebuilt
  (rebuilding changes the recorded build id) - otherwise every compile would
  force a re-baseline for no benefit.
* **Verification is charged outside the measured loop.** The timed region is
  unchanged; a 10-second run still reports ~10.0 s elapsed and a 1-minute
  sustained run ~60.1 s. The fingerprint costs roughly 70 ms for the default
  ray-trace configuration, re-checked every round.

`--no-verify` disables all of this and prints every raw number.

**What has a reference, and what does not.** Twelve of the fifteen workloads are
verified: the ray traces, the three fractals, AES, SHA3, ZLib, PhotoWorxx, Queen,
CheckMate and Memory Latency. The three **memory bandwidth** tests (read, write,
copy) deliberately have none: throughput is a timing measurement with no
canonical correct value, and hashing N per-worker buffers would make the
fingerprint depend on the thread count. Their correctness is covered by
`selftest_memory.exe`.

**Demonstrating it.** Set `CST_TEST_CORRUPT=1` and every verified test is perturbed
by one bit - exactly what a miscomputing CPU produces - so the whole path can be
exercised on a healthy machine:

```powershell
$env:CST_TEST_CORRUPT='1'
.\build\cst.exe run all --seconds 2        # all verified tests score 0, exit 3
Remove-Item Env:CST_TEST_CORRUPT
```

## 6. Calibration against AIDA64

Without a calibration file the tool prints its own physically meaningful
throughput. To make the number directly comparable with AIDA64, record AIDA64's
score for the same test on the same machine:

```
scale     = aida64_score / cst_measured_value
reference = aida64_score
```

then put it in `calibration.txt`:

```
# test          scale        reference    unit
f64ray          0.0001234    45678.0      KRay/s
```

From then on `AIDA64 index` is reported next to the raw throughput. Rows that
still carry the template values (scale 1.0, no reference) are ignored: the tool
prints `[uncalibrated]` - and the GUI shows 未校准 in the index column - instead
of presenting the raw throughput under the name "AIDA64 index". `cst run ...
--emit-calibration calibration.txt` writes a template with every test id.

**Honest caveats.** This is a re-implementation, not AIDA64's code, and AIDA64 is
not required. The calibrated index is a rescaling of *this tool's* measurement,
not a prediction of what AIDA64 would print for an arbitrary machine. Scene
geometry, compressor internals and iteration counts cannot match AIDA64 exactly,
because those are unpublished. Use the index for before/after comparisons on one
machine; use the raw units for absolute claims.

## 7. Verification of the kernels themselves

Each module ships a standalone self-test with real known answers:

```powershell
.\build\selftest_crypto.exe   # AES/SHA3/DEFLATE round trips
.\build\selftest_image.exe    # Queen 8x8=92, 10x10=724 (all-10! SIMD pass cross-checked); CheckMate chunk walks (8x8=92, 9x9=352, 10x10=724 solutions, positions=N!, slice partition, 16x16 probe + first-solution chunk); image transforms
.\build\selftest_memory.exe   # alignment, pattern round-trip, latency chain shape
```

Beyond those, three checks validate against an *independent* implementation — CPython's `zlib` for the compressor and decoder, and a scalar reference for the
fractal and ray-tracing kernels. See [tools/README.md](tools/README.md).

What these caught during development, and are retained to keep caught:

| bug | symptom | check that finds it |
|---|---|---|
| `select()` passed an all-ones comparison result to `blendv_pd`, which tests the sign bit | every `select` returned its false operand; fractals rendered all zeros | `verify_fractal` |
| `any_true` used `_mm256_testz_pd` | reported "no lane active" for a mask of 1.0 values | `verify_fractal` |
| a comparison mask used directly as a multiplier (`all-ones` is `-nan`) | NaN propagated into every accumulated colour | `verify_rt` |
| `0 * inf` in the Blinn-Phong half-vector normalisation | NaN in whole pixels | `verify_rt` |
| compressor emitted matches with distance 0 (self-match) and beyond the output start | "invalid distance too far back" | `validate_zlib` |
| code-length RLE expanded past `HLIT+HDIST` | "invalid bit length repeat" | `validate_zlib` |
| Adler-32 read at the wrong bit offset | round-trip comparison always failed | `verify_inflate` |
| ray-trace scene mutated by one worker while others read it | intermittent access violation at 8 threads | smoke test |

## 8. Layout

```
src/
  cst.h              core types: BenchInfo, Benchmark, Result
  common.{h,cpp}     timing, CPU feature detection, worker threads, hashing
  simd.h             SIMD vector wrappers (D4/F8 AVX2, F16/D8 AVX-512)
  raytrace.{h,cpp}   scene/camera definitions and CPU dispatch
  raytrace_kernel.h  the batch ray-trace kernel (traits + Tracer template)
  raytrace_avx512.cpp  AVX-512 kernels (F16/D8), compiled /arch:AVX512
  bench_fp.cpp       f64ray, f32ray, julia, mandel, sinjulia
  bench_crypto.cpp   aes, sha3, zlib  (+ keccak, deflate, inflate)
  bench_image.cpp    photoworxx, queen, checkmate
  bench_memory.cpp   memread, memwrite, memcopy, memlat
  registry.cpp       test registry
  calibration.{h,cpp} index calibration
  reference.{h,cpp}  golden-reference verification (score 0 on mismatch)
  session.{h,cpp}    the measurement session shared by the CLI and the GUI
  main.cpp           command line front end (cst::cli_main)
  cli.h              declaration of cst::cli_main for the GUI entry point
  selftest_*.cpp     standalone correctness self-tests
vs/                  Visual Studio projects: cst (app), cst_engine (static lib),
                     self-tests, frame dumper + shared CppSettings.props
gui/                 WinUI 3 front end (C++/WinRT, compiled into cst.exe)
CpuStressTest.sln    Visual Studio solution (x64)
tools/               PE analysis + independent regression checks + gui_smoke.ps1
docs/                reverse-engineering notes
reference.txt        recorded golden fingerprints (created by --accept-reference)
calibration.txt      optional AIDA64 index calibration
```

`tools/pedump.py` is a standalone PE32+ reader and is reusable for other
targets; see [docs/reverse-engineering.md](docs/reverse-engineering.md).
