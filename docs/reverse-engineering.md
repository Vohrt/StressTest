# Reverse-engineering notes: `aida_bench64.dll`

These notes record what static analysis of the locally installed AIDA64 benchmark
engine revealed, and — just as importantly — what it did not. They are the basis
for `README.md` §1 and for the workload parameters chosen in `src/`.

Target: `C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll`
- AIDA64 Extreme 8.40.8500
- `aida_bench64.dll` "AIDA64 Benchmark Module 4.7", 4.7.931.8, 13,020,264 bytes
- PE32+, image base `0x140000000`, sections: `.text` 0x997e6a, `.rdata` 0x4c59a,
  `.data` 0x2a5550, `.pdata` 0x37f8

## Method

`tools/pedump.py` is a small PE32+ reader (sections, RVA/file-offset mapping,
`.pdata` function boundaries, string search). `tools/inventory.py` and
`tools/entropy.py` profile the sections; `tools/benchstrings.py`,
`tools/raystrings.py` and `tools/params.py` extract the plaintext descriptor
strings.

## Findings

### 1. The benchmark kernels are assembled at runtime

The DLL exports nothing. Applying MSVC RTTI recovery to it surfaces classes such
as:

```
.?AVBenchAsm_Inst_Base@@
.?AV?$BenchAsm_Inst_4@VBenchAsm_REX@@G@@
.?AV?$BenchAsm_Inst_4M@VBenchAsm_EVEX@@K@@
.?AV?$BenchAsm_Inst_Ext@VBenchAsm_VEX@@@@
.?AVBenchAsm_Inst_Oio@@
```

i.e. a small x86 instruction encoder with REX / VEX / EVEX encodings. Together
with the `CodeDump_*` keys (`CodeDump_RayTrace_%s_%05d.bin`,
`CodeDump_NGFractal_%s_%c%03d.bin`, `CodeDump_MTMBW_%s_%s_%s_%03d.bin`, and the
`*DumpCode` flags) this shows AIDA64 **generates** its benchmark kernels in
memory for the detected CPU rather than shipping them as code.

Consequence: there is no static `FP64 Ray-Trace` function to disassemble. The
algorithm lives in C++ generator code, and recovering it means decompiling the
generator — a much larger and more speculative job than re-implementing the
documented workload. This project therefore re-implements, and calibrates.

### 2. Section entropy rules out packing

Per-4 KiB-block Shannon entropy:

| Section | Entropy range | Zero bytes |
|---|---|---|
| `.text` | 5.7–6.5 (x86 code) | 7.1 % |
| `.rdata` | 0.9–7.9 (strings/tables) | 24.6 % |
| `.data` | 0.7–3.4 (lookup tables) | 59.5 % |

So the image is plaintext — the earlier suspicion of an encrypted overlay was
wrong (a `.data` virtual size larger than its raw size just reflects section
alignment). Everything above comes from readable data.

### 3. Ray-trace workload parameters

These are the exact key names the module exposes (`.rdata`, ~0x9d7f38–0x9d80d0):

```
RayTraceBMPFlag      RayTraceLSBMaskFlag   RayTraceDumpCode
RayTraceProcMask     RayTraceX            RayTraceY
RayTraceScene        RayTraceDepth        RayTraceStepMethod
RayTraceErrMargin    RayTraceStartMethod  RayTraceMinRunTime
RayTraceMethName     RayTraceUsedProcMask RayTraceUsedScene
RayTraceUsedMethod   RayTraceCompletedLines
RayTraceKiloRayPerSec  RayTraceRunTime    BenchRayTraceResults
```

Per-method shape, from the numbered keys:

```
RayTraceFunctionName%02d      RayTraceFunctionXSize%02d   RayTraceFunctionYSize%02d
RayTraceFunctionMaxDepth%02d  RayTraceFunctionMinDepth%02d
RayTraceFunctionDefDepth%02d  RayTraceFunctionMaxScene%02d
RayTraceFunctionMinScene%02d  RayTraceFunctionDefScene%02d
RayTraceFunctionErrMargin%02d%02d
```

Two conclusions shaped this project:

- the metric is **kilo-rays per second** (`RayTraceKiloRayPerSec`), not frames;
- progress is counted in **scanlines** (`RayTraceCompletedLines`), matching the
  documented per-line work partitioning. `src/raytrace.cpp` counts one work unit
  per traced primary ray and the benchmark traces scanline chunks, mirroring both.

Methods are named `RayTraceDouble` and `RayTraceSingle`; the parallel SIMD
implementations are named `TP<n>_...` / `Base<n>_...` for SSE2, SSE3, SSSE3,
SSE4.1, AVX, AVX2, FMA3/FMA4, XOP and AVX-512F/BWDQ/ER — the same shape used for
the method strings this tool reports.

### 4. Which tests exist, and their shared classes

`Bench_*` symbol names in `.rdata`:

```
Bench_Init        Bench_SST          Bench_SST2      Bench_RayTrace
Bench_NGFractal   Bench_NGImage2D    Bench_NGPack    Bench_NGQueen
Bench_NGAES       Bench_Hash         Bench_Hash2     Bench_MTMBW
Bench_MTMBW2      Bench_NGMemLat     Bench_InstLat   Bench_NGInstLat
```

`Julia`, `Mandel` and `SinJulia` appear as *method names* of the single
`Bench_NGFractal` class (`NGFractalMethodName%02d`, `NGFractalMaxIter%02d`,
`NGFractalMethodXSize%02d`, `NGFractalMethodYSize%02d`), which is why this tool
groups them as related fractal workloads with per-method parameters.

`Base_ZLib_1_2_5` is present verbatim, confirming the ZLib test is pinned to zlib
1.2.5 — `src/bench_crypto.cpp` targets that library's level-9/32 KiB-window
semantics for the reason documented in §15.

### 5. Scoring is not published

The module computes a rate per test, but no normalisation factor, reference
machine or formula is exposed as data, and FinalWire does not document one. That
is why this tool reports a physically meaningful rate as its primary number and
offers an explicit, user-supplied calibration (README §5) for an
AIDA64-comparable index.

### 6. Not recovered

- The RLE/code-length structures behind the fractal generator (the AES kernels
  turned out to be static in the image; see §12).
- The scoring normalisation for Queen / CheckMate / Julia / Mandel / SinJulia /
  Ray-Trace.
- AIDA64 Extreme's command line has no benchmark switches (the executable's
  string table is obfuscated, and FinalWire documents CLI benchmarking only for
  Engineer/Business/Network Audit editions). The ray-trace kernel and its scene
  were nevertheless exercised directly — see §7 — by loading the module and
  driving its benchmark entry points under a debugger.

## 7. The ray-trace kernel and scene (confirmed from the code dump)

`RayTraceDumpCode` makes the DLL emit the kernel it assembles for the detected
CPU. Capturing `CodeDump_RayTrace_X64_01070.bin` and disassembling it pinned down
the shape of the workload `src/raytrace_kernel.h` replicates:

- One kernel call handles a batch of **16 rays**. Recursion depth is checked
  first; a batch with nothing to shade returns black.
- The nearest-hit sweep walks the scene's **linked list of sphere nodes** and
  stores, per lane, the address of the node that produced the first hit. The
  hit mask is tested with `vptestmd` / `kortestw` before any shading.
- The winning object's centre, radius, reciprocal radius, shininess, mirror
  term and albedo are then fetched **per lane with `vgatherdps`** (five gathers
  in the dumped variant), rather than blended per object inside the sweep.
- Each white point light gets a shadow ray traced with the same sweep code; the
  loop breaks early once every lane is occluded (`kortestw` + `jb`), and
  visible = hit ∧ ¬occluded.
- Shading normalises the light vector, computes `diff = max(L̂·N, 0)` and
  `spec_base = max(L̂·R, 0)`, calls an out-of-line `pow(spec_base, shininess)`
  routine, and accumulates `albedo·diff + pow` per light, zeroing lanes that
  are shadowed.
- One reflection ray per bounce is traced recursively — direction
  `D − 2(N·D)N` scaled by 1000 — masked by `mirror != 0` and weighted by the
  mirror term on the way back.
- The kernel times itself with `rdtsc` around the batch loop; the score path
  reports through `RayTraceKiloRayPerSec` / `RayTraceCompletedLines`.

> Note: this repository does not redistribute AIDA64's kernel dump
> (`CodeDump_*.bin`) - it is AIDA64's machine code. Re-produce it from your own
> installation with AIDA64's `RayTraceDumpCode` and the tools in `tools/`. The
> rendered frame capture (`docs/aida/aida_image.bin`) is included, because the
> pixel comparison above is against it.

### Independent re-check of the dump (2026-10)
Re-disassembling the 4,071-byte dump confirms the structural claims above and
pins down some limits:

| check | result |
|---|---|
| `vptestmd` | 2 |
| `kortestw` | 6 |
| `vgatherdps` | 5 |
| `vfmadd*` | 59 |
| `vsqrtps` / `vdivps` | 3 / 5 |
| `zmm` register references | 959 |

Two corrections, one of which is a correction to this document rather than to the
code:

1. **The dump is a single-precision kernel.** Every packed arithmetic opcode in
   it is `ps` — 50 `vaddps`, 50 `vmulps`, 40 `vfmadd231ps`, 38 `vsubps`,
   5 `vdivps`, 3 `vsqrtps`, 5 `vgatherdps` — with **zero** `pd` instructions
   (`vcmppd`, `vmovupd`, `vsqrtpd`, `vdivpd`, `vgatherdpd` all count 0). So it
   evidences the FP32 method (`RayTraceSingle`). An earlier revision of this
   document labelled `CodeDump_RayTrace_X64_01070.bin` as the double-precision
   reference (`Base512_AVX512F_F64`); it is not. The FP64 kernel was never
   captured, and the FP64 path in `src/` is a port of the same structure at
   double width, which is why the two agree to within the 1.35/255 tolerance
   above.
2. **The dump carries no scene data.** All 263 memory operands are register-based
   (`[rsp+…]`, `[rbp+…]`, …); there are **no** RIP-relative operands, so no
   embedded constants and no sphere/light numbers. The scene is passed in by
   pointer.

The scene values in `src/raytrace.h` did not come from the dump — they came from
the debug session that also produced `aida_image.bin`, by reading the DLL's own
scene structure out of the process. That is what `§7` means by "recovered from
the scene the DLL builds in memory at run time", and the framebuffer comparison
above is what validates it.

### The pixel comparison reproduces

`docs/` previously recorded that rendering the reconstructed scene and comparing
against the AIDA64 framebuffer `aida_image.bin` gives a mean absolute difference
of **1.35/255** per channel, 93.8 % of pixels within 2 levels.

**This reproduces exactly** (`tools/verify_compare.cpp`, FP32 kernel, capture read
as BGRA **with the row order flipped** — see below):

| statistic | recorded | re-measured |
|---|---|---|
| mean absolute difference | 1.3459 | **1.303 / 1.270 / 1.465** (R/G/B) |
| pixels within 2 levels | 93.8 % | **93.8 %** |
| pixels within 8 levels | 97.1 % | **97.1 %** |
| pixels within 1 level | — | 84.9 % |

Two traps that make this comparison fail badly if you get them wrong, both worth
knowing before re-running it:

- **The AIDA64 framebuffer is stored bottom-up.** The renderer writes image row 0
  at the far end of the buffer, so the capture must be flipped vertically before
  differencing. Skipping the flip gives a mean absolute difference of ~49/255 and
  13.4 % within 2 levels — i.e. it looks like a completely wrong scene rather
  than a row-order mistake.
- **The pixel format is BGRA** (`byte0=B, byte1=G, byte2=R, byte3=0`), from the
  kernel's `(R<<16)|(G<<8)|B` packing. Reading it as RGBA scores mean|d| ≈ 119 on
  the red/blue channels.

Single-precision and double-precision renders of this scene agree to within the
tolerance above, so which of the two kernels produced the capture is not
observable at 8-bit — consistent with the two being structurally the same kernel.

The scene is therefore corroborated: the four spheres, camera basis, light
positions and the `[5e-6, 1]` segment window in `src/raytrace.h` reproduce
AIDA64's own framebuffer to within 1.35/255 on average, with the residual on
one-pixel silhouette edges.

Two implementation notes carried into `src/`:

- A double→float conversion plus broadcast per sphere field, per sphere, per
  sweep dominated the gather cost, so the scene is narrowed to the kernel's
  scalar type once per frame (`PackedScene`).
- MSVC only models `__m512` as a native 512-bit register when AVX-512 is part
  of the ISA baseline; under `/arch:AVX2` it splits every 512-bit value into two
  256-bit halves and spills them, which cost roughly 3× on this kernel. The
  AVX-512 kernel therefore lives in its own translation unit built with
  `/arch:AVX512` (`src/raytrace_avx512.cpp`). It is *not* the default: even
  with a native 512-bit baseline MSVC still splits 512-bit live ranges through
  the stack and round-trips the mask logic through GPRs, which makes the wide
  kernel 2–3× slower than the AVX2 one (see the README section on the ray-trace
  dispatch), so `src/raytrace.cpp` selects the AVX2+FMA kernels unless
  `CST_FORCE_AVX512=1` is set. Both paths produce bit-identical output.

## 8. The NGFractal kernels (confirmed from the code dumps)

The `NGFractalDumpCode` channel emits one binary per *method* as
`CodeDump_NGFractal_X64_<family><mmm>.bin`. Capturing the base methods of the
Julia / Mandel / SinJulia families (`F000`, `D000`, `L000`) and disassembling
them together with the class init / prepare / advance functions pinned the whole
workload down:

| | Julia | Mandel | SinJulia |
|---|---|---|---|
| code dump | `F000` (f32, SSE 1x4) | `D000` (f64, SSE 1x2) | `L000` (x87 80-bit) |
| image | 1024 x 1024, 999 iterations | 1024 x 1024, 999 iterations | 256 x 256, 70 iterations |
| iteration | z' = z^2 + c, z0 = pixel | z' = z^2 + c, z0 = c = pixel | z' = c*sin(z), z0 = pixel |
| constant | c = -0.770885 + 0.1i | - | c = 0.98059 + 0.45075i |
| escape | \|z\|^2 < 4, counted before each update | same | Re(sin z) + Im(sin z) >= 64 or \|Im z'\| >= 708.390625 |

Details that matter:

- **Pixel mapping.** The view is a square of half-extent `ext` centred on
  (cx, cy), rasterised from its top-left corner with *no* half-pixel offset:
  pixel (x, y) = corner + x*colstep + y*rowstep, where corner =
  (cx - (W/2)*s, cy + (H/2)*s) and s = ext/(W/2).  Julia keeps this whole state
  block in f32; Mandel and SinJulia use f64.
- **Mandel starts from z0 = c = pixel.** The kernel loads its "z" registers from
  the same row base as its "c" registers, so pixels with |c| >= 2 fail the very
  first escape test and count 0.  That is the black disc outside the radius-2
  circle visible in AIDA64's own frame dumps.
- **Count semantics.** Julia and Mandel count the iterations for which the
  \|z\|^2 < 4 test held (interior pixels therefore reach 999).  SinJulia counts
  *completed* iterations of z' = c*sin(z).
- **Animation.** Every frame rotates the view about the complex origin by
  pi/400 rad — the f32 constant 0x3C00ADFD in the Julia class and the f64
  constant 0x3F8015BF9217271A in the double classes.  The extent is multiplied by
  the zoom factor F = 1.0 per frame, so the rotation is the only animation and
  the per-frame work is constant.
- **Plumbing.** Method entries are 48 bytes (name / code start / code end /
  flags / class id); the class id drives a 0x31-entry jump table at 0x4A2608
  which selects the init, per-frame prepare and advance functions.  The kernel
  itself is memcpy'd into a 64 KiB executable buffer and called once per frame as
  `kernel(&params, &scratch)`; a BMP writer then scales the count buffer and
  writes it bottom-up as a 32 bpp BMP (`X64_<F|D|L>_M%03d_P%016I64X_F%04d.bmp`).

### Frame-dump comparison

`tools/fractal_dump.cpp` renders a frame with the same kernels the benchmark
runs (raw uint32 counts + a PGM preview), for direct comparison against the BMP
frames AIDA64 saves.  The comparison has to flip one of the two images
vertically: the BMPs are stored bottom-up, and it has to compare the *count* of
each pixel, not its colour (the palette ramp is a separate, unaligned step).
Measured against the reference frames kept from the capture session:

| frame | mask | IoU | notes |
|---|---|---|---|
| mandel f0000 | interior (count == 999) | **1.00000** | 176 162 pixels, exact |
| mandel f0000 | count == 0 (black disc) | **1.00000** | 73 165 pixels, exact |
| julia f0000 | interior (count == 999) | 0.98346 | all differences within 2 px of the boundary |
| julia f0000 | count == 0 (corners) | 0.99986 | one pixel off |
| julia f0191 | interior | 0.98280 | confirms the pi/400 rotation phase |
| sinjulia f0000 | all counts == 70 | exact | AIDA64's frame is a uniform colour |

The Julia residual is boundary rounding: its state block is f32, so a different
evaluation order (SIMD width, FMA availability) flips individual boundary
pixels.  After a 2-pixel dilation of either mask the difference disappears
entirely, which is why `tools/verify_fractal.cpp` instead checks the SIMD
kernels bit-exactly against a scalar re-implementation of the same formulas.

## 9. The MTMBW memory kernels (confirmed from the code dumps)

The `MTMBWDumpCode` channel emits one binary per *direction* as
`CodeDump_MTMBW_%s_%s_%s_%03d.bin` (ISA, direction, level).  Driving it with
`<bench_mtmbw></bench_mtmbw><bench_mtmbw_domeasure></bench_mtmbw_domeasure>
<mtmbwminruntime>0.5</mtmbwminruntime><mtmbwprocmask0>0xFFFF</mtmbwprocmask0>
<mtmbwmethod>4096|8192|12288</mtmbwmethod><mtmbwdumpcode>1</mtmbwdumpcode>` pins the
settings reader at 0x9759D0 and the measure loop at 0x46AD10 (it raises error
codes 0x1404 / 0x1405 when the settings or the DoMeasure flag are missing).
Method values are 0x1000 = Read, 0x2000 = Write, 0x3000 = Copy; the "Mem" level
of an AVX-512 CPU selects the `ZMM_DQ64` kernel variant.

| | Read | Write | Copy |
|---|---|---|---|
| buffer | 96 MiB/thread | 96 MiB/thread | 48 + 48 MiB/thread |
| block | 128 x 64 B (8 KiB) | 128 x 64 B (8 KiB) | 64 x (64 + 64 B) (4 KiB) |
| access | `vmovntdqa` | `vmovntdq` | `vmovntdqa` + `vmovntdq` |
| stored data | - | constant register set, zeroed with `vpxord` | the loaded bytes |
| AIDA64 "BS" | 32 | 128 | 64 |

Per-repetition plumbing: `clflushopt` over the buffer in 512-byte blocks, then
`mfence`, `cpuid`+`rdtsc`, the sweep, `cpuid`+`rdtsc`; the timestamps land at
`[rbp+rsi*8+0x10]` / `[rbp+rsi*8+0x60]`.  The dumped code still contains
placeholder immediates - the framework patches the buffer VA and size in at run
time (hence the 0x6000000 = 96 MiB literal in the dumps).

### The all-zero store path (Zen5)

Standalone probes on the Ryzen 7 9800X3D, one fixed 96 MiB sweep, single thread,
the same instruction shape in every variant:

| data stored by the non-temporal sweep | GB/s |
|---|---|
| all-zero register (what AIDA64's write kernel stores) | 86 - 87 |
| constant non-zero register | 30.4 |
| `w[j] = j` ramp, 8 independent chains | 30.3 - 30.4 |

- The buffer size does not matter: at 512 MiB the numbers are the same, so this
  is a store-path property, not cache residency.
- It is not a codegen artefact: the zero-store and the ramp loop compile to the
  same 5-instruction body, and a fully unrolled 8-chain ramp is just as slow.
- The same gap appears in a copy: non-temporally copying an all-zero source runs
  at 77 - 79 GB/s single-threaded, copying a `w[j] = j` source at 30.4.
- Consequently AIDA64's own Mem-level write/copy numbers on this host (78208 and
  60838 MiB/s) are only reachable on the all-zero path, and its write kernel
  indeed zeroes its registers with `vpxord`.  Our reimplementations store
  all-zero blocks for the same reason: a pattern would report roughly a third of
  the AIDA64 figure they are meant to be comparable with.  Read bandwidth does
  not depend on the data, so `memread` keeps its deterministic `w[j] = j`
  buffer and its full-coverage digest.

### Alignment and deviations

- Same working sets (96 / 96 / 48 MiB per thread), 64 KiB alignment and
  non-temporal 64-byte accesses; 512 bytes (8 registers) per iteration instead
  of AIDA64's fully unrolled 8 KiB swing - the DRAM access pattern is identical.
- No `clflushopt` phase: AIDA64 flushes the buffer before every timed sweep but
  excludes that phase from its timestamps, and our repetitions run back to back
  exactly like the timed part does.
- Digest: `memread` folds every loaded vector; `memwrite` / `memcopy` end each
  pass with a sampled read-back (one 4 KiB page per MiB, 0.4 % of the buffer)
  because all-zero store data carries nothing to fold.  AIDA64's kernels do no
  verification of any kind.

### Measured against AIDA64 (same host, AVX-512, Mem level)

AIDA64's per-level processor counts are the popcounts of its `usedprocmask0`
values: 0xAAAA = 8 threads for the read row, 0xA000 = 2 threads for write and
copy.  Running our kernels at the same counts:

| direction | AIDA64 | this reimplementation |
|---|---|---|
| read | 63797.9 MiB/s (8 threads) | 62.6 GB/s (8 threads) |
| write | 78208.2 MiB/s (2 threads) | 84.5 GB/s (2 threads) |
| copy | 60838.1 MiB/s (2 threads) | 60.3 GB/s (2 threads) |

## 10. The NGQueen kernels (confirmed from the code dumps)

The `NGQueenDumpCode` channel emits `CodeDump_NGQueen_X64_%05d.bin` per method,
where `%05d` is the *encoded StartMethod value* (01000, 01001, ...), not a
small index.  Driving it needs the DoMeasure flag and a settings string whose
keys appear in reader order - the tag lookup shares one search offset and a
failed lookup leaves it unchanged (reader at 0x9764D0):

```
<bench_ngqueen></bench_ngqueen><bench_ngqueen_domeasure></bench_ngqueen_domeasure>
<ngqueenusedgroups>1</ngqueenusedgroups><ngqueenprocmask0>0xFFFF</ngqueenprocmask0>
<ngqueenminruntime>0.5</ngqueenminruntime><ngqueennextcheckrank>0</ngqueennextcheckrank>
<ngqueensize>10</ngqueensize><ngqueenstartmethod>1000</ngqueenstartmethod>
<ngqueenstepmethod>8</ngqueenstepmethod><ngqueendumpcode>1</ngqueendumpcode>
<ngqueenlsbmaskflag>0</ngqueenlsbmaskflag>
```

Config fields: +0x100 minruntime (double), +0x108 nextcheckrank, +0x110 size,
+0x114 startmethod, +0x118 stepmethod, +0x11C dumpcode, +0x120 lsbmaskflag.
`StartMethod` is 1000 + method index, validated to [1000..1056] by 0x8A0D60
(anything else raises 0x1104); `StepMethod` is how many method slots the runner
probes forward.  On this Zen5 host only two NGQueen methods exist - slot 0 =
SSE2, slot 1 = SSSE3 - and probing 8 slots writes six empty result sets.
Error state lives at 0xC67CD8: 0x1100 ok, 0x1101 GetInfo failed, 0x1102
settings read failed (0x8A314C), 0x1104 bad StartMethod / no methods
(0x8A3140), 0x1105/0x1106 null arguments in the method-list builder 0x8A14C0.

**Algorithm.**  Both dumps walk *every* one of the N! column permutations of
the board - no pruning, every complete candidate is screened - with the tree
levels striding by a factorial table at 0xC376A4 (0!..13!: each level computes
`limit = (clamp(cursor, start, end) - value) / factorial[level]`, a work
partition, not a correctness bound).  A leaf is a full board:

* state vector: 16 bytes, one lane per row, lanes >= N unused, each lane holds
  the column placed in that row;
* `d1 = state - [15,14,...,0]` (bias table 0xC36B00) -> constant on one
  diagonal family (`col + row - 2N + 1`);
* `d2 = state - [0,1,...,15]` (0xC36B10) -> constant on the other
  (`col - row`);
* SSSE3 method: rotate the lane vector cyclically with `pshufb` (index vector
  `[1..N-1,0,N..15]`, entry `(size-1)*16` of the table at 0xC36BD0), compare
  against the *unrotated* difference vectors, `por` the two compares,
  `pmovmskb`, mask with `(1 << N) - 1`, or into an accumulator; loop count
  `N/2` (= 5) because one cyclic distance covers both lane orders;
* SSE2 method: the same check with `psrldq` shifts, loop count `N-1` (= 9) and
  a shrinking mask (`mask >>= 1` per step) so the zero bytes shifted in are
  never tested;
* the board is a solution exactly when the accumulator stays 0 - 724 of the
  3,628,800 permutations qualify;
* `[rbp+168h]` (checked positions) is incremented once per leaf check; reaching
  the end of a full pass increments `[rbp+180h]` (completed boards).

**Score semantics.**  `NGQueenKPosPerSec` counts "checked positions" where one
position = one candidate board (one leaf).  A one-CPU run makes that exact:
90,720,000 checked = 25 x 3,628,800 = 25 completed boards.  The reported
runtime is the *sum* of the per-CPU runtimes and
`kPos/s = checkedPos * N / runtimeSum` = checkedPos / per-CPU time - the
aggregate throughput of N parallel CPUs, which is exactly the quantity this
project's harness prints (`total work / wall time`), so the figures compare
directly at equal thread counts.

Measured on the Ryzen 7 9800X3D, 16 CPUs (mask 0xFFFF), minruntime 0.5,
dumpcode off:

| method | kPos/s | runtime (sum) | checked positions | boards |
|---|---|---|---|---|
| SSE2 | 1,392,560 | 8.023 | 698,317,200 | 188 |
| SSSE3 | 1,892,374 | 8.012 | 947,570,400 | 255 |

### Alignment and deviations

- Same candidate space (all N! permutations, each exactly once), same board
  representation and diagonal test (bias tables, 5 cyclic rotations, constant
  `(1 << N) - 1` lane mask), same definition of a "position".
- Enumeration differs: Heap's algorithm with one `pshufb` lane swap per
  candidate instead of the factoradic tree with per-level cursor bookkeeping.
  The candidate multiset is identical, the per-candidate work is lighter, so
  the reimplementation runs a little above AIDA64's figure - 2.31 M kPos/s at
  8 threads and 2.69 M at 16 on the same host (SSSE3-shaped leaf check).
- Verification: AIDA64's kernel checks nothing.  Ours counts the 724 solutions
  per pass and fingerprints the solution set (splitmix64 of each packed board,
  summed - order independent), cross-checked in `selftest_image.exe` against an
  independent scalar scan of all 10! permutations.

## 11. The NGImage2D kernels (CPU PhotoWorxx)

CPU PhotoWorxx is AIDA64's `NGImage2D` benchmark: a photo pipeline over 32-bit
RGB images.  Driving the DLL through the shared-memory settings channel
(`StartMethod = 2000*k + variant`, `StepMethod`, `DumpCode`, `MinRunTime`) and
reading the method descriptor tables in the image mapped the whole thing out.

### Stage list, sizes and buffer counts

`0xBF8970` is an 11-entry descriptor table (entry size 0x28, read by the
settings-driven driver at `0x529040` / `0x528D20`).  Two details differ from a
naive reading:

- The **name pointer stored in entry `k` labels method `k+1`**; method 0's
  name is the "HWRandom" string that precedes the table.  The DLL's own
  `NGImage2DMethName` result confirms this ("*SWRandom_<variant>" for a run
  that selected method 1).
- Each entry's dword at +0x1C is the kernel's **buffer count code**
  (0 -> 1 buffer, 1 -> 2, 2 -> 3, 3 -> 4 plus a scale word at ctx+0x28), not a
  flag.

| # | stage | image (X*Y) | buffers | kernel semantics (from the dumps) |
|---|---|---|---|---|
| 0 | HWRandom | 4096x3072 | 1 | hardware RNG pixel fill (VIA XSTORE / RDRAND / RDSEED) |
| 1 | SWRandom | 4096x3072 | 1 | software RNG fill: 4-tap fixed-point polynomial recurrence, NT stores |
| 2 | Rot90 | 3072x2048 | 2 | `dst[x][h-1-y] = src[y][x]`, tiled transpose, NT stores |
| 3 | Rot180 | 3072x2048 | 2 | `dst[i] = src[n-1-i]`, 64-pixel `vpermd` blocks, NT stores |
| 4 | Diff | 2048x2048 | 3 | `dst = |a-b|` per channel, alpha byte cleared |
| 5 | RGB2YUV | 3072x2048 | 4 | BT.601 studio YUV 4:2:0 (see below) |
| 6 | VP8YUV | 3072x2048 | 4 | the same conversion; only a per-run scale word (set to 1) differs |
| 7 | Pattern | 4096x3072 | 1 | `value = (y<<16) | (x<<8) | ((x+y)&255)` gradient |
| 8 | Access | 4096x3072 | 1 | linear read |
| 9 | Fill | 4096x3072 | 1 | zero fill |
| 10 | Move | 3072x2048 | 2 | linear copy |

The kernel ABI is `void kernel(ctx)` with `ctx[0]`, `ctx[8]`, `ctx[16]`,
`ctx[24]` = buffers, `ctx[32]` = X, `ctx[36]` = Y, `ctx[40]` = the scale word.
Method 0 (HWRandom) only runs on hosts with a hardware RNG; its "auto" pick is
an empty kernel whose feature check fails on this host, exactly as in the GUI.

### The colour conversion (stages 5 and 6)

Identical fixed-point tables in both stages (`.rdata` 0xC19A80 region), 16-
and 32-bit coefficients, 15-bit scale (`0x84000 = 16.5<<15`,
`0x404000 = 128.5<<15`):

```
Y = ( 8421*R + 16515*G +  3211*B + 0x84000 ) >> 15     # BT.601 studio
U = ( -4850*R - 9536*G + 14385*B + 0x404000) >> 15
V = (14385*R - 12059*G -  2327*B + 0x404000) >> 15
```

Chroma is 4:2:0: a 2x2 block is averaged as two rounded column averages, each
`(x+1)>>1`, then those two with the same rounding.  Pixels are BGRA
(`0x00RRGGBB`), the alpha byte is never read by any stage.

### Score semantics

Every stage is timed in its own window and reported as
`rate = frames * X * Y / window_seconds` (MPixel/s); `NGImage2DRunTime` is the
sum of the per-thread CPU seconds (`~threads * wall`), `MiBPerSec` is the same
rate expressed in `X*Y*4` bytes.  The GUI figure is the sum of the ten stage
rates.  AIDA64 gives each stage a 0.3..0.5 s window per cycle.

### Memory behaviour (why the reference is DRAM bound)

The DLL's per-run working set is several hundred megabytes (the framework
allocates image/workspace buffers well past any current L3; the image buffers
alone are `X*Y*4` each with more than one live at a time) and the stage
kernels run long enough that each frame re-reads data the last-level cache can
no longer hold.  On the 9800X3D (96 MB L3) this shows up as every stage
sitting at 30-70 GB/s - DRAM numbers - while the same code on a cached 50 MB
buffer would run 5-10x faster.

Measured through the settings channel on the Ryzen 7 9800X3D, 16 CPUs (mask
0xFFFF), minruntime 0.5, auto method selection (`StartMethod 0x8000 + k`):

| stage | auto variant (sub) | MPixel/s |
|---|---|---|
| HWRandom | `X64_1x1_Empty` (feature check fails, GUI skips) | - |
| SWRandom | `X64_16x1_AVX512Quad_NT` (67) | 8,454 |
| Rot90 | `X64_AVX512F_8_NT___32x_4K` (1368) | 6,076 |
| Rot180 | `X64__64x1_AVX512F_PERM_NT` (415) | 8,393 |
| Diff | `X64__8x1_AVXDual_NT_4096` (125) | 5,817 |
| RGB2YUV | `X64_64x2_AVX512VNNINT_64` (174) | 15,238 |
| VP8YUV | `X64_64x2_AVX512VNNINT_64` (174) | 13,863 |
| Pattern | `X64_1x1_Base` (0) | 7,599 |
| Access | `X64_1x1_MOV` (0) | 15,960 |
| Fill | `X64_1x1_MOV` (0) | 8,785 |
| Move | `X64_1x1_MOV` (0) | 5,664 |

Sum of the ten scored stages: **95,849 MPixel/s** on this host.

### Alignment and deviations

The reimplementation runs the same ten stages, on the same image sizes, with
the same per-stage buffers and the same score definition (sum of the
per-stage rates; a wall-clock schedule gives each stage an equal window and
the reported work is scaled by the stage count, so the harness's single
`work/elapsed` division lands on that sum).

- **NT stores everywhere a stage writes**, matching the `*_NT` variants
  AIDA64 selects and keeping every write stage DRAM bound.
- **Read-only stages read through rings** (8x the image) whose reading window
  advances one image per pass, so a pass's data is always evicted before it
  comes around again - the DLL achieves the same effect by having a much
  larger footprint.
- **Access is a scalar one-dword-per-iteration loop** (`volatile`, no
  vectorisation) because the DLL's selected Access kernel is; a vector load
  would stream 3x faster and stop resembling the reference.
- **SWRandom** uses splitmix64 instead of the 4-tap polynomial recurrence, and
  **VP8YUV** shares the RGB2YUV kernel (its scale word is 1); pixel values are
  not part of anyone's score and the self test pins the deterministic digest.
- **HWRandom** is skipped, as the DLL does on hosts without a hardware RNG.
- RGB2YUV/VP8YUV run scalar (AIDA64 uses AVX-512 VNNI there); they are the
  slowest stages relative to the reference (0.31x/0.33x).  The memory stages
  land within 1.0-2.1x.

Measured on the same host, 16 threads, `cst run photoworxx --seconds 15`:

| stage | this port | AIDA64 | ratio |
|---|---|---|---|
| SWRandom | 7,903 | 8,454 | 0.93 |
| Rot90 | 6,953 | 6,076 | 1.14 |
| Rot180 | 7,886 | 8,393 | 0.94 |
| Diff | 7,838 | 5,817 | 1.35 |
| RGB2YUV | 4,665 | 15,238 | 0.31 |
| VP8YUV | 4,637 | 13,863 | 0.33 |
| Pattern | 7,906 | 7,599 | 1.04 |
| Access | 34,233 | 15,960 | 2.14 |
| Fill | 17,080 | 8,785 | 1.94 |
| Move | 7,414 | 5,664 | 1.31 |
| **sum** | **106,515** | **95,849** | **1.11** |

The port lands 11% above the reference sum on this host; the two totals are
directly comparable at equal thread counts.

`cst run photoworxx` needs at least one full stage cycle (10 x 0.3 s = 3 s);
the default 15 s measurement covers five cycles, and the numbers above were
taken with `--seconds 6..15`.  Shorter runs report fewer stages' rates in the
numerator but still divide by the full elapsed time, so they read low.

## 12. The NGAES kernels (CPU AES)

CPU AES is AIDA64's `NGBase_AES`: every worker encrypts (and separately
decrypts) its own buffer of `NGAESSizeInBytes` bytes with AES-256 in ECB.  The
settings channel drives it exactly like the other benches
(`<bench_ngaes>`, `<bench_ngaes_domeasure>`, `<ngaesprocmask0>`,
`<ngaesstartmethod>`, `<ngaesstepmethod>`, `<ngaessizeinbytes>`,
`<ngaesdumpcode>`, ...); the reader lives at `0x975E70` and fills a config
struct at `+0x100 EnAESMinRunTime`, `+0x108 DeAESMinRunTime`,
`+0x110 SizeInBytes`, `+0x118 StartMethod`, `+0x11C StepMethod`,
`+0x120 DumpCode`, `+0x124..+0x128` the usual five flags, plus the proc mask.

### Selecting a variant

`StartMethod` is `1000 + index` for the 28 kernels (`0x8000` is the automatic
selection; anything else fails with error `0x100B`).  The index order is the
memory order of the descriptor table at file offset `0xBB8E20` (entry stride
`0x40`, name pointer at `+0x30`, encrypt/decrypt code ranges at
`+0x08/+0x10` and `+0x18/+0x20`, CPU feature word at `+0x28`):

| idx | name | idx | name |
|---|---|---|---|
| 0 | VIAAES (VIA XSTORE, skipped on x86) | 14 | X64_KlG_SSE41 |
| 1 | X64_Base_5K | 15 | X64_4in_SSE41 |
| 2 | X64_Base_8K | 16 | X64_4ii_SSE41 |
| 3 | X64_Keylast | 17 | VAES_YMM_4 |
| 4 | X64_KeylastG | 18 | VAES_YMM_8 |
| 5 | X64_4indep | 19 | VAES_EVEX_XMM_4 |
| 6 | X64_4ind_i | 20 | VAES_EVEX_XMM_8 |
| 7-9 | AESNI_4 / _8 / _16 | 21 | VAES_EVEX_XMM_16 |
| 10-12 | AESNI_AVX_4 / _8 / _16 | 22-24 | VAES_EVEX_YMM_4 / _8 / _16 |
| 13 | X64_Kl_SSE41 | 25-27 | VAES_EVEX_ZMM_4 / _8 / _16 |

On this Zen5 host the automatic selection lands on index 27,
`NGAES_VAES_EVEX_ZMM_16` (its own kernel self-check picks the best supported
one).  The `DumpCode` channel writes `CodeDump_NGAES_X64_En/De%03d.bin` — note
those are the **X64 self-check kernels**, not the selected variant's.

### Kernel shape and score semantics

The kernel ABI is `kernel(ctx)` with `ctx[0] = bytes`, `ctx[8] = source`,
`ctx[16] = destination`, `ctx[64..] = the 240-byte AES-256 round schedule`.
The picked VAES kernel (at `0x493000`, `0x1E9` bytes) processes 512 bytes per
iteration: eight `zmm` registers loaded through `vpxord` with the first round
key, then a round loop of `vbroadcasti32x4 zmm0,[rax+16]` plus eight
`vaesenc`, one `vaesenclast` round, eight aligned 64-byte stores.  Decrypt is
the same shape with `aesdec`/`aesdeclast` and the inverse (`aesimc`-folded)
schedule.  ECB means neither direction chains anything: the whole point of
the test is raw AES throughput.

Results come back as `EnAESMibPerSec` / `DeAESMibPerSec`, with
`NGAESCompletedEnAES` frames, `EnAESRunTime` (the sum of per-thread CPU
seconds) and `enAESClksPerByte`/`PerBlock`.  The arithmetic is

```
rate = completed frames * SizeInBytes / 2^20 / wall_seconds
```

and `completed frames` is the **sum over workers** of their completed buffer
passes.  A buffer size sweep through the settings channel exposes the layout:

| workers | size/worker | working set | En MiB/s |
|---|---|---|---|
| 1 | 1 MiB | 1 MiB | 44,657 |
| 1 | 8 MiB | 8 MiB | 45,056 |
| 4 | 8 MiB | 32 MiB | 87,327 |
| 16 | 1 MiB | 16 MiB | 316,494 |
| 16 | 8 MiB | 128 MiB | 46,733 |
| 16 | 16 MiB | 256 MiB | 21,875 |

The single-thread rate (~45 GiB/s) is the VAES-512 ALU ceiling; everything
above it tracks the cache hierarchy (16 MiB total stays in L3, 128 MiB is
DRAM bound).  This proves `NGAESSizeInBytes` is a **per-worker** size - the
framework allocates one source and one destination buffer of that size per
thread - and that a "frame" is one pass over a worker's own buffer.

### Alignment and deviations

The port (`src/bench_crypto.cpp`) runs AES-256 in ECB over one 8 MiB source
and destination pair per worker, uses the same round structure (round key
broadcast + eight 512-bit AES instructions per round), keeps the AES-NI and
scalar paths for CPUs without VAES, and reports the same
`frames * size / 2^20 / second` figure in MiB/s.  `full_digest()` also
decrypts the ciphertext and folds the round trip in, and the FIPS-197 C.3
vectors pin both directions in `selftest_crypto`.

Measured on the Ryzen 7 9800X3D against the same host's DLL (auto variant,
8 MiB per worker, auto method selection):

| threads | AIDA64 En | AIDA64 De | this port (En) | ratio |
|---|---|---|---|---|
| 1 | 45,056 | 45,565 | 40,780 | 0.90 |
| 8 | 69,945 | 71,439 | 86,510 | 1.24 |
| 16 | 46,733 | 54,519 | 66,500 | 1.42 |

The single-thread figures agree within 10%; at higher thread counts the DLL's
per-frame bookkeeping (its threads only scale +4% from one to sixteen) leaves
the port ahead, the same kind of documented gap as the other reworked tests.

## 13. The Hash2 kernels (CPU SHA3)

CPU SHA3 is AIDA64's `Bench_Hash2`.  Its settings reader sits at `0x976C50`
and fills `+0x00` proc mask, `+0x100 Hash2MinRunTime` (double), `+0x108
Hash2SizeInBytes`, `+0x110 Hash2StartMethod`, `+0x114 Hash2StepMethod`,
`+0x118 Hash2DumpCode`, `+0x11C Hash2LSBMaskFlag`; the results land at
`+0x1C8 Hash2RunTime`, `+0x1D0 Hash2MibPerSec`, `+0x1D8 Hash2ClksPerByte`,
`+0x1E0 Hash2Rounds`, `+0x1E8 Hash2UsedSizeInBytes`.

The older `Bench_Hash` ("CPU Hash") is a separate bench with its own reader at
`0x975470` (`HashMinRunTime` at cfg `+0x00`, `HashProcMask` `+0x08`,
`HashMethod` `+0x108`, `HashDumpCode` `+0x10C`, `SizeInBytes` `+0x110`); its
method list carries SHA3-256/SHA3-512 plus the historical SHA1/SHA-256 kernel
families (`INTC_506Ex_*`, `X64_BP_SSE2_*`, ...).  The GUI's CPU SHA3 is the
Hash2 bench; CPU Hash is not covered by this port.

### Method selection

The mapper at `0x40003C10` answers `0x8000` -> band 0, `0x8001` -> band 1,
`1000..1459` -> band 0 and `2000..2459` -> band 1 (460 methods per band).
Both bands hash 8 KiB messages; band 0 is **SHA3-512** (the result digest has
eight 64-bit words) and band 1 is **SHA3-256** (four words).  Explicit ids
work: 1408 selects exactly what `0x8000` auto-selects, 2408 what `0x8001`
does.  On this Zen5 host both bands pick index 408,
`*AVX512VLBW_Zen4_4_U`; the neighbouring variants are the same family with a
different interleave depth (`_U/_4/_2/_1`) across `_2.._6` generation slots,
with `X64_*`, `BMI_*`, `BMI2_*`, `AMD_800Fxx_*`, `INTC_*` and `X64_Pair*`
families behind them.

### Kernel structure and score semantics

The picked SHA3-512 kernel lives at `0x18CD80` (12928 bytes; `Hash2DumpCode=1`
writes exactly that range as `CodeDump_Hash2_X64_01408.bin`).  It keeps the
Keccak state in pairs of 64-bit lanes across ~20 xmm registers, so every stage
is two lanes per instruction: theta is `vpternlogq`/`vprolq` with
`vshufpd`/`vpunpcklqdq` lane glue, rho applies `vprolvq` against constant
vector tables, chi is `vpternlogq` with immediate `0xD2`, and 72-byte rate
blocks are absorbed with a 64-byte `vpxor` plus a broadcast.  Measured against
the resident clock this kernel runs at ~954 cycles per permutation per thread
(0.86-0.88 clocks per byte, normalised), and the whole `AVX512VLBW_Zen4`
family lands in a narrow 5385..5737 MiB/s window.

The result arithmetic is:

```
Hash2Rounds     = sum over workers of completed 8 KiB messages
Hash2MibPerSec  = Hash2Rounds * UsedSizeInBytes / 2^20 / wall_seconds
Hash2ClksPerByte= clock / aggregate bytes per second   (not per-thread cycles)
Hash2RunTime    = sum of per-thread CPU seconds (16 * wall in every capture)
```

`Hash2UsedSizeInBytes` defaults to **8192** when the settings omit the size,
so the GUI drives 8 KiB messages.  In our captures the DoMeasure path always
ran every CPU - setting `<hash2procmask00>` did not restrict the worker set -
so all DLL figures here are 16-thread numbers.

### Alignment and deviations

The port (`src/bench_crypto.cpp`) hashes independent 8 KiB messages, counts
rounds, and reports `rounds * 8192 / 2^20 / second` in MiB/s, the same figure
the DLL writes back.  `Hash2Digest`-style verification is served by
`full_digest()` (one single-threaded 8 KiB message, decrypt-free but
bit-exact).  Two Keccak-f[1600] implementations are kept: an unrolled scalar
one and an AVX-512 one (state in five zmm rows: theta from `vpternlogq` +
`vpermq`, rho+pi gathered with four `vpermi2q` nodes and `vprolvq`, chi as
five `vpternlogq`).  The AVX-512 workload keeps **two messages in flight** per
worker, the interleaving depth knob the DLL family exposes as `_1/_2/_4`,
because a single chain is permutation-latency bound (~1430 cycles).

Measured on the Ryzen 7 9800X3D against the same host's DLL (SHA3-512 band,
auto method; the DLL ran all 16 workers for every capture):

| threads | AIDA64 MiB/s | this port (MiB/s) | ratio |
|---|---|---|---|
| 16 | 5,360 - 5,441 | 4,632 - 4,664 | 0.86 |
| 8 | n/a | 3,050 | - |
| 1 | n/a | 432 | - |

The SHA3-256 band of the DLL runs at 10,090-10,206 MiB/s (136-byte rate
blocks) and is documented here for reference; the port keeps the SHA3-512
band that the GUI measures.  The remaining ~14% gap matches the distance
between any single AVX-512 chain and the DLL's tuned two-lane xmm kernel.

## 14. The NGMemLat kernels (Memory Latency)

Memory Latency is AIDA64's `NGMemLat`.  Its settings reader sits at `0x975770`
and fills `+0x00` proc mask, `+0x100 NGMemLatMinRunTime` (double), `+0x108
NGMemLatTSCRatio` (double), `+0x110 SizeInBytes`, `+0x118 StrideInBytes`,
`+0x11C StartMethod`, `+0x120 StepMethod`, `+0x124 CacheLevel`, `+0x128
DistanceInBytes`, `+0x12C DumpCode`, `+0x130 TLBWindow`, `+0x134 MaxPMC`,
`+0x138` PMC-index array and `+0x178..+0x17C` the large-page / invalidate /
fence / sleep / prework flags.  The pre-reader initialiser at `0x89C220`
defaults everything to zero except `DistanceInBytes = 8` and `TSCRatio = 1.0`.

### Method selection

The mapper at `0x89C120` answers `0x8000..0x8007` with bands `0..7`, and the
explicit blocks `1000/2000/.../8000` also with bands `0..7`.  The band table at
`0x140C2B8E0` (stride `0x28`: name, `0x40`, shared sub-descriptor, code
pointer, band id) holds:

| band | auto | id | name | measured ns |
|---|---|---|---|---|
| 0 | 0x8000 | 1000 | `Page_Random` | 79.18 |
| 1 | 0x8001 | 2000 | `Linear_Forward` | 2.35 |
| 2 | 0x8002 | 3000 | `Linear_Backward` | 6.48 |
| 3 | 0x8003 | 4000 | `Random_Full` | 92.26 |
| 4 | 0x8004 | 5000 | `CacheLine_Forward` | 1.18 |
| 5 | 0x8005 | 6000 | `CacheLine_Backward` | 2.50 |
| 6 | 0x8006 | 7000 | `Page_Random 2CL` | 79.75 |
| 7 | 0x8007 | 8000 | `Page_Random 1CL` | 78.96 |

The GUI's Memory Latency test leaves `StartMethod` at the auto code, so band 0
(`Page_Random`, explicit id 1000) is the row this port aligns to; all bands ran
with `UsedSizeInBytes = 536870912`, `UsedStrideInBytes = 128`,
`UsedBlockSizeInBytes = 8388608`, `UsedMovs = 64` and `UsedReps = 4194304`.

### The timed kernel and the score

`NGMemLatDumpCode = 1` writes the generated measurement kernel (324 bytes, file
`CodeDump_NGMemLat_X64_Mem_M000.bin`): it reads TSC, then runs an inner loop of
exactly `Reps` (`0x400000` = 2^22) iterations of 64 back-to-back
`mov rcx,[rcx]` dependent loads - 2^28 loads per pass - repeats the pass until a
TSC deadline expires, and stores start/end TSC plus the pass count into the
context.  The result arithmetic is therefore

```
ResultInNanoSec = (end_tsc - start_tsc) / tsc_hz * 1e9 / (Reps * Movs)
ResultInClks    = ResultInNanoSec * clock_hz
RunTime         = the same seconds figure, and Reps*Movs*ns == RunTime
```

with `Reps * Movs` = 268,435,456 on every method.  The chain itself is a
pointer cycle over the 512 MiB region at 128-byte node spacing; `[ctx+8]` is a
relative deadline, so a pass always runs to completion (~20 s at 79 ns).

### Validation surface (error codes at `140C67CD8`)

`0x1300` OK; `0x1301` GetInfo path; `0x1302` settings reader failed; `0x1305`
large pages requested but unavailable; `0x1306` missing buffer argument;
`0x1309`/`0x130A`/`0x130B` context/CPU-count checks; `0x130D` bad StartMethod;
`0x130E` **the configured mask must cover exactly one CPU** (the driver
measures one logical CPU at a time, which is why `NGMemLatProcMask` accepts a
single bit); `0x130F`/`0x1310` MinRunTime range; `0x1311` size above the
method maximum; `0x1312`/`0x1313` SizeInBytes below 1024 or not 1 KiB-aligned;
`0x1314` CacheLevel >= the method's cache-level count; `0x1315`/`0x1317`
StrideInBytes below 8 or not a power of two; `0x1318`-`0x131A` DistanceInBytes
outside [8, 512] or not a multiple of 8; `0x131B` TLBWindow vs size (bands
6/7); `0x131C` mask array differs from the framework's CPU set.  `TSCRatio`
must not be zero: with `0.0` the DLL crashed (access violation) in every
attempt, with `1.0` every band produced clean results.

### Alignment and deviations

The port (`src/bench_memory.cpp`) reproduces band 0 as a 512 MiB Hamiltonian
cycle: 4 KiB pages, 128-byte node spacing, page order shuffled inside 8 MiB
windows, **one node (one cache line) per page per pass** so the page's other
lines are never batched by the L2 region prefetcher, with the window keeping
the TLB warm.  This layout was chosen by measurement: a page-local walk that
takes several lines per page is served from L2 at ~21-23 ns on this Zen5 host
(the region prefetcher covers the rest of the page), and only the
one-line-per-page-per-pass structure reproduces the DLL's 79 ns.  The port
reports `ns/access = elapsed * 1e9 / dependent loads` (single-threaded by
default, `default_threads = 1`), builds the chain and runs one warm-up
traversal before the harness starts its clock, and lets the harness average
whole traversals.

Measured on the Ryzen 7 9800X3D against the same host's DLL (band 0, auto
method):

| run | this port | AIDA64 |
|---|---|---|
| 2 s / 8 s / 20 s single-thread | 78.3 - 81.6 ns | 79.18 ns |

Deviations: nodes carry 32-bit successor indices instead of 8-byte pointers
(the loaded value still drives the next address; one LEA off the critical
path); the buffer is not forced into the low 4 GiB the DLL patches into its
stub; one repetition is a whole 4,194,304-load traversal while AIDA64 times a
single 2^28-load TSC pass (same per-load average); the DLL runs its chain
build outside its TSC window and so does the port.  The other seven bands are
documented above but not ported - they are deterministic access-pattern
curves, not the test the GUI reports.

## 15. The NGPack zlib methods (CPU ZLib)

CPU ZLib is a method of AIDA64's `NGPack` family. `NGPack` is not an
interpreter: the DLL embeds a genuine zlib 1.2.5 build (copyright banner at
file `0x9da5d9`, `z_errmsg` table, `inflate 1.2.5` reply), compiled with
32-bit `unsigned long` so `sizeof(z_stream) = 0x58` - exactly the value passed
to `deflateInit_`/`inflateInit_`.

Settings reader `0x975530` fills:

| offset | tag | meaning |
|---|---|---|
| `+0x00` | `NGPackMinRunTime` | pack budget, seconds (double) |
| `+0x08` | `NGUnPackMinRunTime` | unpack budget, seconds (double) |
| `+0x10` | `SizeInBytes` | source size |
| `+0x18` | `NGPackProcMask` | per-group CPU masks |
| `+0x118` | `NGPackMethod` | method selection |

The tag scan is a forward cursor over the settings blob, so a capture script
must write the tags in exactly the reader's order
(`ngpackminruntime`, `ngunpackminruntime`, `ngpackprocmask0`, `ngpackmethod`,
`sizeinbytes`); with the mask tag ahead of the two budgets the reader fails
with `0x0F02` even though every tag is present.  The section command tag is
`<bench_ngpack>` plus `<bench_ngpack_domeasure>`.

Error codes at `140C67CD8`: `0x0F00` OK, `0x0F02` reader failed, `0x0F07`
mask array missing, `0x0F08` no CPUs in range, `0x0F09` group masks wider than
the CPU set, `0x0F0B`-`0x0F0D` size below `0x8000`, above `0x2000000` or not
32 KiB aligned, `0x0F10` both budgets zero, `0x0F12` computed allocation above
the memory limit, `0x0F13` worker start timeout.

### Method selection and the default workload

`Bench_NGPack_DoMeasure` (`0x89FC10`) resolves `NGPackMethod` like the other
benches: the auto code `0x8000` (and explicit ids from 1000) select
`Base_ZLib_1_2_5`, the single entry of the zlib table at `0x140C2DAE0`
(stride `0x20`: name, deflate pointer, inflate pointer, feature mask).  For
any resolved method id >= `0x2000` the driver also overrides `SizeInBytes`
with `0x2000000`, so the GUI's CPU ZLib always compresses **32 MiB per pass**;
ids below `0x2000` keep the ini size (reader default `0x8000` = 32 KiB).

The per-worker context is 0x190 bytes (worker thread `0x89E690`, context
fields: `+0x18` src buffer, `+0x20` dst buffer, `+0x50`/`+0x58` pack/unpack
budgets in clock ticks, `+0x60` size, `+0x70` CPU mask, `+0x170` table index,
`+0x174` first-worker flag).

### The input corpus

No settings tag feeds the data: the worker generates it itself before the
timed window.  The source buffer is filled one 16-byte block at a time with
four identical little-endian dwords

```
N = size / 4
k = N + i + 3                                            (i = block index)
t = (k * (k + 1)) mod 2^32                               (32-bit multiply)
v = ((t >> 1) - 0x21523F22) mod 2^32                     (32-bit subtract)
```

so 16 bytes carry `v v v v`.  The multiply, the shift and the subtract all
wrap in 32 bits, which makes the sequence deterministic but not monotonic.
Every worker compresses the same corpus.  Cross-checked through the
`PackedSizeInBytes` field: a 32 MiB pass compresses to 9,222,583 bytes
(ratio 0.2749) and CPython zlib at level 9 with the DLL's chunking produces
9,227,339 bytes (0.05 % apart, library-version level).

### The timed loops and the score

`deflate` (`0x89E2E0`) runs one `deflateInit_(&strm, 9, "1.2.5", 0x58)` per
pass - level 9, windowBits 15, memLevel 8, default strategy - then hands the
whole source buffer to `deflate()` in 32 KiB steps with `Z_SYNC_FLUSH`
between them and `Z_FINISH` on the last, calls `deflateEnd`, counts the pass
and repeats until the pack budget (`clock() + ticks(NGPackMinRunTime)`)
expires, always running at least one pass.  `inflate` (`0x89E400`) mirrors
that with `inflateInit_` and its own deadline loop over the compressed
buffer.  The worker brackets both calls with `[140C67E00h]` timestamps.

AIDA64 reports the *sum of the per-thread rates*, with 10^6 bytes per
"MiB" and the per-thread seconds of the measured window:

```
MiBPackPerSec   = sum over CPUs ( reps_i * size * 1e-6 / t_i )
MiBUnPackPerSec = sum over CPUs ( unp_reps_i * size * 1e-6 / t_i )
```

verified against the capture: one CPU with a 0.5 s budget runs 2 pack passes
in 0.561159 s -> `2 * 32 MiB * 1e-6 / 0.561159 = 119.59`, exactly the
reported figure; the 16-thread run matches the same way.  `NGPackRunTime` is
the sum of the per-thread pack times, `PackedSizeInBytes` the compressed size
of one pass, `NGPackCompletedPack`/`Unpack` the pass counts.  With
`NGPackMinRunTime = 0` the pack phase is skipped entirely, and a pack budget
above one pass time makes the loop run the corresponding number of passes
(1.0 s -> 3 passes per CPU) - all confirmed by capture.

Captured on the Ryzen 7 9800X3D (auto method, 32 MiB, all 16 CPUs unless
noted):

| measurement | value |
|---|---|
| `Base_ZLib_1_2_5`, 1 CPU, 0.5 s budgets | pack 119.59, unpack 392.95 |
| 16 CPUs, 0.5 s budgets | pack 1436.2-1444.9, unpack 5270-5337 |
| `PackedSizeInBytes` | 9222583 (every run) |
| pack passes with 0.5 s / 1.0 s budget | 2 / 3 per CPU |

### Alignment and deviations

The port (`src/bench_crypto.cpp`, class `BenchZlib`) implements DEFLATE from
RFC 1951/1950 only - no third-party code - and now runs it the way the DLL
drives zlib: level 9 from zlib's `configuration_table[9]` (chain 4096, lazy
258, nice 258, good 32 including the `chain >>= 2` short-circuit), 32 KiB
window and memLevel-8 block cadence, fresh state per pass, and the 32 MiB
corpus above fed in 32 KiB `Z_SYNC_FLUSH` steps with `Z_FINISH` on the last
one.  The workload class repeats whole passes like the DLL does and lets the
harness sum the worker throughput, so the reported MB/s is directly
comparable with `MiBPackPerSec`.

| run (same host) | this port | AIDA64 (DLL) |
|---|---|---|
| 1 thread, 32 MiB pass | 149.8 MB/s | 119.6 MB/s |
| 8 threads | 1.00 GB/s | 728.7 MB/s |
| 16 threads | 1.53-1.55 GB/s | 1.436-1.445 GB/s |
| compressed size per pass | 9,247,657 B (0.2756) | 9,222,583 B (0.2749) |
| dynamic blocks per pass | 1024 | 1024 (one per 32 KiB step) |

Deviations: the port's encoder writes the code-length tables as plain 3-bit
literals instead of using the 16/17/18 repeat codes (about +0.3 % output
size, no measurable time change); it never emits stored blocks (never the
best choice for this corpus); and it hashes and matches the caller's buffer
in place while zlib slides a 64 KiB window and copies input into it - with
equal match statistics (0.3 % output delta, identical block cadence) that
copy traffic and zlib 1.2.5's older inner loops are the reason the port is
about 1.2x faster per thread.  AIDA64 also times the decompression phase,
but the CPU ZLib figure the GUI shows is the compression rate, so the port
measures compression.  AIDA64 labels both rates "MiB/s" while its arithmetic
divides by 10^6; the port keeps the identical arithmetic and calls the unit
MB/s.

## 16. CPU CheckMate (the v7.35 NGQueen flavour)

AIDA64 v7.35 replaced CPU Queen with CPU CheckMate on the same NGQueen class.
The GUI's CheckMate row drives NGQueen with `NGQueenSize = 16`, an auto
`NGQueenStartMethod = 0x8000` that resolves to the AVX512_VPOPCNTDQ kernel on
this host (method index 32, code dump `CodeDump_NGQueen_X64_01032.bin`) and
`NGQueenNextCheckRank = 0`.  Section 10 covers the class; this section covers
what the CheckMate configuration does with it.

### What the kernel enumerates

The kernel is not a pruned solver: it walks the complete 16! permutation space
of the board in fixed-size chunks.  The top `N - 3` ranks are an ordered
selection of distinct columns (a factoradic tree); every such prefix forms one
chunk whose evaluation covers the 3! = 6 completions of the last three ranks in
a single eight-lane vector step.  The "checked positions" counter therefore
advances by 6 per chunk - one unit per permutation - which is directly
observable in a capture: driven with `NGQueenSize = 8` the DLL completes a
board every 8! = 40,320 checked positions (6,720 chunks) with exactly 92
solutions, while a 16 x 16 run can never complete one (16! / 6 =
3,491,317,114,880 chunks), so `NGQueenCompletedBoards` stays 0 and the rate is
the whole story.  `NGQueenNextCheckRank = 0` means no pruning at all: the GUI
never sets it (the published CheckMate range, 2,930-20,751 MP/s over notebooks,
and the 12,710 MP/s local GUI score match the unpruned kernel; the `ncr >= 2`
configuration lands near 1.6 GPos/s, eight times below).

The eight-lane step builds the queens' `(col +/- row)` difference values for
the six tail permutations with `vpmultishiftqb`/`vpshufb` gather tables and
reduces diagonal duplicates with `vpopcntq`, one chunk per iteration.

### The score

The GUI prints the NGQueen rate in MP/s (millions of checked positions per
second); the DLL-side figure is the same quantity in thousands.  The rule
(section 10) is the sum of the per-CPU rates over the per-CPU measured
windows:

```
kPosPerSec = checkedPositions * nCPU / runtimeSum      (DLL, thousands)
```

The port reports `positions * 1e-6 / wall`, which is the same aggregate when
every worker runs the same window.

### Alignment and deviations

The port (`src/bench_image.cpp`, class `CheckMateChase<16>`) reproduces the
accounting: an iteratively advanced factoradic prefix over the top 13 ranks
with the two prefix diagonal masks carried incrementally, one chunk = one
prefix plus the six completions of the last three ranks, +6 positions per
chunk.  Worker slices are contiguous ranges of the 16!/6 chunk ordinals,
unranked through the factoradic digits - the same partition the DLL applies to
its index space.  The self-test walks whole small boards (8 x 8 = 92, 9 x 9 =
352, 10 x 10 = 724 solutions, positions = N!), checks that four slices
partition the 10! space exactly, probes the first 2^16 chunks of the 16 x 16
space against an independent scalar walk, and locates the chunk of the
lexicographically first 16 x 16 solution through an independently written
factoradic rank (rank 16,661,499,344) where the kernel must score it.

| run (same host) | this port | AIDA64 |
|---|---|---|
| 1 thread | 1.28-1.33 GPos/s | 1.18 GPos/s (DLL capture) |
| 8 threads | 8.9-9.6 GPos/s | 6.38 GPos/s (DLL capture) |
| 16 threads | 13.4-14.0 GPos/s | 12.2-12.4 GPos/s (DLL capture), 12.71 (GUI) |

Deviations: the port evaluates the six completions as a branch-free scalar
block (two prefix-mask tests and three pair distances per completion) instead
of the DLL's eight-lane `vpopcntq` step, and it skips the evaluation for
chunks whose outer eleven-queen prefix is already in conflict, where the DLL
always runs its vector step.  That shortcut is why the port lands a little
above the DLL per thread; nothing else is dropped - every chunk is enumerated
and counted, no subtree is pruned, and a complete board still costs exactly N!
checked positions.  The per-chunk cost note matters for multi-threading: the
worker state is padded to 128 bytes so no two workers share a cache line that
a chunk writes (`CheckMateWorker` in `src/bench_image.cpp`), which is worth
about 3x at 16 threads.

## Reusable tooling

`tools/pedump.py` is standalone and reusable for other PE32+ targets:

```python
import sys; sys.path.insert(0, "tools")
from pedump import PEFile
pe = PEFile(r"...\aida_bench64.dll")
print(pe.image_base, len(pe.functions()))
pe.read_rva(0x9daf60, 32)          # byte range by RVA
pe.find_ascii(b"RayTraceDumpCode") # raw file offsets
```
