"""Analyse the captured AIDA64 ray-trace kernel.

Disassembles CodeDump_RayTrace_X64_01070.bin, reports the instruction mix, checks
the AVX-512 features the project docs claim (vptestmd/kortestw, vgatherdps,
masked compares), and inspects register widths to decide whether this really is
a double-precision variant.
"""
import collections
import re

import iced_x86

PATH = (r"C:\Users\Voheart\AppData\Local\Temp\opencode\bench_scratch"
        r"\CodeDump_RayTrace_X64_01070.bin")
code = open(PATH, "rb").read()

BASE = 0x140000000
dec = iced_x86.Decoder(64, code, ip=BASE)
fmt = iced_x86.Formatter(iced_x86.FormatterSyntax.INTEL)

op_count = collections.Counter()
feat = collections.Counter()
widths = collections.Counter()
rip = []

for ins in dec:
    text = fmt.format(ins)
    op = text.split(" ", 1)[0].lower()
    op_count[op] += 1

    # Register width actually used, which settles pd-vs-ps questions.
    for m in re.finditer(r"\b(zmm|ymm|xmm)\d+", text):
        widths[m.group(1)] += 1

    for key in ("vptestmd", "kortestw", "vgatherdps", "vgatherdpd",
                "vsqrtpd", "vsqrtps", "vdivpd", "vdivps", "vfmadd",
                "vcmppd", "vcmpps", "vpcmp", "vpbroadcastsd", "vpbroadcastss",
                "vpbroadcastd", "vmovupd", "vmovups", "kmov", "vaddpd",
                "vaddps", "vmulpd", "vmulps", "vsubpd", "vsubps", "vmaxpd",
                "vminpd", "vpermpd", "vblendmpd", "vmovddup", "vcvt",
                "vrndscale", "vexp2", "vgetmant", "rdtsc", "pause", "lock",
                "vbroadcastsd", "vbroadcastss"):
        if key in text:
            feat[key] += 1

    if ins.is_ip_rel_memory_operand:
        rip.append((ins.ip, text, ins.ip_rel_memory_address))

print(f"instructions: {sum(op_count.values())}, bytes: {len(code)}")

print("\n--- claimed-feature check ---")
for k in ("vptestmd", "kortestw", "vgatherdps", "vgatherdpd", "vcmppd", "vcmpps",
          "vpcmp", "vmovupd", "vmovups", "vsqrtpd", "vsqrtps", "vdivpd", "vdivps",
          "vfmadd", "kmov"):
    print(f"  {k:14s} {feat.get(k, 0)}")

print("\n--- register widths used ---")
for k, v in widths.most_common():
    print(f"  {k:4s} {v}")

print("\n--- top 20 opcodes ---")
for k, v in op_count.most_common(20):
    print(f"  {k:16s} {v}")

print(f"\n--- RIP-relative operands: {len(rip)} ---")
for ip, text, tgt in rip[:30]:
    print(f"  {ip:#012x}  {text:56s} -> {tgt:#x}")
