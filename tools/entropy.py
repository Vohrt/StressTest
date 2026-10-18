import math
import sys

sys.path.insert(0, r"C:\Users\Voheart\source\repos\CpuStressTest\tools")
from pedump import PEFile

pe = PEFile(r"C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll")
d = pe.data


def entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = [0] * 256
    for x in b:
        counts[x] += 1
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def profile(name: str, start: int, length: int, blk: int = 4096):
    print(f"\n### {name}  file [{start:#x}, {start+length:#x})")
    hist = {}
    for off in range(start, start + length, blk):
        e = round(entropy(d[off:off + blk]), 1)
        hist[e] = hist.get(e, 0) + 1
    for e in sorted(hist):
        print(f"   entropy {e:4.1f}: {hist[e]*blk/1024:8.0f} KB")
    # zero-byte fraction
    z = d[start:start + length].count(0)
    print(f"   zero bytes: {z/ (length or 1):.2%}")


profile(".text", pe.text.raw, pe.text.rawsize)
for s in pe.sections:
    if s.name in (".data", ".rdata"):
        profile(s.name, s.raw, s.rawsize)

# resource directory contents
rva, size = pe.dirs[2]
print(f"\n.rsrc dir rva={rva:#x} size={size:#x}")
print(repr(d[pe.rva2off(rva):pe.rva2off(rva) + 0x200]))

# Are there executable-looking byte patterns in the .data overlay?
blob = d[pe.sections[6].raw:pe.sections[6].raw + pe.sections[6].rawsize]
for pat, label in [(b"\xC5", "VEX2/3 prefix C5"), (b"\x62", "EVEX 62"), (b"\x0F\x28", "movaps"),
                   (b"\xC4", "VEX3 C4"), (b"\x66\x0F", "66 0F")]:
    c = blob.count(pat)
    print(f"overlay pattern {label}: {c} ({c/len(blob):.4%})")
