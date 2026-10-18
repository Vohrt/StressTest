import re
import sys

sys.path.insert(0, r"C:\Users\Voheart\source\repos\CpuStressTest\tools")
from pedump import PEFile

pe = PEFile(r"C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll")
d = pe.data

for s in pe.sections:
    blob = d[s.raw:s.raw + s.rawsize]
    strs = set(m.group().decode() for m in re.finditer(rb"[\x20-\x7e]{4,60}", blob))
    print(f"--- {s.name}: {len(strs)} ascii strings in {len(blob)} bytes")

print()
for frag in [b"Ray-Trace", b"RayTrace", b"Julia", b"Queen", b"Photo", b"CheckMate",
             b"SHA3", b"ZLIB", b"ZLib", b"FP64", b"bench", b"Bench", b"Score", b"score",
             b"AES", b"Fractal", b"Memory"]:
    offs = []
    p = 0
    while True:
        p = d.find(frag, p)
        if p < 0:
            break
        offs.append(p)
        p += 1
    print(f"{frag!r}: {len(offs)} -> {[hex(o) for o in offs[:8]]}")

print()
# sample distinct long strings from .data
blob = d[pe.sections[6].raw:pe.sections[6].raw + pe.sections[6].rawsize]
strs = sorted(set(m.group().decode() for m in re.finditer(rb"[\x20-\x7e]{8,80}", blob)))
print(f".data long strings: {len(strs)}")
for x in strs[:60]:
    print("   ", x)
