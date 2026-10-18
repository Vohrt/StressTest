import re
import sys

sys.path.insert(0, r"C:\Users\Voheart\source\repos\CpuStressTest\tools")
from pedump import PEFile

pe = PEFile(r"C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll")
d = pe.data

print("### strings in .rdata that mention a bench/fractal/ray/param keyword\n")
blob_start = 0x99a000
blob_end = 0x9e6600
seen = []
for m in re.finditer(rb"[\x20-\x7e]{4,90}", d[blob_start:blob_end]):
    off = blob_start + m.start()
    s = m.group().decode()
    if re.search(r"RayTrace|Ray|Fractal|Julia|Mandel|Queen|NG|Bench|Photo|Check|AES|SHA|ZLib|Zlib|Pack|Image|Mem|SST|Hash|InstLat|Width|Height|Iterat|Loop|Thread|Size|KByte|MByte|Count", s):
        seen.append((off, s))
print(f"{len(seen)} hits")
for off, s in seen:
    print(f"{off:#08x}  {s}")
