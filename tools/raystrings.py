import re
import sys

sys.path.insert(0, r"C:\Users\Voheart\source\repos\CpuStressTest\tools")
from pedump import PEFile

pe = PEFile(r"C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll")
d = pe.data
blob_start, blob_end = 0x99a000, 0x9e6600
allstr = []
for m in re.finditer(rb"[\x20-\x7e]{3,120}", d[blob_start:blob_end]):
    allstr.append((blob_start + m.start(), m.group().decode()))

print("### everything mentioning RayTrace / Ray\n")
for off, s in allstr:
    if re.search(r"Ray", s):
        print(f"{off:#08x}  {s}")

print("\n### region 0x9d5400 - 0x9d6000 (bench descriptors)\n")
for off, s in allstr:
    if 0x9d5400 <= off < 0x9d6000:
        print(f"{off:#08x}  {s}")
