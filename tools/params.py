import re
import sys

sys.path.insert(0, r"C:\Users\Voheart\source\repos\CpuStressTest\tools")
from pedump import PEFile

pe = PEFile(r"C:\Users\Voheart\OneDrive\Tools\aida64\aida_bench64.dll")
d = pe.data
blob_start, blob_end = 0x99a000, 0x9e6600
allstr = [(blob_start + m.start(), m.group().decode())
          for m in re.finditer(rb"[\x20-\x7e]{3,120}", d[blob_start:blob_end])]

print("### RayTrace config area 0x9d7f00-0x9d8100\n")
for off, s in allstr:
    if 0x9d7f00 <= off < 0x9d8150:
        print(f"{off:#08x}  {s}")

print("\n### keys mentioning MinRunTime / StepMethod / KiloRay / CompletedLines\n")
for off, s in allstr:
    if re.search(r"MinRunTime|StepMethod|KiloRay|CompletedLines|UsedScene|UsedMethod|MethName", s):
        print(f"{off:#08x}  {s}")

print("\n### zlib / photo / pack / memory method names\n")
for off, s in allstr:
    if re.search(r"PHOTO|Photo|Worxx|NGPack|ZLib|Zlib|MTMBW", s) and off < 0x9d9000:
        print(f"{off:#08x}  {s}")
