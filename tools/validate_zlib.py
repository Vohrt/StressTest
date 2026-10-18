"""Validate the C++ DEFLATE compressor against CPython's zlib for many sizes.

Runs tools/verify_compress.exe (built from tools/verify_compress.cpp) once per size and asks
zlib to decode the result, reporting the first size at which the compressor's
output is not a valid zlib stream.
"""
import os
import subprocess
import zlib

EXE = r"C:\Users\Voheart\source\repos\CpuStressTest\build\verify_compress.exe"
SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127,
         128, 129, 255, 256, 257, 511, 1000, 4096, 20000, 100000]

bad = 0
for n in SIZES:
    env = dict(os.environ, CST_N=str(n))
    out = subprocess.run([EXE], capture_output=True, text=True, env=env, check=False).stdout
    tok = dict(kv.split("=", 1) for kv in out.strip().split() if "=" in kv)
    clen = int(tok.get("LEN", "0"))
    data = bytes.fromhex(tok.get("HEX", ""))
    try:
        dec = zlib.decompress(data)
        ok = len(dec) == n
        status = "OK" if ok else f"LEN MISMATCH ({len(dec)})"
    except zlib.error as exc:
        status = f"FAIL {exc}"
        ok = False
    if not ok:
        bad += 1
    print(f"n={n:7d} comp={clen:7d} {status}")
    if bad >= 3:
        print("stopping after 3 failures")
        break

print("failures:", bad)
