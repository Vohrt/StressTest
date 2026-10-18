"""Minimal PE32+ reader for static analysis of aida_bench64.dll.

Provides: section mapping, RVA<->file-offset translation, .pdata function
boundary listing, ASCII/UTF-16 string scanning and xref search.
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass


@dataclass
class Section:
    name: str
    va: int          # virtual address (RVA)
    vsize: int
    raw: int         # file offset of raw data
    rawsize: int

    @property
    def end(self) -> int:
        return self.va + max(self.vsize, self.rawsize)


class PEFile:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            self.data = fh.read()
        self._parse()

    # ---------------------------------------------------------------- parsing
    def _parse(self):
        d = self.data
        if d[:2] != b"MZ":
            raise ValueError("not a PE file")
        e_lfanew = struct.unpack_from("<I", d, 0x3C)[0]
        if d[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            raise ValueError("missing PE signature")
        coff = e_lfanew + 4
        (self.machine, self.nsec, self.timestamp, self.symptr, self.nsym,
         self.opt_size, self.characteristics) = struct.unpack_from("<HHIIIHH", d, coff)
        opt = coff + 20
        magic = struct.unpack_from("<H", d, opt)[0]
        if magic != 0x20B:
            raise ValueError(f"expected PE32+ (0x20B), got {magic:#x}")
        self.image_base = struct.unpack_from("<Q", d, opt + 24)[0]
        self.entry_rva = struct.unpack_from("<I", d, opt + 16)[0]
        self.size_of_image = struct.unpack_from("<I", d, opt + 56)[0]
        n_dd = struct.unpack_from("<I", d, opt + 108)[0]
        self.dirs = [struct.unpack_from("<II", d, opt + 112 + 8 * i) for i in range(n_dd)]

        sec_off = opt + self.opt_size
        self.sections: list[Section] = []
        for i in range(self.nsec):
            off = sec_off + 40 * i
            name = d[off:off + 8].rstrip(b"\0").decode("latin1")
            vsize, va, rawsize, raw = struct.unpack_from("<IIII", d, off + 8)
            self.sections.append(Section(name, va, vsize, raw, rawsize))

    # ------------------------------------------------------------- addressing
    def sec_of_rva(self, rva: int) -> Section | None:
        for s in self.sections:
            if s.va <= rva < s.end:
                return s
        return None

    def off2rva(self, off: int) -> int:
        for s in self.sections:
            if s.raw <= off < s.raw + s.rawsize:
                return s.va + (off - s.raw)
        raise ValueError(f"file offset {off:#x} not in any section")

    def rva2off(self, rva: int) -> int:
        s = self.sec_of_rva(rva)
        if s is None or rva - s.va >= s.rawsize:
            raise ValueError(f"rva {rva:#x} not mapped to raw data")
        return s.raw + (rva - s.va)

    def read_rva(self, rva: int, size: int) -> bytes:
        off = self.rva2off(rva)
        return self.data[off:off + size]

    def va2off(self, va: int) -> int:
        return self.rva2off(va - self.image_base)

    @property
    def text(self) -> Section:
        return next(s for s in self.sections if s.name == ".text")

    # ------------------------------------------------------------- .pdata
    def functions(self) -> list[tuple[int, int]]:
        """Return (start_rva, end_rva) for every function in the exception directory."""
        if len(self.dirs) < 4:
            return []
        rva, size = self.dirs[3]
        if not rva or not size:
            return []
        out = []
        for i in range(size // 12):
            begin, end, unwind = struct.unpack_from("<III", self.data, self.rva2off(rva) + 12 * i)
            if begin and end > begin:
                out.append((begin, end))
        out.sort()
        return out

    def func_at(self, rva: int) -> tuple[int, int] | None:
        for b, e in self.functions():
            if b <= rva < e:
                return (b, e)
        return None

    # ------------------------------------------------------------- strings
    def find_ascii(self, needle: bytes, min_off: int = 0) -> list[int]:
        out, pos = [], min_off
        while True:
            pos = self.data.find(needle, pos)
            if pos < 0:
                return out
            out.append(pos)
            pos += 1


if __name__ == "__main__":
    pe = PEFile(sys.argv[1])
    print(f"image base {pe.image_base:#x}, image size {pe.size_of_image:#x}, entry {pe.entry_rva:#x}")
    for s in pe.sections:
        print(f"  {s.name:10s} rva={s.va:#09x} vsize={s.vsize:#09x} raw={s.raw:#09x} rawsize={s.rawsize:#09x}")
    fns = pe.functions()
    print(f"functions in .pdata: {len(fns)}")
    tot = sum(e - b for b, e in fns)
    print(f"total function bytes: {tot} ({tot/1024/1024:.2f} MB)")
