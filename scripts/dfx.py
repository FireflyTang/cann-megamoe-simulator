#!/usr/bin/env python3
"""Decode current MegaMoE DFX marks from npusim record databases."""
import math
import os
import re
import sqlite3
import struct
import subprocess

LLVM_OBJDUMP_CANDIDATES = [
    os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "tools/bisheng_compiler/bin/llvm-objdump"),
    os.path.expanduser("~/Ascend/9.2.0-weekly.20260902.01/cann-9.2.0/tools/bisheng_compiler/bin/llvm-objdump"),
    "llvm-objdump",
]
DISASM_RE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)$")
MOVI_RE = re.compile(r"^MOVI\s+X(\d+),\s*#(\d+)")
DFX_RE = re.compile(r"^DFX_REGION[\w.]*\s+X(\d+)")
CHIP_MEM_SIZE = 0x10000000000
AIC_SIM_LOAD_BASE = 0
AIV_SIM_LOAD_BASE = 0x100000


def find_llvm_objdump():
    for candidate in LLVM_OBJDUMP_CANDIDATES:
        if candidate and (os.path.isfile(candidate) or shutil_which(candidate)):
            return candidate
    raise RuntimeError("llvm-objdump not found; source the CANN environment")


def shutil_which(command):
    from shutil import which
    return which(command) is not None


def static_mark_map(elf_path):
    out = subprocess.run([find_llvm_objdump(), "-d", str(elf_path)], capture_output=True, text=True, check=True).stdout
    program = []
    for line in out.splitlines():
        match = DISASM_RE.match(line)
        if match:
            program.append((int(match.group(1), 16), match.group(2), match.group(3)))
    reg_val = {}
    mark_map = {}
    for pc, mnemonic, args in program:
        movi = MOVI_RE.match(f"{mnemonic} {args}")
        if mnemonic == "MOVI" and movi:
            reg_val[int(movi.group(1))] = int(movi.group(2))
            continue
        dfx = DFX_RE.match(f"{mnemonic} {args}")
        if mnemonic.startswith("DFX_REGION") and dfx:
            reg = int(dfx.group(1))
            imm = reg_val.get(reg)
            if imm is not None and (imm & 0xC00) in (0x400, 0xC00):
                mark_map[pc] = (imm & 0xF3FF, (imm & 0xC00) == 0x400)
    return mark_map


def elf_mixed_entries(elf_path, tiling_key):
    data = open(elf_path, "rb").read()
    if data[:5] != b"\x7fELF\x02":
        raise ValueError("Only little-endian 64-bit ELF is supported")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    entsize, count, names_idx = struct.unpack_from("<HHH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(count)]
    symtab = next((s for s in sections if s[1] == 2 and s[6] != 0), None)
    if symtab is None:
        raise ValueError("ELF symbol table not found")
    strtab = sections[symtab[6]]
    strings = data[strtab[4]:strtab[4] + strtab[5]]
    entries = {}
    for off in range(symtab[4], symtab[4] + symtab[5], symtab[9]):
        name_off, _, _, _, value, _ = struct.unpack_from("<IBBHQQ", data, off)
        name = strings[name_off:].split(b"\0", 1)[0].decode(errors="ignore")
        for kind in ("aic", "aiv"):
            if name.endswith(f"_{tiling_key}_mix_{kind}"):
                entries[kind] = value
    if set(entries) != {"aic", "aiv"}:
        raise ValueError(f"Missing MIX entries for tiling key {tiling_key}: {sorted(entries)}")
    return entries


def decode_db_marks(db_path, mark_map, mix_entries, rank=0):
    aiv_entry = mix_entries.get("aiv", 0)
    aic_entry = mix_entries.get("aic", 0)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""SELECT ExecInstrTickStart, SourceInstrAddr, ExecInstrName,
                  ExecInstrCoreId, CoreTypeId,
                  json_extract(ExtendParams,'$.core_type'),
                  json_extract(ExtendParams,'$.queue_type')
           FROM ExecutedInstructions
           WHERE ExecInstrName IN ('DFX_REGION','DFX_REGION_V')""").fetchall()
    conn.close()
    marks = []
    for tick, pc, name, core, ctype, core_type, queue in rows:
        if not isinstance(tick, int):
            continue
        resolved = None
        for candidate in (
            pc,
            pc - AIC_SIM_LOAD_BASE + aic_entry,
            pc - AIV_SIM_LOAD_BASE + aiv_entry,
        ):
            if candidate in mark_map:
                resolved = mark_map[candidate]
                break
        if resolved is None:
            continue
        raw_id, is_start = resolved
        marks.append({
            "tick": tick, "pc": pc, "id": raw_id, "raw_id": raw_id,
            "is_start": is_start,
            "core": core, "core_type": core_type or f"CT{ctype}", "queue": queue,
        })
    return marks


def pair_marks(marks):
    intervals, open_by_key = [], {}
    for mark in sorted(marks, key=lambda item: (item["core"], item["core_type"], item["tick"])):
        key = (mark["core"], mark["core_type"], mark["id"])
        if mark["is_start"]:
            if key in open_by_key:
                raise ValueError(f"DFX id {mark['id']} reopened on core {mark['core']}")
            open_by_key[key] = mark
        else:
            start = open_by_key.pop(key, None)
            if start is None:
                raise ValueError(f"DFX stop without start: id={mark['id']} core={mark['core']}")
            if mark["tick"] <= start["tick"]:
                raise ValueError(f"Nonpositive DFX interval: id={mark['id']} core={mark['core']}")
            intervals.append({
                "core": mark["core"], "core_type": mark["core_type"], "id": mark["id"],
                "start_tick": start["tick"], "end_tick": mark["tick"],
                "start_pc": start["pc"], "end_pc": mark["pc"],
            })
    if open_by_key:
        raise ValueError(f"Unclosed DFX marks: {sorted(open_by_key)}")
    return intervals


def validate_nesting(intervals):
    by_lane = {}
    for interval in intervals:
        by_lane.setdefault((interval["core"], interval["core_type"]), []).append(interval)
    for lane_intervals in by_lane.values():
        stack = []
        for interval in sorted(lane_intervals, key=lambda item: (item["start_tick"], -item["end_tick"], item["id"])):
            while stack and stack[-1]["end_tick"] <= interval["start_tick"]:
                stack.pop()
            if stack and interval["end_tick"] > stack[-1]["end_tick"]:
                raise ValueError("Improperly overlapping DFX intervals")
            stack.append(interval)


PALETTE = ["generic_work", "iowait", "rpc", "async_tts", "good", "bad",
           "light_memory", "vsync", "grid_row"]


def lane_name(core, core_type):
    return f"AIC0_{core:02d}" if core_type == "AIC" else f"{core_type}-{core:02d}"


def build_events(intervals, names, coeff):
    validate_nesting(intervals)
    by_lane = {}
    for interval in intervals:
        by_lane.setdefault((interval["core"], interval["core_type"]), []).append(interval)
    group_pid = 900
    events = [
        {"name": "process_name", "ph": "M", "pid": group_pid, "args": {"name": "MegaMoE DFX"}},
        {"name": "process_sort_index", "ph": "M", "pid": group_pid, "args": {"sort_index": group_pid}},
    ]
    lane_tid = 0
    role_order = {"AIC": 0, "AIV0": 1, "AIV1": 2}
    for (core, core_type), lane_intervals in sorted(
            by_lane.items(), key=lambda item: (item[0][0], role_order.get(item[0][1], 99), item[0][1])):
        lane_tid += 1
        tid = group_pid + lane_tid
        lane = lane_name(core, core_type)
        events.append({"name": "thread_name", "ph": "M", "pid": group_pid, "tid": tid, "args": {"name": lane}})
        events.append({"name": "thread_sort_index", "ph": "M", "pid": group_pid, "tid": tid,
                       "args": {"sort_index": lane_tid}})
        for interval in sorted(lane_intervals, key=lambda item: (item["start_tick"], -item["end_tick"])):
            label = names.get(interval["id"], f"id{interval['id']}")
            start_us = interval["start_tick"] * coeff
            end_us = interval["end_tick"] * coeff
            duration = end_us - start_us
            while start_us + duration > end_us:
                duration = math.nextafter(duration, 0.0)
            events.append({
                "name": label, "cat": core_type, "ph": "X", "pid": group_pid, "tid": tid,
                "ts": start_us, "dur": duration,
                "cname": PALETTE[interval["id"] % len(PALETTE)],
                "args": {
                    "id": interval["id"], "core": core, "core_type": core_type,
                    "ticks": interval["end_tick"] - interval["start_tick"],
                    "start_tick": interval["start_tick"], "end_tick": interval["end_tick"],
                    "attribution": "marked_stage", "start_pc": interval["start_pc"],
                    "end_pc": interval["end_pc"],
                },
            })
    return events


def parse_stage_header(header_path):
    text = open(header_path).read()
    body = re.search(r"enum\s+MoeProfileStage\s*:\s*uint32_t\s*\{(.*?)\}", text, re.S)
    if not body:
        raise ValueError(f"MoeProfileStage enum not found in {header_path}")
    names = {}
    for name, value in re.findall(r"([A-Z][A-Z0-9_]*)\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*,", body.group(1)):
        number = int(value, 0)
        if number > 0xFFFF:
            raise ValueError(f"Stage ID does not fit DFX uint16: {name}={number}")
        names[number] = name
    if not names:
        raise ValueError("No stage IDs parsed")
    return names


def parse_stage_order(header_path):
    text = open(header_path).read()
    body = re.search(r"enum\s+MoeProfileStage\s*:\s*uint32_t\s*\{(.*?)\}", text, re.S)
    if not body:
        raise ValueError(f"MoeProfileStage enum not found in {header_path}")
    names = [name for name, _ in re.findall(
        r"([A-Z][A-Z0-9_]*)\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*,", body.group(1)
    )]
    if not names:
        raise ValueError("No stage IDs parsed")
    return names
