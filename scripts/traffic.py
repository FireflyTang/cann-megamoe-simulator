#!/usr/bin/env python3
"""Rebuild npusim memory-path and point-to-point rank traffic lanes.

This is an instruction-level estimate, not a hardware bandwidth counter.  The
input run must already contain the raw SQLite databases and a merged
``trace.json``.  The output is a new Chrome trace with additional counter
lanes; the original trace is never modified.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sqlite3
from pathlib import Path
from typing import Any

CHIP_MEM_SIZE = 0x10000000000
DATA_REGION_OFFSET = 0x8000000000
PEERMEM_WINDOW_OFFSET = 60 * 1024

PATH_KEYS = (
    "OUT_to_L1", "OUT_to_UB", "UB_to_OUT", "L0C_to_OUT", "L0C_to_UB",
    "L1_to_L0A", "L1_to_L0B", "L1_to_UNKNOWN",
    "RV_VLD", "RV_VLDI", "RV_VST", "RV_VSTI", "OTHER_ANNOTATED",
)
PATH_LABELS = {
    "OUT_to_L1": "OUT -> L1",
    "OUT_to_UB": "OUT -> UB",
    "UB_to_OUT": "UB -> OUT",
    "L0C_to_OUT": "L0C -> OUT",
    "L0C_to_UB": "L0C -> UB",
    "L1_to_L0A": "L1 -> L0A",
    "L1_to_L0B": "L1 -> L0B",
    "L1_to_UNKNOWN": "L1 -> UNKNOWN",
    "RV_VLD": "RV_VLD",
    "RV_VLDI": "RV_VLDI",
    "RV_VST": "RV_VST",
    "RV_VSTI": "RV_VSTI",
    "OTHER_ANNOTATED": "OTHER annotated",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def memory_path(name: str, extend: dict[str, Any]) -> str | None:
    """Classify one annotated instruction into a memory-path lane."""
    src = extend.get("src_mem")
    dst = extend.get("dst_mem")
    if not int(extend.get("processed_bytes") or 0):
        return None
    if src == "OUT" and dst == "L1":
        return "OUT_to_L1"
    if src == "OUT" and dst == "UB":
        return "OUT_to_UB"
    if src == "UB" and dst == "OUT":
        return "UB_to_OUT"
    if src == "L0C" and dst == "OUT":
        return "L0C_to_OUT"
    if src == "L0C" and dst == "UB":
        return "L0C_to_UB"
    if src == "L1" and dst == "L0A":
        return "L1_to_L0A"
    if src == "L1" and dst == "L0B":
        return "L1_to_L0B"
    if src == "L1" and dst == "UNKNOWN":
        return "L1_to_UNKNOWN"
    if name == "RV_VLD":
        return "RV_VLD"
    if name == "RV_VLDI":
        return "RV_VLDI"
    if name == "RV_VST":
        return "RV_VST"
    if name == "RV_VSTI":
        return "RV_VSTI"
    return "OTHER_ANNOTATED"


def rank_window(address: Any, world: int, size: int) -> int | None:
    if not isinstance(address, int):
        return None
    for rank in range(world):
        base = DATA_REGION_OFFSET + rank * CHIP_MEM_SIZE + PEERMEM_WINDOW_OFFSET
        if base <= address < base + size:
            return rank
    return None


def instruction_rows(run: Path) -> list[tuple[int, str, dict[str, Any], int, int]]:
    rows: list[tuple[int, str, dict[str, Any], int, int]] = []
    pattern = str(run / "data/raw/capture/*/record/chip*.db")
    databases = sorted(Path(item) for item in glob.glob(pattern))
    if not databases:
        raise FileNotFoundError(pattern)
    for chip, database in enumerate(databases):
        connection = sqlite3.connect(database)
        try:
            for name, extend_json, start, end in connection.execute(
                "SELECT ExecInstrName, ExtendParams, ExecInstrTickStart, ExecInstrTickEnd "
                "FROM ExecutedInstructions"
            ):
                rows.append((chip, name, json.loads(extend_json or "{}"), start, end))
        finally:
            connection.close()
    if not rows:
        raise RuntimeError("No instruction rows found")
    return rows


def build_bins(rows, *, world: int, window_size: int, bin_us: float, ticks_per_us: float):
    bin_ticks = max(1, int(round(bin_us * ticks_per_us)))
    first_tick = min(row[3] for row in rows)
    last_tick = max(row[4] for row in rows)
    count = max(1, (last_tick - first_tick + bin_ticks - 1) // bin_ticks)
    memory = {key: [0.0] * count for key in PATH_KEYS}
    memory_counts = {key: [0] * count for key in PATH_KEYS}
    pairs = [(src, dst) for src in range(world) for dst in range(world) if src != dst]
    rank = {pair: [0.0] * count for pair in pairs}
    rank_counts = {pair: [0] * count for pair in pairs}

    for chip, name, extend, start, end in rows:
        if end <= start:
            continue
        path = memory_path(name, extend)
        bytes_total = int(extend.get("processed_bytes") or 0)
        src_rank = rank_window(extend.get("src_addr"), world, window_size)
        dst_rank = rank_window(extend.get("dst_addr"), world, window_size)
        if src_rank is not None and dst_rank is not None:
            pair = (src_rank, dst_rank)
        elif src_rank is not None:
            pair = (src_rank, chip)
        elif dst_rank is not None:
            pair = (chip, dst_rank)
        else:
            pair = None
        if path is None and pair is None:
            continue
        first = max(0, (start - first_tick) // bin_ticks)
        last = min(count - 1, (end - first_tick) // bin_ticks)
        for index in range(first, last + 1):
            bin_start = first_tick + index * bin_ticks
            bin_end = bin_start + bin_ticks
            overlap = max(0, min(end, bin_end) - max(start, bin_start))
            if not overlap:
                continue
            fraction = overlap / float(end - start)
            if path is not None and bytes_total:
                memory[path][index] += bytes_total * fraction
                memory_counts[path][index] += 1
            if pair is not None and pair in rank and bytes_total:
                rank[pair][index] += bytes_total * fraction
                rank_counts[pair][index] += 1

    memory_bins = []
    rank_bins = []
    for index in range(count):
        start_tick = first_tick + index * bin_ticks
        end_tick = start_tick + bin_ticks
        row = {
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_us": start_tick / ticks_per_us,
            "end_us": end_tick / ticks_per_us,
        }
        for key in PATH_KEYS:
            row[f"{key}_bytes"] = memory[key][index]
            row[f"{key}_GBps"] = memory[key][index] / bin_us / 1000.0
            row[f"{key}_count"] = memory_counts[key][index]
        memory_bins.append(row.copy())
        for pair in pairs:
            key = f"rank{pair[0]}_to_rank{pair[1]}"
            row[f"{key}_bytes"] = rank[pair][index]
            row[f"{key}_GBps"] = rank[pair][index] / bin_us / 1000.0
            row[f"{key}_count"] = rank_counts[pair][index]
        rank_bins.append({key: value for key, value in row.items() if key.startswith("rank") or key in ("start_tick", "end_tick", "start_us", "end_us")})
    return memory_bins, rank_bins


def append_counter_lanes(trace: dict[str, Any], memory_bins, rank_bins, *, world: int) -> dict[str, Any]:
    events = trace.setdefault("traceEvents", [])

    memory_pid = "Memory Traffic"
    events.append({"name": "process_name", "ph": "M", "pid": memory_pid, "args": {"name": memory_pid}})
    events.append({"name": "process_sort_index", "ph": "M", "pid": memory_pid, "args": {"sort_index": 2000}})
    memory_lanes = list(PATH_KEYS)
    memory_lanes.extend(("explicit_external", "all_annotated"))
    for index, key in enumerate(memory_lanes):
        tid = 100 + index
        label = PATH_LABELS.get(key, "External total (explicit OUT)" if key == "explicit_external" else "All annotated")
        events.append({"name": "thread_name", "ph": "M", "pid": memory_pid, "tid": tid, "args": {"name": label}})
        events.append({"name": "thread_sort_index", "ph": "M", "pid": memory_pid, "tid": tid, "args": {"sort_index": 1000 + index}})
        for row in memory_bins:
            if key == "explicit_external":
                value = sum(row[f"{item}_GBps"] for item in ("OUT_to_L1", "OUT_to_UB", "UB_to_OUT", "L0C_to_OUT"))
            elif key == "all_annotated":
                value = sum(row[f"{item}_GBps"] for item in PATH_KEYS)
            else:
                value = row[f"{key}_GBps"]
            events.append({"name": label, "ph": "C", "pid": memory_pid, "tid": tid,
                           "ts": row["start_us"], "args": {"GB/s": round(value, 6)}})

    rank_pid = "Rank Traffic"
    events.append({"name": "process_name", "ph": "M", "pid": rank_pid, "args": {"name": rank_pid}})
    events.append({"name": "process_sort_index", "ph": "M", "pid": rank_pid, "args": {"sort_index": 2100}})
    pairs = [(src, dst) for src in range(world) for dst in range(world) if src != dst]
    for index, pair in enumerate(pairs):
        key = f"rank{pair[0]}_to_rank{pair[1]}"
        label = f"rank{pair[0]} -> rank{pair[1]}"
        tid = 200 + index
        events.append({"name": "thread_name", "ph": "M", "pid": rank_pid, "tid": tid, "args": {"name": label}})
        events.append({"name": "thread_sort_index", "ph": "M", "pid": rank_pid, "tid": tid, "args": {"sort_index": 2000 + index}})
        for row in rank_bins:
            events.append({"name": label, "ph": "C", "pid": rank_pid, "tid": tid,
                           "ts": row["start_us"], "args": {"GB/s": round(row[f"{key}_GBps"], 6)}})

    trace["memoryTrafficEstimate"] = {
        "schema": "megamoe_npusim_memory_traffic_v1",
        "source": "npusim ExecutedInstructions ExtendParams",
        "paths": [PATH_LABELS[key] for key in PATH_KEYS],
        "limits": [
            "instruction-level model estimate",
            "not a hardware bandwidth counter",
            "only instructions with processed_bytes are counted",
        ],
    }
    trace["rankTrafficEstimate"] = {
        "schema": "megamoe_npusim_rank_traffic_v1",
        "source": "npusim src_addr/dst_addr in UBMEM peermem windows",
        "directions": [f"rank{src}_to_rank{dst}" for src, dst in pairs],
        "limits": [
            "instruction-level model estimate",
            "not a hardware interconnect counter",
            "only point-to-point lanes are generated",
        ],
    }
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bin-us", type=float, default=0.25)
    args = parser.parse_args()

    run = args.run.resolve()
    config = read_json(run / "config.json")
    identity = read_json(run / "results/analysis/case_identity.json")
    trace_path = (args.trace or run / "results/report/trace.json").resolve()
    output_path = (args.output or run / "results/report/trace_with_traffic.json").resolve()
    ticks_per_us = float(config.get("simulation", {}).get("ticks_per_us", 1650))
    window_size = int(identity.get("peermem_bytes", 0))
    if window_size <= 0:
        raise ValueError("case_identity.json has no positive peermem_bytes")

    rows = instruction_rows(run)
    memory_bins, rank_bins = build_bins(
        rows,
        world=int(config["case"]["ep"]),
        window_size=window_size,
        bin_us=args.bin_us,
        ticks_per_us=ticks_per_us,
    )
    trace = read_json(trace_path)
    trace = append_counter_lanes(trace, memory_bins, rank_bins, world=int(config["case"]["ep"]))
    write_json(output_path, trace)
    print(json.dumps({
        "trace": str(output_path),
        "memory_bins": len(memory_bins),
        "rank_bins": len(rank_bins),
        "instruction_rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
