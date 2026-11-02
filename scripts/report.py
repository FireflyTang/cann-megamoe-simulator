#!/usr/bin/env python3
"""Merge current MegaMoE DFX stages with the native npusim instruction trace."""
import argparse
import collections
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_io import load_config
import dfx


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--stage-header", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    run = args.run.resolve()
    config = load_config(args.config)
    capture = json.loads((run / "results/analysis/capture.json").read_text())
    archive = Path(capture["archive"])
    identity = json.loads((run / "results/analysis/case_identity.json").read_text())
    tiling_key = identity["tiling_key"]
    stage_order = dfx.parse_stage_order(args.stage_header)
    names = {index: name for index, name in enumerate(stage_order)}
    mapping = dfx.static_mark_map(args.kernel)
    entries = dfx.elf_mixed_entries(args.kernel, tiling_key)
    if not mapping:
        raise RuntimeError(
            "Static DFX marker map is empty; the selected llvm-objdump cannot decode "
            "DFX_REGION.MTE2. Set MEGAMOE_LLVM_OBJDUMP to a compatible local objdump."
        )
    stages = []
    marks = {}
    for rank in range(config["case"]["ep"]):
        db = archive / f"record/chip{rank}.db"
        decoded = dfx.decode_db_marks(db, mapping, entries, rank)
        intervals = dfx.pair_marks(decoded)
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM ExecutedInstructions "
                "WHERE ExecInstrName IN ('DFX_REGION','DFX_REGION_V')"
            ).fetchone()[0]
        if not count or count != len(decoded) or count != 2 * len(intervals):
            raise RuntimeError(f"Rank {rank}: incomplete DFX markers")
        if not all(interval["id"] in names for interval in intervals):
            raise RuntimeError(f"Rank {rank}: unknown DFX stage ID")
        coeff = 1 / config.get("simulation", {}).get("ticks_per_us", 1650)
        events = dfx.build_events(intervals, names, coeff)
        for event in events:
            event["pid"] = f"MegaMoE rank{rank}"
            if event["name"] == "process_name":
                event["args"]["name"] = event["pid"]
            elif event["name"] == "process_sort_index":
                event["args"]["sort_index"] = 10000 + rank
        stages.extend(events)
        marks[rank] = {
            "raw_marks": count, "paired_intervals": len(intervals),
            "labels": dict(collections.Counter(names[interval["id"]] for interval in intervals)),
        }
        write_json(run / f"results/analysis/intervals_rank{rank}.json", intervals)

    raw_files = [archive / "record" / (f"chip{rank}_instr.bin" if config["case"]["ep"] > 1 else "instr.bin")
                 for rank in range(config["case"]["ep"])]
    native_root = run / "data/raw/msprof"
    if native_root.exists():
        raise FileExistsError(native_root)
    soc_version = config.get("simulation", {}).get("soc_version", "Ascend950PR_957C")
    if soc_version.startswith("Ascend950PR"):
        soc_version = "dav_3510"
    command = [
        args.python, str(Path(__file__).with_name("export_msprof.py")), *map(str, raw_files),
        "--core-id", "all", "--soc-version", soc_version,
        "--output", str(native_root),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    native_file = Path(result.stdout.strip().splitlines()[-1])
    trace = json.loads(native_file.read_text())
    native_pids = {event.get("pid") for event in trace.get("traceEvents", [])}
    if native_pids.intersection(event["pid"] for event in stages):
        raise RuntimeError("Stage process IDs collide with native instruction processes")
    trace["traceEvents"].extend(stages)
    write_json(run / "results/report/trace.json", trace)
    write_json(run / "results/analysis/marks.json", marks)
    write_json(run / "status.json", {
        "state": "passed", "backend": "npusim",
        "trace": str(run / "results/report/trace.json"),
    })
    print(json.dumps({"trace": str(run / "results/report/trace.json"), "marks": marks}, indent=2))


if __name__ == "__main__":
    main()
