#!/usr/bin/env python3
"""Organize a MegaMoE npusim Chrome trace for review.

The original per-core instruction lanes are collapsed under one ``Raw
Instructions`` process.  MegaMoE DFX, memory-path and point-to-point rank
traffic remain separate processes.  Aggregate rank-total lanes are removed;
only per-pair traffic is retained.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CORE_PID = re.compile(r"^core\d+\.(?:cubecore0|veccore0|veccore1)$")
RANK_TOTAL = re.compile(r"(?:cross[- ]rank[ _-]*)?total", re.I)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def organize(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("traceEvents", [])
    output = []
    raw_threads: dict[str, int] = {}
    raw_thread_names: dict[str, str] = {}

    for event in events:
        pid = event.get("pid")
        if isinstance(pid, str) and CORE_PID.fullmatch(pid):
            # Keep the original per-core pid as the thread id, so every core is
            # still one lane under the shared Raw Instructions process.
            new_event = dict(event)
            new_event["pid"] = "Raw Instructions"
            new_event["tid"] = pid
            if event.get("ph") == "M" and event.get("name") in ("thread_name", "thread_sort_index"):
                # These metadata events are regenerated below.
                continue
            output.append(new_event)
            raw_threads.setdefault(pid, len(raw_threads) + 1)
            raw_thread_names[pid] = pid
            continue

        if event.get("pid") == "Rank Traffic":
            if event.get("ph") == "M" and event.get("name") == "thread_name":
                label = str(event.get("args", {}).get("name", ""))
                if RANK_TOTAL.search(label):
                    continue
            if event.get("ph") == "C" and RANK_TOTAL.search(str(event.get("name", ""))):
                continue
        output.append(event)

    # Shared process metadata for raw instructions.
    output.append({"name": "process_name", "ph": "M", "pid": "Raw Instructions",
                   "args": {"name": "Raw Instructions"}})
    output.append({"name": "process_sort_index", "ph": "M", "pid": "Raw Instructions",
                   "args": {"sort_index": 5000}})
    for tid, sort_index in raw_threads.items():
        output.append({"name": "thread_name", "ph": "M", "pid": "Raw Instructions", "tid": tid,
                       "args": {"name": raw_thread_names[tid]}})
        output.append({"name": "thread_sort_index", "ph": "M", "pid": "Raw Instructions", "tid": tid,
                       "args": {"sort_index": sort_index}})

    # Stable process ordering: DFX ranks first, then traffic, then raw instructions.
    process_order = {
        "MegaMoE rank0": 1000,
        "MegaMoE rank1": 1100,
        "Rank Traffic": 1200,
        "Memory Traffic": 1300,
        "Raw Instructions": 5000,
    }
    for event in output:
        pid = event.get("pid")
        if event.get("ph") == "M" and event.get("name") == "process_sort_index" and pid in process_order:
            event["args"]["sort_index"] = process_order[pid]

    trace["traceEvents"] = output
    trace["traceOrganization"] = {
        "schema": "megamoe_npusim_trace_organization_v1",
        "rawInstructionsProcess": "Raw Instructions",
        "rawThreadIdentity": "original coreX.cubecore/veccore pid",
        "rankTraffic": "point-to-point only; aggregate total lanes removed",
    }
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = read_json(args.trace.resolve())
    organized = organize(trace)
    write_json(args.output.resolve(), organized)
    print(json.dumps({"trace": str(args.output.resolve()),
                      "events": len(organized["traceEvents"])}, indent=2))


if __name__ == "__main__":
    main()
