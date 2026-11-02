#!/usr/bin/env python3
"""Restore npusim binary records and export the native msprof trace."""
import argparse
import collections
import hashlib
import json
import mmap
import os
import struct
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-id", default="all")
    parser.add_argument("--soc-version", default="dav_3510")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Export timeout in seconds; 0 disables the timeout")
    args = parser.parse_args()
    raw_files = [path.resolve(strict=True) for path in args.raw_files]
    if len(set(raw_files)) != len(raw_files):
        parser.error("Duplicate input file")
    combined = len(raw_files) > 1
    wanted = None if args.core_id == "all" else {int(x) for x in args.core_id.split(",")}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump = out / "dump"
    dump.mkdir()
    record = struct.Struct("<QIIQB200s200s7x")
    for raw in raw_files:
        if not raw.stat().st_size or raw.stat().st_size % record.size:
            raise ValueError(f"Empty or incomplete binary record file: {raw}")
    counts = collections.Counter()
    streams = {}
    sources = []
    owners = {}
    try:
        for raw in raw_files:
            with raw.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                digest = hashlib.sha256(mm).hexdigest()
                sources.append({"raw_file": str(raw), "raw_sha256": digest})
                for tick, core, sub, pc, kind, dec, exe in record.iter_unpack(mm):
                    local_core = core % 32
                    if wanted is not None and local_core not in wanted:
                        continue
                    if sub not in (0, 1, 2) or kind not in (0, 1):
                        raise ValueError(f"Unsupported raw record: sub={sub}, kind={kind}")
                    core = core if combined else local_core
                    if core in owners and owners[core] != raw:
                        raise ValueError("Input chips have colliding global core IDs")
                    owners[core] = raw
                    unit = ("cubecore0", "veccore0", "veccore1")[sub]
                    suffix = "instr_popped_log" if kind else "instr_log"
                    name = f"core{core}.{unit}.{suffix}.dump"
                    if name not in streams:
                        streams[name] = (dump / name).open("w", encoding="utf-8")
                        os.chmod(dump / name, 0o600)
                    dec = dec.split(b"\0", 1)[0].decode("utf-8")
                    exe = exe.split(b"\0", 1)[0].decode("utf-8")
                    line = f"[info] {dec}" if "++++++++++++++++++++++" in dec else f"[info] [{tick:08d}] {dec}  {exe}"
                    streams[name].write(line + "\n")
                    counts[name] += 1
    finally:
        for stream in streams.values():
            stream.close()
    if not counts:
        raise ValueError("No records selected")
    command = build_command(dump, out / "output", args.soc_version, args.core_id, wanted is not None)
    with (out / "export.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout or None)
    reports = list((out / "output").glob("OPPROF_*/simulator/trace.json"))
    manifest = {
        "source_files": sources, "combined_global_core_ids": combined,
        "core_sources": {str(k): str(v) for k, v in owners.items()},
        "records_by_file": dict(counts), "command": command, "returncode": result.returncode,
        "reports": [str(path.relative_to(out)) for path in reports],
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode or len(reports) != 1 or "Profiling data parse finished." not in (out / "export.log").read_text():
        raise RuntimeError(f"Native export failed; see {out / 'export.log'}")
    data = json.loads(reports[0].read_text())
    if not any(event.get("ph") == "X" for event in data.get("traceEvents", [])):
        raise RuntimeError("Native report contains no duration events")
    print(reports[0])


def build_command(dump, output, soc_version, core_id, explicit_core):
    command = [
        "msprof", "op", "simulator",
        f"--export={dump}", f"--output={output}",
        f"--soc-version={soc_version}", "--aic-metrics=PipeUtilization",
    ]
    if explicit_core:
        command.append(f"--core-id={core_id}")
    return command


if __name__ == "__main__":
    main()
