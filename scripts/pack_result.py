#!/usr/bin/env python3
"""Pack the minimal reproduction materials of one npusim run.

The final ``trace.json`` and traffic traces stay uncompressed outside the
archive.  This archive contains the configuration, raw SQLite/instruction
records, stage header, logs and analysis metadata needed to reproduce or
re-export the result offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def selected_files(run: Path, include_raw: bool) -> list[Path]:
    files: set[Path] = set()
    exact = (
        "config.json",
        "status.json",
        "build/profile_stages.h",
        "logs/npusim_record.log",
    )
    for name in exact:
        path = run / name
        if path.is_file():
            files.add(path)
    for pattern in ("results/analysis/*.json", "results/analysis/*.csv"):
        files.update(path for path in run.glob(pattern) if path.is_file())
    if include_raw:
        for pattern in ("data/raw/capture/*/record/*.db",
                        "data/raw/capture/*/record/*instr.bin"):
            files.update(path for path in run.glob(pattern) if path.is_file())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compress-level", type=int, choices=range(1, 10), default=6)
    parser.add_argument("--exclude-raw", action="store_true",
                        help="Do not package raw SQLite/instruction records")
    args = parser.parse_args()

    run = args.run.resolve()
    output = (args.output or run / "reproduce.tar.gz").resolve()
    manifest_path = (args.manifest or run / "reproduce_manifest.json").resolve()
    if output.exists():
        raise FileExistsError(output)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    files = selected_files(run, not args.exclude_raw)
    if not files:
        raise RuntimeError("No reproduction files found")

    members = {}
    for path in files:
        members[str(path.relative_to(run))] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=args.compress_level) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(run)), recursive=False)

    manifest = {
        "schema": "megamoe_npusim_reproduce_pack_v1",
        "run": str(run),
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": digest(output),
        "compress_level": args.compress_level,
        "includes_raw_records": not args.exclude_raw,
        "excludes": ["results/report/trace.json", "results/report/trace_with*.json"],
        "members": members,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "archive": str(output),
        "manifest": str(manifest_path),
        "members": len(members),
        "bytes": manifest["archive_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
