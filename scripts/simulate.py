#!/usr/bin/env python3
"""Run one npusim case and preserve task-completion evidence."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from config_io import load_config
from execution_contract import execution_evidence, require_remote_simulation


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def find_task_done(root):
    found = {}
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "[TASK_DONE]" in text:
            found[str(path)] = text.count("[TASK_DONE]")
    return found


def prepare_local_camodel(run, cann_env):
    """Copy the simulator model into the run so stars logs stay reproducible."""
    cann_root = Path(cann_env).resolve().parent
    source = cann_root / "tools/simulator/dav_3510/camodel"
    local = run / "build/toolkit/tools/simulator/dav_3510/camodel"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    for item in source.iterdir():
        generated = item.name.startswith("chip") and item.name.endswith("_instr.bin")
        generated = generated or item.name == "instr.bin" or item.name.startswith("stars_log")
        if item.is_file() and not item.name.startswith(".") and not generated:
            shutil.copy2(item, local / item.name)
    return run / "build/toolkit"


def recover_camodel_logs(run, cann_env):
    """Compatibility recovery for runs started before camodel isolation."""
    cann_root = Path(cann_env).resolve().parent
    source = cann_root / "tools/simulator/dav_3510/camodel"
    local = run / "build/toolkit/tools/simulator/dav_3510/camodel"
    local.mkdir(parents=True)
    for item in source.glob("stars_log*"):
        if item.is_file() and not (local / item.name).exists():
            shutil.copy2(item, local / item.name)
    return local


def collect_instruction_records(run, archive, cann_env, world):
    """Move npusim instruction records from camodel into the run archive."""
    local = run / "build/toolkit/tools/simulator/dav_3510/camodel"
    source_root = Path(cann_env).resolve().parent / "tools/simulator/dav_3510/camodel"
    record = archive / "record"
    names = [f"chip{rank}_instr.bin" for rank in range(world)] if world > 1 else ["instr.bin"]
    for name in names:
        destination = record / name
        if destination.exists():
            continue
        source = local / name
        if not source.exists():
            source = source_root / name
        if source.exists():
            shutil.move(str(source), str(destination))
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--npusim", default="npusim")
    parser.add_argument("--cann-env", type=Path,
                        help="Remote CANN set_env.sh; defaults to runtime.cann_env from the config")
    parser.add_argument("--recover", action="store_true",
                        help="Validate an existing capture without rerunning npusim")
    args = parser.parse_args()
    require_remote_simulation('npusim_execution')
    run = args.run.resolve()
    config = load_config(args.config)
    cann_env = str(args.cann_env.resolve()) if args.cann_env is not None else config["runtime"]["cann_env"]
    case_dir = run / "data/inputs"
    output = run / "data/raw/capture"
    log = run / "logs/npusim_record.log"
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    if args.recover:
        if not output.is_dir():
            raise FileNotFoundError(output)
        recover_camodel_logs(run, cann_env)
        result = argparse.Namespace(returncode=0)
    else:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        local_home = prepare_local_camodel(run, cann_env)
        command = (f"source {cann_env} && export ASCEND_HOME_PATH={local_home} && "
                   f"exec {args.npusim} record -c {case_dir} -o {output} -g")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            timeout_seconds = int(config.get("simulation", {}).get("timeout_seconds", 0))
            result = subprocess.run(["bash", "-c", command], stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout_seconds or None)
    archives = sorted(path for path in output.iterdir() if path.is_dir())
    if result.returncode or not archives:
        raise RuntimeError(f"npusim failed with exit code {result.returncode}; see {log}")
    if len(archives) != 1:
        raise RuntimeError(f"Expected one npusim archive, found {len(archives)}")
    archive = archives[0]
    instruction_records = collect_instruction_records(
        run, archive, cann_env, config["case"]["ep"]
    )
    task_done = find_task_done(archive)
    task_done.update(find_task_done(run / "build/toolkit/tools/simulator/dav_3510/camodel"))
    write_json(run / "results/analysis/capture.json", {
        "archive": str(archive), "returncode": result.returncode,
        "task_done": task_done, "case_dir": str(case_dir), "recovered": args.recover,
        "instruction_records": instruction_records,
        "execution": execution_evidence('npusim_execution', 'model1'),
        "cann_env": cann_env,
    })
    if not task_done:
        raise RuntimeError("No [TASK_DONE] marker found in npusim output")
    print(json.dumps({"archive": str(archive), "returncode": result.returncode,
                      "task_done_files": len(task_done), "recovered": args.recover}, indent=2))


if __name__ == "__main__":
    main()
