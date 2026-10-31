---
name: megamoe-remote-sim
description: MegaMoE 950PR multi-P npusim simulation workflow, from SIM DFX instrumentation through trace export, traffic reconstruction and reproduction packing.
---

# MegaMoE 多 P npusim 仿真

## 1. Authority and scope

Current common-repository baseline: `f38d7f8`.

The authoritative implementation and detailed workflow live in:

```text
/home/t00826093/sources/common/megamoe-sim
```

If `/Volumes/tang-home` is mounted, this repository is expected to be mirrored
or symlinked at:

```text
/Volumes/tang-home/sources/common/megamoe-sim
```

This skill is only the process guide.  Do not copy the scripts into a run
directory.  Always use the versioned scripts in the common repository and record
that repository’s Git revision in the run evidence.

The detailed operating contract is:

```text
/home/t00826093/sources/common/megamoe-sim/docs/workflow.md
```

## 2. Boundaries

- This is offline CANN / npusim simulation.  It does not access real NPU
  hardware and therefore does not use `qrun`.
- Any real NPU or GPU execution still requires `qrun`.
- Target platform: Ascend 950PR / DAV_3510.
- Current rank-traffic reconstruction is valid for the MTE/UBMEM topology
  (`topoType == TOPO_TYPE_MTE`).  A URMA topology needs a separate adapter.
- Use the `tang` environment only:
  - local: `/home/t00826093/miniforge3/envs/tang/bin/python`
  - model1: `/home/t00826093/miniconda3/envs/tang/bin/python`
- One CANN environment should run only one simulation process at a time.

## 3. Workflow

All commands assume `<source-repo>` is the MegaMoE source repository and
`<run>` is a fresh, absolute run root.

### 3.1 Build the SIM device image

```bash
/path/to/tang/python scripts/build_device.py \
  --repo <source-repo> \
  --precision w8a8 \
  --mode sim \
  --topk-dtype bfloat16 \
  --output <run>/device \
  --generated <run>/generated \
  --build-dir <run>/cmake
```

### 3.2 Build the host bridge

```bash
/path/to/tang/python scripts/build_host.py \
  --repo <source-repo> \
  --official-build <run>/cmake \
  --output <run>/host \
  --precision w8a8 \
  --topk-dtype bfloat16
```

### 3.3 Generate the UBMEM case

```bash
/path/to/tang/python scripts/case.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run \
  --host <run>/host/libmegamoe_host.so \
  --kernel <run>/device/bin/MegaMoe_<hash>.o \
  --stage-header <source-repo>/megamoe_profile/include/profile_stages.h
```

### 3.4 Run npusim

```bash
/path/to/tang/python scripts/simulate.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run
```

If only the Python wrapper was interrupted and the simulator completed, use
`--recover`; recovery never reruns `npusim`.

### 3.5 Export and merge the trace

```bash
source /path/to/cann/set_env.sh
/path/to/tang/python scripts/report.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run \
  --kernel <run>/device/bin/MegaMoe_<hash>.o \
  --stage-header <source-repo>/megamoe_profile/include/profile_stages.h
```

### 3.6 Reconstruct traffic

```bash
/path/to/tang/python scripts/traffic.py \
  --run <run>/run
```

### 3.7 Organize the trace

```bash
/path/to/tang/python scripts/organize_trace.py \
  --trace <run>/run/results/report/trace_with_traffic.json \
  --output <run>/run/results/report/trace_organized.json
```

### 3.8 Pack the minimal reproduction archive

```bash
/path/to/tang/python scripts/pack_result.py \
  --run <run>/run
```

## 4. Key contracts

- Final trace JSON files remain uncompressed.
- Minimal reproduction materials are stored separately in `reproduce.tar.gz`.
- Raw SQLite and instruction records are retained by default.
- Traffic values are instruction-level model estimates, not hardware counters.
- Rank traffic is point-to-point only; aggregate total lanes are removed.
- `status.json` must become `passed` only after structural and evidence checks
  succeed.

## 5. Acceptance

A run is accepted only when:

1. every rank has `[TASK_DONE]`,
2. `npusim` exit code is 0,
3. raw SQLite and instruction records exist,
4. all DFX start/stop marks close,
5. every DFX id maps to a known stage in `profile_stages.h`,
6. `results/report/trace.json` contains both native instruction events and
   MegaMoE DFX lanes,
7. if requested, `trace_with_traffic.json` and `trace_organized.json` parse.

Compiler success, process exit, or database existence alone is not sufficient.

## 6. Continuous improvement

During use, if you find any defect or capability gap in this skill and it
belongs to the common simulation capability, supplement and fix this skill
directly.  Record the issue, the affected stage, and the corrected workflow in
the common repository’s documentation before the next run.
