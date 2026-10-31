# MegaMoE npusim common simulation capability

> Canonical path on the user’s Mac: `/Volumes/tang-home/sources/common/megamoe-sim`.
> Equivalent path in this Linux work environment: `/home/t00826093/sources/common/megamoe-sim`.

This repository is the common, version-controlled entry for the current MegaMoE
offline npusim workflow.  It contains the executable scripts used to:

1. build the SIM device image and host bridge,
2. generate a multi-P UBMEM case,
3. run `npusim`,
4. export and merge the native instruction trace with MegaMoE DFX stages,
5. reconstruct instruction-level memory-path and point-to-point rank traffic,
6. organize the Chrome trace into readable process groups, and
7. pack the minimal reproduction evidence.

The repository is intentionally independent of a particular MegaMoE source
tree.  Build scripts take an explicit `--repo`; simulation scripts take a run
directory and explicit artifacts.  The current rank-traffic reconstruction is
validated for the MTE/UBMEM topology (`topoType == TOPO_TYPE_MTE`).  A URMA
topology needs a separate adapter or explicit “unavailable” handling.

## Repository layout

```text
scripts/
  build_device.py        Build the SIM device image from a source repository
  build_host.py          Build the private host bridge
  case.py                Generate one multi-P UBMEM simulation case
  simulate.py            Run npusim and preserve task evidence
  export_msprof.py       Export native instruction records through msprof
  report.py              Merge native instructions with MegaMoE DFX stages
  traffic.py             Rebuild memory-path and point-to-point rank traffic
  organize_trace.py      Group raw instructions and remove aggregate lanes
  pack_result.py         Create the compressed minimal reproduction archive
docs/
  workflow.md            Detailed end-to-end workflow and acceptance notes
```

All scripts are runnable directly with the `tang` Python interpreter.  They do
not access real NPU hardware, so simulation does not use `qrun`.

## Prerequisites

- Ascend 950PR / DAV_3510 CANN with `npusim` and `msprof op simulator`.
- Python `tang` environment:
  - local: `/home/t00826093/miniforge3/envs/tang/bin/python`
  - model1: `/home/t00826093/miniconda3/envs/tang/bin/python`
- `json5`, `torch`, and `numpy` in the selected `tang` environment.
- Source CANN environment, for example:
  `/home/t00826093/Ascend/9.2.0-weekly.20260902.01/cann-9.2.0/set_env.sh`.

## End-to-end flow

Replace `<source-repo>` with the MegaMoE source repository and `<run>` with a
fresh, absolute output directory.  Never reuse a build or run directory from
another mode or attempt.

### 1. Build the SIM device image

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

`--mode sim` keeps the existing MegaMoE stage IDs and maps the profile macros
to DFX `TRACE_START/TRACE_STOP`.  The compiler guard verifies that
`MEGAMOE_SIM_DFX=1` is actually present in the generated recipe.

### 2. Build the host bridge

```bash
/path/to/tang/python scripts/build_host.py \
  --repo <source-repo> \
  --official-build <run>/cmake \
  --output <run>/host \
  --precision w8a8 \
  --topk-dtype bfloat16
```

The host bridge is used only for CPU tiling and case generation.

### 3. Generate the UBMEM case

```bash
/path/to/tang/python scripts/case.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run \
  --host <run>/host/libmegamoe_host.so \
  --kernel <run>/device/bin/MegaMoe_<hash>.o \
  --stage-header <source-repo>/megamoe_profile/include/profile_stages.h
```

The case generator writes per-rank input images, task metadata, kernel images
and the shared `Mc2MoeContext`.  It checks that the parameter/IO/workspace and
kernel-image HBM ranges do not overlap.

### 4. Run npusim

```bash
/path/to/tang/python scripts/simulate.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run
```

`simulate.py` copies the simulator model into `<run>/build/toolkit`, overrides
`ASCEND_HOME_PATH`, runs `npusim record`, moves `chip*_instr.bin` into the run
archive, and records `[TASK_DONE]` evidence.

For a completed run that was interrupted only at the Python wrapper level, use
`--recover`.  This validates the existing capture and does not rerun `npusim`.

### 5. Export and merge the trace

```bash
source /path/to/cann/set_env.sh
/path/to/tang/python scripts/report.py \
  --config <source-repo>/megamoe_profile/configs/sim/min_w8a8_2p.json \
  --run <run>/run \
  --kernel <run>/device/bin/MegaMoe_<hash>.o \
  --stage-header <source-repo>/megamoe_profile/include/profile_stages.h
```

This stage:

- decodes the DFX stage marks from the per-rank SQLite databases,
- exports the native instruction timeline with `msprof op simulator`,
- merges both into `results/report/trace.json`,
- writes `results/analysis/intervals_rank*.json`, and
- sets `status.json` to `passed` only after the structural checks succeed.

### 6. Rebuild traffic lanes

```bash
/path/to/tang/python scripts/traffic.py \
  --run <run>/run
```

This creates `results/report/trace_with_traffic.json` with:

- one `Memory Traffic` process containing explicit memory-path lanes,
- one `Rank Traffic` process containing only point-to-point lanes,
- instruction-level bandwidth estimates, not hardware counters.

The memory lanes are derived from `processed_bytes`, `src_mem`, `dst_mem`,
`src_addr`, and `dst_addr` in the raw instruction database.  The rank lanes are
derived from the UBMEM peermem windows inferred from the run identity.

### 7. Organize the trace

```bash
/path/to/tang/python scripts/organize_trace.py \
  --trace <run>/run/results/report/trace_with_traffic.json \
  --output <run>/run/results/report/trace_organized.json
```

This groups all original `coreX.cubecore0` / `coreX.veccore0` /
`coreX.veccore1` instruction lanes under a single `Raw Instructions` process.
MegaMoE DFX, memory-path traffic, and point-to-point rank traffic remain
separate.  Aggregate rank-total lanes are removed.

### 8. Pack the minimal reproduction archive

```bash
/path/to/tang/python scripts/pack_result.py \
  --run <run>/run
```

The archive is written to `<run>/run/reproduce.tar.gz` and a manifest to
`<run>/run/reproduce_manifest.json`.  The final trace JSONs remain uncompressed
outside the archive.  The archive also retains the kernel ELF, host bridge,
generated profiler headers and build evidence needed for offline re-export.
Raw SQLite and instruction records are included by default; use
`--exclude-raw` only when a caller explicitly does not need offline re-export.

## Acceptance

A run is accepted only when all of the following are true:

1. every rank log contains `[TASK_DONE]`,
2. `npusim` exit code is 0,
3. each rank has a complete SQLite instruction database and instruction record,
4. every DFX start/stop pair closes into a stage defined by
   `profile_stages.h`,
5. `results/report/trace.json` contains both native instruction events and
   MegaMoE DFX lanes,
6. `status.json` is `passed`,
7. if traffic reconstruction is requested, `trace_with_traffic.json` and
   `trace_organized.json` are present and parseable.

Never report compiler success, process exit, or the existence of a database as
business success by itself.
