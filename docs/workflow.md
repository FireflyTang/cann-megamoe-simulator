# MegaMoE npusim workflow

This document is the detailed operating contract for the scripts in this
repository.  The companion skill points here and does not duplicate the
implementation details.

## 0. Source and instrumentation contract

Before using this workflow, the MegaMoE source tree must already contain:

1. a stable `profile_stages.h` enum,
2. SIM-safe DFX macros that emit `TRACE_START` / `TRACE_STOP`,
3. compiler barriers around the marker calls so different stage markers cannot
   be tail-merged,
4. a generated `profile_events_generated.h` with compact DFX IDs,
5. a device build mode that sets `MEGAMOE_PROFILE_BACKEND=1`,
6. a host bridge that can produce tiling metadata for case generation.

The current DFX mapping is:

- stage order in `profile_stages.h` defines DFX IDs `0..N-1`,
- start marker: `DfxId(stage) | 0x400`,
- stop marker: `DfxId(stage) | 0xc00`,
- AIC runtime-to-ELF base: `0`,
- AIV runtime-to-ELF base: `0x100000`,
- no rank offset is added,
- the outer `KERNEL` envelope is intentionally omitted in SIM DFX.

Dynamic `DISPATCH_LOCAL` / `DISPATCH_XFER` and
`DISPATCH_DEDUP_LOCAL` / `DISPATCH_DEDUP_XFER` sites must remain explicit
stage calls under `MEGAMOE_PROFILE_BACKEND=1`.

## 1. Build

Build, instrumentation and case generation are local-only.  Use the local
interpreter:

```text
/home/t00826093/miniforge3/envs/tang/bin/python
```

`build_device.py`, `build_host.py` and `case.py` reject the model1 interpreter.
Never synchronize the MegaMoE source tree to model1 for compilation, and never
run CMake, the official compile script, or the host bridge build on model1.

Use a fresh build directory.  The scripts refuse to reuse an existing output.
`build_device.py` performs a fresh configure and compile transaction and writes
`build_evidence.json`.  `build_host.py` links a private host bridge from the
official build tree.

`build_device.py` requires `--config <source-repo>/megamoe_profile/...json`.
It generates `profile_events_generated.h`, `pair_map.json`, `case_config.h`
and `profiler_config.h` into the fresh `--generated` output before CMake
configure; the op_host diagnostic contract rejects a build if this preflight is
skipped.

The same transaction also builds the official `cust_opmaster` host objects that
`build_host.py` later re-links into the private host bridge.

Research kernel variants must pass caller-owned integer macros with repeated
`--kernel-definition NAME=INTEGER` arguments.  `build_device.py` validates the
names, injects them into the private compiler recipe, and records both the
requested values and the exact compiler flags in `build_evidence.json`.  The
build is rejected if the generated recipe does not contain the private options
assignment that can be bound.

If the source tree does not contain `third_party/cann-cmake`, pass its local
checkout explicitly with `--cann-cmake <path>`.  The path must exist before the
build transaction starts and is recorded in the build evidence.

A successful build must leave:

- the device ELF under `<run>/device/bin/`,
- the host shared library under `<run>/host/`,
- the build transaction and evidence JSON files,
- the generated profiler headers.

## 2. Case generation

`case.py` creates one directory per rank under `<run>/run/data/inputs/chip_N`.
It writes the task JSON, kernel images, parameter arrays and input tensors.
It also copies the current `profile_stages.h` into
`<run>/run/build/profile_stages.h` for provenance.

The generated case uses a `Mc2MoeContext`, not the generic HCCL parameter
structure.  The context contains per-peer UBMEM base addresses.  The case
generator checks that the kernel image, parameter arrays, IO tensors and
workspace do not overlap.

## 3. Simulation

Only this stage runs on model1.  After local case generation:

1. upload `<run>/run` to a fresh remote run path,
2. ensure the common simulation repository on model1 is at the recorded Git
   revision,
3. run `simulate.py` with the model1 `tang` interpreter and
   `<remote-run>/config.json`,
4. retrieve the completed run in a compressed transfer,
5. extract it over the matching local `<run>/run` before local export and
   analysis.

`simulate.py` rejects the local build interpreter.  It is the only common
workflow entry that is expected to execute on model1.

The JSON config retains the local CANN build environment.  If model1 uses a
different CANN installation, pass its `set_env.sh` explicitly to `simulate.py`
with `--cann-env`; the actual execution path is recorded in
`results/analysis/capture.json`.

`simulate.py`:

1. copies the simulator model into `<run>/build/toolkit`,
2. overrides `ASCEND_HOME_PATH` to that private copy,
3. runs `npusim record -c ... -o ... -g`,
4. moves `chip*_instr.bin` into `data/raw/capture/*/record/`,
5. searches the archive and camodel logs for `[TASK_DONE]`,
6. writes `results/analysis/capture.json`.

If the Python wrapper was interrupted but `npusim` itself completed, use
`--recover` to validate the existing capture.  Recovery never reruns the
simulator.

## 4. Native export and DFX merge

`report.py` reads the raw SQLite databases, decodes the DFX marks, pairs them,
and writes `results/analysis/intervals_rank*.json`.  It then calls
`export_msprof.py`, which converts the raw instruction records into the native
msprof instruction trace.

`export_msprof.py` uses:

- `msprof op simulator --export ...`,
- `--soc-version=dav_3510`,
- `--aic-metrics=PipeUtilization`,
- all-core export by omitting `--core-id`.

The merged output is `results/report/trace.json`.  It contains:

- the native instruction events,
- MegaMoE DFX stage events,
- process/thread metadata.

## 5. Traffic reconstruction

`traffic.py` reads the raw SQLite instruction records again and builds two
additional trace processes.

### Memory-path lanes

The script classifies instructions that carry `processed_bytes` and
`src_mem`/`dst_mem` into:

- `OUT -> L1`
- `OUT -> UB`
- `UB -> OUT`
- `L0C -> OUT`
- `L0C -> UB`
- `L1 -> L0A`
- `L1 -> L0B`
- `L1 -> UNKNOWN`
- `RV_VLD`
- `RV_VLDI`
- `RV_VST`
- `RV_VSTI`
- `OTHER annotated`

Two summary lanes are also emitted:

- `External total (explicit OUT)`
- `All annotated`

These values are model-level estimates, not hardware counters.

### Point-to-point rank lanes

For the current MTE/UBMEM topology, each rank has a peermem window at:

```text
base = 0x8000000000 + rank * 0x10000000000 + 60 KiB
size = case_identity.peermem_bytes
```

`traffic.py` maps `src_addr` and `dst_addr` into those windows.  If one side is
inside a window and the other side is not, the script infers the missing rank
from the executing chip.  If both sides are inside windows, the source and
destination ranks are taken directly from the addresses.

For world size `P`, the output contains `P*(P-1)` point-to-point lanes.  No
aggregate total lane is generated.

This reconstruction is currently valid for `topoType == TOPO_TYPE_MTE`.
URMA runs need a separate adapter; do not silently reuse the MTE address-window
logic.

## 6. Trace organization

`organize_trace.py` groups all original instruction lanes under one
`Raw Instructions` process.  Each original `coreX.*` pid becomes a thread id
inside that process.  MegaMoE DFX, memory-path traffic and point-to-point rank
traffic remain separate processes.

The script also removes any aggregate rank-total lane.  The final hierarchy is:

```text
MegaMoE rank0
MegaMoE rank1
Rank Traffic
Memory Traffic
Raw Instructions
```

## 7. Packing

`pack_result.py` writes a compressed `reproduce.tar.gz` containing:

- `config.json`
- `status.json`
- `build/profile_stages.h`
- `logs/npusim_record.log`
- `results/analysis/*.json`
- raw SQLite databases
- raw instruction records

It intentionally excludes the uncompressed final trace JSONs.  Those remain as
first-class outputs outside the archive.

## 8. Acceptance

Do not report success until all of these pass:

1. every rank has `[TASK_DONE]`,
2. `npusim` returned 0,
3. raw SQLite and instruction records exist,
4. DFX start/stop marks all close,
5. every DFX id maps to a known stage,
6. `results/report/trace.json` contains both instruction and DFX events,
7. `status.json` is `passed`,
8. if requested, `trace_with_traffic.json` and `trace_organized.json` parse.

## 10. Continuous improvement

During use, any defect or capability gap that belongs to this common simulation
capability must be fixed directly in this repository and in the corresponding
skill.  Record the issue, the affected workflow stage, and the corrected
process before the next run.  Do not leave a discovered gap only in chat history
or in a private run directory.

## 9. Versioning

This repository should be committed after every meaningful change.  A run’s
`case_identity.json` and build evidence must record the exact script and source
commit used.  Do not copy the scripts into a run directory; record the common
repository revision instead.
