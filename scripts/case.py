#!/usr/bin/env python3
"""Build a current-repo MegaMoE npusim case from formal host tiling."""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from config_io import load_config
from execution_contract import require_local_build
from routing import make_routing
from structured_weights import make_weights

CHIP_MEM_SIZE = 0x10000000000
DATA_REGION_OFFSET = 0x8000000000
KERNEL_BIN_OFFSET = 0x40000000
KERNEL_SUB_BIN_OFFSET = 0x40100000
HBM_PARA_BASE_OFFSET = 0x8100000000
STACK_PHY_BASE_OFFSET = 0x8200000000
SQ_BASE_OFFSET = 0x9010000000
CQ_BASE_OFFSET = 0x9018000000
SQ_NS_SWAP_BUFFER_OFFSET = 0x9028000000
IO_BASE_OFFSET = 0x9001000000
ALIGN = 0x10000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git_identity(path):
    def run(*arguments):
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True, text=True, capture_output=True,
        )
        return result.stdout.strip()

    return {
        "path": str(Path(path).resolve()),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def tensor_bytes(tensor):
    import torch
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def make_context(rank, world):
    # Mc2MoeContext: uint32 rank, uint32 rankSize, uint64 kfc, then two
    # 1024-entry address arrays. MegaMoE does not use the generic HcclOpParam.
    data = bytearray(16 + 2 * 1024 * 8)
    struct.pack_into("<IIQ", data, 0, rank, world, 0)
    for peer in range(world):
        struct.pack_into("<Q", data, 16 + 8 * peer, DATA_REGION_OFFSET + peer * CHIP_MEM_SIZE)
    return bytes(data)


def elf_mixed_images(path, tiling_key):
    """Extract suffix images for one formal tiling key from the ELF .o."""
    data = Path(path).read_bytes()
    if data[:5] != b"\x7fELF\x02":
        raise ValueError("Only little-endian 64-bit ELF is supported")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    entsize, count, names_idx = struct.unpack_from("<HHH", data, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(count)]
    shstrtab = sections[names_idx]

    def section_name(section):
        start = shstrtab[4] + section[0]
        return data[start:data.find(b"\0", start)].decode(errors="ignore")

    wanted = {".text", ".data"}
    named = {section_name(section): section for section in sections if section_name(section) in wanted}
    if set(named) != wanted:
        raise ValueError(f"ELF image sections not found: {sorted(set(wanted) - set(named))}")
    image_size = max(section[3] + section[5] for section in named.values())
    image = bytearray(image_size)
    for section in named.values():
        image[section[3]:section[3] + section[5]] = data[section[4]:section[4] + section[5]]
    symtab = next((s for s in sections if s[1] == 2 and s[6] != 0), None)
    if symtab is None:
        raise ValueError("ELF symbol table not found")
    strtab = sections[symtab[6]]
    strings = data[strtab[4]:strtab[4] + strtab[5]]
    images = {}
    for off in range(symtab[4], symtab[4] + symtab[5], symtab[9]):
        name_off, _, _, _, value, size = struct.unpack_from("<IBBHQQ", data, off)
        name = strings[name_off:].split(b"\0", 1)[0].decode(errors="ignore")
        for kind in ("aic", "aiv"):
            suffix = f"_{tiling_key}_mix_{kind}"
            if name.endswith(suffix):
                if kind in images:
                    raise ValueError(f"Duplicate {kind} image for tiling key {tiling_key}")
                images[kind] = bytes(image[value:])
    if set(images) != {"aic", "aiv"}:
        raise ValueError(f"Missing MIX image for tiling key {tiling_key}: {sorted(images)}")
    return images


def configure_host(host, case):
    host.megamoe_session_configure.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int32]
    host.megamoe_session_configure.restype = ctypes.c_int32
    for name in ("megamoe_tiling_size", "megamoe_session_tiling_key", "megamoe_peermem_size"):
        getattr(host, name).restype = ctypes.c_uint64
    host.megamoe_session_block_dim.restype = ctypes.c_uint32
    host.megamoe_make_tiling.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    host.megamoe_make_tiling.restype = ctypes.c_uint64
    host.megamoe_profile_dedup_mode.argtypes = [ctypes.c_void_p]
    host.megamoe_profile_dedup_mode.restype = ctypes.c_int32
    host.megamoe_profile_wave_groups.argtypes = [ctypes.c_void_p]
    host.megamoe_profile_wave_groups.restype = ctypes.c_uint32
    dims = [case[k] for k in ("tokens", "hidden", "intermediate", "experts", "topk", "ep", "shared_experts")]
    code = host.megamoe_session_configure((ctypes.c_int64 * 7)(*dims), case["force_dedup_mode"])
    if code:
        raise RuntimeError(f"Formal host tiling configuration failed: {code}")


def make_tensor_list(cursor, payload):
    """Return descriptor+payload and advance the 64 KiB-aligned cursor."""
    addr = cursor
    size = 512 + len(payload)
    data = struct.pack("<QQ", 8, addr + 512) + bytes(496) + payload
    return addr, data, align_up(cursor + size, ALIGN)


def validate_task_regions(task, images):
    """Reject overlapping parameter/IO and kernel-image HBM ranges."""
    ranges = []
    for section in ("input_para_array", "output_para_array"):
        for item in task[section]:
            size = int(item["size"], 0)
            if size:
                ranges.append((int(item["addr"], 0), int(item["addr"], 0) + size,
                               item["name"], section))
    for name, key in (("aic.bin", "addr"), ("aiv.bin", "sub_addr")):
        addr = int(task["BIN"][key], 0)
        image_key = "aic" if name == "aic.bin" else "aiv"
        ranges.append((addr, addr + len(images[image_key]), name, "kernel_image"))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        if left[1] > right[0]:
            raise ValueError(f"Overlapping HBM regions: {left} vs {right}")


def generate(config, run, host_path, kernel_path, stage_header):
    import torch
    run = Path(run).resolve()
    host_path = Path(host_path).resolve()
    kernel_path = Path(kernel_path).resolve()
    stage_header = Path(stage_header).resolve()
    case = config["case"]
    world = case["ep"]
    local = case["experts"] // world
    hidden, intermediate, shared = case["hidden"], case["intermediate"], case["shared_experts"]
    dtype = getattr(torch, case["dtype"].replace("fp8", "float8"))

    host = ctypes.CDLL(str(host_path))
    configure_host(host, case)
    tiling = ctypes.create_string_buffer(host.megamoe_tiling_size())
    workspace_bytes = host.megamoe_make_tiling(tiling, 0)
    tiling_key = host.megamoe_session_tiling_key()
    block_dim = host.megamoe_session_block_dim()
    actual_mode = host.megamoe_profile_dedup_mode(tiling)
    if workspace_bytes <= 0 or actual_mode != case["force_dedup_mode"]:
        raise RuntimeError("Formal tiling validation failed")
    images = elf_mixed_images(kernel_path, tiling_key)

    inputs = run / "data/inputs"
    outputs = run / "data/outputs"
    raw = run / "data/raw"
    report = run / "results/report"
    analysis = run / "results/analysis"
    logs = run / "logs"
    build = run / "build"
    for path in (inputs, outputs, raw, report, analysis, logs, build):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage_header, run / "build/profile_stages.h")
    write_json(run / "config.json", config)

    top = {}
    tensors = {}
    for rank in range(world):
        chip = inputs / f"chip_{rank}"
        chip.mkdir(parents=True, exist_ok=True)
        base = rank * CHIP_MEM_SIZE
        cursor = base + IO_BASE_OFFSET
        task = {
            "sqe_type": "AIC",
            "mix": 1,
            "ratio": 2,
            "block_dim": block_dim,
            "hbm_para_addr": hex(base + HBM_PARA_BASE_OFFSET),
            "sub_hbm_para_addr": hex(base + HBM_PARA_BASE_OFFSET),
            "stack_phy_base": hex(base + STACK_PHY_BASE_OFFSET),
            "latency": 1,
            "schem": 1,
            "task_type": 6,
            "group_dim": 1,
            "group_slice": 32,
            "aic_icache_prefetch_cnt": 0,
            "aiv_icache_prefetch_cnt": 0,
            "aic_dcache_prefetch_cnt": 64,
            "aiv_dcache_prefetch_cnt": 64,
            "input_para_array": [],
            "output_para_array": [],
            "ast_output_array": [],
        }

        def add(slot, name, payload, output=False):
            nonlocal cursor
            size = len(payload)
            addr = cursor
            cursor = align_up(cursor + max(size, 1), ALIGN)
            item = {
                "name": name,
                "size": hex(size),
                "addr": hex(addr),
                "para_offset": slot,
                "hbm_para_offset_en": 1,
                "para_type": 0,
            }
            target = task["output_para_array" if output else "input_para_array"]
            target.append(item)
            (chip / name).write_bytes(payload)
            if output:
                task["ast_output_array"].append({"name": name, "addr": hex(addr), "size": hex(size)})
            return addr

        def placeholder(slot):
            task["input_para_array"].append({
                "size": "0x0", "addr": "0x0", "para_offset": slot,
                "hbm_para_offset_en": 1, "para_type": 0,
            })

        def tensor_list(slot, name, payload):
            nonlocal cursor
            addr, blob, cursor = make_tensor_list(cursor, payload)
            item = {
                "name": name, "size": hex(len(blob)), "addr": hex(addr), "para_offset": slot,
                "hbm_para_offset_en": 1, "para_type": 0,
            }
            task["input_para_array"].append(item)
            (chip / name).write_bytes(blob)

        add(0, "context.bin", make_context(rank, world))
        x = torch.ones((case["tokens"], hidden), dtype=torch.bfloat16)
        ids = make_routing(case, rank).to(torch.int32)
        gates = torch.full((case["tokens"], case["topk"]), 1 / case["topk"],
                           dtype=getattr(torch, case.get("topk_weights_dtype", "bfloat16")))
        add(1, "x.bin", tensor_bytes(x))
        add(2, "topk_ids.bin", tensor_bytes(ids))
        add(3, "topk_weights.bin", tensor_bytes(gates))

        w1, w2 = make_weights(local, hidden, intermediate, dtype, rank * local, case.get("weight_dtype"))
        s1 = torch.full((local, 2 * intermediate, (hidden + 63) // 64, 2), 127, dtype=torch.uint8)
        s2 = torch.full((local, hidden, (intermediate + 63) // 64, 2), 127, dtype=torch.uint8)
        for slot, name, tensor in ((4, "weight1.bin", w1), (5, "weight2.bin", w2),
                                   (6, "weight_scales1.bin", s1), (7, "weight_scales2.bin", s2)):
            tensor_list(slot, name, tensor_bytes(tensor))

        for slot in (8, 9, 10, 11):
            placeholder(slot)
        if shared:
            sw1, sw2 = make_weights(shared, hidden, intermediate, dtype, rank * local, case.get("weight_dtype"))
            ss1 = torch.full((shared, 2 * intermediate, (hidden + 63) // 64, 2), 127, dtype=torch.uint8)
            ss2 = torch.full((shared, hidden, (intermediate + 63) // 64, 2), 127, dtype=torch.uint8)
            for slot, name, tensor in ((12, "shared_weight1.bin", sw1), (13, "shared_weight2.bin", sw2),
                                       (14, "shared_weight_scales1.bin", ss1), (15, "shared_weight_scales2.bin", ss2)):
                tensor_list(slot, name, tensor_bytes(tensor))
        else:
            for slot in range(12, 16):
                placeholder(slot)
        for slot in (16, 17, 18):
            placeholder(slot)

        y = torch.zeros((case["tokens"], hidden), dtype=torch.bfloat16)
        counts = torch.zeros(local, dtype=torch.int32)
        add(19, "y.bin", tensor_bytes(y), output=True)
        add(20, "expert_token_nums.bin", tensor_bytes(counts), output=True)
        add(21, "workspace.bin", bytes(workspace_bytes))
        add(22, "tiling.bin", bytes(tiling))
        (chip / "aic.bin").write_bytes(images["aic"])
        (chip / "aiv.bin").write_bytes(images["aiv"])
        task["BIN"] = {
            "name": "aic.bin", "addr": hex(base + KERNEL_BIN_OFFSET),
            "sub_name": "aiv.bin", "sub_addr": hex(base + KERNEL_SUB_BIN_OFFSET),
        }
        validate_task_regions(task, images)
        write_json(chip / "mega_task.json", task)
        write_json(chip / "mega_sq.json", {"name": "mega_sq", "tasks": [{"sqe_type": "AIC", "name": "mega"}]})
        write_json(chip / "sq_top.json", {
            "name": "sq_top",
            "sq_base_addr": hex(base + SQ_BASE_OFFSET),
            "cq_base_addr": hex(base + CQ_BASE_OFFSET),
            "sq_ns_swap_buffer_addr": hex(base + SQ_NS_SWAP_BUFFER_OFFSET),
            "sq": [{"name": "mega", "id": 0, "stack_phy_base": hex(base + STACK_PHY_BASE_OFFSET)}],
        })
        top[f"chip{rank}"] = {"die0": {"case_path": str(chip) + "/", "sq_file": "sq_top.json"}}
        tensors[rank] = {
            "x": list(x.shape), "topk_ids": list(ids.shape), "topk_weights": list(gates.shape),
            "weight1": [local, 2 * intermediate, hidden], "weight2": [local, hidden, intermediate],
            "weight_scales1": list(s1.shape), "weight_scales2": list(s2.shape),
            "shared_weight1": [shared, 2 * intermediate, hidden] if shared else [],
            "shared_weight2": [shared, hidden, intermediate] if shared else [],
            "y": list(y.shape), "expert_token_nums": list(counts.shape),
        }
        (raw / f"tiling_rank{rank}.bin").write_bytes(bytes(tiling))

    write_json(inputs / "top.json", top)
    write_json(run / "results/report/tensors.json", tensors)
    write_json(run / "results/analysis/case_identity.json", {
        "schema": "megamoe_npusim_case_v1",
        "config": str(run / "config.json"),
        "host_library": str(host_path), "host_sha256": digest(host_path),
        "kernel_object": str(kernel_path), "kernel_sha256": digest(kernel_path),
        "stage_header": str(stage_header), "stage_header_sha256": digest(stage_header),
        "tiling_key": tiling_key, "block_dim": block_dim,
        "workspace_bytes": workspace_bytes, "peermem_bytes": host.megamoe_peermem_size(),
        "actual_dedup_mode": actual_mode, "world": world,
        "source_repository": git_identity(stage_header.parents[2]),
        "common_repository": git_identity(SCRIPTS.parent),
    })
    return {
        "case_dir": str(inputs), "tiling_key": tiling_key, "block_dim": block_dim,
        "workspace_bytes": workspace_bytes, "world": world,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--stage-header", type=Path, required=True)
    args = parser.parse_args()
    require_local_build('case_generation')
    result = generate(load_config(args.config), args.run, args.host, args.kernel, args.stage_header)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
