"""Construct the sparse structured fixture without full-size FP32 weight buffers."""
import torch


def make_weights(experts, hidden, intermediate, dtype, expert_offset=0, weight_dtype=None):
    if weight_dtype == "fp4_e2m1":
        return make_fp4_weights(experts, hidden, intermediate, expert_offset)
    # Both supported FP8 formats represent 0, 1 and 2 exactly. Build their bytes
    # directly, retaining the original fixture while reducing host peak memory.
    one, two = torch.tensor([1., 2.]).to(dtype).view(torch.uint8).tolist()
    first = torch.zeros((experts, 2 * intermediate, hidden), dtype=torch.uint8)
    second = torch.zeros((experts, hidden, intermediate), dtype=torch.uint8)
    idx = torch.arange(intermediate)
    for expert in range(experts):
        first[expert, idx, idx % hidden] = one
        first[expert, idx + intermediate, idx % hidden] = one
        second[expert, idx % hidden, idx] = two if (expert_offset + expert) % 2 else one
    return first.view(dtype), second.view(dtype)


def make_fp4_weights(experts, hidden, intermediate, expert_offset=0):
    """Same sparse logical values as FP8, packed into A8W4 NZ C0=32 on CPU.

    For logical W[n,k], element offset = (k//32)*N*32 + n*32 + k%32.
    Even logical offsets occupy low nibble; E2M1 represents 1 as 2, 2 as 4.
    This matches frame_layout_format<zn_layout_ptn, Int<32>>(K,N).
    Kernel prologue expands FP4 to FP8/64; Fixpipe multiplies by 64, so the
    existing E8M0 scale byte 127 (unit scale) remains correct.
    """
    if hidden % 64 or intermediate % 128:
        raise ValueError('A8W4 fixture requires H64 and I128 alignment')
    def sparse_nz(ncols, kcols, rows, cols, value):
        packed = torch.zeros((experts, kcols // 32, ncols, 16), dtype=torch.uint8)
        offsets = (cols // 32) * (ncols * 32) + rows * 32 + cols % 32
        byte_offsets = offsets // 2
        shift = (offsets % 2) * 4
        for expert in range(experts):
            nibble = value(expert)
            flat = packed[expert].view(-1)
            # Fixture nonzeros have distinct packed bytes for these shapes.
            if torch.unique(byte_offsets).numel() != byte_offsets.numel():
                raise ValueError('Fixture packed-byte collision')
            flat[byte_offsets] |= (nibble << shift).to(torch.uint8)
        return packed
    idx = torch.arange(intermediate)
    first = sparse_nz(2*intermediate, hidden, torch.cat((idx, idx+intermediate)),
                      torch.cat((idx % hidden, idx % hidden)), lambda expert: 2)
    second = sparse_nz(hidden, intermediate, idx % hidden, idx,
                       lambda expert: 4 if (expert_offset+expert) % 2 else 2)
    return first, second
