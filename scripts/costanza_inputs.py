"""Pinned random input recipe, independent of the third-party runtime/harness.

Matches ae244472 defaults: BF16 uniform X, FP32 uniform gate weights,
per-expert NumPy SeedSequence/PCG64 random weight bytes and unit MX scales.
All generation and golden calculations below are CPU-only.
"""
from functools import lru_cache

RECIPE = 'costanza_random_v1'
REFERENCE = 'costanza_random_cpu_mx_v1'
COMMIT = 'ae2444725b4e7e0b22b0b211fb401e27535111bd'


def enabled(case):
    return case.get('input_recipe') == RECIPE


def contract(case):
    if not enabled(case):
        raise ValueError('Explicit costanza_random_v1 input recipe required')
    for key in ('tokens', 'hidden', 'intermediate', 'experts', 'topk', 'ep'):
        if type(case.get(key)) is not int or case[key] <= 0:
            raise ValueError('Invalid random input dimension: ' + key)
    if case['experts'] % case['ep'] or case['topk'] > case['experts']:
        raise ValueError('Invalid random input expert partition')
    if type(case.get('shared_experts', 1)) is not int or case.get('shared_experts', 1) < 0:
        raise ValueError('Invalid shared expert count')
    if case.get('routing') not in ('costanza_balanced_v1', 'costanza_unbalanced_v1'):
        raise ValueError('Random reference inputs require the reference routing mode')
    if (case.get('topk_weights_dtype') != 'float32' or case.get('topk_weights_prefetch', False)
            or case.get('reference_rng_prefix', 'random_x') != 'random_x'):
        raise ValueError('Reference inputs require FP32 gates, prefetch OFF and random-X RNG order')
    if case.get('dtype') != 'fp8_e4m3fn' or case.get('weight_dtype') not in ('fp8_e4m3fn', 'fp4_e2m1'):
        raise ValueError('Reference inputs support E4M3 activations and E4M3/E2M1 weights')
    if type(case.get('combine_comm_mode', 0)) is not int or case.get('combine_comm_mode', 0) not in (0, 1):
        raise ValueError('Random reference supports Combine modes 0 or 1 only')
    if case.get('topk_weights_type', False) or case.get('comm_quant_mode', 0) != 0:
        raise ValueError('Reference requires Combine-stage gates and BF16 communication')
    if case['hidden'] % 64 or case['intermediate'] % 128:
        raise ValueError('Reference inputs require H64 and I128 alignment')
    seed = case.get('seed', 831)
    if type(seed) is not int or not 0 <= seed <= 0xffffffff:
        raise ValueError('Reference seed must be in 0..2^32-1')


@lru_cache(maxsize=1)
def _tokens(tokens, hidden, experts, topk, world, seed, routing):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.empty((tokens * world, hidden), dtype=torch.bfloat16)
    x.uniform_(-0.1, 0.1, generator=generator)
    if routing == 'costanza_unbalanced_v1':
        from costanza_unbalanced_routing import sample_global_ids
        ids = sample_global_ids(tokens, experts, topk, world, generator).reshape(tokens * world, topk).long()
    else:
        from costanza_balanced_routing import make_routing
        case = dict(tokens=tokens, experts=experts, topk=topk, ep=world)
        ids = torch.cat([make_routing(case, r) for r in range(world)])
    # Default active masks are all ones and consume no RNG draws.
    gates = torch.empty((tokens * world, topk), dtype=torch.float32)
    gates.uniform_(-0.1, 0.1, generator=generator)
    return x, ids, gates


def rank_inputs(case, rank):
    contract(case)
    if type(rank) is not int or not 0 <= rank < case['ep']:
        raise ValueError('Invalid reference input rank')
    arrays = _tokens(case['tokens'], case['hidden'], case['experts'], case['topk'],
                     case['ep'], case.get('seed', 831), case['routing'])
    start = rank * case['tokens']
    return tuple(array[start:start + case['tokens']].clone() for array in arrays)


def expert_codes(case, slot, index):
    """Global weight slot includes each rank's shared experts BEFORE routed ones."""
    import numpy as np
    contract(case)
    slots = case['experts'] + case['ep'] * case.get('shared_experts', 1)
    if type(slot) is not int or not 0 <= slot < slots or type(index) is not int or index not in (1, 2):
        raise ValueError('Invalid expert weight identity')
    seed = case.get('seed', 831)
    rng = np.random.default_rng(np.random.SeedSequence([
        seed & 0xffffffff, (seed >> 32) & 0xffffffff,
        slot & 0xffffffff, (slot >> 32) & 0xffffffff, index]))
    h, n = case['hidden'], case['intermediate']
    shape = (1, 2 * n, h) if index == 1 else (1, h, n)
    codes = rng.integers(0, 256, size=shape, dtype=np.uint8)
    codes &= 0x0b if case['weight_dtype'] == 'fp4_e2m1' else 0xbf
    return codes[0]


def slot_for(case, rank, local_expert, shared=False):
    contract(case)
    count = case['experts'] // case['ep']
    shared_count = case.get('shared_experts', 1)
    limit = shared_count if shared else count
    if (type(rank) is not int or not 0 <= rank < case['ep']
            or type(local_expert) is not int or not 0 <= local_expert < limit):
        raise ValueError('Invalid local expert identity')
    return rank * (count + shared_count) + (0 if shared else shared_count) + local_expert


def pack_fp4_nz(codes):
    """Logical [N,K] low-nibble values to A8W4 FRACTAL_NZ_C0_32 bytes."""
    import numpy as np
    import torch
    n, k = codes.shape
    if k % 32:
        raise ValueError('FP4 NZ K must be divisible by 32')
    nz = np.ascontiguousarray(codes.reshape(n, k // 32, 32).transpose(1, 0, 2))
    return torch.from_numpy((nz[..., 0::2] | (nz[..., 1::2] << 4)).copy())


def rank_weights(case, rank, shared=False):
    import torch
    contract(case)
    count = case.get('shared_experts', 1) if shared else case['experts'] // case['ep']
    matrices = [[], []]
    for expert in range(count):
        slot = slot_for(case, rank, expert, shared)
        for index in (1, 2):
            codes = expert_codes(case, slot, index)
            value = (pack_fp4_nz(codes) if case['weight_dtype'] == 'fp4_e2m1'
                     else torch.from_numpy(codes).view(torch.float8_e4m3fn))
            matrices[index - 1].append(value)
    if count == 0:
        raise ValueError('Do not request weights for an empty partition')
    return tuple(torch.stack(values) for values in matrices)


def logical_weights(case, slot):
    import torch
    lookup = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    result = []
    for index in (1, 2):
        raw = torch.from_numpy(expert_codes(case, slot, index))
        result.append(lookup[raw.long()] if case['weight_dtype'] == 'fp4_e2m1'
                      else raw.view(torch.float8_e4m3fn).float())
    return tuple(result)


def quantize_mx(values):
    """Reference E4M3 block-32 quantize/dequantize, after the caller's BF16 rounding."""
    import torch
    if values.shape[-1] % 32 or not bool(torch.isfinite(values).all()):
        raise ValueError('MX reference requires finite aligned rows')
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = (torch.floor(torch.log2(maximum)) - 8).clamp(min=-127)
    scale = torch.pow(2., exponent)
    normalized = (blocks / scale).clamp(-448, 448)
    return (normalized.to(torch.float8_e4m3fn).float() * scale).reshape(values.shape)


def _ffn(rows, first, second):
    import torch
    gmm1 = (rows @ first.T).to(torch.bfloat16).float()
    gate, up = gmm1.chunk(2, dim=-1)
    activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
    return (quantize_mx(activated) @ second.T).to(torch.bfloat16)


def combine_rows(members, ids, gates, shared_rows, combine_dedup):
    """Preserve top-k order, per-owner expert ordering, and BF16 wire rounding."""
    import torch
    b, k, h = members.shape
    result = torch.zeros((b, h), dtype=torch.float32)
    if not combine_dedup:
        for position in range(k):
            result.add_(members[:, position].float() * gates[:, position, None])
    else:
        # Caller passes owner IDs separately via an additional final axis.
        experts, owners = ids[..., 0], ids[..., 1]
        for position in range(k):
            canonical = (owners[:, :position] != owners[:, position, None]).all(dim=1)
            tokens = torch.where(canonical)[0]
            if not len(tokens):
                continue
            same = owners[tokens] == owners[tokens, position, None]
            count = same.sum(dim=1)
            order = torch.where(same, experts[tokens], torch.iinfo(torch.int64).max).argsort(dim=1, stable=True)
            partial = members[tokens, order[:, 0]].float() * gates[tokens, order[:, 0], None]
            for offset in range(1, int(count.max())):
                active = torch.where(count > offset)[0]
                token = tokens[active]; slot = order[active, offset]
                partial[active] += members[token, slot].float() * gates[token, slot, None]
            merged = count > 1
            partial[merged] = partial[merged].to(torch.bfloat16).float()
            result[tokens] += partial
    for rows in shared_rows:
        result.add_(rows.float())
    return result.to(torch.bfloat16).float()


def golden_rank(case, rank, x=None, ids=None, gates=None):
    """Dense CPU reference, bounded to one expert's weights and 128-row matmuls."""
    import torch
    contract(case)
    if x is None and ids is None and gates is None:
        x, ids, gates = rank_inputs(case, rank)
    elif x is None or ids is None or gates is None:
        raise ValueError('Reference X, IDs and gates must be supplied together')
    if type(rank) is not int or not 0 <= rank < case['ep']:
        raise ValueError('Invalid reference input rank')
    if (x.shape != (case['tokens'], case['hidden'])
            or ids.shape != (case['tokens'], case['topk'])
            or gates.shape != ids.shape or x.dtype != torch.bfloat16
            or gates.dtype != torch.float32
            or ids.dtype not in (torch.int32, torch.int64)
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(gates).all())
            or bool(((ids < 0) | (ids >= case['experts'])).any())):
        raise ValueError('Invalid random reference input tensors')
    quantized = quantize_mx(x.float())
    b, k, h = case['tokens'], case['topk'], case['hidden']
    members = torch.empty((b, k, h), dtype=torch.bfloat16)
    local = case['experts'] // case['ep']
    for expert in ids.unique().tolist():
        token, position = torch.where(ids == expert)
        owner, index = divmod(expert, local)
        weights = logical_weights(case, slot_for(case, owner, index))
        for start in range(0, len(token), 128):
            rows = token[start:start + 128]
            members[rows, position[start:start + 128]] = _ffn(quantized[rows], *weights)
    shared_rows = []
    for expert in range(case.get('shared_experts', 1)):
        weights = logical_weights(case, slot_for(case, rank, expert, shared=True))
        output = torch.empty((b, h), dtype=torch.bfloat16)
        for start in range(0, b, 128):
            output[start:start + 128] = _ffn(quantized[start:start + 128], *weights)
        shared_rows.append(output)
    owner_ids = torch.stack((ids, ids // local), dim=-1)
    return combine_rows(members, owner_ids, gates, shared_rows, bool(case.get('combine_comm_mode', 0)))


def provenance(case):
    import numpy as np
    import torch
    contract(case)
    return dict(input_recipe=RECIPE, reference=REFERENCE, reference_commit=COMMIT,
                seed=case.get('seed', 831), x_dtype='bfloat16', x_distribution='uniform(-0.1,0.1)',
                gate_dtype='float32', gate_distribution='uniform(-0.1,0.1)',
                weight_generation='per-expert SeedSequence PCG64 uint8 mask',
                weight_storage_mask=11 if case['weight_dtype'] == 'fp4_e2m1' else 191,
                mx_scale_byte=127, fast_random_weight=True, transpose_weight=1,
                is_interleaved=False, shared_slot_order='shared before routed, for each rank',
                torch_version=torch.__version__, numpy_version=np.__version__)


def _golden_cache_identity(case, rank, generator_identity):
    """Small mathematical identity; caller supplies the already frozen code ID."""
    import json
    import numpy as np
    import torch
    contract(case)
    if type(rank) is not int or not 0 <= rank < case['ep']:
        raise ValueError('Invalid reference input rank')
    if not isinstance(generator_identity, (str, dict)) or not generator_identity:
        raise ValueError('A nonempty frozen generator identity is required')
    try:
        # Normalize JSON tuples, etc., so written and freshly computed identities
        # compare identically. This never reads or fingerprints source/data files.
        generator = json.loads(json.dumps(generator_identity, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Generator identity must be finite JSON metadata') from error
    mathematical = {key: case[key] for key in
        ('tokens', 'hidden', 'intermediate', 'experts', 'topk', 'ep',
         'routing', 'dtype', 'weight_dtype', 'topk_weights_dtype')}
    mathematical.update(shared_experts=case.get('shared_experts', 1),
        seed=case.get('seed', 831), reference_rng_prefix=case.get('reference_rng_prefix', 'random_x'),
        combine_comm_mode=case.get('combine_comm_mode', 0),
        topk_weights_prefetch=case.get('topk_weights_prefetch', False),
        topk_weights_type=case.get('topk_weights_type', False),
        comm_quant_mode=case.get('comm_quant_mode', 0), input_recipe=RECIPE, rank=rank)
    # Dispatch dedup changes no math. In particular force_dedup_mode=1 and =3
    # intentionally share this identity when Combine mode and input recipe match.
    return dict(schema='costanza_golden_cache_v1', reference=REFERENCE,
        generator_identity=generator, mathematical=mathematical,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        storage=dict(dtype='bfloat16', byteorder='little',
                     shape=[case['tokens'], case['hidden']],
                     bytes=case['tokens'] * case['hidden'] * 2))


def golden_rank_cached(case, rank, cache_root, generator_identity, *, return_info=False, on_phase=None):
    """Temporary runtime cache, atomically publishing BF16 data plus identity.

    No source/data hash, readback, fsync, WAL or per-file logs. The small identity
    JSON alone is digested for an address. Existing malformed entries are errors,
    never cache misses, and cannot silently substitute a different reference.
    """
    from time import perf_counter
    started = perf_counter()
    info = dict(enabled=True, hit=False, generation_seconds=0.0, load_seconds=0.0, write_seconds=0.0)
    def finish(value):
        info['total_seconds'] = perf_counter() - started
        return (value, info) if return_info else value
    import errno
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import tempfile
    import numpy as np
    import torch

    identity = _golden_cache_identity(case, rank, generator_identity)
    serialized = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    address = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    base = Path(cache_root)
    base.mkdir(parents=True, exist_ok=True)
    entry = base / address
    info.update(root=str(base), entry=str(entry), raw_bytes=identity['storage']['bytes'])

    def read():
        if on_phase is not None:
            on_phase('golden_cache_load')
        load_started = perf_counter()
        try:
            recorded = json.loads((entry / 'identity.json').read_text())
            if recorded != identity:
                raise ValueError('Golden cache identity mismatch: ' + str(entry))
            raw = entry / 'expected.bf16'
            expected_bytes = identity['storage']['bytes']
            if raw.stat().st_size != expected_bytes:
                raise ValueError('Golden cache byte size mismatch: ' + str(entry))
            codes = np.fromfile(raw, dtype='<u2')
            if codes.size * 2 != expected_bytes:
                raise ValueError('Golden cache changed while reading: ' + str(entry))
            expected = torch.from_numpy(codes.astype(np.uint16, copy=False)).view(torch.bfloat16)
            expected = expected.reshape(identity['storage']['shape']).float()
            if not bool(torch.isfinite(expected).all()):
                raise ValueError('Golden cache contains nonfinite values: ' + str(entry))
            info['load_seconds'] += perf_counter() - load_started
            return expected
        except (OSError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError('Invalid golden cache entry: ' + str(entry)) from error

    if entry.exists():
        info['hit'] = True
        return finish(read())
    if on_phase is not None:
        on_phase('golden_cache_generate')
    generation_started = perf_counter()
    expected = golden_rank(case, rank)
    info['generation_seconds'] = perf_counter() - generation_started
    wanted_shape = tuple(identity['storage']['shape'])
    if tuple(expected.shape) != wanted_shape or not bool(torch.isfinite(expected).all()):
        raise ValueError('Generated golden has invalid shape or nonfinite values')
    if on_phase is not None:
        on_phase('golden_cache_write')
    write_started = perf_counter()
    packed = expected.to(torch.bfloat16).contiguous()
    temporary = Path(tempfile.mkdtemp(prefix='.golden-', dir=base))
    try:
        packed.view(torch.uint16).numpy().astype('<u2', copy=False).tofile(temporary / 'expected.bf16')
        (temporary / 'identity.json').write_text(serialized + '\n')
        try:
            os.rename(temporary, entry)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another process may have atomically published the same identity.
            # Validate and reuse it rather than replacing either writer's files.
            info['concurrent_publication'] = True
            info['write_seconds'] = perf_counter() - write_started
            return finish(read())
        value = packed.float()
        info['write_seconds'] = perf_counter() - write_started
        return finish(value)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
