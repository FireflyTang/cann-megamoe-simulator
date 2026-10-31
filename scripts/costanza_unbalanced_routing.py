"""CPU hotspot routing matching reference ae244472; never runs external code."""
from functools import lru_cache
import math

MODE = 'costanza_unbalanced_v1'
REFERENCE_COMMIT = 'ae2444725b4e7e0b22b0b211fb401e27535111bd'


def contract(case):
    for key in ('tokens', 'experts', 'topk', 'ep', 'hidden'):
        if type(case.get(key)) is not int or case[key] <= 0:
            raise ValueError('Hotspot routing requires positive integer ' + key)
    if case['experts'] % case['ep'] or case['topk'] > case['experts']:
        raise ValueError('Invalid hotspot expert partition or top-k')
    if case.get('global_expert_balance', 'exact') != 'exact' or case.get('remote_topk') is not None:
        raise ValueError('Hotspot routing cannot force expert/card-pair balance')
    seed = case.get('seed', 831)
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError('Reference random_seed must be in [0, 2**32-1]')
    prefix = case.get('reference_rng_prefix', 'random_x')
    if prefix not in ('random_x', 'none'):
        raise ValueError('reference_rng_prefix must be random_x or none')
    return tuple(case[key] for key in ('tokens', 'experts', 'topk', 'ep', 'hidden')) + (seed, prefix)


@lru_cache(maxsize=2)
def _global_ids(tokens, experts, topk, world, hidden, seed, prefix):
    import torch
    rng = torch.Generator(device='cpu').manual_seed(seed)
    if prefix == 'random_x':
        # The reference calls gen_x before routing. Preserve its CPU RNG state
        # while leaving this harness's structured all-ones input unchanged.
        temporary = torch.empty((tokens * world, hidden), dtype=torch.bfloat16)
        temporary.uniform_(-0.1, 0.1, generator=rng)
        del temporary
    return sample_global_ids(tokens, experts, topk, world, rng)


def sample_global_ids(tokens, experts, topk, world, rng):
    """Continue a caller-owned RNG after generating the actual reference X."""
    import torch
    hot = min(8, experts)
    warm = min(16, max(experts - hot, 0))
    scale = experts / max(topk, 1)
    weights = torch.ones(experts, dtype=torch.float32)
    weights[0] = 20.0 * scale
    if hot > 1:
        weights[1:hot] = torch.logspace(math.log10(max(1.25 * scale, 1.0)),
            math.log10(max(0.42 * scale, 1.0)), hot - 1)
    weights[hot:hot + warm] = max(0.21 * scale, 1.0)
    global_cold = min(round(experts * 0.005), max(experts - hot - warm, 0), max(experts - topk, 0))
    if global_cold:
        weights[-global_cold:] = 0
    ordinary = torch.arange(hot + warm, experts - global_cold)
    rank_cold = min(round(experts * 0.015), ordinary.numel(), max(experts - topk - global_cold, 0))
    permutation = torch.randperm(experts, generator=rng)
    ranks = []
    for _ in range(world):
        rank_weights = weights.clone()
        if rank_cold:
            cold = ordinary[torch.randperm(ordinary.numel(), generator=rng)[:rank_cold]]
            rank_weights[cold] = 0
        rank_weights = rank_weights[permutation]
        chunks = []
        for start in range(0, tokens, 4096):
            size = min(4096, tokens - start)
            chunks.append(torch.multinomial(rank_weights.expand(size, -1), topk,
                replacement=False, generator=rng).to(torch.int32))
        ranks.append(torch.cat(chunks, dim=0))
    return torch.stack(ranks)


def make_routing(case, rank):
    import torch
    key = contract(case)
    if type(rank) is not int or not 0 <= rank < case['ep']:
        raise ValueError('Rank outside hotspot routing world')
    return _global_ids(*key)[rank].to(dtype=torch.int64).clone()


def provenance(case):
    key = contract(case)
    return dict(construction_version=MODE, reference_commit=REFERENCE_COMMIT,
        seed=key[-2], reference_rng_prefix=key[-1], reference_x_dtype='bfloat16',
        global_cold_expert_ratio=0.005, rank_cold_expert_ratio=0.015,
        sample_chunk_size=4096, sampling='weighted_without_replacement',
        balance_verified=False, global_generation=True,
        bitwise_scope='Same PyTorch version, seed, BF16 random-X prefix and rank/token shapes')
