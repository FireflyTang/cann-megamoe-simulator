"""CPU-only, reproducible token routing; shared experts are outside this ID space."""


def short_balanced(case):
    """Version 2 short period: E256 / EP4 / topK6 / BS64."""
    return (case["tokens"], case["experts"], case["topk"], case["ep"]) == (64, 256, 6, 4)


def short_period_balanced(case):
    return case['tokens'] % case['experts'] != 0


def validate_balanced(case):
    """Require exact global expert and card-pair balance; source counts differ by at most one."""
    if case.get('routing') != 'balanced':
        return
    for key in ('tokens', 'experts', 'topk', 'ep'):
        if type(case[key]) is not int or case[key] <= 0:
            raise ValueError(f'balanced routing requires positive integer {key}')
    if case['experts'] % case['ep']:
        raise ValueError('balanced routing requires experts % ep == 0')
    policy = case.get('global_expert_balance', 'exact')
    if policy not in ('exact', 'floorceil'):
        raise ValueError('global_expert_balance must be exact or floorceil')
    # A full token rotation balances card pairs even when K itself is not
    # divisible by EP (for example BS4/K6/EP4). Keep the established integer
    # construction; only globally indivisible expert loads use floor/ceil.
    if policy == 'floorceil' and case['tokens'] * case['topk'] % case['ep']:
        raise ValueError('floorceil requires tokens * topk divisible by ep')
    if short_period_balanced(case) and (
            (case['tokens'] % case['ep'] and case['topk'] % case['ep']) or
            (policy == 'exact' and case['tokens'] * case['topk'] * case['ep'] % case['experts'])):
        raise ValueError('Short balanced period cannot evenly cover card pairs and global experts')
    if not 1 <= case['topk'] <= case['experts']:
        raise ValueError('balanced routing requires 1 <= topk <= experts')
    if case['topk'] < case['ep'] and (
            case['tokens'] % case['ep'] or
            case['tokens'] * case['topk'] * case['ep'] % case['experts']):
        raise ValueError('Sparse balanced routing requires whole card rotations and global expert balance')
    if case.get('remote_topk') is not None:
        raise ValueError('balanced routing does not accept remote_topk')


def routing_audit(case, rank, ids):
    """Independent counts from actual routing, with the device int32 byte hash."""
    import hashlib
    import os
    import torch
    experts, world = case['experts'], case['ep']
    owners = ids // (experts // world)
    counts = torch.bincount(ids.flatten(), minlength=experts)
    per_destination = torch.bincount(owners.flatten(), minlength=world)
    unique = [(owners == dst).any(dim=1).sum().item() for dst in range(world)]
    unique_experts = bool((ids.sort(dim=1).values.diff(dim=1) > 0).all())
    audit = dict(rank=rank, mode=case.get('routing', 'cyclic'),
                 expert_counts=counts.tolist(), destination_contributions=per_destination.tolist(),
                 destination_unique_tokens=unique, unique_experts_per_token=unique_experts,
                 ids_int32_bytes=ids.numel()*4)
    if case.get('routing') == 'costanza_balanced_v1':
        from costanza_balanced_routing import provenance, make_routing as reference_routing
        if ids.shape != (case['tokens'], case['topk']) or not unique_experts or not torch.equal(ids, reference_routing(case, rank)):
            raise ValueError('Reference balanced routing IDs differ from the pinned construction')
        audit.update(provenance(case), balance_verified=True)
    elif case.get('routing') == 'costanza_unbalanced_v1':
        from costanza_unbalanced_routing import provenance
        if not unique_experts or ids.shape != (case['tokens'], case['topk']) or not bool(((ids >= 0) & (ids < experts)).all()):
            raise ValueError('Invalid hotspot routing IDs')
        audit.update(provenance(case))
    elif os.environ.get('MEGAMOE_IO_POLICY') != 'exit_count_total_logical_bytes_v1':
        audit['ids_int32_sha256'] = hashlib.sha256(ids.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    if case.get('routing') == 'balanced':
        validate_balanced(case)
        expected = case['tokens'] * case['topk'] // experts
        if short_period_balanced(case):
            upper = (case['tokens'] * case['topk'] + experts - 1) // experts
            assert unique_experts and bool(((counts == expected) | (counts == upper)).all())
        else:
            assert unique_experts and bool((counts == expected).all())
        assert bool((per_destination == case['tokens'] * case['topk'] // world).all())
        assert unique == [case['tokens'] * min(case['topk'], world) // world] * world
        audit.update(balance_verified=True, period_tokens=case['tokens'] if short_period_balanced(case) else experts, seed_used=False,
                     rank_symmetry=('destination rank rotation; source-rank local-expert offsets'
                                    if case['topk'] < world or (short_period_balanced(case) and not short_balanced(case)) else
                                    'destination rank rotation; local expert indices identical'),
                     per_source_expert=[expected, upper] if short_period_balanced(case) else expected,
                     global_per_expert=case["tokens"] * case["topk"] * world // experts,
                     construction_version=("balanced_sparse_v4" if case['topk'] < world else
                                           "balanced_short_v2" if short_balanced(case) else
                                           "balanced_short_v3" if short_period_balanced(case) else "balanced_v1"))
    if case.get('routing') == 'balanced' and case.get('global_expert_balance', 'exact') == 'floorceil':
        total = case['tokens'] * case['topk'] * world
        audit.update(global_expert_balance='floorceil',
                     global_per_expert=[total // experts, (total + experts - 1) // experts],
                     construction_version='balanced_floorceil_v5')
    return audit


def make_balanced_ids(case, rank):
    """Pure CPU integer construction, shared by the torch runtime and offline checks."""
    validate_balanced(case)
    world, local = case['ep'], case['experts'] // case['ep']
    if type(rank) is not int or not 0 <= rank < world:
        raise ValueError('rank outside ep')
    short = short_balanced(case)
    if case['topk'] < world or (short_period_balanced(case) and not short):
        # Each source/destination has C rows. Give its ordinal stream the
        # contiguous interval [rank*C,(rank+1)*C), reduced modulo local experts.
        # Global intervals tile exactly; within a token each destination sees
        # consecutive, distinct local experts. The source counts differ by <=1.
        pair_rows = case['tokens'] * case['topk'] // world
        ordinals = [rank * pair_rows] * world
        rows = []
        for token in range(case['tokens']):
            row = []
            for slot in range(case['topk']):
                owner = (rank + token % world + slot) % world
                row.append(owner * local + ordinals[owner] % local)
                ordinals[owner] += 1
            rows.append(row)
        return rows
    return [[((rank + token % world + slot) % world) * local +
             ((token + (slot // world) * (local // 2)) % local if short else
              (token // world + slot // world) % local)
             for slot in range(case['topk'])] for token in range(case['tokens'])]


def make_routing(case, rank):
    import torch

    tokens, experts, topk = case['tokens'], case['experts'], case['topk']
    world = case['ep']
    local = experts // world
    mode = case.get('routing', 'cyclic')
    if mode == 'costanza_balanced_v1':
        from costanza_balanced_routing import make_routing as make_reference
        return make_reference(case, rank)
    if mode == 'costanza_unbalanced_v1':
        from costanza_unbalanced_routing import make_routing as make_hotspot
        return make_hotspot(case, rank)
    if mode == 'balanced':
        validate_balanced(case)
        if type(rank) is not int or not 0 <= rank < world:
            raise ValueError('rank outside ep')
        return torch.tensor(make_balanced_ids(case, rank), dtype=torch.int64)
    if mode == 'random':
        generator = torch.Generator(device='cpu').manual_seed(case.get('seed', 0) + rank)
        # Bound temporary memory independently of the full batch size.
        chunks = []
        for start in range(0, tokens, 1024):
            scores = torch.rand((min(1024, tokens-start), experts), generator=generator)
            chunks.append(scores.topk(topk, dim=1).indices)
        return torch.cat(chunks, dim=0)
    offsets = torch.arange(tokens)[:, None]
    remote_topk = case.get('remote_topk')
    if remote_topk is not None:
        local_ids = (offsets + torch.arange(topk-remote_topk)[None, :]) % local + rank*local
        peer_ids = (offsets + torch.arange(remote_topk)[None, :]) % local + ((rank+1)%world)*local
        return torch.cat((local_ids, peer_ids), dim=1)
    if mode == 'cyclic':
        return (offsets + torch.arange(topk)[None, :] + ((rank+1)%world)*local) % experts
    raise NotImplementedError(f'Routing mode {mode!r} is not implemented')


def validate_global_histogram(case, histogram):
    """Check actual combined ranks, never silently weaken exact balance."""
    if case.get('routing') in ('costanza_balanced_v1', 'costanza_unbalanced_v1'):
        import torch
        expected = torch.bincount(torch.cat([make_routing(case, r).flatten() for r in range(case['ep'])]),
                                  minlength=case['experts']).tolist()
        if list(histogram) != expected:
            raise ValueError('Reference routing global count mismatch')
        return
    validate_balanced(case)
    total = case['tokens'] * case['topk'] * case['ep']
    experts = case['experts']
    low, remainder = divmod(total, experts)
    assert len(histogram) == experts and sum(histogram) == total, 'Global routing coverage mismatch'
    if case.get('global_expert_balance', 'exact') == 'floorceil':
        assert all(n in (low, low + bool(remainder)) for n in histogram), 'Global floorceil imbalance'
    else:
        assert remainder == 0 and histogram == [low] * experts, 'Global expert load is not balanced'
