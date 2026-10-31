"""Deterministic global routing matching reference is_topk_balance=True."""

MODE = 'costanza_balanced_v1'
REFERENCE_COMMIT = 'ae2444725b4e7e0b22b0b211fb401e27535111bd'


def contract(case):
    for key in ('tokens', 'experts', 'topk', 'ep'):
        if type(case.get(key)) is not int or case[key] <= 0:
            raise ValueError('Reference balanced routing requires positive integer ' + key)
    if case['experts'] % case['ep'] or case['topk'] > case['experts']:
        raise ValueError('Invalid reference balanced expert partition or top-k')
    if case.get('remote_topk') is not None or case.get('global_expert_balance', 'exact') != 'exact':
        raise ValueError('Reference routing cannot impose the legacy balance policy')


def make_routing(case, rank):
    import torch
    contract(case)
    if type(rank) is not int or not 0 <= rank < case['ep']:
        raise ValueError('Rank outside reference balanced routing world')
    # Equivalent to concatenating arange(round, round+E) % E globally,
    # then reshaping into global tokens and splitting contiguous rank batches.
    start = rank * case['tokens'] * case['topk']
    flat = torch.arange(start, start + case['tokens'] * case['topk'], dtype=torch.int64)
    return ((flat % case['experts'] + flat // case['experts']) % case['experts']).reshape(
        case['tokens'], case['topk'])


def provenance(case):
    contract(case)
    total = case['tokens'] * case['topk'] * case['ep']
    low, remainder = divmod(total, case['experts'])
    return dict(construction_version=MODE, reference_commit=REFERENCE_COMMIT,
                reference_switch=dict(is_topk_balance=True), seed_used=False,
                global_generation=True, card_pair_balance_required=False,
                global_per_expert=[low, low + bool(remainder)],
                balance_scope='global expert counts; no per-token/card-pair balance guarantee')
