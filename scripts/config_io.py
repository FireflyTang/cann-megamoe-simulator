"""Read JSON5 with an additional shell-style # line-comment syntax."""
from pathlib import Path

import json5
from measurement_validation import validate_measurement_config


def strip_hash_comments(text):
    """Preserve strings, JSON5 comments and source positions while masking # comments."""
    result = list(text)
    state = None
    i = 0
    while i < len(text):
        char = text[i]
        pair = text[i:i + 2]
        if state in ('"', "'"):
            if char == '\\':
                i += 2
                continue
            if char == state:
                state = None
        elif state == 'block':
            if pair == '*/':
                state = None
                i += 2
                continue
        elif state in ('line', 'hash'):
            if char in '\r\n\u2028\u2029':
                state = None
            elif state == 'hash':
                result[i] = ' '
        elif char in ('"', "'"):
            state = char
        elif pair in ('//', '/*'):
            state = 'line' if pair == '//' else 'block'
            i += 2
            continue
        elif char == '#':
            state = 'hash'
            result[i] = ' '
        i += 1
    return ''.join(result)


def load_config(path):
    config = json5.loads(strip_hash_comments(Path(path).read_text(encoding='utf-8')))
    validate_measurement_config(config)
    from numerical_validation import validate_policy
    validate_policy(config['validation'])
    from profile_modes import instrumentation_mode
    mode = instrumentation_mode(config)
    metrics = config.get('profiling', {}).get('aic_metrics', 'pipe_utilization')
    if metrics not in ('pipe_utilization', 'none'):
        raise ValueError('profiling.aic_metrics must be pipe_utilization or none')
    if mode != 'full' and any(config.get('diagnostics', {}).values()):
        raise ValueError('Reduced instrumentation cannot enable diagnostic paths')
    runtime = config.get('runtime', {})
    if 'golden_cache_root' in runtime:
        cache_root = runtime['golden_cache_root']
        if (not isinstance(cache_root, str) or not cache_root or not Path(cache_root).is_absolute()
                or config['case'].get('input_recipe') != 'costanza_random_v1' or mode != 'off'):
            raise ValueError('golden_cache_root requires absolute path, random recipe and OFF capture')
    repeats = runtime.get('formal_repeats', 1)
    if type(repeats) is not int or repeats not in (1, 3, 10):
        raise ValueError('runtime.formal_repeats must be 1, 3 or explicit grouped 10')
    if repeats == 3 and (runtime.get('launch_mode') != 'graph_capture' or
                         config.get('profiling', {}).get('msprof', False) or
                         config.get('diagnostics', {}).get('mte_probe', False)):
        raise ValueError('repeat3 requires graph_capture and no external msprof or mte_probe')
    from sampling_contract import grouped_external
    grouped_external(config)
    torch_profile = config.get('profiling', {}).get('torch_profiler', False)
    if type(torch_profile) is not bool:
        raise ValueError('profiling.torch_profiler must be boolean')
    if torch_profile and config.get('profiling', {}).get('msprof', False):
        raise ValueError('torch_profiler and external msprof cannot own the same capture')
    from host_device_profile import analysis_location
    analysis_location(config)
    entry_sync = config.get('diagnostics', {}).get('entry_rank_sync', False)
    if type(entry_sync) is not bool:
        raise ValueError('diagnostics.entry_rank_sync must be boolean')
    if entry_sync and config['case']['ep'] not in (2, 4):
        raise ValueError('Entry rank sync diagnostic supports only the 2P/4P MTE harness')
    if runtime.get('launch_mode', 'eager') not in ('eager', 'graph_capture'):
        raise ValueError('runtime.launch_mode must be eager or graph_capture')
    presync = runtime.get('graph_presync', False)
    sync_count = runtime.get('graph_presync_count', 1)
    if type(sync_count) is not int or sync_count not in (1, 2):
        raise ValueError('graph_presync_count must be 1 or 2')
    sync_kind = runtime.get('graph_presync_kind', 'allreduce')
    if sync_kind not in ('allreduce', 'light_signal'):
        raise ValueError('graph_presync_kind must be allreduce or light_signal')
    if sync_kind == 'light_signal' and (not presync or sync_count != 1 or entry_sync):
        raise ValueError('light_signal requires presync count1 and internal entry sync OFF')
    graph_warmup = runtime.get('graph_warmup', 0)
    if type(presync) is not bool or type(graph_warmup) is not int or graph_warmup < 0:
        raise ValueError('graph_presync must be boolean; graph_warmup must be nonnegative integer')
    if (presync or graph_warmup) and runtime.get('launch_mode') != 'graph_capture':
        raise ValueError('Graph presync/warmup requires graph_capture')
    if presync and (entry_sync or graph_warmup < 1):
        raise ValueError('Graph presync comparison requires internal entry sync OFF and a warm replay')
    if 'memory_abort_on_swap' in runtime:
        if type(runtime['memory_abort_on_swap']) is not bool:
            raise ValueError('memory_abort_on_swap must be boolean')
        if not ('memory_min_available_gib' in runtime or 'memory_max_rss_gib' in runtime):
            raise ValueError('Swap policy requires memory guard')
    if 'memory_max_rss_gib' in runtime:
        import math
        value = runtime['memory_max_rss_gib']
        reserve = runtime.get('memory_abort_available_gib', 16)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError('RSS limit and reserve must be finite and positive')
    balance = config.get('case', {}).get('global_expert_balance', 'exact')
    if balance not in ('exact', 'floorceil'):
        raise ValueError('case.global_expert_balance must be exact or floorceil')
    if balance != 'exact' and config['case'].get('routing') != 'balanced':
        raise ValueError('global_expert_balance requires balanced routing')
    case = config.get('case', {})
    if case.get('routing') == 'costanza_balanced_v1':
        from costanza_balanced_routing import contract
        contract(case)
    if case.get('routing') == 'costanza_unbalanced_v1':
        from costanza_unbalanced_routing import contract
        contract(case)
    if case.get('topk_weights_dtype', 'bfloat16') not in ('bfloat16', 'float32'):
        raise ValueError('case.topk_weights_dtype must be bfloat16 or float32')
    weight_dtype = case.get('weight_dtype', case.get('dtype'))
    if weight_dtype == 'fp4_e2m1':
        if case.get('dtype') != 'fp8_e4m3fn':
            raise ValueError('A8W4 requires fp8_e4m3fn activation and fp4_e2m1 weight')
        if case['hidden'] % 64 or case['intermediate'] % 128:
            raise ValueError('A8W4 requires hidden aligned to 64 and intermediate aligned to 128')
    elif weight_dtype != case.get('dtype'):
        raise ValueError('A8W8 requires matching FP8 activation and weight formats')
    numerical_reference(config.get('case', {}))
    requested_dedup_mode(config.get('case', {}))
    return config


def numerical_reference(case):
    recipe = case.get('input_recipe', 'structured_v1')
    if recipe == 'structured_v1':
        return 'structured_cpu_formula'
    if recipe == 'costanza_random_v1':
        from costanza_inputs import contract, REFERENCE
        contract(case)
        return REFERENCE
    raise ValueError('Unsupported case.input_recipe: ' + repr(recipe))


def requested_dedup_mode(case):
    """Resolve an exact forced mask; PR13576 schema 2; explicit zero is a forced baseline."""
    if 'force_dedup_mode' in case:
        mode = case['force_dedup_mode']
        if type(mode) is not int or mode not in (0, 1, 2, 3):
            raise ValueError('case.force_dedup_mode must be integer 0, 1, 2 or 3')
        if case.get('force_dedup') is not True:
            raise ValueError('case.force_dedup_mode requires force_dedup=true')
        if case.get('combine_comm_mode', 1) != int(bool(mode & 1)):
            raise ValueError('combine_comm_mode must match force_dedup_mode combine bit')
        return mode
    if case.get('force_dedup', False):
        return 3
    return 2 | (1 if case.get('combine_comm_mode', 1) == 1 else 0)


def validate_dedup_gate(case, gate):
    """Check captured tiling, not merely the requested config switches."""
    if not isinstance(gate, dict):
        raise ValueError('Missing captured dedup_gate metadata')
    if gate.get('mode_schema') != 'pr13576_combine1_dispatch2_v1':
        raise ValueError('Missing/unsupported dedup mode schema; do not reinterpret legacy masks')
    mode = gate['actual_mode']
    if type(mode) is not int or mode not in (0, 1, 2, 3):
        raise ValueError('Invalid captured dedup mode')
    requested = requested_dedup_mode(case)
    native_requested = 2 | (1 if case.get('combine_comm_mode', 1) == 1 else 0)
    if gate.get('native_requested_mode') != native_requested or gate.get('requested_mode') != requested:
        raise ValueError('Captured native/forced request mode mismatch')
    if case.get('force_dedup', False) and mode != requested:
        raise ValueError(f'Forced dedup mode mismatch: requested {requested}, actual {mode}')
    for field, expected in (('request_dispatch', bool(requested & 2)),
                            ('request_combine', bool(requested & 1)),
                            ('dispatch_enabled', bool(mode & 2)),
                            ('combine_enabled', bool(mode & 1))):
        if gate[field] is not expected:
            raise ValueError(f'Captured dedup field {field} disagrees with mode')
    if not case.get('force_dedup', False) and mode != gate['natural_mode']:
        raise ValueError('Unforced mode differs from captured natural mode')
    if gate.get('savings_gate_policy') == 'pr13576_8f7db7b5':
        import math
        expected_gate = dedup_savings_metadata(case, gate['estimated_saved_bytes'])
        for field, expected in expected_gate.items():
            actual = gate.get(field)
            if isinstance(expected, float):
                valid = isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12)
            else:
                valid = actual == expected
            if not valid:
                raise ValueError('Captured savings gate mismatch: '+field)
        if gate['savings_gate_value'] < gate['savings_gate_required'] and bool(gate['natural_mode'] & 2):
            raise ValueError('Natural dispatch dedup enabled below savings threshold')


def dedup_savings_metadata(case, saved_bytes):
    """PR12213 c54fcb host gate units; natural_mode is still captured from host.

    Mirrors SetDedupModeParams only for reporting its savings comparison; it
    does not replace the host UB safety guards or infer an actual mode.
    """
    import math
    if not math.isfinite(saved_bytes) or saved_bytes < 0:
        raise ValueError('Invalid estimated saved bytes')
    if case['ep'] == 4:
        experts = case['experts'] // case['ep']
        if experts <= 0 or case['experts'] % case['ep']:
            raise ValueError('Expert count must divide world size')
        required = (0.30 if 2 * case['intermediate'] <= 2048 else 2.10) * 1024 * 1024
        value = saved_bytes / experts
        basis = 'bytes_per_local_expert'
        total_required = required * experts
    else:
        required = 60.0 * 1024 * 1024
        value = saved_bytes
        basis = 'total_saved_bytes'
        total_required = required
    return {'savings_gate_policy': 'pr13576_8f7db7b5', 'savings_gate_basis': basis,
            'savings_gate_value': float(value), 'savings_gate_required': float(required),
            'required_saved_bytes': float(total_required)}
