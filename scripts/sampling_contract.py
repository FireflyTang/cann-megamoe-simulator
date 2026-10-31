"""Explicit OFF-only grouped timing; legacy full/repeat3 policies are unchanged."""


def grouped_external(config):
    runtime = config.get('runtime', {})
    profiling = config.get('profiling', {})
    grouped = profiling.get('torch_profiler_grouped', False)
    if type(grouped) is not bool:
        raise ValueError('profiling.torch_profiler_grouped must be boolean')
    if not grouped:
        if runtime.get('formal_repeats', 1) == 10 or runtime.get('formal_validation', 'each') != 'each':
            raise ValueError('Ten samples/last-only validation require explicit grouped external timing')
        return False
    if not (profiling.get('mode') == 'off' and profiling.get('enabled') is False
            and profiling.get('torch_profiler') is True
            and profiling.get('torch_profiler_analysis') == 'local'
            and profiling.get('msprof', False) is False
            and type(runtime.get('formal_repeats')) is int and runtime['formal_repeats'] == 10
            and runtime.get('formal_validation') == 'last_only'
            and runtime.get('launch_mode') == 'graph_capture'
            and config.get('case', {}).get('warmup', 0) == 0
            and runtime.get('graph_warmup') == 3
            and not any(config.get('diagnostics', {}).values())):
        raise ValueError('Grouped timing requires OFF/local profiler, graph warm3/formal10 and last-only validation')
    return True


def formal_count(config):
    return 10 if grouped_external(config) else 1


def validate_grouped_evidence(config, result, launch):
    if not grouped_external(config):
        return
    if not (result.get('formal_sample_count') == 10
            and result.get('validated_formal_indices') == [9]
            and result.get('validation_scope') == 'last_formal_only'
            and result.get('formal_launches') == 10
            and launch.get('formal_launches') == 10
            and launch.get('formal_replays') == 10):
        raise ValueError('Grouped timing sample count/last-only validation evidence differs')
    checks = result.get('numerical_checks', {})
    if checks.get('validation_scope') not in (None, 'last_formal_only'):
        raise ValueError('Grouped timing cannot claim per-sample numerical validation')
