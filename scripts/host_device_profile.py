"""Optional per-rank PyTorch Host/NPU capture, with no extra operator launches."""
from contextlib import contextmanager, nullcontext
from pathlib import Path


def analysis_location(config):
    location = config.get('profiling', {}).get('torch_profiler_analysis', 'remote')
    if location not in ('remote', 'local'):
        raise ValueError('torch_profiler_analysis must be remote or local')
    if location == 'local' and not config.get('profiling', {}).get('torch_profiler', False):
        raise ValueError('Local external analysis requires torch_profiler=true')
    return location


@contextmanager
def formal_profile(config, root, rank):
    if not config['profiling'].get('torch_profiler', False):
        yield
        return
    from grouped_external_capture import grouped_formal_count
    grouped_formal_count(config)
    import torch_npu
    p = torch_npu.profiler
    deferred = analysis_location(config) == 'local'
    directory = Path(root) / 'data/raw/torch_profiler' / f'rank{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    options = dict(
        profiler_level=p.ProfilerLevel.Level1,
        aic_metrics=(p.AiCMetrics.AiCoreNone if config['profiling'].get('aic_metrics') == 'none' else p.AiCMetrics.PipeUtilization),
    )
    if deferred:
        # Retain framework/CANN originals for parsing after compressed return.
        options['data_simplification'] = False
    experimental = p._ExperimentalConfig(**options)
    handler = (p.tensorboard_trace_handler(str(directory), analyse_flag=False)
               if deferred else p.tensorboard_trace_handler(str(directory)))
    # One window records one legacy launch or all explicit grouped replays. Warmups and tensor
    # validation remain outside; step() only closes the active recording window.
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        experimental_config=experimental,
        schedule=p.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=handler,
    ) as capture:
        yield
        capture.step()


def host_range(config, name):
    if not config['profiling'].get('torch_profiler', False):
        return nullcontext()
    import torch
    return torch.profiler.record_function(name)
