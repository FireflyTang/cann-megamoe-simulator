"""Instrumentation identity and sparse-record validation; no NPU imports."""
import json
import re
import shlex
import struct
from measurement_validation import validate_frequency_mhz


def instrumentation_mode(config):
    p = config.get('profiling', {})
    enabled = p.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('profiling.enabled must be boolean')
    mode = p.get('mode', 'full' if enabled else 'off')
    if mode not in ('full', 'kernel_only', 'off'):
        raise ValueError('profiling.mode must be full, kernel_only or off')
    if enabled != (mode != 'off'):
        raise ValueError('profiling.mode conflicts with profiling.enabled')
    return mode


def validate_compile_modes(commands, mode):
    assert commands, 'Missing device compile commands'
    for command in commands:
        definitions = {}
        for token in shlex.split(command):
            if token.startswith('-D'):
                key, _, value = token[2:].partition('=')
                if key in definitions and definitions[key] != value:
                    raise AssertionError(('Conflicting compile macro', key))
                definitions[key] = value or '1'
        assert ('ENABLE_PROFILING' in definitions) == (mode != 'off'), (mode, definitions)
        assert definitions.get('MEGAMOE_PROFILE_DETAIL') == str(int(mode == 'full')), (mode, definitions)


def validate_launch(config, launch):
    """Strict experiment protocol, enabled by explicit experiment metadata only."""
    if config.get('experiment', {}).get('kind') != 'instrumentation_overhead_v1':
        return
    assert config['case']['warmup'] == 0 and config['case']['iterations'] == 1
    assert config['runtime']['graph_warmup'] == 3
    assert launch['warmup_outside_graph'] == 0
    assert launch['graph_warmup_passes'] == [0, 1, 2]
    assert launch['formal_launches'] == launch['formal_replays'] == 1
    assert launch['capture_did_not_execute'] and launch['capture_stream_verified']
    assert launch['graph_presync_kind'] == 'light_signal'
    assert launch['light_sync_epochs'] == [4] * (config['case']['ep'] + 1)
    assert launch['graph_sequence'] == ['MegaMoeLightRankSync', 'MegaMoeProfile']
    assert launch['instrumentation_mode'] == instrumentation_mode(config)
    assert launch['profiler_level'] == 'Level1' and launch['aic_metrics'] == 'none'


def validate_kernel_records(raw, config, rank):
    """Exactly one positive KERNEL pair per physical core; reject detail leakage."""
    cores = config['device']['aic_cores'] + config['device']['aiv_cores']
    slots = config['device']['slots_per_core']
    assert len(raw) == cores * (64 + slots * 16)
    freq = validate_frequency_mhz(config['profiling']['timestamp_frequency_mhz'])
    events = []
    for core in range(cores):
        count = struct.unpack_from('<I', raw, core*64)[0]
        assert count == 2, ('kernel_only records per core', rank, core, count)
        pos = cores*64 + core*slots*16
        begin, end = struct.unpack_from('<IIQ', raw, pos), struct.unpack_from('<IIQ', raw, pos+16)
        assert begin[:2] == (0x1, 0) and end[:2] == (0xff, 0), ('non-KERNEL record', rank, core)
        assert end[2] > begin[2], ('invalid kernel interval', rank, core)
        events.append(dict(name='KERNEL', ph='X', pid=rank, tid=core,
                           ts=begin[2]/freq, dur=(end[2]-begin[2])/freq,
                           args=dict(begin_cycle=begin[2], end_cycle=end[2], instrumentation_mode='kernel_only')))
    return events
