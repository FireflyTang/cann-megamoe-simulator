"""CPU-only compiled profiler ABI binding. Never loads a device library.

freeze_contract is a NEW FREEZE operation (one hash pass over code/library/config).
validate_contract only reads small contract metadata and stats frozen artifacts.
Legacy artifacts require an explicit reviewed registry; this is not compiler proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import uuid

SCHEMA = 'megamoe_compiled_profiler_contract_v1'
MACROS = ('PROF_N_CORES', 'PROF_AIV_CORES', 'PROF_SLOTS_PER_CORE',
          'PROF_RECORD_BYTES', 'PROF_RINGIDX_ALIGN')
KEYS = {'w8a8': [664612, 34219044], 'w4a8': [665640, 34220072]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def layout(header):
    text = Path(header).read_text()
    values = {}
    for key in MACROS:
        matches = re.findall(r'^\s*#define\s+' + key + r'\s+([0-9]+)\s*$', text, re.M)
        require(len(matches) == 1, 'Missing/ambiguous nonliteral profiler macro: ' + key)
        values[key] = int(matches[0])
    n, v, slots, record, align = (values[k] for k in MACROS)
    require((n, v, record, align) == (84, 56, 16, 64),
            'Unsupported compiled core/record/counter ABI; requires explicit new implementation')
    require(slots >= 2, 'Invalid compiled slots_per_core')
    require(n * (align + slots * record) <= 0xffffffff, 'Profiler uint32 offset overflow')
    return dict(aic_cores=n-v, aiv_cores=v, cores=n, slots_per_core=slots,
                record_bytes=record, ringidx_align=align, ringidx_region=n*align,
                data_offset=n*align, core_stride=slots*record,
                total_bytes=n*(align+slots*record), record_format='<IIQ',
                counter_format='<I', core_mapping='AIV[0,56);AIC[56,84)', block_dim=28)


def events(header):
    pairs = re.findall(r'\b([A-Z][A-Z0-9_]+_(?:BEGIN|END))\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)',
                       Path(header).read_text())
    result = {name: int(value, 0) for name, value in pairs}
    require(len(result) == len(pairs) and len(result) >= 2, 'Invalid generated event schema')
    require(result.get('KERNEL_BEGIN') == 1 and result.get('KERNEL_END') == 255,
            'Kernel event ABI mismatch')
    require(len(set(result.values())) == len(result), 'Duplicate event IDs')
    return result


def check_device(device, abi):
    for name in ('aic_cores', 'aiv_cores', 'slots_per_core'):
        require(type(device.get(name)) is int and device[name] == abi[name],
                'Runtime ' + name + ' differs from compiled profiler layout')


def check_config(config, contract):
    check_device(config['device'], contract['layout'])
    required_enabled=contract['mode']!='off'
    require(config['profiling']['mode'] == contract['mode'] and
            config['profiling'].get('enabled') is required_enabled, 'Instrumentation mode mismatch')
    case = config['case']
    expected_weight = 'fp4_e2m1' if contract['precision'] == 'w4a8' else 'fp8_e4m3fn'
    require(case.get('dtype') == 'fp8_e4m3fn' and case.get('weight_dtype') == expected_weight,
            'Compiled FP8 activation / weight precision mismatch')
    mode = case.get('force_dedup_mode')
    require(type(mode) is int and mode in (0, 1, 2, 3) and case.get('force_dedup') is True and
            case.get('topk_weights_prefetch') is False and case.get('combine_comm_mode') == (mode & 1),
            'Unsupported or inconsistent forced dedup/prefetch template')


def freeze_contract(root, output, reviewed_registry=None, build_contract=None, frozen_fingerprints=None):
    """Called exactly once for a new execution package; never edits its parent."""
    root, output = Path(root).resolve(), Path(output)
    require(not output.exists(), 'Refuse to overwrite frozen contract')
    def frozen_digest(path):
        relative = str(Path(path).relative_to(root))
        if frozen_fingerprints is not None:
            require(relative in frozen_fingerprints, 'Missing one-time freeze fingerprint: '+relative)
            return frozen_fingerprints[relative]
        return digest(path)
    package = read(root/'package.json')
    registry = read(root/'build/artifact_registry.json')
    binary = package['binary']
    require(binary == 'build/kernel.o', 'Unsupported binary location')
    binary_sha = frozen_digest(root/binary)
    artifact = registry['binaries'].get(binary_sha)
    require(artifact is not None and artifact == package['artifact_identity'],
            'Binary is not the registered compiled artifact identity')
    mode = package['mode']
    precision = package.get('precision', artifact.get('precision', 'w8a8'))
    require(mode in ('full', 'kernel_only','off') and mode == artifact['mode'],
            'Compiled artifact mode mismatch')
    require(precision in KEYS, 'Unsupported precision')
    metadata_sha = frozen_digest(root/'build/kernel.json')
    metadata = read(root/'build/kernel.json')
    require(metadata_sha == artifact['metadata_sha256'] and metadata['sha256'] == binary_sha,
            'Compiler JSON / binary registration mismatch')
    require(metadata['kernelList'] == package['kernel_list'], 'Kernel registration mapping differs')
    require(sorted(x['tilingKey'] for x in package['kernel_list']) == KEYS[precision]
            and sorted(artifact['tiling_keys']) == KEYS[precision], 'Precision / template key mismatch')
    generated = {p.name: frozen_digest(p) for p in sorted((root/'generated').iterdir()) if p.is_file()}
    require(generated == registry['generated_files'], 'Generated headers are not registered compiler inputs')
    source_proof = None
    if build_contract is not None:
        proof = read(build_contract)
        require(proof.get('schema') == 'megamoe_profiler_build_contract_v1' and
                proof.get('status') == 'pass', 'Missing completed build attestation')
        for key, expected in [('binary_sha256', binary_sha), ('mode', mode),
                              ('metadata_sha256', metadata_sha), ('precision', precision),
                              ('generated_files', generated)]:
            require(proof.get(key) == expected, 'Build attestation mismatch: ' + key)
        require(proof.get('profiling_enabled') is (mode!='off') and proof.get('effective_include_verified') is True and proof.get('writer_guard_verified') is True,
                'Actual compile profile flags/include are unverified')
        if mode=='off':require(proof.get('profiling_macro_defined') is False,'OFF build must leave ENABLE_PROFILING undefined')
        source_proof = dict(kind='compiler_build_attestation', path=str(build_contract),
                            sha256=digest(build_contract))
    else:
        require(reviewed_registry is not None, 'Fail closed: compiled proof or explicit legacy review required')
        reviewed = read(reviewed_registry)
        require(reviewed.get('schema') == 'megamoe_legacy_profiler_review_v1' and
                reviewed.get('registry_sha256') == digest(root/'build/artifact_registry.json') and
                binary_sha in reviewed.get('accepted_binary_sha256', []),
                'Legacy review does not authorize this exact compiled registry/artifact')
        require(bool(reviewed.get('evidence')) and bool(reviewed.get('limitations')),
                'Legacy review requires evidence and provenance limits')
        source_proof = dict(kind='explicit_legacy_registry_review', review=reviewed,
                            limitation='Not independent reconstruction of compiler include resolution')
    abi = layout(root/'generated/profiler_config.h')
    contract = dict(schema=SCHEMA, contract_id=uuid.uuid4().hex, precision=precision, mode=mode,
                    binary=binary, binary_sha256=binary_sha, metadata_sha256=metadata_sha,
                    artifact_identity=artifact, kernel_list=package['kernel_list'],
                    generated_files=generated, event_ids=events(root/'generated/profile_events_generated.h'),
                    layout=abi, build_provenance=source_proof,
                    submit_allowed=source_proof['kind'] == 'compiler_build_attestation',
                    frozen_sizes={name:(root/name).stat().st_size for name in
                                  [binary, 'build/kernel.json', 'build/artifact_registry.json']})
    check_device(package['device'], abi)
    manifest = read(root/'manifest.json')
    for item in manifest['cases']:
        check_config(item['config'], contract)
        require(read(root/'cases'/item['id']/'config.json5') == item['config'], 'Saved config drift')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(contract, stream, indent=2)
        stream.write('\n')
    return contract


def validate_contract(root, contract_path=None, offline_only=False):
    """Pre-submit and pre-device CPU validation, no content hashing or device calls."""
    root = Path(root)
    contract = read(contract_path or root/'profiler_contract.json')
    require(contract.get('schema') == SCHEMA, 'Missing / unknown compiled profiler contract')
    require(offline_only or contract.get('submit_allowed') is True,
            'Legacy provenance is offline-only: independently bound compiler contract required before new submit')
    package = read(root/'package.json')
    for field in ('mode', 'binary', 'artifact_identity', 'kernel_list'):
        require(package.get(field) == contract[field], 'Frozen package contract drift: '+field)
    require(package.get('precision', contract['artifact_identity'].get('precision', 'w8a8')) ==
            contract['precision'], 'Frozen precision drift')
    require(package['files'].get(contract['binary']) == contract['binary_sha256'], 'Binary identity drift')
    require(package['files'].get('build/kernel.json') == contract['metadata_sha256'], 'Metadata identity drift')
    for name, size in contract['frozen_sizes'].items():
        require((root/name).stat().st_size == size, 'Frozen artifact size changed: '+name)
    require(layout(root/'generated/profiler_config.h') == contract['layout'], 'Generated layout changed')
    require(events(root/'generated/profile_events_generated.h') == contract['event_ids'], 'Generated event ABI changed')
    for name, value in contract['generated_files'].items():
        require(package['files'].get('generated/'+name) == value, 'Generated identity drift: '+name)
    check_device(package['device'], contract['layout'])
    for item in read(root/'manifest.json')['cases']:
        check_config(item['config'], contract)
        require(read(root/'cases'/item['id']/'config.json5') == item['config'], 'Saved config drift')
    return contract


def allocation_bytes(config, contract):
    check_config(config, contract)
    return contract['layout']['total_bytes']


def inspect_capture(path, contract):
    """Read existing valid record slots only; no hashing or device execution."""
    abi, ids = contract['layout'], set(contract['event_ids'].values())
    require(Path(path).stat().st_size == abi['total_bytes'], 'Capture byte size differs from compiled layout')
    if contract['mode']=='off':
        raw=Path(path).read_bytes()
        require(not any(raw),'OFF capture buffer must be entirely zero')
        return dict(status='pass',cores=abi['cores'],total_bytes=abi['total_bytes'],min_records=0,max_records=0,mode='off')
    counts = []
    with Path(path).open('rb') as stream:
        for core in range(abi['cores']):
            stream.seek(core*abi['ringidx_align'])
            count, = struct.unpack('<I', stream.read(4))
            require(2 <= count < abi['slots_per_core'], 'Missing/overflow core records')
            if contract['mode'] == 'kernel_only':
                require(count == 2, 'KO must have exactly two KERNEL records')
            stream.seek(abi['data_offset']+core*abi['core_stride'])
            records = list(struct.iter_unpack('<IIQ', stream.read(count*16)))
            require(records[0][0] == 1 and records[-1][0] == 255, 'Missing kernel endpoints')
            require(all(r[0] in ids for r in records), 'Unknown event / wrong record stride')
            require(all(a[2] <= b[2] for a,b in zip(records, records[1:])), 'Non-monotonic cycles')
            counts.append(count)
    return dict(status='pass', cores=len(counts), total_bytes=abi['total_bytes'],
                min_records=min(counts), max_records=max(counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--freeze-output', type=Path)
    parser.add_argument('--legacy-review', type=Path)
    parser.add_argument('--build-contract', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--offline-only', action='store_true')
    args = parser.parse_args()
    contract = (freeze_contract(args.root, args.freeze_output, args.legacy_review, args.build_contract)
                if args.freeze_output else validate_contract(args.root, args.contract, args.offline_only))
    result = dict(status='pass', contract_id=contract['contract_id'], layout=contract['layout'])
    if args.capture:
        result['capture'] = inspect_capture(args.capture, contract)
    print(json.dumps(result, indent=2))
