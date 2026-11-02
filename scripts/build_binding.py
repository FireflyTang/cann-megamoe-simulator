"""Future-build hook: inject a compile-time guard, then attest successful artifacts.

The hook modifies only an explicitly new generated compiler recipe and attempt dir.
It must run inside the build owner's configure->compile worktree lock. It does not
compile itself. begin() returns a token; finish() is called only after check=True
compile success. Failed compiles never generate a passing build contract.
"""
import ast
import json
from pathlib import Path
import re
import shutil
from profiler_contract import digest, layout, require


def begin(recipe, generated, attempt, mode, precision, writer):
    recipe, generated, attempt = map(Path, (recipe, generated, attempt))
    require(mode in ('full', 'kernel_only','off','sim') and precision in ('w8a8', 'w4a8'), 'Bad build mode')
    snapshot = attempt/'profiler-compiled-inputs'
    snapshot.mkdir(exist_ok=False)
    for name in ('profiler_config.h', 'profile_events_generated.h', 'pair_map.json'):
        shutil.copy2(generated/name, snapshot/name)
    abi = layout(snapshot/'profiler_config.h')
    text = recipe.read_text()
    matches = list(re.finditer(r'^(\s*)custom_compile_options = (.+),\s*$', text, re.M))
    require(len(matches) == 1, 'Unsupported compiler recipe; cannot enforce actual compile flags')
    match = matches[0]
    options = ast.literal_eval(match.group(2))
    require(set(options) == {'__ALLSOC__'}, 'Unexpected SOC-specific flags')
    flags = options['__ALLSOC__']
    writer = Path(writer).resolve()
    writer_text = writer.read_text()
    for name in ('PROF_N_CORES','PROF_AIV_CORES','PROF_SLOTS_PER_CORE','PROF_RECORD_BYTES','PROF_RINGIDX_ALIGN'):
        require('static_assert('+name+' == MEGAMOE_COMPILED_'+name in writer_text, 'Actual writer is missing compiled ABI assertion: '+name)
    require('-I'+str(writer.parent) in flags, 'Actual compiler recipe does not reference guarded writer directory')
    detail = 1 if mode in ('full', 'sim') else 0
    if mode=='off':
        require(not any(x.startswith('-DENABLE_PROFILING') for x in flags) and
                any(x in ('-UENABLE_PROFILING','-U',) or x.startswith('-UENABLE_PROFILING=') for x in flags) and
                flags.count('-DMEGAMOE_PROFILE_DETAIL=0')==1,
                'OFF recipe must leave ENABLE_PROFILING undefined and DETAIL=0')
    else:
        require(flags.count('-DENABLE_PROFILING=1') == 1 and
                flags.count('-DMEGAMOE_PROFILE_DETAIL='+str(detail)) == 1,
                'Actual generated recipe does not enable requested instrumentation')
    require(not any(x.startswith('-include') for x in flags),
            'Ambiguous forced include or profiling undefine')
    # Exact compiled config is forced first. Its pragma once makes later inclusion
    # of this same header a no-op; later contradictory config fails macro/guard checks.
    guard = snapshot/'compiled_profiler_guard.h'
    guard_text=('#pragma once\n#include '+json.dumps(str(snapshot/'profiler_config.h'))+'\n' +
        ('#ifdef ENABLE_PROFILING\n#error profiling_must_be_undefined\n#endif\n' if mode=='off' else '#if !defined(ENABLE_PROFILING) || ENABLE_PROFILING != 1\n#error profiling_disabled\n#endif\n') +
        '#if !defined(MEGAMOE_PROFILE_DETAIL) || MEGAMOE_PROFILE_DETAIL != '+str(detail)+'\n'
        '#error instrumentation_mode_mismatch\n#endif\n'+
        ('#if !defined(MEGAMOE_PROFILE_BACKEND) || MEGAMOE_PROFILE_BACKEND != 1\n#error sim_backend_disabled\n#endif\n' if mode=='sim' else
         '#if !defined(MEGAMOE_PROFILE_BACKEND) || MEGAMOE_PROFILE_BACKEND != 0\n#error board_backend_disabled\n#endif\n')+
        ''.join('#define MEGAMOE_COMPILED_'+name+' '+str(value)+'\nstatic_assert('+name+' == '+str(value)+', "compiled profiler ABI mismatch");\n'
        for name,value in [('PROF_N_CORES',abi['cores']),('PROF_AIV_CORES',abi['aiv_cores']),
         ('PROF_SLOTS_PER_CORE',abi['slots_per_core']),('PROF_RECORD_BYTES',16),('PROF_RINGIDX_ALIGN',64)]))
    guard.write_text(guard_text)
    # Remove only the verified former generated include, preserving all other flags.
    old = '-I'+str(generated.resolve())
    require(flags.count(old) == 1, 'Recipe does not reference selected generated input exactly once')
    flags[flags.index(old)] = '-I'+str(snapshot.resolve())
    expected_backend = '-DMEGAMOE_PROFILE_BACKEND='+str(1 if mode == 'sim' else 0)
    existing_backend = [x for x in flags if x.startswith('-DMEGAMOE_PROFILE_BACKEND=')]
    if existing_backend:
        require(existing_backend == [expected_backend], 'Wrong profiling backend in generated recipe')
    else:
        flags.append(expected_backend)
    flags.extend(['-include', str(guard.resolve())])
    edited = text[:match.start()]+match.group(1)+'custom_compile_options = '+repr(options)+','+text[match.end():]
    recipe.write_text(edited)
    token = dict(schema='megamoe_profiler_build_token_v1', mode=mode, precision=precision,
                 profile_backend=1 if mode == 'sim' else 0,
                 recipe=str(recipe), recipe_sha256=digest(recipe), writer=str(writer), writer_sha256=digest(writer), writer_guard_verified=True, guard=str(guard),
                 guard_sha256=digest(guard), layout=abi,
                 generated_files={p.name:digest(p) for p in snapshot.iterdir() if p.name != guard.name},
                 generated_directory=str(snapshot), compile_options=flags)
    (attempt/'profiler-build-token.json').write_text(json.dumps(token,indent=2)+'\n')
    return token


def finish(token, binary, metadata, output, compile_exit_code):
    require(compile_exit_code == 0, 'Cannot attest failed compile')
    require(not Path(output).exists(), 'Refuse to overwrite completed build contract')
    require(token.get('writer_guard_verified') is True and digest(token['writer']) == token['writer_sha256'], 'Actual profiler writer changed or is unguarded')
    require(digest(token['recipe']) == token['recipe_sha256'] and
            digest(token['guard']) == token['guard_sha256'], 'Compiler inputs changed during build')
    generated = Path(token['generated_directory'])
    require({name:digest(generated/name) for name in token['generated_files']} == token['generated_files'],
            'Generated profiler headers changed during compile')
    binary_sha = digest(binary)
    require(json.loads(Path(metadata).read_text())['sha256'] == binary_sha, 'Compiler artifact identity mismatch')
    result = dict(token, schema='megamoe_profiler_build_contract_v1', status='pass',
                  binary_sha256=binary_sha, metadata_sha256=digest(metadata),
                  profiling_enabled=token['mode']!='off', profiling_macro_defined=token['mode']!='off', effective_include_verified=True,
                  verification='actual generated compiler recipe forced-include compile-time ABI guard')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result
