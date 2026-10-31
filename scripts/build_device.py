"""Compile only the two formal W8A8 or W4A8 index templates, in an immutable attempt.

Run after the repository's ascend950 --ophost build, under local-build-env.sh.
The official generated parameter recipe is selected explicitly, never by shape.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from build_transaction import BuildTransaction

def compile_spec(precision):
    if precision not in ('w4a8','w8a8'):raise ValueError('Unsupported precision')
    weight=40 if precision=='w4a8' else 36
    template = ','.join((
    f'TILINGKEY_MOE_WEIGHT_DTYPE={weight}', f'TILINGKEY_SHARED_WEIGHT_DTYPE={weight}',
    'TILINGKEY_DISPATCH_QUANT_MODE=4', 'TILINGKEY_DISPATCH_QUANT_OUT_DTYPE=4',
    'TILINGKEY_SHARED_QUANT_OUT_DTYPE=4', 'TILINGKEY_COMBINE_QUANT_OUT_TYPE=0',
    'TILINGKEY_COMM_MODE=0', 'TILINGKEY_TOPK_WEIGHTS_TYPE=0'))
    parameter = 'MegaMoe_e089a1fd8cd418cf3727844acea9f941' if precision=='w4a8' else 'MegaMoe_9d10941d738bc0a718602c71c5d93ae8'
    keys=[665640,34220072] if precision=='w4a8' else [664612,34219044]
    return template,parameter,keys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=['w4a8','w8a8'],default='w4a8')
    p.add_argument('--mode', choices=['full', 'kernel_only', 'off', 'sim'], required=True)
    p.add_argument('--topk-dtype', choices=['bfloat16','float32'], default='bfloat16')
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generated', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True, help='Private CMake tree; never reuse another mode build')
    a = p.parse_args()
    template,parameter,expected_keys=compile_spec(a.precision)
    allowed_tang = {
        Path('/home/t00826093/miniforge3/envs/tang'),
        Path('/home/t00826093/miniconda3/envs/tang'),
    }
    if Path(sys.prefix).resolve() not in allowed_tang:
        raise RuntimeError('Local or model1 tang is required')
    repo = args.repo.resolve()
    build, out = a.build_dir.resolve(), a.output.resolve()
    if build.exists():
        raise ValueError('Build directory must be a fresh isolated path')
    inputs = [repo/'mc2', repo/'common', repo/'include', repo/'megamoe_profile/include', repo/'cmake', repo/'CMakeLists.txt', repo/'scripts']
    # The W4 configure path consumes this shared FetchContent source tree.
    shared_cmake = repo/'third_party/cann-cmake'
    if shared_cmake.exists():
        inputs.append(shared_cmake)
    with BuildTransaction(repo, out, generated=a.generated, inputs=inputs) as transaction:
        (out/'bin').mkdir()
        env = dict(os.environ, MEGAMOE_SESSION_PROFILE_MODE=a.mode,
                   MEGAMOE_SESSION_GENERATED=str(transaction.generated),
                   HI_PYTHON=sys.executable, TILINGKEY_PAR_COMPILE='1', BIN_FILENAME_HASHED='1',
                   ASCEND_CUSTOM_OPP_PATH=str(build/'custom'), ASCEND_PROCESS_LOG_PATH=str(out/'plog'))
        evidence = dict(mode=a.mode, soc='Ascend950PR_957c', precision=a.precision, template_filter=template,
                        started_at=time.time(), commands=[], phases={})
        def run(label, command):
            evidence['commands'].append(command)
            started = time.monotonic()
            with (out/(label+'.log')).open('x') as log:
                transaction.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(transaction.lock_fd,))
            evidence['phases'][label] = time.monotonic()-started
        try:
            run('configure', ['cmake','-S',str(repo),'-B',str(build),'-DASCEND_COMPUTE_UNIT=ascend950',
                             '-DENABLE_HOST_TILING=ON', '-DASCEND_OP_NAME=mega_moe', '-DENABLE_OPS_HOST=ON', '-DENABLE_OPS_KERNEL=ON', '-DFETCHCONTENT_SOURCE_DIR_CANN-CMAKE='+str(shared_cmake), '-DKERNEL_TEMPLATE_INPUT='+template])
            run('prepare', ['cmake','--build',str(build),'-j1','--target','generate_compile_cmd',
                            'common_src_copy','mega_moe_apt_ascend950_src_copy',
                            'mega_moe_apt_ascend950_py_copy','generate_ops_info'])
            recipes = list((build/'binary/ascend950').rglob('MegaMoe-mega_moe_apt-*.sh'))
            if a.topk_dtype=='bfloat16':
                recipe, = [x for x in recipes if parameter+'_param.json' in x.read_text()]
            else:
                matches=[]
                for candidate in recipes:
                    for name in set(re.findall(r'(MegaMoe_[0-9a-f]+)_param\.json',candidate.read_text())):
                        params=list((build/'binary/ascend950').rglob(name+'_param.json'))
                        if len(params)!=1:continue
                        raw=json.loads(params[0].read_text())
                        def named_topk(v):
                            if isinstance(v,dict):
                                if v.get('name')=='topk_weights':yield v
                                for item in v.values():yield from named_topk(item)
                            elif isinstance(v,list):
                                for item in v:yield from named_topk(item)
                        found=list(named_topk(raw))
                        reference=list((build/'binary/ascend950').rglob(parameter+'_param.json'))
                        if not found or len(reference)!=1 or not all(str(v.get('dtype','')).lower() in ('float32','float') for v in found):continue
                        base=json.loads(reference[0].read_text())
                        def normalize(v):
                            if isinstance(v,dict):
                                return {k:('TOPK_DTYPE' if k=='dtype' and v.get('name')=='topk_weights' else normalize(x))
                                        for k,x in v.items() if k not in ('bin_filename','binFileName','bin_filename_hash')}
                            if isinstance(v,list):return [normalize(x) for x in v]
                            return v
                        if normalize(raw)==normalize(base):matches.append((candidate,name,params[0]))
                if len(matches)!=1:raise ValueError('Expected unique official FP32 topk recipe')
                recipe,parameter,selected_params=matches[0]
            script = recipe.read_text().replace('Ascend950PR_9599', 'Ascend950PR_957c')
            script = 'set -o pipefail\n' + script
            for key, name in [('PID_FILE','compile_pids.txt'),('STOP_FILE','compile_stop.flag')]:
                script = re.sub(r'^'+key+r'=.*$', key+'="'+str(out/name)+'"', script, flags=re.M)
            selected = out/'selected_compile.sh'
            selected.write_text(script)
            source = build/'binary/ascend950/src/mega_moe_apt/MegaMoe.py'
            # Fresh official src_copy retains arch35; the generated recipe's
            # legacy source search does not descend into that architecture folder.
            kernel_source = source.parent/'op_kernel/arch35/mega_moe_apt.cpp'
            if not kernel_source.is_file():
                raise ValueError('Missing copied arch35 kernel source: '+str(kernel_source))
            env['BUILD_KERNEL_SRC'] = str(kernel_source)
            evidence['official_script'] = dict(path=str(recipe), sha256=sha(recipe))
            evidence['selected_script_sha256'] = sha(selected)
            evidence['sources'] = transaction.record['compile_inputs_sha256']
            evidence['transaction_id'] = transaction.record['attempt_id']
            from build_binding import begin, finish
            binding = begin(source, transaction.generated, out, a.mode, a.precision, repo/'megamoe_profile/include/profiler_core.h')
            transaction.watch_inputs([build/'binary/ascend950/src', recipe, selected])
            run('compile', ['bash',str(selected),str(source),str(out/'bin')])
            finish(binding, out/'bin'/(parameter+'.o'), out/'bin'/(parameter+'.json'),
                   out/'profiler_build_contract.json', 0)
            metadata = json.loads((out/'bin'/(parameter+'.json')).read_text())
            keys = [x['tilingKey'] for x in metadata['kernelList']]
            if sorted(keys) != expected_keys:
                raise RuntimeError('Unexpected formal template keys: '+repr(keys))
            evidence.update(status='pass', keys=keys,
                            artifacts={x.name:sha(x) for x in (out/'bin').iterdir() if x.is_file()})
        except BaseException as exc:
            evidence.update(status='failed', error=repr(exc))
            raise
        finally:
            evidence['ended_at'] = time.time()
            (out/'build_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__ == '__main__':
    main()
