"""Build the private bridge using the official single-op host recipe and objects."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from build_transaction import BuildTransaction
from execution_contract import execution_evidence, require_local_build


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--official-build', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--precision', choices=['w8a8','w4a8'], default='w8a8')
    p.add_argument('--topk-dtype', choices=['bfloat16','float32'], default='bfloat16')
    args = p.parse_args()
    require_local_build('build_host')
    repo = args.repo.resolve()
    build = args.official_build.resolve()
    out = args.output.resolve()
    link_path=build/'CMakeFiles/ophost_transformer.dir/link.txt'
    if not link_path.is_file():link_path=build/'CMakeFiles/cust_opmaster.dir/link.txt'
    inputs = [repo/'mc2/mega_moe', repo/'megamoe_profile/session/host_bridge.cpp',
              repo/'megamoe_profile/include', build/'compile_commands.json',link_path]
    with BuildTransaction(repo, out, inputs=inputs) as transaction:
        entries = json.loads((build/'compile_commands.json').read_text())
        recipe, = [e for e in entries if e['file'].endswith('/arch35/mega_moe_tiling.cpp')]
        command = shlex.split(recipe['command'])
        obj = out/'host_bridge.o'
        command[command.index('-o')+1] = str(obj)
        command[command.index('-c')+1] = str(repo/'megamoe_profile/session/host_bridge.cpp')
        command.insert(1, '-I'+str(repo))
        command += ['-DMEGAMOE_PROFILE_ABI=1', '-I'+str(repo), '-MMD', '-MF', str(out/'host_bridge.d')]
        link = shlex.split(link_path.read_text())
        selected = [i for i, arg in enumerate(link) if arg.endswith('/arch35/mega_moe_tiling.cpp.o')]
        if len(selected) != 1:
            raise RuntimeError('Expected one formal arch35 tiling object')
        link[selected[0]] = str(obj)
        library = out/'libmegamoe_host.so'
        link[link.index('-o')+1] = str(library)
        link = [x.replace('-Wl,-soname,libophost_transformer.so', '-Wl,-soname,libmegamoe_host.so') for x in link]
        link += ['-lplatform', '-lgraph_base', '-lmetadef', '-lexe_graph']
        command += ['-DMEGAMOE_SESSION_W4='+str(int(args.precision == 'w4a8'))]
        command += ['-DMEGAMOE_SESSION_TOPK_FP32='+str(int(args.topk_dtype == 'float32'))]
        link_inputs = [(Path(x) if Path(x).is_absolute() else build/x) for x in link[1:] if x.endswith(('.o','.a','.so'))]
        transaction.watch_inputs([build/'CMakeCache.txt', *(p for p in link_inputs if not p.resolve().is_relative_to(out))])
        record = dict(precision=args.precision,topk_weights_dtype=args.topk_dtype,compiler_command=command, link_command=link,
                      execution=execution_evidence('build_host', 'local'),
                      official_compile_recipe=recipe, started_at=time.time())
        (out/'recipe.json').write_text(json.dumps(record, indent=2)+'\n')
        with (out/'build.log').open('w') as log:
            start = time.monotonic()
            transaction.run(command, cwd=recipe['directory'], stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(transaction.lock_fd,))
            record['compile_seconds'] = time.monotonic()-start
            transaction.run(link, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(transaction.lock_fd,))
        # Exact compiler dependency list, plus every object/archive consumed by the link.
        dependencies = shlex.split((out/'host_bridge.d').read_text().replace('\\\n', ' ').split(':', 1)[1])
        paths = [Path(x) if Path(x).is_absolute() else Path(recipe['directory'])/x for x in dependencies]
        paths += [(Path(x) if Path(x).is_absolute() else build/x) for x in link[1:] if x.endswith(('.o','.a','.so'))]
        record.update(status='pass', finished_at=time.time(), library=str(library), library_sha256=digest(library),
                      dependencies={str(x.resolve()): digest(x) for x in paths if x.is_file()})
        (out/'build_evidence.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'library': str(library), 'sha256': record['library_sha256'],
                          'compile_seconds': record['compile_seconds']}))


if __name__ == '__main__':
    main()
