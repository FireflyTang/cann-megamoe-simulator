"""Serialize all mutable official-build state until evidence has been published.

Fallback contract: this is deliberately a global, cooperative lock. Different
output paths and tiling keys do not isolate repo/build or FetchContent trees.
Only callers using this transaction participate; never run a legacy build in
parallel. This module performs no hardware work and never hashes captured data.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

DEFAULT_LOCK = Path('/home/t00826093/.cache/megamoe-build/official-build.lock')


def source_state(roots):
    """Metadata guard against edits during a build, not a content identity."""
    state = {}
    for root in roots:
        root = Path(root).resolve()
        paths = [root] if root.is_file() else sorted(root.rglob('*'))
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                s = path.stat()
                state[str(path)] = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    return state


def source_identity(roots):
    """Once-per-attempt compile input freeze; never call on runtime/raw trees."""
    result = {}
    for name in source_state(roots):
        with open(name, 'rb') as stream:
            result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


class BuildTransaction:
    def __init__(self, repo, output, *, generated=None, inputs=(), lock_path=DEFAULT_LOCK):
        self.repo = Path(repo).resolve()
        self.output = Path(output).resolve()
        self.generated = Path(generated).resolve() if generated is not None else None
        self.inputs = [Path(p).resolve() for p in inputs]
        self.lock_path = Path(lock_path).resolve()
        self._lock = None
        self._old_env = {}
        self._watched = []
        self._active_groups = {}
        self.record = {'schema': 'megamoe_build_transaction_v1', 'attempt_id': uuid.uuid4().hex,
                       'policy': 'global_cooperative_lock_configure_through_evidence',
                       'repo': str(self.repo), 'output': str(self.output),
                       'lock_path': str(self.lock_path), 'pid': os.getpid(),
                       'status': 'pending', 'requested_at': time.time()}

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.lock_path.open('a+')
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX)
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            if self.generated is not None:
                self.generated.mkdir(parents=True, exist_ok=False)
            self.record.update(acquired_at=time.time(), status='running')
            roots = self.inputs
            self._input_state = source_state(roots)
            self.record['compile_inputs_sha256'] = source_identity(roots)
            self.record['input_guard'] = 'device_inode_size_mtime_ctime_before_after'
            self.record['generated_output'] = str(self.generated) if self.generated is not None else None
            # Global CANN/temp caches are not shared by concurrent new attempts.
            values = {'TMPDIR': self.output/'tmp', 'TMP': self.output/'tmp',
                      'TEMP': self.output/'tmp', 'XDG_CACHE_HOME': self.output/'cache',
                      'ASCEND_CACHE_PATH': self.output/'ascend-cache',
                      'ASCEND_WORK_PATH': self.output/'ascend-work',
                      'PYTHONPYCACHEPREFIX': self.output/'pycache',
                      'CCACHE_DISABLE': '1'}
            for key, val in values.items():
                self._old_env[key] = os.environ.get(key)
                os.environ[key] = str(val)
                if key != 'CCACHE_DISABLE':
                    Path(val).mkdir(parents=True, exist_ok=True)
            self.record['environment'] = {k: os.environ[k] for k in values}
            self._write()
            return self
        except BaseException:
            self._restore()
            raise

    def __exit__(self, kind, error, traceback):
        failure = error
        try:
            if failure is None:
                roots = self.inputs
                if source_state(roots) != self._input_state:
                    failure = RuntimeError('Compile inputs changed during transaction; output rejected')
                elif any(source_state(roots) != old for roots, old in self._watched):
                    failure = RuntimeError('Generated build intermediates changed during compilation; output rejected')
            self.record.update(status='failed' if failure else 'pass', finished_at=time.time())
            if failure:
                self.record['error'] = repr(failure)
                evidence = self.output/'build_evidence.json'
                if evidence.exists():
                    try:
                        old = json.loads(evidence.read_text())
                    except (ValueError, OSError):
                        old = {}
                    old.update(status='failed', transaction_error=repr(failure))
                    evidence.write_text(json.dumps(old, indent=2)+'\n')
            self._write()
        finally:
            self._restore()
        if error is None and failure is not None:
            raise failure
        return False

    @property
    def lock_fd(self):
        if self._lock is None:
            raise RuntimeError('Build lock is not held')
        return self._lock.fileno()

    def run(self, command, *, check=True, **kwargs):
        """Run a non-detaching command; retain the lock until its group stops."""
        if 'start_new_session' in kwargs:
            raise ValueError('Build commands must use a private process group')
        passed = tuple(kwargs.pop('pass_fds', ()))
        process = subprocess.Popen(command, start_new_session=True,
                                   pass_fds=tuple(set((*passed, self.lock_fd))), **kwargs)
        self._active_groups[process.pid] = process
        try:
            code = process.wait()
            # A wrapper returning without waiting for compiler descendants is
            # not a completed build, even if its own exit code was zero.
            remaining = self._live_group_members(process.pid)
            if remaining:
                raise RuntimeError('Build command left active descendants: '+repr(remaining))
            if check and code:
                raise subprocess.CalledProcessError(code, command)
            return subprocess.CompletedProcess(command, code)
        finally:
            self._drain_group(process.pid)

    @staticmethod
    def _live_group_members(group):
        members = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc/'stat').read_text().rsplit(') ', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == group and fields[0] not in ('Z', 'X'):
                members.append(int(proc.name))
        return members

    def _drain_group(self, group):
        process = self._active_groups.get(group)
        if process is None:
            return
        # Do not let a second interrupt unlock while the first is cleaning up.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            if self._live_group_members(group):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            # Reaping our immediate child is insufficient: descendants may
            # close inherited FDs. Zombies cannot write shared build state.
            while self._live_group_members(group):
                time.sleep(0.02)
            del self._active_groups[group]
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def watch_inputs(self, roots):
        roots = [Path(p).resolve() for p in roots]
        self._watched.append((roots, source_state(roots)))
        self.record.setdefault('generated_compile_inputs_sha256', {}).update(source_identity(roots))

    def _write(self):
        target = self.output/'build_transaction.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2)+'\n')
        temporary.replace(target)

    def _restore(self):
        for group in list(self._active_groups):
            self._drain_group(group)
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None


def require_complete_transaction(output):
    """Call before publishing a new build artifact into an execution registry."""
    output = Path(output).resolve()
    value = json.loads((output/'build_transaction.json').read_text())
    evidence = json.loads((output/'build_evidence.json').read_text())
    if value.get('status') != 'pass' or value.get('output') != str(output):
        raise RuntimeError('Build transaction incomplete or mismatched')
    if evidence.get('transaction_id') != value.get('attempt_id'):
        raise RuntimeError('Build evidence belongs to a different attempt')
    if evidence.get('status') != 'pass':
        raise RuntimeError('Build evidence is not successful')
    return value
