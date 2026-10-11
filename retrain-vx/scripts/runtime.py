"""Standard-library batch orchestration helpers and allocation checks."""
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def read(p):
    return json.loads(Path(p).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def verify_files():
    path = ROOT / 'provenance/freeze.json'; frozen = read(path)
    for name, digest in frozen['files'].items():
        if sha(PROJECT / name) != digest:
            raise ValueError('Frozen file changed: ' + name)
    return sha(path)


def topology(records, job_id, fingerprint):
    if len(records) != 8 or sorted(r['rank'] for r in records) != list(range(8)):
        raise ValueError('Exactly eight unique ranks required')
    if any(r['world'] != 8 or r['job_id'] != job_id or r['fingerprint'] != fingerprint for r in records):
        raise ValueError('GPU worker belongs to another job/configuration')
    if len({r['gpu_uuid'] for r in records}) != 8:
        raise ValueError('GPU oversubscription or duplicate devices')
    hosts = {r['host'] for r in records}
    if len(hosts) != 2 or any(sorted(r['local_rank'] for r in records if r['host'] == h) != [0, 1, 2, 3] for h in hosts):
        raise ValueError('Must use exactly two nodes with four GPU workers each')
    if any('GH200' not in r['gpu_name'] for r in records):
        raise ValueError('GPU hardware differs from qualified GH200 implementation')


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=30)


def run_command(command, deadline, log_path, check=None):
    with Path(log_path).open('ab', buffering=0) as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while p.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('Bounded runtime deadline reached')
                if check:
                    check(p)
                time.sleep(2)
            if p.returncode:
                raise RuntimeError(f'Command failed ({p.returncode}); inspect {log_path}')
        except BaseException:
            stop_process(p); raise


def owned_bytes():
    return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
