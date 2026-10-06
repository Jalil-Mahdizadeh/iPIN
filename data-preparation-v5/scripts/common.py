import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.part')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True)+'\n').encode())

def read_json(path):
    return json.loads(Path(path).read_text())

def write_csv(path, fields, rows, delimiter=','):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.part')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def fasta_read(path):
    seqs = {}; name = None; parts = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                seq = ''.join(parts)
                assert name not in seqs or seqs[name] == seq, name
                seqs[name] = seq
            name = line[1:].split()[0]
            if '|' in name:
                name = name.split('|')[1]
            parts = []
        elif line.strip(): parts.append(line.strip())
    if name is not None:
        seq = ''.join(parts)
        assert name not in seqs or seqs[name] == seq, name
        seqs[name] = seq
    return seqs

def fasta_write(path, seqs):
    atomic_bytes(path, ''.join(f'>{p}\n{seqs[p]}\n' for p in sorted(seqs)).encode())

def mark(stage, status, **kwargs):
    obj = dict(stage=stage,status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    write_json(ROOT/'work'/f'{stage}.status.json',obj)
    print(json.dumps(obj),flush=True)

def pair(a,b):
    return (a,b) if a <= b else (b,a)
