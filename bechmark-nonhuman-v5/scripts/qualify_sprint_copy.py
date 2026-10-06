"""Check memory-mapped block copying against the previous exact byte-copy loop."""
import datetime
import json
import os
import time
from pathlib import Path
from sprint_hsp_mmap import copy_blocks, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'qualification/sprint-copy'


def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def legacy(raw, dest, blocks):
    with raw.open('rb') as source, dest.open('wb') as target:
        for key, (offset, size, count) in sorted(blocks.items()):
            source.seek(offset)
            while size:
                data = source.read(min(size, 8 * 1024**2))
                assert data
                target.write(data)
                size -= len(data)


def compare(name, raw, blocks):
    a, b = OUT / (name + '-legacy.hsp'), OUT / (name + '-mmap.hsp')
    started = time.monotonic()
    legacy(raw, a, blocks)
    legacy_seconds = time.monotonic() - started
    started = time.monotonic()
    copy_blocks(raw, b, sorted(blocks.items()))
    mmap_seconds = time.monotonic() - started
    assert a.read_bytes() == b.read_bytes()
    return {'case': name, 'source': record(raw), 'legacy': record(a), 'mmap': record(b),
            'blocks': len(blocks), 'identical_bytes': True,
            'legacy_seconds': legacy_seconds, 'mmap_seconds': mmap_seconds}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / 'predictions/sprint/raw.hsp'
    blocks, key = {}, None
    with source_path.open('rb') as source:
        while True:
            offset = source.tell()
            line = source.readline()
            assert line
            if line.startswith(b'>'):
                if key is not None:
                    blocks[key][1] = offset - blocks[key][0]
                if len(blocks) == 10000:
                    break
                fields = line.split()
                assert len(fields) == 4 and fields[2] == b'and'
                key = (fields[1], fields[3])
                assert key not in blocks
                blocks[key] = [offset, 0, 0]
            else:
                assert key is not None
                blocks[key][2] += 1
        source.seek(0)
        raw = OUT / 'real-10000-blocks.hsp'
        raw.write_bytes(source.read(offset))
    cases = [compare('real-10000-blocks', raw, blocks)]
    real_bytes, real_blocks = raw.read_bytes(), sorted(blocks.items())
    # Exercise the legacy loop's >8 MiB branch and differently ordered blocks.
    raw = OUT / 'large-block.hsp'
    header = b'> p00002 and p00003\n'
    a = header + b'0 0 20\n' * 1300000
    b = b'> p00000 and p00000\n0 0 12\n'
    raw.write_bytes(a + b)
    blocks = {(b'p00002', b'p00003'): [0, len(a), 1300000],
              (b'p00000', b'p00000'): [len(a), len(b), 1]}
    cases.append(compare('large-block-and-short-self', raw, blocks))
    # Put alternating real keys on opposite sides of a >8 MiB block. This
    # reproduces the large-file buffer refills absent from a tiny fixture.
    dispersed = OUT / 'dispersed-real-blocks.hsp'
    scattered, offset = {}, 0
    with dispersed.open('wb') as stream:
        for half in [0, 1]:
            if half == 1:
                payload = b'> p99999 and p99999\n' + b'0 0 20\n' * 1300000
                stream.write(payload)
                scattered[(b'p99999', b'p99999')] = [offset, len(payload), 1300000]
                offset += len(payload)
            for key, (old_offset, size, count) in real_blocks[half::2]:
                payload = real_bytes[old_offset:old_offset + size]
                stream.write(payload)
                scattered[key] = [offset, size, count]
                offset += size
    cases.append(compare('dispersed-real-blocks', dispersed, scattered))
    invalid = OUT / 'invalid.tmp'
    rejected = []
    for name, items in [
        ('out_of_bounds', [(('a', 'b'), (0, raw.stat().st_size + 1, 1))]),
        ('incomplete_coverage', [(('a', 'b'), (0, 1, 1))]),
        ('duplicate_key', [(('a', 'b'), (0, 1, 1)), (('a', 'b'), (1, 1, 1))]),
    ]:
        try:
            copy_blocks(raw, invalid, items)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Did not reject ' + name)
    code = [record(ROOT / 'scripts' / n) for n in
            ['sprint_hsp_mmap.py', 'run_sprint_cpu_mmap.py', 'qualify_sprint_copy.py']]
    q = {'passed': True, 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'native_block_copy_identical_bytes': True, 'cases': cases,
         'invalid_inputs_rejected': rejected, 'code': code,
         'model_or_hsp_calculation_changed': False}
    path = OUT / 'qualification.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(q, indent=2) + '\n')
    os.replace(tmp, path)
    print(json.dumps({'passed': True, 'cases': [{k: c[k] for k in
                      ['case', 'blocks', 'identical_bytes', 'legacy_seconds', 'mmap_seconds']}
                      for c in cases], 'invalid_inputs_rejected': rejected}), flush=True)


if __name__ == '__main__':
    main()
