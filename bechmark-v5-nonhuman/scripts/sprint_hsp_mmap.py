"""Copy already validated HSP blocks without buffered random-read amplification."""
import hashlib
import json
import mmap
import os
import time
from pathlib import Path


def digest(path, length=None):
    h = hashlib.sha256()
    remaining = Path(path).stat().st_size if length is None else length
    with Path(path).open('rb') as stream:
        while remaining:
            block = stream.read(min(remaining, 16 * 1024**2))
            assert block, 'Unexpected end of file'
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def copy_blocks(raw, destination, ordered):
    """Keep every byte within each block and the caller's existing key order."""
    raw, destination = Path(raw), Path(destination)
    assert raw.resolve() != destination.resolve()
    source_bytes = raw.stat().st_size
    started = time.monotonic()
    written = blocks = 0
    next_report = 1024**3
    with raw.open('rb') as source, destination.open('wb') as target:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            view = memoryview(mapping)
            try:
                previous = None
                for key, (offset, size, count) in ordered:
                    assert previous is None or previous < key
                    assert 0 <= offset and 0 < size <= source_bytes - offset and count > 0
                    assert target.write(view[offset:offset + size]) == size
                    previous = key
                    written += size
                    blocks += 1
                    if written >= next_report:
                        print({'stage': 'copy_hsp_mmap', 'bytes': written,
                               'total_bytes': source_bytes, 'blocks': blocks,
                               'seconds': time.monotonic() - started}, flush=True)
                        next_report = written + 1024**3
            finally:
                view.release()
        target.flush()
        os.fsync(target.fileno())
    assert written == source_bytes == raw.stat().st_size == destination.stat().st_size
    print({'stage': 'copy_hsp_mmap_complete', 'bytes': written, 'blocks': blocks,
           'seconds': time.monotonic() - started}, flush=True)


def verify_qualification(root):
    q = json.loads((root / 'qualification/sprint-copy/qualification.json').read_text())
    assert q['passed'] and q['native_block_copy_identical_bytes']
    for item in q['code']:
        assert digest(item['path']) == item['sha256']


def verify_previous_prefix(root, candidate):
    path = root / 'provenance/sprint-copy-recovery.json'
    if not path.exists():
        return
    item = json.loads(path.read_text())['preserved_previous_prefix']
    assert Path(item['path']).stat().st_size == item['bytes']
    assert digest(item['path']) == item['sha256']
    assert digest(candidate, item['bytes']) == item['sha256'], 'Previous canonical prefix differs'
    print({'stage': 'previous_canonical_prefix_verified', 'bytes': item['bytes'],
           'identical': True}, flush=True)
