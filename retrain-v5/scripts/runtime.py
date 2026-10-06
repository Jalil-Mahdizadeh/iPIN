"""Verify a production release and any existing run before starting its worker."""
import json
import sys
from pathlib import Path
from contracts import RUNS, core_hashes, sha256
from release import verify_release

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retraining-v5')


def verify_run(release, name, verify_assets=True):
    assert name in RUNS
    config = release / 'configs' / (name + '.json')
    manifest = verify_release(ROOT, config, release / 'code', verify_assets=verify_assets)
    out = ROOT / 'runs' / name
    if (out / 'contract.json').exists():
        old = json.loads((out / 'contract.json').read_text())
        assert old['qualification'] is False and old['tiny'] is False
        assert old['configuration'] == json.loads(config.read_text()), 'Resume configuration mismatch'
        assert old['code'] == core_hashes(release / 'code'), 'Resume code mismatch'
        assert old['data_manifest_sha256'] == sha256(ROOT / 'data/prepared/manifest.json')
        assert old['initialization_manifest_sha256'] == sha256(ROOT / 'provenance/downloads.json')
    return manifest


if __name__ == '__main__':
    release, name = Path(sys.argv[1]).resolve(), sys.argv[2]
    verify_run(release, name)
    print(json.dumps({'event': 'v5_release_verified', 'run': name, 'automatic_benchmark': False}), flush=True)
