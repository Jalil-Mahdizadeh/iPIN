"""Verify the single final training release; benchmarking is separate."""
import json
import sys
from pathlib import Path
from launch import sha, verify

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')
RUN = 'clean-residue-mean-official-seed2'


def verify_final(release):
    manifest = verify(release)
    assert manifest['stage'] == 'final-single-run-training'
    assert manifest['enabled_runs'] == [RUN]
    assert manifest['automatic_final_benchmark'] is False
    cfg = json.loads((release / 'configs' / f'{RUN}.json').read_text())
    assert cfg['partition'] == 'official' and cfg['readout'] == 'residue_mean'
    assert cfg['learning_rate'] == 2e-5 and cfg['total_updates'] == 12745
    assert cfg['warmup_updates'] == 2000 and cfg['seed'] == 2
    assert cfg['world_size'] == 4 and cfg['train_cap_residues'] is None
    return manifest


if __name__ == '__main__':
    m = verify_final(Path(sys.argv[1]).resolve())
    print(json.dumps({'event': 'final_release_verified', 'runs': m['enabled_runs'],
                      'automatic_benchmark': False}), flush=True)
