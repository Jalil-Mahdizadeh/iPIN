"""Verify the single final release and pinned automatic benchmark inputs."""
import json
import sys
from pathlib import Path
from launch import sha, verify

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')
BENCH = ROOT.parent / 'benchmark-v3'
RUN = 'clean-residue-mean-official-seed2'


def verify_final(release):
    manifest = verify(release)
    assert manifest['stage'] == 'final-single-run-closeout'
    assert manifest['enabled_runs'] == [RUN]
    for name, digest in manifest['benchmark_input_sha256'].items():
        assert sha(BENCH / name) == digest, ('Frozen final benchmark input changed', name)
    cfg = json.loads((release / 'configs' / f'{RUN}.json').read_text())
    plan = json.loads((BENCH / 'provenance/plan.json').read_text())
    assert cfg == plan['training_configuration']
    assert cfg['partition'] == 'official' and cfg['readout'] == 'residue_mean'
    assert cfg['learning_rate'] == 2e-5 and cfg['total_updates'] == 12745
    assert cfg['warmup_updates'] == 2000 and cfg['seed'] == 2
    assert cfg['world_size'] == 4 and cfg['train_cap_residues'] is None
    assert sha(release / 'code/model.py') == sha(BENCH / 'scripts/frozen_model.py')
    assert sha(release / 'code/data.py') == sha(BENCH / 'scripts/training_data.py')
    return manifest


if __name__ == '__main__':
    m = verify_final(Path(sys.argv[1]).resolve())
    print(json.dumps({'event': 'final_release_verified', 'runs': m['enabled_runs'],
                      'benchmark_files': len(m['benchmark_input_sha256'])}), flush=True)
