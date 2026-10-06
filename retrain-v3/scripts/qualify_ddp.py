"""Bounded, disposable v3 clean-BCE qualification on four actual GPUs."""
import json
import os
import subprocess
import sys
from pathlib import Path
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use the bounded qualification allocation'
    qual = ROOT / 'qualification'
    cfg = json.loads((ROOT / 'configs/qualification.json').read_text())
    assert cfg['classification_corruption'] is False and cfg['mlm_weight'] == 0
    assert not (qual / 'ddp-clean-run-full').exists(), 'Keep existing qualification evidence'

    def run(name, config, stop=None, tiny=False, updates=4):
        path = qual / (name.split('-run-')[0] + '-config.json')
        if path.exists():
            assert json.loads(path.read_text()) == config
        else:
            atomic_json(path, config)
        args = ['torchrun', '--standalone', '--nnodes=1', '--nproc-per-node=4',
                str(ROOT / 'scripts/train.py'), '--qualification', '--config', str(path),
                '--output', str(qual / name), '--max-updates', str(updates),
                '--train-limit', '67', '--val-limit', '8', '--qualification-max-tokens', '384']
        if stop is not None:
            args += ['--stop-after-updates', str(stop)]
        if tiny:
            args += ['--tiny']
        subprocess.run(args, check=True)

    # Production global batch and both orientations; cross a sampler cycle.
    # Stop at a scheduled validation boundary and replay that pending validation.
    left, right = 'ddp-clean-run-full', 'ddp-clean-run-resumed'
    run(left, cfg)
    run(right, cfg, stop=2)
    stopped = json.loads((qual / right / 'stopped.json').read_text())
    assert stopped['state']['pending_validation'] and stopped['state']['update'] == 2
    run(right, cfg)
    report = qual / 'ddp-clean-comparison.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/compare_resume.py'),
                    str(qual / left), str(qual / right), '--report', str(report)], check=True)
    # Cover both staged head optimizers with uneven accumulation and a dummy rank.
    for head, pairs in [('cls_mlp', 9), ('residue_mean', 3)]:
        trial = {**cfg, 'readout': head, 'global_pairs_per_update': pairs, 'train_max_pairs': 1}
        name = f'ddp-{head}-run-tiny'
        run(name, trial, tiny=True, updates=2)
        done = json.loads((qual / name / 'completed.json').read_text())
        assert len(set(done['model_hashes_by_rank'])) == 1
    atomic_json(qual / 'ddp.json', {
        'passed': True, 'production_training': False, 'job_id': os.environ['SLURM_JOB_ID'],
        'world_size': 4, 'full_650m_clean_resume': True, 'learning_rate': cfg['learning_rate'],
        'global_pairs_per_update': 64, 'uninterrupted_updates': 4, 'stop_after_update': 2,
        'training_rows': 67, 'validation_rows': 8, 'maximum_pair_tokens': 384,
        'scope': 'Full 650M short-input clean-BCE restart; both residual heads on tiny ESM under DDP. Not scientific validation.',
        'interrupted_validation_recovered': True, 'head_ddp_uneven_and_dummy_ranks': True,
        'code': {n: sha256(ROOT / 'scripts' / n) for n in ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']},
        'data_manifest_sha256': sha256(ROOT / 'data/prepared/manifest.json'),
        'source_sha256': {'scripts/qualify_ddp.py': sha256(Path(__file__)),
                          'configs/qualification.json': sha256(ROOT / 'configs/qualification.json')}
    })
    print('V3 four-GPU qualification passed; zero production runs.', flush=True)


if __name__ == '__main__':
    main()
