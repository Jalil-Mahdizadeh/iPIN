"""Full-size readout optimizer/resume checks on short inputs, never production."""
import json
import os
import subprocess
import sys
from pathlib import Path
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
CORE = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']


def main():
    assert os.environ.get('SLURM_JOB_ID')
    reports = []
    for head, arm in [('cls_mlp', 'cls-mlp'), ('residue_mean', 'residue-mean')]:
        prefix = f'readout-{arm}'
        cfg = json.loads((ROOT / 'configs' / f'clean-{arm}-lr2e-5-fold0-seed2.json').read_text())
        cfg.update(global_pairs_per_update=9, warmup_updates=1, validate_every_updates=2,
                   checkpoint_every_updates=1, log_every_updates=1, train_max_pairs=1)
        config = ROOT / 'qualification' / (prefix + '-config.json')
        assert not config.exists()
        atomic_json(config, cfg)
        def run(suffix, stop=None):
            out = ROOT / 'qualification' / (prefix + suffix)
            args = ['torchrun', '--standalone', '--nnodes=1', '--nproc-per-node=4',
                    str(ROOT / 'scripts/train.py'), '--config', str(config), '--output', str(out),
                    '--qualification', '--max-updates', '4', '--train-limit', '11', '--val-limit', '8',
                    '--qualification-max-tokens', '384']
            if stop is not None:
                args += ['--stop-after-updates', str(stop)]
            subprocess.run(args, check=True)
        run('-full')
        run('-resumed', 2)
        stopped = json.loads((ROOT / 'qualification' / (prefix + '-resumed') / 'stopped.json').read_text())
        assert stopped['state']['pending_validation'] and stopped['state']['update'] == 2
        run('-resumed')
        report = ROOT / 'qualification' / (prefix + '-comparison.json')
        subprocess.run([sys.executable, str(ROOT / 'scripts/compare_resume.py'),
                        str(ROOT / 'qualification' / (prefix + '-full')),
                        str(ROOT / 'qualification' / (prefix + '-resumed')), '--report', str(report)], check=True)
        reports.append(str(report.relative_to(ROOT)))
    atomic_json(ROOT / 'qualification/readout-ddp.json', {
        'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'production_training': False,
        'full_650m_heads': ['cls_mlp', 'residue_mean'], 'world_size': 4,
        'updates': 4, 'stop_at': 2, 'pending_validation_recovered': True,
        'global_pairs_per_update': 9, 'short_input_maximum_tokens': 384,
        'scope': 'Both actual-size clean residual heads: four-GPU uneven accumulation and bitwise interrupted/resumed state. Disposable short-input qualification.',
        'comparison_reports': reports,
        'code': {n: sha256(ROOT / 'scripts' / n) for n in CORE},
        'data_manifest_sha256': sha256(ROOT / 'data/prepared/manifest.json'),
        'source_sha256': {'scripts/qualify_readout_ddp.py': sha256(Path(__file__))}})
    print('Both full-size readout heads passed exact four-GPU resume.', flush=True)


if __name__ == '__main__':
    main()
