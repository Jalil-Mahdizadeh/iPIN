"""Two actual-size, four-GPU exact-resume checks; never production training."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retraining-v5')
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from contracts import RUNS, core_hashes, sha256


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_GPUS_ON_NODE') == '4'
    assert CODE.is_relative_to(ROOT / 'qualification')
    reports, reused = [], []
    for name in RUNS:
        cfg = json.loads((CODE.parent / 'configs' / (name + '.json')).read_text())
        cfg.update(global_pairs_per_update=9, warmup_updates=1, validate_every_updates=2,
                   checkpoint_every_updates=1, log_every_updates=1, train_max_pairs=1, validation_updates=[2,4])
        config = ROOT / 'qualification' / (name + '-resume-config.json')
        report = ROOT / 'qualification' / (name + '-resume-comparison.json')
        if config.exists():
            assert json.loads(config.read_text()) == cfg
        else:
            write(config, cfg)
        if report.exists():
            previous = json.loads(report.read_text())
            assert previous['passed'] and previous['bitwise_identical']
            for suffix in ['-full', '-resumed']:
                old = ROOT / 'qualification' / (name + suffix)
                contract = json.loads((old / 'contract.json').read_text())
                assert contract['code'] == core_hashes(CODE)
                assert contract['configuration'] == cfg and contract['world_size'] == 4
                assert contract['data_manifest_sha256'] == sha256(ROOT / 'data/prepared/manifest.json')
                done = json.loads((old / 'completed.json').read_text())
                assert done['state']['update'] == 4 and not done['state']['pending_validation']
            reused.append(name); reports.append(str(report.relative_to(ROOT)))
            print('Preserving already passed exact-resume check: ' + name, flush=True)
            continue
        def run(suffix, stop=None):
            out = ROOT / 'qualification' / (name + suffix)
            command = ['bash', str(CODE / 'container.sh'), cfg['backbone'],
                'python', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4',
                str(CODE / 'train.py'), '--config', str(config), '--output', str(out),
                '--qualification', '--max-updates', '4', '--train-limit', '11', '--val-limit', '8',
                '--qualification-max-tokens', '384']
            if stop is not None: command += ['--stop-after-updates', str(stop)]
            subprocess.run(command, check=True)
        run('-full')
        run('-resumed', 2)
        stopped = json.loads((ROOT / 'qualification' / (name + '-resumed') / 'stopped.json').read_text())
        assert stopped['state']['pending_validation'] and stopped['state']['update'] == 2
        run('-resumed')
        subprocess.run(['bash', str(CODE / 'container.sh'), cfg['backbone'], 'python', '-B',
            str(CODE / 'compare_resume.py'), str(ROOT / 'qualification' / (name + '-full')),
            str(ROOT / 'qualification' / (name + '-resumed')), '--report', str(report)], check=True)
        result = json.loads(report.read_text())
        assert result['passed'] and result['bitwise_identical']
        reports.append(str(report.relative_to(ROOT)))
    value = {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'world_size': 4, 'actual_size_backbones': ['esm2-650m', 'esmc-600m'],
        'conditions': ['esm2', 'esmc'], 'production_training': False,
        'updates_per_trajectory': 4, 'stop_at': 2, 'pending_validation_recovered': True,
        'global_pairs_per_update': 9, 'short_input_max_tokens': 384,
        'reused_previously_passed_identical_core_checks': reused,
        'scope': 'Actual full-size models: uneven accumulation and bitwise model/optimizer/RNG/sampler/selection/validation-prediction resume on the tested GH200 runtime.',
        'code': core_hashes(CODE), 'comparison_reports': reports,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                         [Path(__file__), CODE / 'container.sh', CODE / 'compare_resume.py']}}
    value['source_sha256'].update({name: sha256(ROOT / name) for name in reports})
    write(ROOT / 'qualification/ddp-resume.json', value)
    print('Both full-size v5 models passed bitwise four-GPU resume.', flush=True)


if __name__ == '__main__': main()
