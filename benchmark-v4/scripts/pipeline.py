"""Run only the selected C0 checkpoint; reuse every earlier model's predictions."""
import json
import datetime
import os
import subprocess
import sys
from common import ROOT, atomic_json, require_continue, stop_requested


def main():
    import prepare_selected
    prepare_selected.main()
    if (ROOT / 'completed.json').exists():
        from write_report import finish_closeout
        finish_closeout(json.loads((ROOT / 'completed.json').read_text()))
        print('V4 benchmark already completed; verified final artifacts.', flush=True)
        return
    require_continue()
    import qualify
    qualify.main()
    require_continue()
    execution = json.loads((ROOT / 'provenance/execution-plan.json').read_text())
    cfg = json.loads((ROOT / 'config.json').read_text())
    assert cfg['world_size'] == execution['logical_shards'] == 4
    import torch
    assert torch.cuda.device_count() == execution['physical_gpus']
    if execution['mode'] == 'four_fixed_shards_parallel_four_gpus':
        assert execution['physical_gpus'] == 4
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nnodes=1', '--nproc-per-node=4', str(ROOT / 'scripts/infer.py')], check=True)
    else:
        assert execution['mode'] == 'four_fixed_shards_sequential_single_gpu'
        assert execution['physical_gpus'] == 1
        run_sequential(cfg)
    require_continue()
    import analyze
    analyze.main()
    require_continue()
    import protein_macro
    protein_macro.main()
    require_continue()
    import write_report
    write_report.main()


def run_sequential(cfg):
    timing_path = ROOT / 'provenance/shard-execution.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    for rank in range(cfg['world_size']):
        require_continue()
        environment = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(cfg['world_size']), LOCAL_RANK='0')
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(json.dumps({'event': 'sequential_shard_start', 'rank': rank, 'time_utc': started}), flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/infer.py')], env=environment, check=True)
        timings.append({'rank': rank, 'started_at_utc': started,
            'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'job_id': os.environ.get('SLURM_JOB_ID')})
        atomic_json(timing_path, timings)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError:
        if stop_requested(): raise SystemExit(75)
        raise
