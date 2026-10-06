"""Resumable independent GPU shards; no fitting or test-based model selection."""
import datetime
import fcntl
import gc
import json
import os
import socket
import time

import numpy as np
import torch

from common import ROOT, PairData, atomic_json, configure, load_model, predict, prediction_fingerprint, sha256, require_continue


def main():
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    lock = (ROOT / 'logs' / f'inference-rank-{rank:02d}.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = configure(local)
    cfg = json.loads((ROOT / 'config.json').read_text())
    assert world == cfg['world_size'], 'Shard count is frozen for resumption'
    qualification = json.loads((ROOT / 'provenance/qualification.json').read_text())
    assert qualification['passed']
    assert qualification['prediction_fingerprint'] == prediction_fingerprint()
    selection = json.loads((ROOT / 'provenance/selection.json').read_text())
    for name, digest in selection['data']['sha256'].items():
        assert sha256(ROOT / 'data' / name) == digest, name
    contract = prediction_fingerprint()
    started = time.monotonic()
    task_specs = [(name, split) for name in cfg['models'] for split in cfg['splits_by_model'][name]]
    current = None
    model = None
    for name, split in task_specs:
        require_continue()
        data = PairData(ROOT / 'data', split)
        indices = np.arange(rank, len(data), world, dtype=np.int64)
        indices = indices[np.argsort(data.lengths[indices], kind='stable')]
        task = f'{name}-{split}'
        directory = ROOT / 'predictions' / task
        directory.mkdir(exist_ok=True)
        done_path = directory / f'rank-{rank:02d}.done.json'
        if done_path.exists():
            previous = json.loads(done_path.read_text())
            assert previous['fingerprint'] == contract
            for item in previous['chunks']:
                assert sha256(directory / item['file']) == item['sha256']
            print(json.dumps({'event': 'verified_complete_task', 'task': task, 'rank': rank}), flush=True)
            continue
        if current != name:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            model = load_model(name, device)
            current = name
            print(json.dumps({'event': 'model_loaded', 'name': name, 'rank': rank,
                              'sha256': selection['models'][name]['sha256']}), flush=True)
        task_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        chunks = []
        for number, begin in enumerate(range(0, len(indices), cfg['commit_rows'])):
            require_continue()
            rows = indices[begin:begin + cfg['commit_rows']]
            destination = directory / f'rank-{rank:02d}-chunk-{number:04d}.npz'
            meta_path = destination.with_suffix('.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['fingerprint'] == contract and sha256(destination) == meta['sha256']
                with np.load(destination, allow_pickle=False) as saved:
                    old = saved['predictions']
                assert old.shape == (len(rows), 4) and np.isfinite(old).all()
                assert np.array_equal(old[:, 0], rows) and np.array_equal(old[:, 1], data.rows[rows, 2])
            else:
                predictions = []
                for micro in data.microbatches(rows, cfg['token_budget'], cfg['max_pairs']):
                    require_continue()
                    predictions.append(predict(model, data, micro, device))
                predictions = np.concatenate(predictions)
                assert np.array_equal(predictions[:, 0], rows)
                temporary = destination.with_suffix('.partial')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, predictions=predictions)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                meta = {'file': destination.name, 'sha256': sha256(destination), 'rows': len(rows),
                        'fingerprint': contract, 'rank': rank, 'world_size': world}
                atomic_json(meta_path, meta)
            chunks.append(meta)
            progress = {'event': 'progress', 'task': task, 'rank': rank,
                        'rows_done': min(begin + cfg['commit_rows'], len(indices)), 'rows_total': len(indices),
                        'elapsed_seconds': time.monotonic() - task_started,
                        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            atomic_json(ROOT / 'logs' / f'progress-rank-{rank:02d}.json', progress)
            print(json.dumps(progress), flush=True)
        atomic_json(done_path, {'task': task, 'rank': rank, 'world_size': world,
            'fingerprint': contract, 'rows': len(indices), 'chunks': chunks,
            'elapsed_seconds': time.monotonic() - task_started,
            'gpu_peak_bytes': torch.cuda.max_memory_allocated(device), 'hostname': socket.gethostname(),
            'gpu_uuid': str(torch.cuda.get_device_properties(device).uuid),
            'job_id': os.environ.get('SLURM_JOB_ID'), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'precision': 'FP32 weights; BF16 autocast; TF32 disabled'})
        print(json.dumps({'event': 'task_complete', 'task': task, 'rank': rank}), flush=True)
    print(json.dumps({'event': 'worker_complete', 'rank': rank,
                      'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
