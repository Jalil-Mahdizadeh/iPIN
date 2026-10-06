"""Resumable controlled PPI fine-tuning; qualification and production are explicit."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import expit
from torch.nn.parallel import DistributedDataParallel as DDP
from data import PairData
from model import PairModel
from state import Checkpoints, atomic_json, sha256, fsync_dir
from contracts import CORE, verify_inputs


def rank_ordered_average(group, bucket):
    """Deterministic rank-order FP32 sums independent of ring/bucket rebuilds."""
    buf = bucket.buffer()
    world = dist.get_world_size(group)
    gathered = torch.empty(world * buf.numel(), device=buf.device, dtype=buf.dtype)
    work = dist.all_gather_into_tensor(gathered, buf, group=group, async_op=True)
    def reduce_in_order(future):
        shards = gathered.view(world, -1)
        buf.copy_(shards[0])
        for rank in range(1, world): buf.add_(shards[rank])
        return buf.div_(world)
    return work.get_future().then(reduce_in_order)


def configuration_checks(cfg):
    assert cfg['readout'] in ['cls_linear', 'cls_mlp', 'residue_mean']
    assert cfg['attention_mode'] in ['standard', 'chain_aware']
    assert cfg['attention_backend'] in ['efficient', 'math']
    assert cfg['partition'] in ['official', 'fold-0', 'fold-1', 'fold-2']
    assert isinstance(cfg['classification_corruption'], bool)
    assert cfg['positive_weight'] > 0 and cfg['classification_weight'] > 0 and cfg['mlm_weight'] >= 0
    assert cfg['train_cap_residues'] in [None, 2193]
    for key in ['total_updates', 'global_pairs_per_update', 'validate_every_updates',
                'checkpoint_every_updates', 'train_token_budget', 'eval_token_budget']:
        assert cfg[key] > 0, key
    assert cfg['selection_metric'] == 'pooled_ap'
    assert cfg['backbone'] in ['esm2', 'esmc']
    assert cfg['initialization'] == cfg['backbone'] + '_pretrained'
    assert cfg['readout'] == 'residue_mean' and cfg['mlm_weight'] == 0
    assert not cfg['classification_corruption'] and cfg['positive_weight'] == 1
    assert cfg['train_cap_residues'] is None and cfg['partition'] == 'official'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--qualification', action='store_true')
    mode.add_argument('--production', action='store_true')
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--max-updates', type=int)
    p.add_argument('--stop-after-updates', type=int)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--qualification-max-tokens', type=int, default=1024)
    p.add_argument('--sleep-after-update', type=float, default=0.)
    args = p.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text())
    configuration_checks(cfg)
    root, out = Path(cfg['root']).resolve(), Path(args.output).resolve()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.qualification:
        assert out.is_relative_to(root / 'qualification')
        assert args.max_updates and 1 <= args.max_updates <= 8
        assert args.train_limit and 2 <= args.train_limit <= 128
        assert args.val_limit and 2 <= args.val_limit <= 32
        assert args.qualification_max_tokens <= 4096
    else:
        assert out.is_relative_to(root / 'runs')
        assert not any([args.tiny, args.max_updates, args.stop_after_updates, args.train_limit,
                        args.val_limit, args.sleep_after_update])
        assert world == cfg['world_size'] == 4
        # A qualified immutable release, not a mere config file, enables production.
        from release import verify_release
        verify_release(root, config_path, Path(__file__).resolve().parent)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(8)
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if world > 1: dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10), device_id=device)
    attempt = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.environ.get("SLURM_JOB_ID", "local")}'
    started = time.monotonic()
    requested = False
    def event(kind, **details):
        if rank == 0:
            value = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'attempt': attempt, 'event': kind, **details}
            with (out / 'events.jsonl').open('a') as f:
                f.write(json.dumps(value, allow_nan=False) + '\n'); f.flush()
            print(json.dumps(value, allow_nan=False), flush=True)
    def handle(signum, frame):
        nonlocal requested
        requested = True
    for signum in [signal.SIGTERM, signal.SIGUSR1, signal.SIGINT]: signal.signal(signum, handle)
    def local_stop(): return requested or (out / 'REQUEST_STOP').exists() or (out / 'REQUEST_REQUEUE').exists()
    def stopping():
        flag = torch.tensor(int(local_stop()), device=device)
        if world > 1: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    manifest = json.loads((root / 'data/prepared/manifest.json').read_text())
    if rank == 0:
        verify_inputs(root, backbone=cfg['backbone'], images=False)
    if world > 1: dist.barrier()
    train = PairData(root / 'data/prepared', cfg['partition'], 'train', cfg['train_cap_residues'])
    val = PairData(root / 'data/prepared', cfg['partition'], 'val')
    if args.qualification:
        for data, limit in [(train, args.train_limit), (val, args.val_limit)]:
            rng = np.random.default_rng(1729)
            eligible = data.lengths <= args.qualification_max_tokens
            selected = np.concatenate([rng.choice(np.flatnonzero(eligible & (data.rows[:, 2] == label)),
                limit // 2 + (limit % 2 if label else 0), replace=False) for label in [0, 1]])
            selected.sort()
            data.rows, data.lengths = data.rows[selected], data.lengths[selected]
    total = args.max_updates if args.qualification else cfg['total_updates']
    code_names = CORE
    image = json.loads((root / 'provenance/images.json').read_text())[cfg['backbone']]
    assert os.environ.get('PLMI_V4_IMAGE_SHA256') == image['sha256'], 'Use the pinned container wrapper'
    contract = {'configuration': cfg, 'qualification': args.qualification, 'tiny': args.tiny,
        'train_limit': args.train_limit, 'val_limit': args.val_limit,
        'qualification_max_tokens': args.qualification_max_tokens if args.qualification else None,
        'total_steps': total, 'world_size': world,
        'data_manifest_sha256': sha256(root / 'data/prepared/manifest.json'),
        'initialization_manifest_sha256': sha256(root / 'provenance/downloads.json'),
        'code': {name: sha256(Path(__file__).with_name(name)) for name in code_names},
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'image_sha256': image['sha256'],
        'packages': {n: importlib.metadata.version(n) for n in
                     ['transformers', 'tokenizers', 'numpy'] + (['esm'] if cfg['backbone'] == 'esmc' else [])},
        'selection': 'maximum pooled validation AP; strict greater-than keeps earlier update on exact ties',
        'test_evaluation': 'forbidden in this trainer'}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        import fcntl
        lock_handle = (out / 'RUNNING.lock').open('a+')
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_handle.seek(0); lock_handle.truncate(); lock_handle.write(attempt); lock_handle.flush()
        if (out / 'contract.json').exists():
            assert json.loads((out / 'contract.json').read_text())['fingerprint'] == fingerprint, 'Run contract mismatch'
        atomic_json(out / 'contract.json', {'fingerprint': fingerprint, **contract})
        (out / 'stopped.json').unlink(missing_ok=True)
    model = PairModel(root / 'assets' / cfg['backbone'], cfg, tiny=args.tiny).to(device)
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim >= 2], weight_decay=cfg['weight_decay']),
              dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim < 2], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], foreach=False)
    checkpoints = Checkpoints(out, rank, world, fingerprint)
    state, selection = checkpoints.load(model, optimizer)
    if state is None:
        state = dict(update=0, cursor={'cycle': 0, 'offset': 0}, epoch=0, next_chunk=0,
            examples_seen=0, best_ap=None, best_update=None, total_steps=total,
            pending_validation=False, last_validation_update=None)
    assert state['total_steps'] == total
    assert state['examples_seen'] == state['update'] * cfg['global_pairs_per_update']
    assert state['cursor']['cycle'] * len(train) + state['cursor']['offset'] == state['examples_seen']
    runtime = {'rank': rank, 'uuid': str(torch.cuda.get_device_properties(device).uuid),
               'device': torch.cuda.get_device_name(device)}
    devices = [None] * world
    if world > 1: dist.all_gather_object(devices, runtime)
    else: devices = [runtime]
    assert len({item['uuid'] for item in devices}) == world, devices
    event('resumed' if selection else 'initialized', state=state, world_size=world, devices=devices,
          checkpoint=selection, train_rows=len(train), val_rows=len(val),
          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Keep each parameter gradient in its own aligned storage. Bucket views
    # change storage offsets when DDP rebuilds its buckets, including after
    # restart; reduction kernels can then round gradient norms differently.
    wrapper = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False,
                  gradient_as_bucket_view=False) if world > 1 else model
    if world > 1: wrapper.register_comm_hook(state=None, hook=rank_ordered_average)
    optimizer.zero_grad(set_to_none=True)
    if selection is None: checkpoints.save(model, optimizer, state)
    last_checkpoint = time.monotonic()
    last_committed = state['update']

    def stop(reason):
        if rank == 0: atomic_json(out / 'stopped.json', {'state': state, 'reason': reason,
                                                       'attempt': attempt, 'checkpoint_committed': True})
        event('stopped', state=state, reason=reason)
        if world > 1: dist.destroy_process_group()
        return 0

    def validate():
        model.eval()
        indices = np.arange(rank, len(val), world, dtype=np.int64)
        indices = indices[np.argsort(val.lengths[indices], kind='stable')]
        predictions, interrupted = [], False
        with torch.inference_mode():
            for positions in val.microbatches(indices, cfg['eval_token_budget'], cfg['eval_max_pairs']):
                if local_stop(): interrupted = True; break
                ids = indices[positions]
                features = val.batch(ids, np.zeros(len(ids), dtype=np.int64), cfg['seed'], False, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model(**features, compute_loss=False)
                assert torch.isfinite(z).all()
                predictions.extend((int(i), int(val.rows[i, 2]), float(a), float(b))
                                   for i, (a, b) in zip(ids, z.float().cpu().numpy()))
        gathered = [None] * world
        if world > 1: dist.all_gather_object(gathered, {'rows': predictions, 'interrupted': interrupted})
        else: gathered = [{'rows': predictions, 'interrupted': interrupted}]
        metrics, best = None, False
        if not any(item['interrupted'] for item in gathered):
            array = np.asarray([row for item in gathered for row in item['rows']], dtype=np.float64)
            array = array[np.argsort(array[:, 0])]
            assert array.shape == (len(val), 4) and np.array_equal(array[:, 0], np.arange(len(val)))
            assert np.array_equal(array[:, 1], val.rows[:, 2]) and np.isfinite(array).all()
            pooled = array[:, 2:4].mean(1)
            metrics = {'rows': len(array), 'pooled_ap': float(average_precision_score(array[:, 1], pooled)),
                'auroc': float(roc_auc_score(array[:, 1], pooled)),
                'brier': float(np.mean((expit(pooled) - array[:, 1]) ** 2)),
                'ab_ap': float(average_precision_score(array[:, 1], array[:, 2])),
                'order_logit_gap_mean': float(np.abs(array[:, 2] - array[:, 3]).mean())}
            best = state['best_ap'] is None or metrics['pooled_ap'] > state['best_ap']
            if best: state['best_ap'], state['best_update'] = metrics['pooled_ap'], state['update']
            state['pending_validation'], state['last_validation_update'] = False, state['update']
            if rank == 0:
                folder = out / 'validation'; folder.mkdir(exist_ok=True)
                target = folder / f'update-{state["update"]:09d}.npz'
                temporary = target.with_suffix('.partial')
                with temporary.open('wb') as f:
                    np.savez_compressed(f, predictions=array); f.flush(); os.fsync(f.fileno())
                os.replace(temporary, target); fsync_dir(folder)
                atomic_json(target.with_suffix('.json'), {'sha256': sha256(target), 'rows': len(array),
                    'update': state['update'], 'fingerprint': fingerprint, 'metrics': metrics})
                event('validation', update=state['update'], metrics=metrics, best=best)
        model.train()
        return metrics, best

    if state['pending_validation']:
        _, best = validate() if not stopping() else (None, False)
        checkpoints.save(model, optimizer, state, best=best)
        last_checkpoint = time.monotonic()
        if state['pending_validation'] or stopping(): return stop('signal_or_REQUEST_STOP')
    while state['update'] < total:
        if stopping():
            checkpoints.save(model, optimizer, state)
            return stop('signal_or_REQUEST_STOP')
        indices, cycles, next_cursor = train.next_batch(state['cursor'], cfg['seed'], cfg['global_pairs_per_update'])
        local_indices, local_cycles = indices[rank::world], cycles[rank::world]
        dummy = len(local_indices) == 0
        if dummy: local_indices, local_cycles = indices[:1], cycles[:1]
        micros = list(train.microbatches(local_indices, cfg['train_token_budget'], cfg['train_max_pairs']))
        optimizer.zero_grad(set_to_none=True)
        model.train()
        started_update = time.monotonic()
        statistics = torch.zeros(3, device=device, dtype=torch.float64)
        for number, positions in enumerate(micros):
            sync = contextlib.nullcontext() if world == 1 or number == len(micros) - 1 else wrapper.no_sync()
            with sync:
                features = train.batch(local_indices[positions], local_cycles[positions], cfg['seed'],
                         cfg['classification_corruption'] or cfg['mlm_weight'] > 0, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, cls, mlm, z = wrapper(**features)
                    scaled = loss * (0. if dummy else world / len(indices))
                scaled.backward()
            if not dummy: statistics += torch.stack((loss.detach().double(), cls.double(), mlm.double()))
            del features, loss, cls, mlm, z, scaled
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'], error_if_nonfinite=True)
        update = state['update']
        warmup = min(cfg['warmup_updates'], max(1, total - 1))
        factor = (update + 1) / warmup if update < warmup else max(0., (total - update) / (total - warmup))
        for group in optimizer.param_groups: group['lr'] = cfg['learning_rate'] * factor
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if world > 1: dist.all_reduce(statistics)
        state['update'] += 1
        state['cursor'] = next_cursor
        state['epoch'], state['next_chunk'] = next_cursor['cycle'], next_cursor['offset']
        state['examples_seen'] += len(indices)
        if rank == 0 and (state['update'] <= 3 or state['update'] % cfg['log_every_updates'] == 0):
            atomic_json(out / 'status.json', {'state': state, 'last_committed_update': last_committed,
                        'attempt': attempt, 'phase': 'training', 'seconds_since_launch': time.monotonic() - started})
            event('update', update=state['update'], cursor=next_cursor, lr=cfg['learning_rate'] * factor,
                mean_loss=float(statistics[0] / len(indices)), mean_bce=float(statistics[1] / len(indices)),
                mean_mlm=float(statistics[2] / len(indices)), grad_norm=float(norm),
                seconds=time.monotonic() - started_update, global_pairs=len(indices),
                batch_digest=hashlib.sha256(np.stack((indices, cycles)).tobytes()).hexdigest(),
                max_tokens=int(train.lengths[indices].max()), gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        forced = args.stop_after_updates is not None and state['update'] >= args.stop_after_updates
        stopped, final = stopping() or forced, state['update'] == total
        due_val = state['update'] % cfg['validate_every_updates'] == 0 or final
        if due_val: state['pending_validation'] = True
        _, best = validate() if due_val and not stopped else (None, False)
        stopped = stopped or stopping()
        if stopped or final or due_val or state['update'] % cfg['checkpoint_every_updates'] == 0 or time.monotonic() - last_checkpoint >= cfg['checkpoint_minutes'] * 60:
            checkpoints.save(model, optimizer, state, best=best)
            last_checkpoint, last_committed = time.monotonic(), state['update']
            event('checkpoint_committed', update=state['update'], best=best)
        if stopped: return stop('qualification_stop' if forced else 'signal_or_REQUEST_STOP')
        if args.sleep_after_update: time.sleep(args.sleep_after_update)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    hashes = [None] * world
    if world > 1: dist.all_gather_object(hashes, digest.hexdigest())
    else: hashes = [digest.hexdigest()]
    assert len(set(hashes)) == 1, hashes
    if rank == 0:
        atomic_json(out / 'completed.json', {'state': state, 'fingerprint': fingerprint,
                    'qualification': args.qualification, 'test_evaluated': False, 'model_hashes_by_rank': hashes})
        atomic_json(out / 'status.json', {'state': state, 'phase': 'complete', 'last_committed_update': state['update']})
    event('complete', state=state)
    if world > 1: dist.destroy_process_group()
    return 0


if __name__ == '__main__': sys.exit(main())
