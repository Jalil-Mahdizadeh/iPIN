"""Fixed, mask-aware local pooling; no trainable parameters or sequence crops."""
import numpy as np
import torch
from study import config, text_hash


def blocks(length, target=8):
    if length < 1:
        raise ValueError('Empty chain')
    n = (length + target - 1) // target
    sizes = np.full(n, length // n, dtype=int)
    sizes[:length % n] += 1
    return np.r_[0, np.cumsum(sizes)].tolist()


def spatial_indices(good, uid, chain_id):
    cfg = config()
    seed = int(text_hash(f"{cfg['seed']}:{cfg['pooling']['spatial_seed_namespace']}:{uid}:{chain_id}")[:16], 16)
    valid = np.flatnonzero(np.asarray(good, dtype=bool))
    indices = np.arange(len(good))
    indices[valid] = np.random.default_rng(seed).permutation(valid)
    return indices


def global_pool(projected, good_a, good_b):
    p = projected[good_a][:, good_b].reshape(-1, 32).float()
    if not len(p) or not torch.isfinite(p).all():
        raise FloatingPointError('Empty or nonfinite inter-chain tensor')
    return torch.cat([p.mean(0), p.std(0, unbiased=False), p.amax(0), torch.quantile(p, .95, dim=0)])


def local_pool(projected, good_a, good_b, global_features=None):
    cfg = config()['pooling']
    if projected.shape != (len(good_a), len(good_b), 32):
        raise ValueError('Projection/mask shape mismatch')
    globals_ = global_features if global_features is not None else global_pool(projected, good_a, good_b)
    aa = blocks(len(good_a), cfg['block_target_length'])
    bb = blocks(len(good_b), cfg['block_target_length'])
    ca = torch.stack([good_a[l:r].sum() for l, r in zip(aa[:-1], aa[1:])])
    cb = torch.stack([good_b[l:r].sum() for l, r in zip(bb[:-1], bb[1:])])
    valid_a = ca >= cfg['minimum_block_quality'] * torch.tensor(np.diff(aa), device=projected.device)
    valid_b = cb >= cfg['minimum_block_quality'] * torch.tensor(np.diff(bb), device=projected.device)
    p = projected.float().masked_fill(~(good_a[:, None] & good_b[None, :])[:, :, None], 0)
    # Separable sums avoid nondeterministic scatter-add atomics and dense tile weights.
    p = torch.stack([p[l:r].sum(0) for l, r in zip(aa[:-1], aa[1:])])
    p = torch.stack([p[:, l:r].sum(1) for l, r in zip(bb[:-1], bb[1:])], dim=1)
    count = ca[:, None] * cb[None, :]
    keep = valid_a[:, None] & valid_b[None, :] & (count > 0)
    means = p[keep] / count[keep][:, None]
    if not len(means) or not torch.isfinite(means).all():
        raise FloatingPointError('No finite qualifying local blocks')
    k = min(cfg['top_k'], len(means))
    high = means.topk(k, dim=0, largest=True).values.mean(0)
    low = means.topk(k, dim=0, largest=False).values.mean(0)
    out = torch.cat([globals_[:64], high, low])
    if not torch.isfinite(out).all():
        raise FloatingPointError('Nonfinite local features')
    return out, {'blocks_total': len(aa[1:]) * len(bb[1:]), 'blocks_qualifying': int(keep.sum()), 'k_used': k}


def pool_views(projected, good_a, good_b, permutation_a, permutation_b):
    g = global_pool(projected, good_a, good_b)
    local, meta = local_pool(projected, good_a, good_b, g)
    pa = torch.as_tensor(permutation_a, device=projected.device)
    pb = torch.as_tensor(permutation_b, device=projected.device)
    if not torch.equal(good_a[pa], good_a) or not torch.equal(good_b[pb], good_b):
        raise ValueError('Spatial permutation changes the quality mask')
    scrambled, sm = local_pool(projected[pa][:, pb], good_a, good_b, g)
    if sm != meta:
        raise ValueError('Spatial null changes block availability')
    return {'global': g, 'local': local, 'scrambled': scrambled}, meta
