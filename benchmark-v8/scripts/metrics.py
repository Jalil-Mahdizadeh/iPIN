"""Frozen-candidate ranking and component-resampled partner diagnostics.

Methods follow the historical OpenPPI estimands. No model fitting or input I/O.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np


def queries(a, b, y):
    by_anchor = defaultdict(list)
    for i, (left, right) in enumerate(zip(a, b)):
        if left == right:
            raise ValueError('Self interaction in partner diagnostic')
        by_anchor[int(left)].append((i, int(right)))
        by_anchor[int(right)].append((i, int(left)))
    result = []
    for anchor, rows in sorted(by_anchor.items()):
        ids, partners = np.asarray(rows, dtype=np.int64).T
        positive = y[ids]
        if positive.any() and (~positive).any():
            result.append((anchor, ids[positive], ids[~positive],
                           partners[positive], partners[~positive]))
    return result


def concordance(p, u, pw=None, uw=None):
    pw = np.ones(len(p)) if pw is None else np.asarray(pw)
    uw = np.ones(len(u)) if uw is None else np.asarray(uw)
    if pw.sum() <= 0 or uw.sum() <= 0:
        return np.nan
    order = np.argsort(u, kind='stable')
    prefix = np.r_[0., np.cumsum(uw[order])]
    left = np.searchsorted(u[order], p, side='left')
    right = np.searchsorted(u[order], p, side='right')
    return float(np.dot(pw, .5 * (prefix[left] + prefix[right])) / (pw.sum() * uw.sum()))


def retrieval(scores, y, ks):
    order = np.argsort(-scores, kind='stable')
    s, positive = scores[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    fraction = np.add.reduceat(positive.astype(float), starts) / (ends - starts)
    discount = np.r_[0., np.cumsum(1 / np.log2(np.arange(len(s)) + 2))]
    result = []
    for k in ks:
        kk = min(k, len(s))
        hits = np.dot(fraction, np.clip(kk - starts, 0, ends - starts))
        dcg = np.dot(fraction, discount[np.minimum(ends, kk)] - discount[np.minimum(starts, kk)])
        result.extend([hits / positive.sum(), dcg / discount[min(kk, int(positive.sum()))]])
    return np.asarray(result)


def component_draws(component, seed, repeats):
    labels = sorted(set(component.tolist()))
    lookup = {c: i for i, c in enumerate(labels)}
    indices = np.array([lookup[c] for c in component], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    selected = rng.integers(0, len(labels), (repeats, len(labels)))
    counts = np.zeros(selected.shape, dtype=np.int32)
    np.add.at(counts, (np.arange(repeats)[:, None], selected), 1)
    return indices, counts, labels


def within_anchor(scores, qs, comp, counts):
    m, repeats = scores.shape[1], len(counts)
    values = np.zeros((len(qs), m))
    total = np.zeros((repeats, m))
    mass = np.zeros(repeats)
    for qi, (anchor, pi, ui, pp, up) in enumerate(qs):
        ac = comp[anchor]
        pw = counts[:, comp[pp]].astype(float)
        uw = counts[:, comp[up]].astype(float)
        pw[:, comp[pp] == ac] = 1
        uw[:, comp[up] == ac] = 1
        den = pw.sum(1) * uw.sum(1)
        valid = den > 0
        anchor_weight = counts[:, ac] * valid
        mass += anchor_weight
        for j in range(m):
            p, u = scores[pi, j], scores[ui, j]
            values[qi, j] = concordance(p, u)
            order = np.argsort(u, kind='stable')
            prefix = np.c_[np.zeros(repeats), np.cumsum(uw[:, order], axis=1)]
            left = np.searchsorted(u[order], p, side='left')
            right = np.searchsorted(u[order], p, side='right')
            numerator = (.5 * (prefix[:, left] + prefix[:, right]) * pw).sum(1)
            ratio = np.divide(numerator, den, out=np.zeros(repeats), where=valid)
            total[:, j] += anchor_weight * ratio
    draws = np.divide(total, mass[:, None], out=np.full_like(total, np.nan), where=mass[:, None] > 0)
    point = values.mean(0) if len(qs) else np.full(m, np.nan)
    return point, draws, values


def early_retrieval(scores, y, qs, comp, counts, ks, minimum_u):
    selected = [q for q in qs if len(q[2]) >= minimum_u]
    values = np.zeros((len(selected), scores.shape[1], 2 * len(ks)))
    for i, (anchor, pi, ui, pp, up) in enumerate(selected):
        ids = np.r_[pi, ui]
        for j in range(scores.shape[1]):
            values[i, j] = retrieval(scores[ids, j], y[ids], ks)
    m, k = scores.shape[1], 2 * len(ks)
    if not selected:
        return np.full((m, k), np.nan), np.full((len(counts), m, k), np.nan), values, selected
    weight = counts[:, comp[[q[0] for q in selected]]].astype(float)
    den = weight.sum(1)
    num = weight @ values.reshape(len(selected), -1)
    boot = np.divide(num, den[:, None], out=np.full_like(num, np.nan), where=den[:, None] > 0)
    return values.mean(0), boot.reshape(len(counts), m, k), values, selected


def select_quartets(a, b, y, uids, seed, maximum, edge_cap, endpoint_cap):
    """Enumerate both crosses; retain only existing, negative-labelled edges."""
    n = max(int(a.max()), int(b.max())) + 1
    codes = np.minimum(a, b) * n + np.maximum(a, b)
    if len(np.unique(codes)) != len(codes):
        raise ValueError('Duplicate unordered candidate pairs must be resolved before diagnostics')
    pids, uids_rows = np.flatnonzero(y), np.flatnonzero(~y)
    order = np.argsort(codes[uids_rows])
    ucodes, urows = codes[uids_rows][order], uids_rows[order]
    candidates = []
    for i, first in enumerate(pids[:-1]):
        second = pids[i + 1:]
        aa, bb = int(a[first]), int(b[first])
        distinct = (a[second] != aa) & (a[second] != bb) & (b[second] != aa) & (b[second] != bb)
        second = second[distinct]
        for flip in (False, True):
            cc, dd = (b[second], a[second]) if flip else (a[second], b[second])
            code1 = np.minimum(aa, dd) * n + np.maximum(aa, dd)
            code2 = np.minimum(cc, bb) * n + np.maximum(cc, bb)
            k1, k2 = np.searchsorted(ucodes, code1), np.searchsorted(ucodes, code2)
            ix = np.flatnonzero((k1 < len(ucodes)) & (k2 < len(ucodes)))
            ix = ix[(ucodes[k1[ix]] == code1[ix]) & (ucodes[k2[ix]] == code2[ix])]
            for j in ix:
                endpoints = (aa, bb, int(cc[j]), int(dd[j]))
                rows = (int(first), int(second[j]), int(urows[k1[j]]), int(urows[k2[j]]))
                key = hashlib.sha256(f'{seed}:{uids[first]}:{uids[second[j]]}:{int(flip)}'.encode()).digest()
                candidates.append((key, endpoints, rows))
    candidates.sort()
    edge_uses, endpoint_uses = Counter(), Counter()
    kept, endpoints = [], []
    for key, ep, rows in candidates:
        if len(kept) >= maximum:
            break
        if any(edge_uses[r] >= edge_cap for r in rows[:2]) or any(endpoint_uses[p] >= endpoint_cap for p in ep):
            continue
        kept.append(rows); endpoints.append(ep)
        edge_uses.update(rows[:2]); endpoint_uses.update(ep)
    return (np.asarray(kept, dtype=np.int64).reshape(-1, 4),
            np.asarray(endpoints, dtype=np.int64).reshape(-1, 4), len(candidates))


def quartet_metrics(scores, rows, endpoints, comp, counts):
    m = scores.shape[1]
    if not len(rows):
        return np.full(m, np.nan), np.full((len(counts), m), np.nan), np.empty((0, m))
    positive = scores[rows[:, 0]] + scores[rows[:, 1]]
    alternate = scores[rows[:, 2]] + scores[rows[:, 3]]
    delta = positive - alternate
    tolerance = 1e-10 * (1 + np.abs(positive) + np.abs(alternate))
    credit = (delta > tolerance).astype(float) + .5 * (np.abs(delta) <= tolerance)
    weight = np.stack([np.prod(counts[:, np.unique(comp[ep])].astype(float), axis=1) for ep in endpoints], axis=1)
    den = weight.sum(1)
    num = weight @ credit
    draws = np.divide(num, den[:, None], out=np.full_like(num, np.nan), where=den[:, None] > 0)
    return credit.mean(0), draws, credit


def summary(point, draws, support, minimum_fraction=.95):
    finite = np.isfinite(draws)
    qualified = bool(support and finite.mean() >= minimum_fraction)
    return {'point': float(point) if np.isfinite(point) else None,
            'ci95': np.quantile(draws[finite], [.025, .975]).tolist() if qualified else None,
            'finite_replicates': int(finite.sum()), 'support_qualified': qualified}
