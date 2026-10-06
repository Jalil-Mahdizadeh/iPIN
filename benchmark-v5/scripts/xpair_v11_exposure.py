"""Audit documented V11 source membership, with exact and Ankh-normalized sequences."""
import csv
import hashlib
import re

import numpy as np

from bench_utils import ROOT, atomic, load_npz, now, read, record, save_npz, sha


def main():
    freeze = read(ROOT / 'provenance/xpair-v11.json')
    hp = freeze['hyper_parameters']
    previous = read(ROOT / 'provenance/human-releases-exposure.json')
    meta = read(ROOT / 'data/sequences.json')
    union = np.load(ROOT / 'data/union.npy')
    mapping = load_npz(ROOT / 'data/pair-mapping.npz')
    def norm(s): return re.sub(r'[UZOBJ]', 'X', s.upper())
    def digest(s): return hashlib.sha256(s.encode()).hexdigest()
    pairs = {tuple(sorted(map(int, r[:2]))): i for i, r in enumerate(union)}
    assert len(pairs) == len(union)
    flags, audits = {}, {}
    for mode, transform in [('exact', lambda s: s), ('ankh-normalized', norm)]:
        lookup = {}
        for i, seq in enumerate(meta['sequence']):
            lookup.setdefault(digest(transform(seq)), []).append(i)
        endpoints = np.zeros(len(meta['sequence']), np.uint8)
        hit = np.zeros(len(union), np.uint8)
        pos, neg = hit.copy(), hit.copy()
        sources = []
        for source in previous['sources']:
            item = source['native_file']
            assert sha(item['path']) == item['sha256']
            bit = source['bit']
            memo = {}
            n = positive = retained = 0
            minlen, maxlen = 10**9, 0
            with open(item['path']) as stream:
                for row in csv.DictReader(stream):
                    seqs = [row['query'], row['text']]
                    label = int(row['label'])
                    n += 1; positive += label
                    minlen = min(minlen, *map(len, seqs)); maxlen = max(maxlen, *map(len, seqs))
                    if any(len(s) < hp['min_len'] or len(s) > hp['max_len'] for s in seqs): continue
                    retained += 1
                    ids = []
                    for s in seqs:
                        if s not in memo: memo[s] = lookup.get(digest(transform(s)), [])
                        ids.append(memo[s])
                        for i in ids[-1]: endpoints[i] |= bit
                    for a in ids[0]:
                        for b in ids[1]:
                            j = pairs.get(tuple(sorted((a, b))))
                            if j is not None:
                                hit[j] |= bit
                                (pos if label else neg)[j] |= bit
            assert n == source['rows'] and positive == source['positives'] and retained == n
            sources.append({'split': source['split'], 'file': item, 'rows': n, 'positives': positive,
                            'length_eligible_rows': retained, 'min_length': minlen, 'max_length': maxlen})
        tests = {}
        for test in ['original', 'ilp']:
            rows = np.load(ROOT / 'data' / f'{test}.npy')
            ids = mapping[test]; y = rows[:, 2]
            remaining = ~(endpoints > 0)[rows[:, :2]].any(1)
            pp = hit[ids] > 0
            tests[test] = {
                'exact_exposed_sequences': int((endpoints[np.unique(rows[:, :2])] > 0).sum()),
                'pairs_exposed': int(pp.sum()), 'positive_pairs_exposed': int((pp & (y == 1)).sum()),
                'negative_pairs_exposed': int((pp & (y == 0)).sum()),
                'pairs_with_exposed_endpoint': int((~remaining).sum()),
                'endpoint_unexposed_pairs': int(remaining.sum()),
                'endpoint_unexposed_positives': int(y[remaining].sum()),
                'test_negative_but_source_positive': int(((pos[ids] > 0) & (y == 0)).sum()),
                'test_positive_but_source_negative': int(((neg[ids] > 0) & (y == 1)).sum()),
            }
        for key, array in [('endpoints', endpoints), ('pairs', hit), ('source_positive', pos), ('source_negative', neg)]:
            flags[mode + '__' + key] = array
        audits[mode] = {'sources': sources, 'union_sequences_exposed': int((endpoints > 0).sum()), 'tests': tests}
    prior = load_npz(previous['flags']['path'])
    assert sha(previous['flags']['path']) == previous['flags']['sha256']
    for key in ['endpoints', 'pairs', 'source_positive', 'source_negative']:
        assert np.array_equal(flags['exact__' + key], prior[key]), key
    old = load_npz(ROOT / 'provenance/exposure-flags.npz')
    identical = all(np.array_equal(flags[mode + '__endpoints'] > 0, old['dscript__exact__endpoints'] > 0)
                    for mode in ['exact', 'ankh-normalized'])
    assert identical, 'Existing D-SCRIPT exclusion must be reviewed before figure labeling'
    path = ROOT / 'provenance/xpair-v11-exposure-flags.npz'
    save_npz(path, **flags)
    atomic(ROOT / 'provenance/xpair-v11-exposure.json', {
        'at_utc': now(), 'model': 'xpair-v11', 'checkpoint': freeze['checkpoint'],
        'audits': audits, 'flags': record(path), 'script': record(__file__),
        'exact_flags_match_prior_human_release_audit': True,
        'existing_dscript_endpoint_mask_removes_all_identified_exposure': identical,
        'limits': 'Documented public source membership, supported by paper and embedded checkpoint paths; exact processed author TSVs are unavailable locally. No claim of complete checkpoint-history proof or homolog/pretraining exclusion.',
    })
    print({'xpair_v11_exposure': audits, 'dscript_mask_sufficient': identical}, flush=True)


if __name__ == '__main__': main()
