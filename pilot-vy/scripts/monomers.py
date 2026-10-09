"""Independent monomer selection and explicit profile/diversity descriptors."""
from collections import Counter, defaultdict
import hashlib
import numpy as np
from study import text_hash

ALPHABET = 'ARNDCEQGHILKMFPSTWYVXBZUO-'
GAP, PAD = 25, 26
LOOKUP = np.full(256, 255, dtype=np.uint8)
for i, c in enumerate(ALPHABET):
    LOOKUP[ord(c)] = i


def encode(sequence):
    a = LOOKUP[np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)]
    if np.any(a == 255):
        raise ValueError('Unsupported cached alphabet')
    return a


def tax_group(tax):
    fields = tax.split(':')
    unknown = {'', 'na', 'n/a', 'none', 'unknown', 'unclassified'}
    return fields[1].strip() if len(fields) > 1 and fields[1].strip().lower() not in unknown else '__unknown__'


def blocks(length):
    return np.array_split(np.arange(length), 4)


def related(row, others, good, identity, comparison_coverage):
    valid = (others[:, good] != GAP) & (row[good] != GAP)
    coverage = valid.sum(1)
    equal = ((others[:, good] == row[good]) & valid).sum(1)
    return (coverage >= comparison_coverage * good.sum()) & (equal >= identity * coverage)


def neff80(tokens, cfg):
    good = tokens[0] != PAD; rows = tokens[1:, good]; n = len(rows)
    if n < 2 or not good.any():
        raise ValueError('Neff requires at least two homologs and valid positions')
    count = np.ones(n, dtype=int); coverages = []; comparable = 0
    for i in range(1, n):
        valid = (rows[:i] != GAP) & (rows[i] != GAP)
        covered = valid.sum(1)
        equal = ((rows[:i] == rows[i]) & valid).sum(1)
        enough = covered >= cfg['comparison_coverage'] * int(good.sum())
        near = enough & (equal >= cfg['neff_identity'] * covered)
        count[i] += int(near.sum()); count[:i] += near.astype(int)
        comparable += int(enough.sum()); coverages.extend((covered / good.sum()).tolist())
    return {'neff80': float((1 / count).sum()), 'fraction_comparable': comparable / (n * (n - 1) / 2),
            'comparison_coverage_quantiles': np.quantile(coverages, [0, .25, .5, .75, 1]).tolist()}


def select(monomer, cfg, seed):
    query, quality = monomer['query'], monomer['mask']; length = len(query)
    if len(quality) != length or set(quality) - {'*', '-'} or text_hash(query) != monomer['query_sha256']:
        raise ValueError('Invalid cached monomer identity or coordinates')
    q = encode(query)
    if np.any(q == GAP):
        raise ValueError('Gap in query sequence')
    good = np.array([c == '*' for c in quality])
    meta = {'query_sha256': monomer['query_sha256'], 'length': length, 'quality_fraction': float(good.mean()),
            'raw_depth': int(monomer['stats']['raw_rows']), 'cached_depth': len(monomer['rows']),
            'filtered_depth': 0, 'retained_depth': 0, 'available': False}
    if length > cfg['maximum_residues']:
        return None, {**meta, 'reason': 'long_monomer'}
    if good.mean() < cfg['minimum_quality_fraction']:
        return None, {**meta, 'reason': 'low_quality_coverage'}
    groups = defaultdict(list); keys = set()
    def priority(key):
        return text_hash(f'{seed}:{monomer["query_sha256"]}:{key}')
    for key, aligned, taxonomy in monomer['rows']:
        if key in keys:
            raise ValueError('Duplicate accession survived original cache')
        keys.add(key); row = encode(aligned)
        if len(row) != length:
            raise ValueError('Invalid cached homolog coordinates')
        if (row[good] != GAP).mean() < cfg['minimum_good_column_coverage']:
            continue
        groups[tax_group(taxonomy)].append((key, row, taxonomy))
    for rows in groups.values():
        rows.sort(key=lambda x: priority(x[0]))
    meta['filtered_depth'] = sum(map(len, groups.values()))
    order = sorted(groups, key=priority)
    selected = np.empty((cfg['maximum_homologs'], length), dtype=np.uint8); retained = []
    done = False
    for i in range(max(map(len, groups.values()), default=0)):
        for group in order:
            if i >= len(groups[group]):
                continue
            key, row, taxonomy = groups[group][i]
            if retained and related(row, selected[:len(retained)], good, cfg['redundancy_identity'], cfg['comparison_coverage']).any():
                continue
            selected[len(retained)] = row; retained.append((key, taxonomy))
            if len(retained) == cfg['maximum_homologs']:
                done = True; break
        if done:
            break
    meta['retained_depth'] = len(retained)
    if len(retained) < cfg['minimum_homologs']:
        return None, {**meta, 'reason': 'shallow_monomer_alignment'}
    tokens = np.vstack([q, selected[:len(retained)]]); tokens[:, ~good] = PAD
    taxa = Counter(tax_group(tax) for _, tax in retained)
    probs = np.array(list(taxa.values()), dtype=float) / len(retained)
    h = tokens[1:, good]; occupancy = (h != GAP).mean(0)
    meta.update(neff80(tokens, cfg))
    meta.update({'available': True, 'reason': 'available', 'selected_keys': [k for k, _ in retained],
                 'selected_taxonomy': [t for _, t in retained], 'taxonomic_families': len(taxa),
                 'taxonomic_effective_number': float(np.exp(-(probs * np.log(probs)).sum())),
                 'unknown_taxonomy_fraction': taxa.get('__unknown__', 0) / len(retained),
                 'valid_residues': int(good.sum()), 'occupancy_mean': float(occupancy.mean()),
                 'occupancy_quantiles': np.quantile(occupancy, [0, .25, .5, .75, 1]).tolist(),
                 'fraction_good_positions_half_depth': float((occupancy >= .5).mean()),
                 'ambiguous_fraction': float(((h >= 20) & (h < GAP)).mean()),
                 'depth_cap_reached': len(retained) == cfg['maximum_homologs'],
                 'block_good_counts': [int(good[b].sum()) for b in blocks(length)],
                 'tokens_sha256': hashlib.sha256(tokens.tobytes()).hexdigest()})
    return tokens, meta


BASE_PROFILE_NAMES = ['log_length', 'log_raw_depth', 'log_cached_depth', 'log_filtered_depth',
                      'log_retained_depth', 'log_neff80', 'fraction_comparable', 'log_taxonomic_families',
                      'log_taxonomic_effective_number', 'unknown_taxonomy_fraction', 'depth_cap_reached', 'quality_fraction']
BLOCK_PROFILE_NAMES = [f'frequency_{aa}' for aa in ALPHABET] + [
    'entropy_mean', 'entropy_std', 'entropy_q25', 'entropy_q75', 'occupancy_mean', 'occupancy_q25',
    'occupancy_min', 'query_agreement', 'ambiguous_fraction', 'good_position_fraction']
PROFILE_NAMES = BASE_PROFILE_NAMES + [f'{block}/{name}' for block in ['global', 'block0', 'block1', 'block2', 'block3'] for name in BLOCK_PROFILE_NAMES]


def profile(tokens, meta):
    if not meta['available']:
        raise ValueError('Profile requested without usable monomer')
    good = tokens[0] != PAD; length = tokens.shape[1]; rows = tokens[1:]
    result = [np.log1p(meta[k]) for k in ['length', 'raw_depth', 'cached_depth', 'filtered_depth', 'retained_depth', 'neff80']]
    result += [meta['fraction_comparable'], np.log1p(meta['taxonomic_families']), np.log1p(meta['taxonomic_effective_number']),
               meta['unknown_taxonomy_fraction'], float(meta['depth_cap_reached']), meta['quality_fraction']]
    for indices in [np.arange(length), *blocks(length)]:
        valid = indices[good[indices]]; value = np.zeros(len(BLOCK_PROFILE_NAMES), dtype=float)
        if len(valid):
            h = rows[:, valid]
            freq = np.array([(h == aa).mean(0) for aa in range(26)]).T
            nongap = freq[:, :25]; denom = nongap.sum(1, keepdims=True)
            conditional = np.divide(nongap, denom, out=np.zeros_like(nongap), where=denom > 0)
            entropy = -(conditional * np.log(np.maximum(conditional, 1e-12))).sum(1)
            occupancy = 1 - freq[:, GAP]
            value = np.r_[freq.mean(0), entropy.mean(), entropy.std(), np.quantile(entropy, [.25, .75]),
                          occupancy.mean(), np.quantile(occupancy, .25), occupancy.min(),
                          (h == tokens[0, valid]).mean(), ((h >= 20) & (h < GAP)).mean(), len(valid) / len(indices)]
        result.extend(value.tolist())
    out = np.asarray(result, dtype=float)
    if out.shape != (192,) or len(PROFILE_NAMES) != 192 or not np.isfinite(out).all():
        raise ValueError('Invalid fixed profile descriptor')
    return out


def pair_gate(a, b):
    if not a['available'] or not b['available']:
        return 0.
    return float(min(1., min(a['neff80'], b['neff80']) / 32) * min(a['quality_fraction'], b['quality_fraction']))
