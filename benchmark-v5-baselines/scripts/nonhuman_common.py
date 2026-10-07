"""Extension metadata; original science/configuration are imported unchanged."""
from common import *

SPECIES = ['mouse', 'fly', 'worm', 'yeast', 'ecoli']
TRAINING = ['v5', 'bernett']
NEURAL = ['ipin-esm2', 'ipin-esmc', 'v2-capped', 'v2-clean-bce', 'native-plm']
ALL_MODELS = [ref + ':' + method for ref in TRAINING for method in BASELINES] + NEURAL
NLABELS = {**LABELS, 'v2-capped': 'iPIN v2 length-capped (Bernett)',
           'v2-clean-bce': 'iPIN v2 clean BCE (Bernett)'}
for ref in TRAINING:
    for method in BASELINES:
        NLABELS[ref + ':' + method] = ('v5' if ref == 'v5' else 'Bernett') + ': ' + LABELS[method]
SPECIES_LABELS = {'mouse': 'Mouse', 'fly': 'Fly', 'worm': 'Worm', 'yeast': 'Yeast', 'ecoli': 'E. coli'}
NONHUMAN = PROJECT / 'benchmark-v5-nonhuman'
HISTORICAL = NONHUMAN / 'v1-v4-comparison'
ARCHIVE = ROOT / 'archive/before-nonhuman'
NDATA = ROOT / 'data/nonhuman'
NMODELS = ROOT / 'models/nonhuman'
NWORK = ROOT / 'work/nonhuman'

def freeze_extension():
    verify_protocol()
    for p in [NDATA, NMODELS, NWORK, ROOT / 'provenance/nonhuman', ROOT / 'logs/nonhuman']:
        p.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'provenance/nonhuman/protocol-freeze.json'
    identity = {'addendum_sha256': sha(ROOT / 'NONHUMAN-ADDENDUM.md'), 'config': CONFIG,
                'training_references': TRAINING, 'species': SPECIES, 'neural_references': NEURAL,
                'original_scientific_scripts': {s: sha(ROOT / 'scripts' / s) for s in ['common.py', 'fit_predict.py', 'search.py', 'evaluate.py']}}
    if path.exists():
        assert read(path)['identity'] == identity, 'Frozen extension changed'
    else:
        atomic(path, {'at_utc': now(), 'identity': identity, 'new_species_baseline_metrics_not_yet_computed': True,
                      'previous_neural_and_FADI_results_known': True})
    return identity

def canonical_source(item):
    # Resolve the documented legacy path without depending on a compatibility symlink.
    p = Path(item['path'].replace('/bechmark-nonhuman-v5/', '/benchmark-v5-nonhuman/'))
    assert p.is_file() and sha(p) == item['sha256'], p
    return p

def verify_legacy_models():
    previous = read(ROOT / 'provenance/predictions-frozen.json')
    for s, digest in previous['scripts'].items(): assert sha(ROOT / 'scripts' / s) == digest
    for item in previous['artifacts']: verify(item)
    return previous
