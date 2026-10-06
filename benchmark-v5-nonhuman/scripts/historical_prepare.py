"""Freeze the two user-selected historical controls without altering the v5 roster."""
import csv, errno, os, shutil
from pathlib import Path
from bench_utils import ROOT, PROJECT, atomic, now, read, record, sha

EXT = ROOT/'v1-v4-comparison'
NAMES = ['v2-capped', 'v2-clean-bce']


def main():
    for name in ['source', 'checkpoints', 'qualification', 'predictions', 'results', 'logs', 'provenance']:
        (EXT/name).mkdir(parents=True, exist_ok=True)
    if (EXT/'provenance/selection.json').exists():
        raise RuntimeError('Selection already frozen; do not overwrite it')
    old = read(PROJECT/'benchmark-v2/provenance/selection.json')
    source_files = {}
    for source, target in [(PROJECT/'benchmark-v2/scripts/frozen_model.py', EXT/'source/model.py'),
                           (PROJECT/'plm-interact-reproducability/scripts/native_model.py', EXT/'source/native_oracle.py')]:
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        source_files[str(target)] = {'origin': record(source), 'copy': record(target)}
    models = {}
    for name in NAMES:
        entry = old['models'][name]
        source = PROJECT/'benchmark-v2'/entry['path']
        assert sha(source) == entry['sha256']
        target = EXT/'checkpoints'/source.name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError as error:
                if error.errno != errno.EXDEV: raise
                # The container exposes the frozen source and output through
                # different mount points. Reuse the immutable prior snapshot.
                target.symlink_to(source)
        assert sha(target) == entry['sha256']
        dev = PROJECT/'benchmark-v2'/entry['selected_validation']['path']
        assert sha(dev) == entry['selected_validation']['sha256']
        models[name] = {**entry, 'checkpoint': record(target), 'selected_dev': record(dev),
                       'reuse_method': 'Verified reference to the frozen benchmark-v2 checkpoint; no weight conversion'}
    runtime = read(ROOT/'provenance/runtime-inputs.json')['items']['native-runtime']
    assert sha(runtime['path']) == runtime['sha256']
    with (PROJECT/'benchmark-v4/results/metrics.csv').open() as f:
        history = [r for r in csv.DictReader(f) if r['view']=='pooled' and r['group']=='all' and r['threshold_rule']=='fixed_0.5']
    native = next(r for r in history if r['model']=='native-bernett')
    closest = min((r for r in history if r['model']!='native-bernett'), key=lambda r: abs(float(r['ap'])-float(native['ap'])))
    assert closest['model'] == 'v2-clean-bce'
    selection = {'at_utc': now(), 'models': models, 'source_files': source_files,
        'user_choice': 'Both: closest available training setup and closest prior Bernett performance; separately reported',
        'selection_reason': {'v2-capped': 'Native ESM2 CLS-linear head, standard attention, masked BCE plus MLM, and the native combined-residue training cap of 2193. Not an exact native trainer reproduction.',
                             'v2-clean-bce': 'Closest prior pooled Bernett AP and AUROC among evaluated v1-v4 models; preserves the existing DEV-selected update 4000.'},
        'prior_bernett_comparison': history, 'source_selection': record(PROJECT/'benchmark-v2/provenance/selection.json'),
        'base': old['base_model_path'], 'runtime': runtime,
        'prepared_inputs': record(ROOT/'provenance/prepared.json'),
        'dev_inputs': {name: record(PROJECT/'benchmark-v2/data'/name) for name in ['val.npy', 'tokens.npy', 'offsets.npy']},
        'precision': 'FP32 weights, BF16 autocast, TF32 disabled', 'world_size': 4,
        'token_budget': 16384, 'max_pairs': 8, 'commit_rows': 512,
        'prediction': 'Mean raw AB/BA logits; both orientations retained; full released sequences',
        'nonhuman_scores_used_for_selection': False, 'new_training': False}
    atomic(EXT/'provenance/selection.json', selection)
    print({n: {'update':models[n]['update'], 'sha256':models[n]['sha256']} for n in NAMES}, flush=True)


if __name__ == '__main__': main()
