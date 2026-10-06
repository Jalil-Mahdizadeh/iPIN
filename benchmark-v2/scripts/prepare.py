"""Freeze all four validation-selected v2 checkpoints and reusable v1 artifacts."""
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OLD = WORK / 'benchmark-v1'
assert not (ROOT / 'provenance/selection.json').exists(), 'Selection already frozen'
for name in ['scripts', 'data', 'checkpoints', 'predictions', 'reused', 'results', 'logs', 'provenance', 'slurm', 'cache']:
    (ROOT / name).mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def read(path): return json.loads(path.read_text())
def save(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

old_manifest = {}
for line in (OLD / 'provenance/artifact-sha256.txt').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    old_manifest[(WORK / name).resolve()] = digest
verified = []

def check_old(path):
    expected = old_manifest[path.resolve()]
    assert sha(path) == expected, str(path)
    verified.append({'path': str(path), 'sha256': expected})
    return expected

def copy_old(source, dest):
    digest = check_old(source)
    shutil.copyfile(source, dest)
    assert sha(dest) == digest
    return digest

old_selection = read(OLD / 'provenance/selection.json')
for name in ['config.json', 'completed.json', 'PROTOCOL.md']:
    copy_old(OLD / name, ROOT / 'provenance' / ('benchmark-v1-' + name))
for name in ['selection.json', 'qualification.json', 'input-audit.json', 'environment.json', 'precision-outlier-audit.json']:
    copy_old(OLD / 'provenance' / name, ROOT / 'provenance' / ('benchmark-v1-' + name))
for name in ['common.py', 'infer.py', 'frozen_data.py', 'frozen_model.py']:
    copy_old(OLD / 'scripts' / name, ROOT / 'scripts' / name)
data_hashes = {}
for name in ['test.npy', 'val.npy', 'tokens.npy', 'offsets.npy', 'prepared-manifest.json']:
    data_hashes[name] = copy_old(OLD / 'data' / name, ROOT / 'data' / name)
    if name in old_selection['data']['sha256']:
        assert data_hashes[name] == old_selection['data']['sha256'][name]

models, reuse = {}, {}
for name, previous in [('native-bernett', 'native-bernett'), ('v1-reference', 'reference-seed2'), ('v1-symmetric', 'symmetric-seed2')]:
    item = dict(old_selection['models'][previous])
    checkpoint = OLD / item['path']
    assert check_old(checkpoint) == item['sha256']
    if previous != 'native-bernett':
        current = read(WORK / 'retrain-v1/runs' / previous / 'best.json')
        assert current['sha256'] == item['sha256'] and current['update'] == item['update']
    target = ROOT / 'checkpoints' / f'{name}{checkpoint.suffix}'
    os.link(checkpoint, target)
    item['path'] = str(target.relative_to(ROOT))
    item['reused_from_model'] = previous
    item['test_predictions_reused'] = True
    models[name] = item
    reuse[name] = {}
    for split in ['test', 'val']:
        source = OLD / 'results' / f'{previous}-{split}.npz' if split == 'test' or previous == 'native-bernett' else OLD / 'predictions' / f'{previous}-selected-validation.npz'
        destination = ROOT / 'reused' / f'{name}-{split}.npz'
        digest = copy_old(source, destination)
        reuse[name][split] = {'source_path': str(source), 'path': str(destination.relative_to(ROOT)), 'sha256': digest}

new_names = []
release = WORK / 'retrain-v2/releases' / (WORK / 'retrain-v2/releases/CURRENT').read_text().strip()
for short, run_name in [('reference', 'reference-official-seed2'), ('capped', 'capped-official-seed2'),
                        ('positive10', 'positive10-official-seed2'), ('clean-bce', 'clean-bce-official-seed2')]:
    name = 'v2-' + short
    new_names.append(name)
    run = WORK / 'retrain-v2/runs' / run_name
    best, completed, contract = [read(run / f'{f}.json') for f in ['best', 'completed', 'contract']]
    assert completed['state']['update'] == 12745 and not completed['state']['pending_validation']
    assert completed['state']['best_update'] == best['update']
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    validations = [e for e in events if e['event'] == 'validation']
    selected = max(validations, key=lambda e: e['metrics']['pooled_ap'])
    assert selected['update'] == best['update'] and selected['metrics']['pooled_ap'] == best['best_ap']
    assert contract['configuration']['readout'] == 'cls_linear' and contract['configuration']['attention_mode'] == 'standard'
    source = run / 'checkpoints' / best['file']
    assert source.stat().st_size == best['bytes'] and sha(source) == best['sha256']
    destination = ROOT / 'checkpoints' / f'{name}-{best["file"]}'
    os.link(source, destination)
    validation = run / 'validation' / f'update-{best["update"]:09d}.npz'
    metadata = read(validation.with_suffix('.json'))
    assert metadata['fingerprint'] == contract['fingerprint'] and sha(validation) == metadata['sha256']
    validation_target = ROOT / 'predictions' / f'{name}-selected-validation.npz'
    shutil.copyfile(validation, validation_target)
    assert sha(validation_target) == metadata['sha256']
    for part in ['contract', 'best', 'completed']:
        shutil.copyfile(run / f'{part}.json', ROOT / 'provenance' / f'{name}-{part}.json')
    models[name] = {'kind': 'local_training_checkpoint', 'path': str(destination.relative_to(ROOT)),
        'source_path': str(source), 'source_manifest': best, 'sha256': best['sha256'], 'update': best['update'],
        'validation_primary_ap': best['best_ap'], 'source_run': run_name, 'configuration': contract['configuration'],
        'snapshot_method': 'hard link to immutable atomically committed checkpoint', 'test_predictions_reused': False,
        'selected_validation': {'path': str(validation_target.relative_to(ROOT)), 'sha256': metadata['sha256'],
                                'source_path': str(validation)}}
for source, dest in [('model.py', 'frozen_v2_model.py'), ('data.py', 'frozen_v2_data.py')]:
    shutil.copyfile(release / 'code' / source, ROOT / 'scripts' / dest)

container = old_selection['container']
assert sha(Path(container['path'])) == container['sha256']
selection = {'selected_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'selection_basis': 'Best checkpoint from every completed initial v2 model, per existing maximum pooled validation AP rule; all four included at user request before new test inference',
    'models': models, 'fresh_inference_models': new_names, 'reused_predictions': reuse,
    'data': {'path': 'data', 'sha256': data_hashes, 'source': str(OLD / 'data')},
    'container': container, 'base_model_path': old_selection['base_model_path'],
    'v2_prepared_path': str(WORK / 'retrain-v2/data/prepared'),
    'historical_test_interpretation': 'Exploratory reuse; this test set has already informed prior research decisions'}
cfg = dict(read(OLD / 'config.json'))
cfg['models'] = new_names
cfg['splits_by_model'] = {name: ['test'] for name in new_names}
cfg['comparison_models'] = list(models)
save(ROOT / 'config.json', cfg)
save(ROOT / 'provenance/reuse-verification.json', {'checked_at_utc': selection['selected_at_utc'],
     'source_artifact_manifest_sha256': sha(OLD / 'provenance/artifact-sha256.txt'),
     'verified_old_files': verified, 'v1_selected_weights_unchanged': True,
     'reason_for_reuse': 'Identical complete input arrays, checkpoint identities, SIF, frozen model/data forward code, BF16/TF32 settings, pooling, shard count and batch settings; prediction coverage rechecked during qualification and analysis'})
save(ROOT / 'provenance/selection.json', selection)
print(json.dumps({'selected_at_utc': selection['selected_at_utc'],
    'fresh_models': {n: {'update': models[n]['update'], 'validation_ap': models[n]['validation_primary_ap']} for n in new_names},
    'reuse_models': list(reuse), 'verified_source_files': len(verified)}, indent=2))
