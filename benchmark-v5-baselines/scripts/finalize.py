"""Final provenance, resource accounting and artifact checks after evaluation."""
import csv
import gzip
import ast
import joblib
import numpy as np
from common import *
from fit_predict import features

def resources(path):
    result = {}
    mapping = {'User time (seconds)': 'user_seconds', 'System time (seconds)': 'system_seconds',
               'Maximum resident set size (kbytes)': 'max_rss_kib', 'Exit status': 'exit_status'}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        for text, key in mapping.items():
            if line.startswith(text + ':'):
                result[key] = float(line[len(text)+1:].strip())
        prefix = 'Elapsed (wall clock) time (h:mm:ss or m:ss):'
        if line.startswith(prefix):
            fields = list(map(float, line[len(prefix):].strip().split(':')))
            elapsed = 0.0
            for field in fields:
                elapsed = elapsed*60 + field
            result['wall_seconds'] = elapsed
    assert result.get('exit_status') == 0, (path, result)
    return result

def main():
    verify_protocol()
    # Verify that this is the requested transfer of fixed methods, not a new model variant.
    inherited = read(ROOT / 'provenance/inherited-code.json')
    for item in inherited['files']:
        assert sha(item['source']) == item['source_sha256']
    for name in ['fit_predict.py', 'search.py']:
        old = (ORIGINAL_BASELINES / 'scripts' / name).read_text().replace('human-train', 'train')
        assert old == (ROOT / 'scripts' / name).read_text(), name
    def nodes(path):
        tree = ast.parse(Path(path).read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name, functions in [('common.py', ['degrees', 'Ranking']),
                            ('evaluate.py', ['write_csv', 'metrics', 'bootstrap'])]:
        original = nodes(ORIGINAL_BASELINES / 'scripts' / name)
        current = nodes(ROOT / 'scripts' / name)
        assert all(original[f] == current[f] for f in functions), name
    assert CONFIG == read(ROOT / 'provenance/original-v11-protocol-freeze.json')['config']
    prepared = read(ROOT / 'provenance/prepared.json')
    predictions = read(ROOT / 'provenance/predictions-frozen.json')
    summary = read(ROOT / 'results/summary.json')
    assert summary['complete'] and set(summary['datasets']) == set(DATASETS)
    for item in prepared['artifacts'] + predictions['artifacts']:
        verify(item)
    for name, digest in predictions['scripts'].items():
        assert sha(ROOT / 'scripts' / name) == digest
    meta = read(ROOT / 'data/sequences.json')
    model = joblib.load(ROOT / 'models/sequence-propensity.joblib')
    indices = np.random.default_rng(19).choice(len(meta['sequence']), 25, replace=False)
    saved = np.load(ROOT / 'models/protein-values.npz', allow_pickle=False)
    assert np.allclose(model.predict(features([meta['sequence'][i] for i in indices])),
                       saved['sequence_propensity'][indices], atol=1e-12, rtol=1e-12)
    rows_checked = 0
    for dataset in DATASETS:
        expected = summary['datasets'][dataset]['rows']
        with gzip.open(ROOT / 'results' / (dataset + '-predictions.csv.gz'), 'rt', newline='') as inp:
            reader = csv.DictReader(inp)
            labels = np.load(ROOT / 'data' / (dataset + '-labels.npy'), allow_pickle=False)
            pairs = np.load(ROOT / 'data' / (dataset + '-pairs.npy'), allow_pickle=False)
            count = 0
            for i, row in enumerate(reader):
                assert int(row['source_row_1based']) == i+1 and int(row['label']) == int(labels[i])
                assert row['protein_a_sha256'] == meta['sha256'][pairs[i, 0]]
                assert row['protein_b_sha256'] == meta['sha256'][pairs[i, 1]]
                count += 1
            assert count == expected
            rows_checked += count
    image = PROJECT / 'images/plm-interact/plm-interact-native-arm64-v1.sif'
    image_record = record(image)
    assert image_record['sha256'] == 'e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'
    resource_data = {name: resources(ROOT / 'logs' / (name + '.resources.txt'))
                     for name in ['prepare', 'search', 'fit-predict', 'evaluate']}
    prior_eval = ROOT / 'logs/export-recovery-evaluate.resources.txt'
    if prior_eval.exists():
        resource_data['first-evaluation-before-export-optimization'] = resources(prior_eval)
        first_points = read(ROOT / 'provenance/first-evaluation-point-metrics.json')
        for dataset, models in first_points.items():
            for model, metrics in models.items():
                assert all(summary['datasets'][dataset]['models'][model][key] == val for key, val in metrics.items())
    atomic(ROOT / 'provenance/runtime.json', {'at_utc': now(), 'container': image_record,
            'environment': predictions['environment'], 'phase_resources': resource_data,
            'new_gpu_computation': False, 'cpu_thread_cap': 16,
            'model_size_bytes': (ROOT / 'models/sequence-propensity.joblib').stat().st_size,
            'new_inference_not_including_evaluation_wall_seconds': sum(resource_data[k]['wall_seconds'] for k in ['search', 'fit-predict']),
            'allocation': 'CPU portion of existing interactive GH200 allocation; no new SLURM job',
            'notes': 'Shared homology search counted once. Resource files include validation and symmetry checks in the scoring phase. No historical neural cost estimate is inferred.'})
    assert all((ROOT / 'results' / ('comparison.' + ext)).is_file() for ext in ['png', 'pdf', 'svg'])
    artifacts = []
    complete = ROOT / 'results/COMPLETE.json'
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == complete:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] == 'cache' or '__pycache__' in rel.parts or path.suffix in ['.lock', '.partial']:
            continue
        if rel.parts[0] == 'work' and rel.as_posix() != 'work/alignments.tsv':
            continue
        if rel.as_posix() in ['logs/run.log', 'logs/finalize.log']:
            continue
        artifacts.append({'path': rel.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    atomic(complete, {'complete': True, 'at_utc': now(), 'artifacts': artifacts,
                       'checks': {**summary['checks'], 'all_archived_and_prepared_inputs_verified': True,
                                  'regressor_reload_predictions_identical': True,
                                  'scientific_implementation_and_config_unchanged_from_v11': True,
                                  'exported_source_rows_verified': rows_checked, 'container_hash_verified': True}})
    print({'complete': True, 'artifacts': len(artifacts), 'exported_rows_checked': rows_checked,
           'resources': resource_data}, flush=True)

if __name__ == '__main__':
    main()
