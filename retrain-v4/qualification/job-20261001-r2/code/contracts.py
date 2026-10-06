"""Small stdlib-only manifest helpers shared by host launchers and the trainer."""
import hashlib
import json
from pathlib import Path

CORE = ['train.py', 'model.py', 'model_esm2.py', 'model_esmc.py',
        'data.py', 'state.py', 'contracts.py', 'release.py']
RUNS = ['esm2-chain-aware-official-seed2', 'esmc-standard-official-seed2',
        'esmc-chain-aware-official-seed2']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def core_hashes(code):
    return {name: sha256(Path(code) / name) for name in CORE}


def production_config(cfg):
    expected = dict(seed=2, partition='official', world_size=4, total_updates=12745,
        global_pairs_per_update=64, learning_rate=2e-5, weight_decay=.01,
        warmup_updates=2000, clip_grad_norm=1., classification_weight=10.,
        positive_weight=1., classification_corruption=False, mlm_weight=0.,
        readout='residue_mean', readout_width=128, attention_backend='efficient',
        train_cap_residues=None, train_token_budget=8192, train_max_pairs=4,
        eval_token_budget=16384, eval_max_pairs=8, checkpoint_every_updates=100,
        checkpoint_minutes=15, validate_every_updates=1000, log_every_updates=10,
        selection_metric='pooled_ap')
    for key, value in expected.items():
        assert cfg[key] == value, ('Production protocol changed', key, cfg[key], value)
    assert cfg['name'] in RUNS
    options = dict(zip(RUNS, [('S1', 'esm2', 'chain_aware'), ('C0', 'esmc', 'standard'),
                            ('C1', 'esmc', 'chain_aware')]))
    assert (cfg['arm'], cfg['backbone'], cfg['attention_mode']) == options[cfg['name']]
    assert cfg['initialization'] == cfg['backbone'] + '_pretrained'


def verify_inputs(root, backbone=None, images=True):
    root = Path(root)
    manifest = json.loads((root / 'data/prepared/manifest.json').read_text())
    assert not manifest['test_imported'] and not manifest['partition_changed']
    for name, expected in manifest['files'].items():
        assert sha256(root / 'data/prepared' / name) == expected, ('Data changed', name)
    for item in json.loads((root / 'provenance/downloads.json').read_text()):
        relevant = backbone is None or item['local_path'].startswith('assets/' + backbone + '/')
        if relevant:
            assert sha256(root / item['local_path']) == item['sha256'], ('Initialization changed', item['local_path'])
    if images:
        for name, info in json.loads((root / 'provenance/images.json').read_text()).items():
            if backbone is None or name == backbone:
                assert sha256(root.parent / info['relative_path']) == info['sha256'], ('SIF changed', name)


def verify_evidence(root, relative, expected):
    p = Path(root) / relative
    assert sha256(p) == expected, ('Qualification report changed', relative)
    value = json.loads(p.read_text())
    assert value['passed'], relative
    for name, digest in value.get('source_sha256', {}).items():
        assert sha256(Path(root) / name) == digest, ('Qualification input changed', name)
    return value
