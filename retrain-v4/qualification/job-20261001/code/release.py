"""Production requires a frozen, hash-bound, qualified three-run release."""
import json
from pathlib import Path
from contracts import RUNS, core_hashes, production_config, sha256, verify_evidence, verify_inputs


def verify_release(root, config_path, code_dir, verify_assets=True):
    root, config_path, code_dir = map(Path, (root, config_path, code_dir))
    release_path = code_dir.parent
    assert release_path.is_relative_to(root / 'releases')
    m = json.loads((release_path / 'release.json').read_text())
    assert m['ready_for_submission'] and m['enabled_runs'] == RUNS
    assert m['automatic_benchmark'] is False and m['production_submitted_at_preparation'] is False
    assert config_path.is_relative_to(release_path / 'configs')
    for name, expected in m['files'].items():
        assert sha256(release_path / name) == expected, ('Frozen release changed', name)
    cfg = json.loads(config_path.read_text())
    production_config(cfg)
    assert cfg['root'] == str(root.resolve())
    assert config_path.name == cfg['name'] + '.json'
    assert sha256(config_path) == m['files'][str(config_path.relative_to(release_path))]
    for name, expected in m['input_manifests'].items():
        assert sha256(root / name) == expected, ('Input manifest changed', name)
    for name, expected in m['qualification_reports'].items():
        report = verify_evidence(root, name, expected)
        if 'code' in report:
            assert report['code'] == core_hashes(code_dir), ('Qualification code differs', name)
    if verify_assets:
        verify_inputs(root, backbone=cfg['backbone'])
    return m
