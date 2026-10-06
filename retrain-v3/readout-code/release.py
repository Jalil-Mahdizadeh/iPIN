"""Verify immutable files and qualified provenance before a production process."""
import json
from pathlib import Path
from state import sha256


def verify_release(root, config_path, code_dir):
    manifest_path = code_dir.parent / 'release.json'
    if not manifest_path.is_file():
        raise RuntimeError('Production requires a frozen, qualified release; use the launch dry run first')
    release = json.loads(manifest_path.read_text())
    assert release['ready_for_submission'] is True
    assert config_path.is_relative_to(code_dir.parent / 'configs')
    assert sha256(config_path) == release['files'][str(config_path.relative_to(code_dir.parent))]
    for name, expected in release['files'].items():
        assert sha256(code_dir.parent / name) == expected, ('Frozen release changed', name)
    assert sha256(root / 'data/prepared/manifest.json') == release['data_manifest_sha256']
    assert sha256(root / 'provenance/downloads.json') == release['initialization_manifest_sha256']
    for relative, expected in release['qualification_reports'].items():
        path = root / relative
        assert sha256(path) == expected, ('Qualification evidence changed', relative)
        assert json.loads(path.read_text())['passed']
    image = root.parent / 'images/plm-interact/plm-interact-native-arm64-v1.sif'
    assert sha256(image) == release['sif_sha256']
    return release
