"""Bind the minimal DDP storage amendment to full-size resume evidence."""
import ast
import json
from pathlib import Path
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
CORE = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']


def main():
    old = ROOT / 'releases/20260930-adaptation/code'
    new = ROOT / 'readout-code'
    before = ast.parse((old / 'train.py').read_text())
    after = ast.parse((new / 'train.py').read_text())
    count = 0
    for node in ast.walk(before):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'DDP':
            for keyword in node.keywords:
                if keyword.arg == 'gradient_as_bucket_view':
                    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    keyword.value.value = False
                    count += 1
    assert count == 1 and ast.dump(before) == ast.dump(after), 'Unexpected trainer changes'
    for name in CORE[1:]:
        assert sha256(old / name) == sha256(new / name), name
    report = json.loads((ROOT / 'qualification/readout-aligned-ddp.json').read_text())
    assert report['passed'] and report['code'] == {n: sha256(new / n) for n in CORE}
    for name in report['comparison_reports']:
        result = json.loads((ROOT / name).read_text())
        assert result['passed'] and result['bitwise_identical']
    files = ['readout-code/' + n for n in CORE]
    files += ['qualification/readout-aligned-ddp.json',
              'qualification/readout-resume-failure-diagnostics.json',
              'qualification/gradient-storage-offset-diagnostic.json',
              'scripts/qualify_readout_engine.py'] + report['comparison_reports']
    atomic_json(ROOT / 'qualification/readout-engine-amendment.json', {
        'passed': True, 'code': report['code'],
        'scope': 'AST verifies the only executable trainer change is DDP gradient_as_bucket_view=True to False; all other core files are identical. Both full-size heads pass four-GPU bitwise resume.',
        'initial_failed_qualification_job': '3196205',
        'corrected_qualification_job': report['job_id'],
        'model_architecture_objective_optimizer_schedule_data_unchanged': True,
        'original_controls_keep_original_code': True,
        'extra_gradient_storage_bytes': 650057090 * 4,
        'limitation': 'Qualifies the observed same-hardware/software resume case, not every possible CUDA/backend version or future workload. Existing controls used bucket views; no claim that all amended versus original trajectories are bitwise identical.',
        'source_sha256': {n: sha256(ROOT / n) for n in files}})
    print('Readout engine amendment is minimal and qualified.')


if __name__ == '__main__':
    main()
