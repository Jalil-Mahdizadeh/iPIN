"""Freeze qualified code, data, and artifacts before any fitted-head evaluation."""
import ast
import json
import subprocess
import sys
from common import ROOT, PROJECT, atomic, now, sha


def main():
    if (ROOT/'provenance/freeze.json').exists():raise RuntimeError('Already frozen; do not silently replace a release')
    protocol=json.loads((ROOT/'provenance/protocol-freeze.json').read_text())
    for rel,digest in protocol['files'].items():
        assert sha(ROOT/rel)==digest,rel
    for path in (ROOT/'scripts').glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    subprocess.run(['bash','-n',str(ROOT/'slurm/pilot.sbatch')],check=True)
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True)
    unit={'passed':True,'at_utc':now(),'tests':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').glob('*.py'))}}
    atomic(ROOT/'qualification/unit.json',unit)
    for rel in ['qualification/encoder.json','qualification/real-encoder.json']:
        assert json.loads((ROOT/rel).read_text())['passed']
    real=json.loads((ROOT/'qualification/real-encoder.json').read_text())
    for rel,digest in real['code_sha256'].items():assert sha(ROOT/rel)==digest,('real qualification code changed',rel)
    for case in real['cases']:
        for rel,digest in case['monomer_files'].items():assert sha(ROOT/rel)==digest,('real qualification input changed',rel)
    max_case=json.loads((ROOT/'qualification/max-unmasked.json').read_text())
    assert max_case['passed'] and max_case['encoder_sha256']==sha(ROOT/'scripts/encoder.py')
    archive=json.loads((ROOT/'results/archive-coverage.json').read_text());assert archive['complete']
    # Compare deterministic cache outputs made before the performance-only parser change.
    before=json.loads((ROOT/'provenance/cache-before-parser-optimization.json').read_text())
    for rel,digest in before['files'].items():assert sha(ROOT/rel)==digest,('parser equivalence',rel)
    paths=[ROOT/'config.json',*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),ROOT/'slurm/pilot.sbatch']
    paths += [ROOT/'data'/n for n in ['sample.json','proteins.json','dev-groups.json','coverage-proteins.json',
                                    'test-sequence-hashes.json','baseline.json','monomer-catalog.json']]
    paths += [ROOT/'qualification'/n for n in ['unit.json','encoder.json','real-encoder.json','max-unmasked.json']]
    paths += [ROOT/'provenance/protocol-freeze.json',ROOT/'results/archive-coverage.json']
    cfg=json.loads((ROOT/'config.json').read_text())
    image=PROJECT/'images/msa-pairformer/msa-pairformer-arm64-v1.sif'
    atomic(ROOT/'provenance/freeze.json',{'at_utc':now(),'files':{str(p.relative_to(PROJECT)):sha(p) for p in paths},
        'artifacts':{str(image.relative_to(PROJECT)):sha(image),
                     str((ROOT/'data'/cfg['archive']['filename']).relative_to(PROJECT)):cfg['archive']['sha256']},
        'test_labels_used':False,'primary_encoder':'trunk only','production_authorized':False,
        'parser_optimization_cache_equivalence_checked':len(before['files'])})
    print(json.dumps({'frozen':True,'files':len(paths),'at_utc':now()}),flush=True)


if __name__=='__main__':main()
