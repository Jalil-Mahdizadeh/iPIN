"""Freeze the fully qualified protocol before any new Bernett head is fitted."""
import ast
import subprocess
import sys
from study import ROOT, PROJECT, config, read, atomic, sha, now, code_hashes, remaining


def main():
    if (ROOT/'provenance/freeze.json').exists():raise RuntimeError('Already frozen')
    for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text())
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True)
    atomic(ROOT/'qualification/unit.json',{'at_utc':now(),'passed':True})
    q=read(ROOT/'qualification/real.json')
    if not q['passed'] or q['labels_used'] or q['code_sha256']!=code_hashes():raise ValueError('Qualification/code mismatch')
    native=read(ROOT/'qualification/native.json')
    if not native['passed'] or native['code_sha256']!=sha(ROOT/'scripts/native.py'):raise ValueError('Native qualification missing or stale')
    if q['conservative_total_seconds']>remaining(config()['budgets']['shutdown_reserve_seconds']):raise TimeoutError('Qualified work no longer fits')
    source=read(ROOT/'provenance/inputs.json'); files={}
    for rel,digest in source['source_files'].items():
        if sha(PROJECT/rel)!=digest:raise ValueError('Source changed: '+rel)
        files[rel]=digest
    paths=[ROOT/'README.md',ROOT/'config.json',PROJECT/'improvment-proposal-v7.md',*sorted((ROOT/'scripts').glob('*.py')),
        *sorted((ROOT/'tests').glob('*.py')),*sorted((ROOT/'data').glob('*')),
        ROOT/'qualification/real.json',ROOT/'qualification/native.json',ROOT/'qualification/unit.json',
        ROOT/'provenance/inputs.json',ROOT/'provenance/diagnostic-source-digests.json',ROOT/'provenance/exposure-audit.json',
        ROOT/'provenance/resource-start.json',ROOT/'provenance/resource-plan-revision.json',ROOT/'provenance/authorization.json']
    for p in paths:
        if p.is_file() and '.tmp-' not in p.name:files[str(p.relative_to(PROJECT))]=sha(p)
    image=PROJECT/config()['image']
    if sha(image)!=config()['image_sha256']:raise ValueError('Pinned SIF mismatch')
    atomic(ROOT/'provenance/freeze.json',{'at_utc':now(),'files':files,'image':config()['image'],
        'image_sha256':config()['image_sha256'],'test_accessed':False,'new_bernett_heads_fitted':False,
        'production_authorized':False})
    print({'frozen':True,'fingerprint':sha(ROOT/'provenance/freeze.json')},flush=True)


if __name__=='__main__':main()
