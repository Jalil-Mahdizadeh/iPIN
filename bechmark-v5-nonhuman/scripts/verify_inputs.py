"""Check the already frozen runtime/checkpoint manifests without rewriting them."""
from bench_utils import ROOT,read,sha
from pathlib import Path

def verify(item):
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],p
 print('Verified',p.name,flush=True)
def main():
 runtime=read(ROOT/'provenance/runtime-inputs.json');assert runtime['verified']
 for name,item in runtime['items'].items():verify(item)
 for model in read(ROOT/'provenance/selection.json')['models'].values():verify(model['checkpoint'])
 print('Frozen input identities verified',flush=True)
if __name__=='__main__':main()
