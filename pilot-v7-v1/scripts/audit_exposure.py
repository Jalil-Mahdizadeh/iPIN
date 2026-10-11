"""Record the inspected released-head provenance without inferring missing membership."""
import hashlib
import urllib.request
from study import ROOT, PROJECT, config, atomic, sha, now


def main():
    commit=config()['encoder']['source_commit']
    urls=[f'https://api.github.com/repos/yoakiyama/MSA_Pairformer/git/trees/{commit}?recursive=1',
        f'https://raw.githubusercontent.com/yoakiyama/MSA_Pairformer/{commit}/MSA_Pairformer/regression.py',
        f'https://raw.githubusercontent.com/yoakiyama/MSA_Pairformer/{commit}/Cell_2026_analysis_notebooks/Figure2/Figure2_fit_ESMC_contact_heads.ipynb']
    sources=[]
    for url in urls:
        with urllib.request.urlopen(url,timeout=25) as f:data=f.read()
        sources.append({'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    atomic(ROOT/'provenance/exposure-audit.json',{'at_utc':now(),'source_commit':commit,
        'weights_revision':config()['encoder']['weights_revision'],'head':'contact.bin:contact_head',
        'sources_inspected':sources,'membership_resolved':False,'overlap_assessable':False,
        'finding':'No verified training-membership list for the released MSA Pairformer C-beta head was identified in these artifacts. The ESMC notebook references an absent MSA_Transformer_training_set.txt, but that is not proof of the MSA Pairformer head membership.',
        'existing_review_sha256':sha(PROJECT/'improvment-proposal-vx.md'),
        'claim_limit':'Exploratory Bernett utility only; no exposure-free generalization claim. No inference that missing membership means zero overlap.',
        'test_accessed':False})
    print({'exposure_audit_recorded':True,'membership_resolved':False},flush=True)


if __name__=='__main__':main()
