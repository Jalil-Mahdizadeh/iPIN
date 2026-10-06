"""Post-hoc score-concordance diagnostic; never selects on labels or AP.

Full-sequence errors have a sharp onset above about 3570 tokens. Probe nearby
token caps against deposited individual scores, retain all candidates, and
verify any inferred cap on every affected row. Original primary outputs remain.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers.utils import logging
from native_model import ROOT,configure,load_model,tokenize
from metrics_eval import metrics
logging.set_verbosity_error()
configure()
d=pd.read_csv(ROOT/'data/Bernett_benchmarking/pairs_uniprot_seqs_test.csv')
full=pd.read_csv(ROOT/'results/bernett_full.csv').set_index('row_id')
ref=pd.read_csv(ROOT/'data/published/figure4_PLM-interact.csv')
errors=np.abs(full.score.to_numpy()-ref.score.to_numpy())
# Boundary probes and long rows chosen using lengths/score discrepancies only.
ids=sorted(set(full[full.original_tokens.between(3569,3577)].index.tolist()+full.nlargest(3,'original_tokens').index.tolist()))
assert len(ids)<30
model,tok,meta=load_model('bernett')
records=[]
with torch.inference_mode():
    for cap in [3569,3570,3571,3572,3573]:
        start=time.monotonic()
        for i in range(0,len(ids),3):
            batch=ids[i:i+3];features=tokenize(tok,[(d.iloc[j]['query'],d.iloc[j]['text']) for j in batch],cap).to('cuda')
            logits=model(features);probs=torch.sigmoid(logits).cpu().tolist()
            for j,v in zip(batch,probs):records.append(dict(row_id=j,cap=cap,score=v,reference_score=float(ref.iloc[j].score),abs_difference=abs(v-float(ref.iloc[j].score))))
        print('Probe cap',cap,'seconds',time.monotonic()-start,flush=True)
probes=pd.DataFrame(records)
probes.to_csv(ROOT/'results/bernett-cap-probes.csv',index=False)
summary=probes.groupby('cap').abs_difference.agg(['mean','max'])
print(summary.to_string(),flush=True)
# A uniquely near-exact cap explains an implementation detail. This is not
# a search for a cap with better classification performance.
candidates=summary.index[summary['max']<2e-5].tolist()
if len(candidates)!=1:
    (ROOT/'results/bernett-cap-diagnostic.json').write_text(json.dumps(dict(status='unresolved',candidate_caps=candidates,probe_rows=ids),indent=2)+'\n')
    raise RuntimeError('No unique cap explains the source scores; retain the unresolved finding.')
cap=int(candidates[0]);affected=full.index[full.original_tokens>cap].tolist()
rows=[];start=time.monotonic()
with torch.inference_mode():
    for i in range(0,len(affected),3):
        batch=affected[i:i+3];f=tokenize(tok,[(d.iloc[j]['query'],d.iloc[j]['text']) for j in batch],cap).to('cuda')
        logits=model(f);values=torch.stack([logits,torch.sigmoid(logits)],dim=1).cpu().tolist()
        for j,v in zip(batch,values):rows.append(dict(row_id=j,label=int(d.iloc[j].label),logit=v[0],score=v[1]))
        if i%90==0:print('Verify inferred cap',cap,i,'/',len(affected),flush=True)
changed=pd.DataFrame(rows).set_index('row_id')
changed.to_csv(ROOT/'results/bernett-inferred-cap-affected.csv')
merged=full.copy();merged.loc[changed.index,['logit','score']]=changed[['logit','score']];merged.loc[changed.index,'tokens']=cap
merged.to_csv(ROOT/'results/bernett-inferred-cap-all-rows.csv')
delta=np.abs(merged.score.to_numpy()-ref.score.to_numpy())
out=dict(status='score_concordance_explained',post_hoc=True,criterion='Unique cap gives max absolute probe score error <2e-5; no labels or AP used for selection.',inferred_token_cap=cap,
         probe_rows=ids,affected_rows=len(affected),full_test_rows=len(merged),
         mean_absolute_score_error=float(delta.mean()),max_absolute_score_error=float(delta.max()),classification_flips_at_0_5=int(((merged.score.to_numpy()>=.5)!=(ref.score.to_numpy()>=.5)).sum()),
         before_mean_absolute_score_error=float(errors.mean()),before_max_absolute_score_error=float(errors.max()),
         metrics=metrics(merged.label,merged.score),model=meta,verification_elapsed_seconds=time.monotonic()-start)
(ROOT/'results/bernett-cap-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
