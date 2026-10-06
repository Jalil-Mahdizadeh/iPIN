"""Count mutations erased by the released tokenizer's longest-first truncation."""
import json
import pandas as pd
from transformers import AutoTokenizer
from transformers.utils import logging
from native_model import ROOT,ASSETS
logging.set_verbosity_error()
tokenizer=AutoTokenizer.from_pretrained(ASSETS/'esm2_650m',local_files_only=True)
tokenizer.pad_token=tokenizer.eos_token
d=pd.read_csv(ROOT/'data/Mutation_effect_dataset/test_mutation_data.csv',keep_default_na=False)
published=set(pd.read_csv(ROOT/'data/mutation-figure5-mapping.csv').row_id)
out=[]
for cap in [None,1603,2196]:
    kw=dict(padding=False,truncation=False) if cap is None else dict(padding=False,truncation='longest_first',max_length=cap)
    w=tokenizer(d.wild_seq.tolist(),d.participant_sequence.tolist(),**kw)['input_ids']
    m=tokenizer(d.mutant_seq.tolist(),d.participant_sequence.tolist(),**kw)['input_ids']
    identical=[i for i,(a,b) in enumerate(zip(w,m)) if a==b]
    out.append(dict(token_cap=cap,identical_wild_mutant_inputs_all841=len(identical),identical_wild_mutant_inputs_published598=len(set(identical)&published),row_ids=identical))
(ROOT/'results/mutation-truncation-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print([{k:v for k,v in r.items() if k!='row_ids'} for r in out])
