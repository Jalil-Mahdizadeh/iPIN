"""Numerical and tokenization qualification against unmodified upstream class AST."""
import ast
import csv
import json
import time
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM
from native_model import ROOT, ASSETS, configure, load_model, tokenize

configure()
model, tok, meta = load_model('human')
tree = ast.parse((ROOT/'provenance/upstream/predict_ddp.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PLMinteract')
namespace = dict(torch=torch, nn=nn, F=F, AutoModelForMaskedLM=AutoModelForMaskedLM)
exec(compile(ast.Module(body=[node], type_ignores=[]), 'upstream/predict_ddp.py', 'exec'), namespace)
with patch.object(AutoModelForMaskedLM, 'from_pretrained', side_effect=lambda *a, **kw: AutoModelForMaskedLM.from_config(kw['config'])):
    upstream = namespace['PLMinteract'](str(ASSETS/'esm2_650m'), 1, model.config, 'cuda', 1280)
upstream.load_state_dict(model.state_dict(), strict=True)
upstream.eval().requires_grad_(False).cuda()
with open(ROOT/'data/cross_species_benchmarking/test/mouse.ppi.qrels.seq.test.csv') as f:
    rows = list(csv.DictReader(f))
indices = [0, 1, 2, 17, 150, max(range(len(rows)), key=lambda i: len(rows[i]['query'])+len(rows[i]['text']))]
pairs = [(rows[i]['query'], rows[i]['text']) for i in indices]
features = tokenize(tok, pairs, 1603).to('cuda')
assert torch.equal(features.input_ids, tok([[a,b] for a,b in pairs], padding=True, truncation=True, max_length=1603, return_tensors='pt').input_ids.cuda())
with torch.inference_mode():
    native = model(features)
    _, reference, p = upstream.forward_test(torch.zeros(len(pairs), device='cuda'), features)
    assert torch.equal(native, reference)
    singleton = torch.cat([model(tokenize(tok, [p], 1603).to('cuda')) for p in pairs])
    max_delta = float((native-singleton).abs().max())
    assert max_delta < 1e-4, max_delta
    # Padding at the right should not change the intended result.
    padded = model(tokenize(tok, pairs, 1603, padding='max_length').to('cuda'))
    pad_delta = float((native-padded).abs().max())
    assert pad_delta < 1e-4, pad_delta
del upstream
torch.cuda.empty_cache()
pilot = sorted(rows[:512], key=lambda r: len(r['query'])+len(r['text']))
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); start=time.monotonic()
with torch.inference_mode():
    for i in range(0,len(pilot),32):
        model(tokenize(tok, [(r['query'],r['text']) for r in pilot[i:i+32]],1603).to('cuda'))
torch.cuda.synchronize(); duration=time.monotonic()-start
out=dict(model=meta, upstream_class_bitwise_equal=True, pair_tokenization_equal=True,
         test_rows=indices, max_singleton_logit_delta=max_delta, max_fixed_padding_logit_delta=pad_delta,
         pilot_pairs=len(pilot), pilot_seconds=duration, pairs_per_second=len(pilot)/duration,
         peak_gpu_bytes=torch.cuda.max_memory_allocated())
(ROOT/'results/qualification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
