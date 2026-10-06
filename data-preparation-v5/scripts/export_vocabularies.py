"""Run under the existing ESMC SIF; load tokenizers only, never model weights."""
from pathlib import Path
import json
from transformers import AutoTokenizer
from esm.tokenization import EsmSequenceTokenizer

root=Path(__file__).resolve().parents[1]
models={'esm2':AutoTokenizer.from_pretrained(root.parent/'retrain-v1/assets/esm2',local_files_only=True),
        'esmc':EsmSequenceTokenizer()}
out={}
for name,t in models.items():
    vocab=t.get_vocab();sequence='ACDEFGHIKLMNPQRSTVWYXBZUO'
    assert t.encode(sequence,add_special_tokens=False)==[vocab[c] for c in sequence]
    out[name]=dict(vocabulary=vocab,cls_token_id=t.cls_token_id,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id,
                   mask_token_id=t.mask_token_id,layout='CLS A EOS B EOS',no_truncation=True)
(root/'environment/vocabularies.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('Both backbone residue vocabularies verified; no model weights loaded.')
