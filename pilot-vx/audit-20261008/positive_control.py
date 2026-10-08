"""Runtime sanity check on upstream 1B70 example, outside Bernett pilot evaluation.

Only this diagnostic calls the released contact head. It is not an independent
generalization test and its results do not enter any Bernett features or scores.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scripts'))
from common import atomic, now, sha, verify_freeze
from encoder import Encoder
from msa import records, encode
from MSA_Pairformer.dataset import prepare_msa_masks


@torch.inference_mode()
def predict(enc,tokens,breakpoint):
    t=torch.as_tensor(tokens.astype(np.int64),device='cuda').unsqueeze(0)
    m=prepare_msa_masks(t,device=enc.device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=enc.model.predict_cb_contacts(msa=torch.nn.functional.one_hot(t,28).float(),
            mask=m[0],msa_mask=m[1],full_mask=m[2],pairwise_mask=m[3],complex_chain_break_indices=[[breakpoint]])
    return result['predicted_cb_contacts'][0,:breakpoint,breakpoint:].float().cpu().numpy()


def main():
    started=time.monotonic();fingerprint=verify_freeze()
    msa_path=HERE/'1B70_A_1B70_B.fas';structure_path=HERE/'1B70.cif'
    alignment=list(records(msa_path.read_text().splitlines()));query=alignment[0][1]
    structure=MMCIFParser(QUIET=True).get_structure('1B70',structure_path)[0]
    chains=[]
    for name in ['A','B']:
        residues=[r for r in structure[name] if r.id[0]==' ' and 'CA' in r]
        seq=''.join(seq1(r.resname) for r in residues)
        coords=np.array([r['CB' if 'CB' in r else 'CA'].coord for r in residues])
        chains.append((seq,coords))
    assert chains[0][0]+chains[1][0]==query, 'MSA query and resolved structure coordinates do not match exactly'
    boundary=len(chains[0][0]);assert boundary==265
    # Deterministic depth cap, retaining the query. No tuning on contact labels.
    indices=np.r_[0,np.random.default_rng(20261008).choice(np.arange(1,len(alignment)),min(127,len(alignment)-1),replace=False)]
    tokens=np.stack([encode(alignment[i][1]) for i in indices])
    null=tokens.copy();permutation=np.random.default_rng(82).permutation(np.arange(1,len(tokens)))
    null[1:,boundary:]=tokens[permutation,boundary:]
    dist=np.linalg.norm(chains[0][1][:,None,:]-chains[1][1][None,:,:],axis=-1)
    truth=dist<8
    enc=Encoder();out={}
    for name,t in [('paired',tokens),('globally_shuffled',null),('query_only',tokens[:1])]:
        pred=predict(enc,t,boundary);assert np.isfinite(pred).all()
        order=np.argsort(-pred.ravel(),kind='stable');y=truth.ravel()
        out[name]={'interface_contact_ap':float(average_precision_score(y,pred.ravel())),
                   'precision_top_10':float(y[order[:10]].mean()),
                   'precision_top_50':float(y[order[:50]].mean()),
                   'maximum_contact_probability':float(pred.max())}
        print(json.dumps({name:out[name]}),flush=True)
    report={'at_utc':now(),'fingerprint':fingerprint,'script_sha256':sha(__file__),
            'source_commit':'875363570df1ae484cf725aba382790444223005',
            'msa_sha256':sha(msa_path),'structure_url':'https://files.rcsb.org/download/1B70.cif',
            'structure_sha256':sha(structure_path),'query_coordinates_exact':True,
            'chain_lengths':[len(x[0]) for x in chains],'msa_depth':len(tokens),
            'contact_definition':'C-beta distance <8 Angstrom; C-alpha for glycine',
            'interface_contacts':int(truth.sum()),'interface_residue_pairs':int(truth.size),
            'results':out,'diagnostic_contact_head_used':True,'bernett_data_used':False,
            'independent_generalization_test':False,'seconds':time.monotonic()-started}
    atomic(HERE/'positive-control.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
