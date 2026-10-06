"""Frozen humanV11 on the Bernett/ILP union; full FP32, resumable four shards."""
import argparse,fcntl,hashlib,json,os,socket,time
import numpy as np
import torch
from scipy.special import expit
from transformers import AutoTokenizer
from bench_utils import ROOT,PROJECT,atomic,cuda,load_npz,now,read,record,save_npz,sha
from pair_infer import data_arrays,module,test_data

MANIFEST=ROOT/'provenance/human-releases.json'
def signature():
    m=read(MANIFEST)
    for p,h in m['inference_code'].items(): assert sha(ROOT/p)==h,p
    return hashlib.sha256(json.dumps({'manifest':sha(MANIFEST),'torch':torch.__version__,
        'precision':'FP32, TF32 disabled','world':4,'token_budget':16384,'max_pairs':8,'chunk_rows':256},sort_keys=True).encode()).hexdigest()

def load_model(device,backend='efficient'):
    m=read(MANIFEST)['models']['native-human'];p=m['checkpoint']['path']
    assert sha(p)==m['checkpoint']['sha256']
    state=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
    cls=module('human_release_native',ROOT/'scripts/native_model/model.py').PairModel
    model=cls(m['base'],mode='reference',backend=backend)
    model.load_state_dict(state,strict=True)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    model.esm_mask.gradient_checkpointing_disable()
    return model.eval().requires_grad_(False).to(device)

@torch.inference_mode()
def predict(model,data,ids,device):
    f=data.batch(ids,np.zeros(len(ids),np.int64),2,False,device)
    assert f['attention_mask'].sum(1).tolist()==[int(data.lengths[i]) for i in ids for _ in range(2)]
    z=model(f['clean_ids'],f['attention_mask']).cpu().numpy().astype(np.float64)
    assert z.shape==(len(ids),2) and np.isfinite(z).all()
    return z

@torch.inference_mode()
def qualify(device):
    contract=signature();model=load_model(device);oracle=load_model(device,'eager')
    tokenizer=AutoTokenizer.from_pretrained(read(MANIFEST)['models']['native-human']['base'],local_files_only=True)
    meta=read(ROOT/'data/sequences.json');data=test_data();order=np.argsort(data.lengths,kind='stable')
    fixtures=[int(order[j]) for j in [0,len(order)//2,int(.95*len(order)),-1]]
    cases=[];gold={}
    for i in fixtures:
        a,b=data.rows[i,:2];ss=[meta['sequence'][a],meta['sequence'][b]]
        inputs=tokenizer([ss[0],ss[1]],[ss[1],ss[0]],padding=True,truncation=False,return_tensors='pt').to(device)
        f=data.batch(np.array([i]),np.zeros(1,np.int64),2,False,device)
        assert torch.equal(inputs['input_ids'],f['clean_ids']) and torch.equal(inputs['attention_mask'],f['attention_mask'])
        z0=oracle(inputs['input_ids'],inputs['attention_mask']).cpu().numpy().astype(np.float64)
        z=predict(model,data,np.array([i]),device);le=float(abs(z-z0).max());pe=float(abs(expit(z)-expit(z0)).max())
        assert le<=.002 and pe<=.0001,(i,le,pe)
        gold[i]=z[0];cases.append({'union_id':i,'tokens':int(data.lengths[i]),'logit_error':le,'probability_error':pe})
    padding=[];ids=np.array(fixtures)
    for positions in data.microbatches(ids,16384,8):
        z=predict(model,data,ids[positions],device);ref=np.array([gold[int(i)] for i in ids[positions]])
        le=float(abs(z-ref).max());pe=float(abs(expit(z)-expit(ref)).max());assert le<=.002 and pe<=.0001
        padding.append({'rows':ids[positions].tolist(),'logit_error':le,'probability_error':pe})
    # Same frozen weights/precision as the independently qualified five-species release.
    other=PROJECT/'bechmark-nonhuman-v5';q=read(other/'qualification/native-human.json');assert q['passed']
    atomic(ROOT/'qualification/native-human.json',{'passed':True,'fingerprint':contract,'at_utc':now(),
        'cases':cases,'padding_checks':padding,'previous_qualification':record(other/'qualification/native-human.json'),
        'native_raw_tokenizer_and_eager_attention':True,'no_truncation':True,'test_metrics_read':False,
        'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})
    print({'qualification':'passed','cases':cases,'padding':padding},flush=True)

@torch.inference_mode()
def infer(device,rank,world):
    assert world==4 and 0<=rank<4
    contract=signature();q=read(ROOT/'qualification/native-human.json');assert q['passed'] and q['fingerprint']==contract
    m=read(MANIFEST)
    for rel,digest in m['frozen_data'].items():assert sha(ROOT/rel)==digest,rel
    model=load_model(device);data=test_data();ids=np.arange(len(data))[rank::world]
    ids=ids[np.argsort(data.lengths[ids],kind='stable')];dest=ROOT/'predictions/native-human';dest.mkdir(exist_ok=True)
    lock=(dest/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();records=[]
    for chunk,start in enumerate(range(0,len(ids),256)):
        subset=ids[start:start+256];path=dest/f'rank-{rank:02d}-chunk-{chunk:04d}.npz';side=path.with_suffix('.json')
        if side.exists():
            saved=read(side);assert saved['fingerprint']==contract and sha(path)==saved['file']['sha256']
            a=load_npz(path);assert np.array_equal(a['indices'],subset) and a['logits'].shape==(len(subset),2)
        else:
            z=np.empty((len(subset),2),np.float64)
            for positions in data.microbatches(subset,16384,8):z[positions]=predict(model,data,subset[positions],device)
            save_npz(path,indices=subset,logits=z);atomic(side,{'fingerprint':contract,'file':record(path),'rows':len(subset),'at_utc':now()})
        records.append(record(side));print({'rank':rank,'rows':start+len(subset),'total':len(ids),'seconds':time.monotonic()-started},flush=True)
    atomic(dest/f'rank-{rank:02d}.done.json',{'model':'native-human','rank':rank,'world':world,'rows':len(ids),
        'fingerprint':contract,'chunks':records,'at_utc':now(),'hostname':socket.gethostname(),
        'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),'seconds':time.monotonic()-started,
        'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');a=p.parse_args()
    d=cuda(int(os.environ.get('LOCAL_RANK','0')))
    if a.qualify:qualify(d)
    else:infer(d,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))
