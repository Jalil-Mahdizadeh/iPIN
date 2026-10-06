"""Original TUnA human seed 47; native full-length FP32 and checked feature reuse."""
import fcntl,hashlib,json,os,sys,time
import numpy as np
import torch
from scipy.special import expit
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,cuda,load_npz,now,read,record,save_npz,sha
os.environ['TUNA_UPSTREAM_DIR']=str(EXTERNAL/'benchmark/tuna/upstream/TUnA/results/bernett/TUnA')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/tuna/matplotlib')
sys.path[:0]=['/opt/tuna/vendor',str(ROOT/'scripts/human_transfer_tuna')]
from adapter import create,endpoint_features,cached_scores,native_scores,enable_sdpa
from esm_cache import load_encoder,enable_fast,embed
MANIFEST=ROOT/'provenance/human-releases.json'

def signature():
    m=read(MANIFEST)
    for rel,h in m['inference_code'].items():assert sha(ROOT/rel)==h,rel
    return hashlib.sha256(json.dumps({'manifest':sha(MANIFEST),'torch':torch.__version__,
        'precision':'FP32, TF32 disabled','batch_max':8,'token_budget':8192,'feature_commit':64},sort_keys=True).encode()).hexdigest()

@torch.inference_mode()
def main():
    contract=signature();m=read(MANIFEST);entry=m['models']['tuna-human'];path=entry['checkpoint']['path']
    assert sha(path)==entry['checkpoint']['sha256']
    for rel,h in m['frozen_data'].items():assert sha(ROOT/rel)==h
    dest=ROOT/'features/tuna-human';dest.mkdir(exist_ok=True)
    lock=(dest/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();device=cuda();meta=read(ROOT/'data/sequences.json');rows=np.load(ROOT/'data/union.npy')
    model=create(checkpoint=path,device=device,hid_dim=256,ff_dim=1024).eval().requires_grad_(False)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    enc,alphabet,_=load_encoder(EXTERNAL/'.private/frozen_pair_models_v1/bundle/encoder',device);enc.requires_grad_(False)
    # Reuse only exact-sequence, same-checkpoint features committed by the prior benchmark.
    other=PROJECT/'bechmark-nonhuman-v5';oldmeta=read(other/'data/sequences.json');lookup={s:i for i,s in enumerate(meta['sha256'])}
    z=np.full((len(meta['sequence']),256),np.nan,np.float32);reused=np.zeros(len(z),bool);cache_sources=[]
    for rank in range(4):
        q=read(other/'qualification'/f'tuna-rank-{rank:02d}.json');d=read(other/'features/tuna'/f'rank-{rank:02d}.done.json')
        assert q['passed'] and d['passed'] and d['signature']==q['signature']
        assert q['models']['tuna-human']['checkpoint']['sha256']==entry['checkpoint']['sha256']
        for item in d['files']:
            assert sha(item['path'])==item['sha256'];a=load_npz(item['path'])
            for j,oldid in enumerate(a['indices']):
                i=lookup.get(oldmeta['sha256'][oldid])
                if i is not None:
                    assert not reused[i] and oldmeta['sequence'][oldid]==meta['sequence'][i]
                    z[i]=a['tuna-human'][j];reused[i]=True
        cache_sources.append(record(other/'features/tuna'/f'rank-{rank:02d}.done.json'))
    lengths=np.array(meta['length']);order=np.argsort(lengths,kind='stable')
    fixtures=[int(order[j]) for j in [0,len(order)//2,-1]]
    if reused.any():fixtures+= [int(np.flatnonzero(reused)[0])]
    fixtures=list(dict.fromkeys(fixtures))
    e0=embed(enc,alphabet,[meta['sequence'][fixtures[1]]])[0];enable_fast(enc)
    e1=embed(enc,alphabet,[meta['sequence'][fixtures[1]]])[0];ee=float((e0-e1).abs().max());assert ee<2e-4
    embs={i:embed(enc,alphabet,[meta['sequence'][i]])[0] for i in fixtures}
    longpair=int(np.argmax(lengths[rows[:,0]]+lengths[rows[:,1]]));pairs=[tuple(map(int,rows[longpair,:2])),(fixtures[0],fixtures[1]),(fixtures[1],fixtures[1])]
    for a,b in pairs:
        for i in (a,b):
            if i not in embs:embs[i]=embed(enc,alphabet,[meta['sequence'][i]])[0]
    original=native_scores(model,[embs[a] for a,b in pairs],[embs[b] for a,b in pairs]).cpu().numpy()
    enable_sdpa(model);qualified={i:endpoint_features(model,e[None],torch.tensor([len(e)],device=device))[0] for i,e in embs.items()}
    table=torch.zeros(len(z),256,device=device)
    for i,value in qualified.items():table[i]=value
    aa=np.array([a for a,b in pairs]);bb=np.array([b for a,b in pairs]);p=cached_scores(model,table,aa,bb,probabilities=True)
    pe=float(abs(p-original).max());assert pe<2e-5,pe
    assert np.array_equal(p,cached_scores(model,table,bb,aa,probabilities=True))
    cache_errors=[]
    for i in fixtures:
        if reused[i]:
            error=float(abs(qualified[i].cpu().numpy()-z[i]).max());assert error<2e-4
            cache_errors.append({'id':i,'error':error})
    sizes=[len(embs[i]) for i in fixtures];padded=torch.zeros(len(fixtures),max(sizes),640,device=device)
    for j,i in enumerate(fixtures):padded[j,:sizes[j]]=embs[i]
    batch=endpoint_features(model,padded,torch.tensor(sizes,device=device))
    padding_error=float((batch-torch.stack([qualified[i] for i in fixtures])).abs().max());assert padding_error<2e-4
    atomic(ROOT/'qualification/tuna-human.json',{'passed':True,'fingerprint':contract,'at_utc':now(),
        'encoder_sdpa_error':ee,'native_probability_error':pe,'padding_error':padding_error,'reuse_checks':cache_errors,
        'longest_pair_union_id':longpair,'pair_lengths':[[int(lengths[a]),int(lengths[b])] for a,b in pairs],
        'checkpoint':entry['checkpoint'],'no_truncation':True,'test_metrics_read':False})
    print({'qualification':'passed','probability_error':pe,'padding_error':padding_error,'reused':int(reused.sum())},flush=True)
    del table,embs,qualified,batch,padded,e0,e1
    missing=np.flatnonzero(~reused);missing=missing[np.argsort(lengths[missing],kind='stable')]
    records=[]
    for begin in range(0,len(missing),64):
        ids=missing[begin:begin+64];file=dest/f'features-{begin:05d}.npz';side=file.with_suffix('.json')
        if side.exists():
            info=read(side);assert info['fingerprint']==contract and sha(file)==info['file']['sha256']
            a=load_npz(file);assert np.array_equal(a['indices'],ids);z[ids]=a['features']
        else:
            start=0
            while start<len(ids):
                end=start+1
                while end<len(ids) and end-start<8 and (end-start+1)*(int(lengths[ids[end]])+2)<=8192:end+=1
                b=ids[start:end];es=embed(enc,alphabet,[meta['sequence'][i] for i in b]);w=max(map(len,es))
                padded=torch.zeros(len(b),w,640,device=device)
                for j,e in enumerate(es):padded[j,:len(e)]=e
                z[b]=endpoint_features(model,padded,torch.tensor([len(e) for e in es],device=device)).cpu().numpy()
                start=end
            save_npz(file,indices=ids,features=z[ids]);atomic(side,{'fingerprint':contract,'file':record(file)})
        records.append(record(side));print({'features':begin+len(ids),'total':len(missing),'seconds':time.monotonic()-started},flush=True)
    assert np.isfinite(z).all();save_npz(dest/'endpoints.npz',features=z,reused=reused)
    scores=cached_scores(model,torch.tensor(z,device=device),rows[:,0],rows[:,1],probabilities=False).astype(np.float64)
    assert all(torch.equal(value,initial[k]) for k,value in model.state_dict().items())
    out=ROOT/'predictions/tuna-human';out.mkdir(exist_ok=True);save_npz(out/'union.npz',scores=scores,probabilities=expit(scores))
    atomic(out/'done.json',{'rows':len(rows),'file':record(out/'union.npz'),'fingerprint':contract,
        'qualification':record(ROOT/'qualification/tuna-human.json'),'checkpoint':entry['checkpoint'],
        'feature_chunks':records,'features':record(dest/'endpoints.npz'),'cache_sources':cache_sources,
        'features_reused':int(reused.sum()),'features_computed':len(missing),'state_unchanged':True,
        'at_utc':now(),'seconds':time.monotonic()-started,'score_kind':'native uncertainty-adjusted logit'})
    print('TUnA human seed 47 complete',flush=True)
if __name__=='__main__':main()
