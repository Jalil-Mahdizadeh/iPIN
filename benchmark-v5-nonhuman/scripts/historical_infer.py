"""Qualified, resumable frozen v2 inference on the existing nonhuman pair union."""
import argparse, fcntl, hashlib, json, os, socket, time
import numpy as np
import torch
from scipy.special import expit
from transformers import AutoTokenizer
from bench_utils import ROOT, PROJECT, atomic, cuda, load_npz, now, read, record, save_npz, sha
import pair_infer as pair

EXT = ROOT/'v1-v4-comparison'


def signature(name):
    paths = [EXT/'provenance/selection.json', EXT/'source/model.py', EXT/'source/native_oracle.py',
             ROOT/'scripts/historical_infer.py', ROOT/'scripts/pair_infer.py', ROOT/'scripts/bench_utils.py',
             ROOT/'scripts/v5_model/data.py', ROOT/'scripts/container.sh', ROOT/'scripts/launch_workers.py']
    return hashlib.sha256(json.dumps({'model':name, 'files':{str(p):sha(p) for p in paths},
        'torch':torch.__version__}, sort_keys=True).encode()).hexdigest()


def verify_inputs():
    sel=read(EXT/'provenance/selection.json')
    assert sha(sel['prepared_inputs']['path'])==sel['prepared_inputs']['sha256']
    for path,digest in read(ROOT/'provenance/prepared.json')['data_files'].items():
        assert sha(ROOT/path)==digest, path
    return sel


def load_model(name, device):
    selection=verify_inputs();entry=selection['models'][name];cp=entry['checkpoint']
    assert sha(cp['path'])==cp['sha256']==entry['sha256']
    saved=torch.load(cp['path'],map_location='cpu',mmap=True,weights_only=False)
    assert saved['training_state']['update']==entry['update']
    assert saved['fingerprint']==entry['source_manifest']['fingerprint']
    assert entry['configuration']['attention_mode']=='standard' and entry['configuration']['readout']=='cls_linear'
    cls=pair.module('historical_frozen_model', EXT/'source/model.py').PairModel
    model=cls(selection['base'], mode='reference', backend='efficient')
    model.load_state_dict(saved['model'],strict=True)
    assert all(torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items())
    model.esm_mask.gradient_checkpointing_disable()
    model.eval().requires_grad_(False).to(device)
    del saved
    return model


@torch.inference_mode()
def predict(model, data, ids, device, fp32=False):
    features=data.batch(ids,np.zeros(len(ids),dtype=np.int64),2,False,device)
    assert features['attention_mask'].sum(1).cpu().tolist()==[int(data.lengths[i]) for i in ids for _ in range(2)]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32):
        z=model(features['clean_ids'],features['attention_mask'])
    z=z.float().cpu().numpy().astype(np.float64)
    assert z.shape==(len(ids),2) and np.isfinite(z).all()
    return z


@torch.inference_mode()
def qualify(name,model,device):
    sel=read(EXT/'provenance/selection.json');entry=sel['models'][name]
    for item in sel['dev_inputs'].values():assert sha(item['path'])==item['sha256']
    item=entry['selected_dev'];assert sha(item['path'])==item['sha256']
    gold=load_npz(item['path'])['predictions'];path=PROJECT/'benchmark-v2/data'
    dev=pair.data_arrays(np.load(path/'val.npy'),np.load(path/'tokens.npy'),np.load(path/'offsets.npy'))
    assert np.array_equal(gold[:,0],np.arange(len(dev))) and np.array_equal(gold[:,1],dev.rows[:,2])
    indices=np.arange(0,len(dev),4);indices=indices[np.argsort(dev.lengths[indices],kind='stable')]
    batches=list(dev.microbatches(indices,16384,8));dev_cases=[]
    for j in sorted({0,len(batches)//2,len(batches)-1}):
        ids=indices[batches[j]];z=predict(model,dev,ids,device);expected=gold[ids,2:4]
        le=float(abs(z-expected).max());pe=float(abs(expit(z)-expit(expected)).max())
        assert le<=.05 and pe<=.005,(name,'dev',le,pe)
        dev_cases.append({'indices':ids.tolist(),'max_tokens':int(dev.lengths[ids].max()),'max_logit_error':le,'max_probability_error':pe})
        print({'model':name,'dev_case':dev_cases[-1]},flush=True)
    data=pair.test_data();meta=read(ROOT/'data/sequences.json');mapping=load_npz(ROOT/'data/pair-mapping.npz')
    ids=[];fixtures=[]
    for test in ['mouse','fly','worm','yeast','ecoli']:
        for row in [0,int(np.argmax(data.lengths[mapping[test]]))]:
            uid=int(mapping[test][row]);ids.append(uid);fixtures.append({'species':test,'source_row':row,'union_id':uid})
    native=pair.module('historical_official_oracle',EXT/'source/native_oracle.py')
    oracle=native.NativePLM();oracle.load_state_dict(model.state_dict(),strict=True);oracle.eval().requires_grad_(False).to(device)
    tokenizer=AutoTokenizer.from_pretrained(sel['base'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    cases=[]
    for uid in sorted(set(ids)):
        ii=np.array([uid]);fp=predict(model,data,ii,device,fp32=True)[0];bf=predict(model,data,ii,device)[0]
        features=data.batch(ii,np.array([0]),2,False,device);a,b=map(int,data.rows[uid,:2]);expected=[]
        for column,(x,y) in enumerate([(a,b),(b,a)]):
            tokens=native.tokenize(tokenizer,[(meta['sequence'][x],meta['sequence'][y])],max_length=None).to(device)
            assert torch.equal(tokens.input_ids[0],features['clean_ids'][column])
            expected.append(float(oracle(tokens)[0]))
        expected=np.array(expected);le=float(abs(fp-expected).max());pe=float(abs(expit(fp)-expit(expected)).max())
        assert le<=.002 and pe<=.0002,(name,'native_fp32',uid,le,pe)
        bpe=float(abs(expit(bf)-expit(expected)).max())
        assert bpe<=.05,(name,'bf16_sensitivity',uid,bpe)
        cases.append({'union_id':uid,'tokens':int(data.lengths[uid]),'native_fp32_logits':expected.tolist(),
            'optimized_fp32_logits':fp.tolist(),'optimized_bf16_logits':bf.tolist(),
            'fp32_max_logit_error':le,'fp32_max_probability_error':pe,'bf16_max_probability_error':bpe})
    atomic(EXT/'qualification'/f'{name}.json',{'passed':True,'at_utc':now(),'model':name,'fingerprint':signature(name),
        'dev_cases':dev_cases,'native_cases':cases,'fixture_selection':fixtures,'fixture_selection_uses_scores':False,
        'original_backbone_fp32_and_raw_string_tokenization_verified':True,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(device),'production_precision':sel['precision']})
    print({'event':'qualification_passed','model':name,'native_cases':len(cases)},flush=True)


@torch.inference_mode()
def infer(name,model,device,rank,world):
    assert world==4
    contract=signature(name);q=read(EXT/'qualification'/f'{name}.json');assert q['passed'] and q['fingerprint']==contract
    data=pair.test_data();ids=np.arange(rank,len(data),world);ids=ids[np.argsort(data.lengths[ids],kind='stable')]
    out=EXT/'predictions'/name;out.mkdir(exist_ok=True)
    lock=(out/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start_time=time.monotonic();records=[]
    for chunk,start in enumerate(range(0,len(ids),512)):
        subset=ids[start:start+512];path=out/f'rank-{rank:02d}-chunk-{chunk:04d}.npz';side=path.with_suffix('.json')
        if side.exists():
            previous=read(side);assert previous['fingerprint']==contract and sha(path)==previous['file']['sha256']
            saved=load_npz(path);assert np.array_equal(saved['indices'],subset) and saved['logits'].shape==(len(subset),2) and np.isfinite(saved['logits']).all()
        else:
            if (EXT/'REQUEST_STOP').exists():raise SystemExit(75)
            logits=np.empty((len(subset),2),dtype=np.float64)
            for pos in data.microbatches(subset,16384,8):logits[pos]=predict(model,data,subset[pos],device)
            save_npz(path,indices=subset,logits=logits)
            atomic(side,{'fingerprint':contract,'file':record(path),'rows':len(subset),'at_utc':now()})
        records.append(record(side))
        print(json.dumps({'event':'chunk','model':name,'rank':rank,'rows':min(start+512,len(ids)),
            'total':len(ids),'seconds':time.monotonic()-start_time}),flush=True)
    atomic(out/f'rank-{rank:02d}.done.json',{'model':name,'rank':rank,'world':world,'rows':len(ids),'fingerprint':contract,
        'chunks':records,'at_utc':now(),'hostname':socket.gethostname(),'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),
        'job_id':os.environ.get('SLURM_JOB_ID'),'seconds':time.monotonic()-start_time,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['v2-capped','v2-clean-bce'],required=True);p.add_argument('--qualify',action='store_true')
    a=p.parse_args();device=cuda(int(os.environ.get('LOCAL_RANK','0')));model=load_model(a.model,device)
    if a.qualify:qualify(a.model,model,device)
    else:infer(a.model,model,device,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))


if __name__=='__main__':main()
