"""Stateless masking and restartable, fixed global-update sampling."""
import hashlib
import numpy as np
import torch

class PairData:
    def __init__(self,path,split):
        self.rows=np.load(path/f'{split}.npy')
        self.tokens=np.load(path/'tokens.npy',mmap_mode='r');self.offsets=np.load(path/'offsets.npy',mmap_mode='r')
        self.lengths=np.diff(self.offsets)[self.rows[:,0]]+np.diff(self.offsets)[self.rows[:,1]]+3
    def __len__(self):return len(self.rows)
    def sequence(self,i):return self.tokens[self.offsets[i]:self.offsets[i+1]].astype(np.int64)
    def plan(self,epoch,seed,global_batch):
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,712]))
        # Sort randomized pools, not the whole epoch: length bucketing without a length curriculum.
        ids=rng.permutation(len(self));chunks=[]
        for start in range(0,len(ids),global_batch*50):
            pool=ids[start:start+global_batch*50];pool=pool[np.argsort(self.lengths[pool],kind='stable')]
            chunks.extend(pool[i:i+global_batch] for i in range(0,len(pool),global_batch))
        rng.shuffle(chunks)
        assert sum(map(len,chunks))==len(self)
        return chunks
    def microbatches(self,indices,token_budget,max_pairs):
        current=[];maximum=0
        for idx in indices:
            l=int(self.lengths[idx]);new_max=max(maximum,l)
            if current and (len(current)>=max_pairs or 2*(len(current)+1)*new_max>token_budget):
                yield current;current=[];maximum=0
            current.append(int(idx));maximum=max(maximum,l)
        if current:yield current
    def batch(self,indices,epoch,seed,training,device):
        inputs=[];targets=[];labels=[]
        for idx in indices:
            a,b,y,row_id=self.rows[idx];aa=self.sequence(a);bb=self.sequence(b)
            chains=[];ys=[]
            for side,t in enumerate([aa,bb]):
                changed=t.copy();target=np.full(len(t),-100,dtype=np.int64)
                if training:
                    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,int(row_id),side,981]))
                    selected=rng.random(len(t))<.15
                    if not selected.any():selected[rng.integers(len(t))]=True
                    target[selected]=t[selected];r=rng.random(len(t))
                    changed[selected&(r<.8)]=32
                    random_positions=selected&(r>=.8)&(r<.9)
                    changed[random_positions]=rng.integers(4,24,random_positions.sum())
                chains.append(changed);ys.append(target)
            for order in [(0,1),(1,0)]:
                x,z=order
                inputs.append(np.concatenate(([0],chains[x],[2],chains[z],[2])))
                targets.append(np.concatenate(([-100],ys[x],[-100],ys[z],[-100])))
            labels.append(y)
        length=max(map(len,inputs));ids=np.full((len(inputs),length),1,dtype=np.int64)
        masks=np.zeros_like(ids);targets_out=np.full_like(ids,-100)
        for i,(x,y) in enumerate(zip(inputs,targets)):
            ids[i,:len(x)]=x;masks[i,:len(x)]=1;targets_out[i,:len(x)]=y
        return {'input_ids':torch.from_numpy(ids).to(device),'attention_mask':torch.from_numpy(masks).to(device),
                'labels':torch.tensor(labels,dtype=torch.float32,device=device),
                'mlm_labels':torch.from_numpy(targets_out).to(device)}
