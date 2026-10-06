"""Resume only default X-PAIR, sharing one native cache across logical shards.

The original four-GPU run finished Bernett but timed out while preparing the
default projection caches. Reuse all native Ankh features and invoke the same
qualified scorer and batch construction. One interactive GPU processes the
four existing logical shards sequentially, avoiding four identical caches.
"""
import os, socket
import torch
import xpair_benchmark as native
from bench_utils import ROOT, atomic, cuda, now, read, record


def main():
    q=read(ROOT/'qualification/xpair.json')
    assert q['passed'] and q['signature']==native.signature()
    assert all((ROOT/'predictions/xpair-bernett'/f'rank-{r:02d}.done.json').exists() for r in range(4))
    native.NAMES={'xpair-default':native.NAMES['xpair-default']}
    original_cache=native.cache
    memo={}
    def shared_cache(model,meta,device):
        state=model.embedding_projection.state_dict()
        if not memo:
            print('Preparing one shared native projection cache for X-PAIR default',flush=True)
            memo['projection_state']={k:v.detach().clone() for k,v in state.items()}
            memo['table']=original_cache(model,meta,device)
            print('Shared projection cache ready',flush=True)
        else:
            assert all(torch.equal(v,memo['projection_state'][k]) for k,v in state.items())
        return memo['table']
    native.cache=shared_cache
    device=cuda(0)
    info={'at_utc':now(),'replaces_timed_out_job':'3382643','job_id':os.environ.get('SLURM_JOB_ID'),
        'host':socket.gethostname(),'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),
        'wrapper':record(__file__),'original_qualification':record(ROOT/'qualification/xpair.json'),
        'native_signature':native.signature(),'new_encoder_inference':False,'changed_model_or_batching':False,
        'physical_gpus':1,'logical_shards':4,'shared_projection_cache_requires_identical_weights':True,
        'state':'running'}
    atomic(ROOT/'provenance/xpair-default-resume.json',info)
    for rank in range(4):
        native.score(rank,4,device)
        info.update(at_utc=now(),completed_logical_shards=rank+1)
        atomic(ROOT/'provenance/xpair-default-resume.json',info)
    info.update(at_utc=now(),state='complete')
    atomic(ROOT/'provenance/xpair-default-resume.json',info)
    print('X-PAIR default complete; four existing logical shards verified',flush=True)


if __name__=='__main__':main()
