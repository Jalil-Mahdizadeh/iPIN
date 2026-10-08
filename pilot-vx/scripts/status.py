"""Read-only pilot status; never starts or resubmits work."""
import json
import subprocess
from common import ROOT


def main():
    out={}
    for name,rel in [('data','provenance/data-prepared.json'),('archive','results/archive-coverage.json'),
                     ('archive_progress','results/archive-progress.json'),('encoder','qualification/encoder.json'),
                     ('real_encoder','qualification/real-encoder.json'),('submission','provenance/submission.json'),
                     ('controller','provenance/controller.json'),('controller_error','provenance/controller-error.json'),
                     ('decision','results/decision.json')]:
        p=ROOT/rel
        if p.exists():
            value=json.loads(p.read_text())
            if name=='archive':value.pop('unmatched_ids',None)
            if name=='data':value={'splits':value['splits'],'test_labels_read':value['test_labels_read']}
            if name=='decision':value.pop('missing_pairs',None)
            out[name]=value
    out['workers']={p.name:json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('worker-*.json'))}
    if 'submission' in out:
        job=out['submission']['job_id']
        q=subprocess.run(['squeue','-j',job,'-h','-o','%i %T %M %R'],text=True,capture_output=True)
        out['scheduler']=q.stdout.strip() or 'not in active queue; inspect sacct and decision.json'
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
