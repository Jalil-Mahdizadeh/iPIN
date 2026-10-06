"""Archive the sealed nine-model comparison and freeze two explicitly requested releases."""
import shutil
from pathlib import Path
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,now,read,record,sha

def main():
    path=ROOT/'provenance/human-releases.json';assert not path.exists(),'Already frozen'
    old=read(ROOT/'artifact-manifest.json');dest=ROOT/'archive/before-human-releases';dest.mkdir(parents=True,exist_ok=True)
    for rel,info in old['files'].items():
        source=ROOT/rel;assert sha(source)==info['sha256'] and source.stat().st_size==info['bytes'],rel
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for rel in ['artifact-manifest.json','completed.json']:shutil.copy2(ROOT/rel,dest/rel)
    other=PROJECT/'benchmark-v5-nonhuman';native=read(other/'provenance/selection.json')['models']['native-human']
    tuna=read(other/'provenance/tuna-human-download.json')
    for item in [native['checkpoint'],tuna]:assert sha(item['path'])==item['sha256']
    target=ROOT/'scripts/human_transfer_tuna';target.mkdir(exist_ok=True)
    inherited=[]
    for name in ['adapter.py','esm_cache.py','common.py']:
        source=other/'scripts/tuna'/name;shutil.copy2(source,target/name)
        inherited.append({'source':record(source),'copy':record(target/name)})
    code=['scripts/native_human_bernett.py','scripts/tuna_human_bernett.py','scripts/pair_infer.py',
          'scripts/bench_utils.py','scripts/container.sh','scripts/native_model/model.py','scripts/v5_model/data.py']
    code += [str(p.relative_to(ROOT)) for p in target.glob('*.py')]
    prep=read(ROOT/'provenance/prepared.json')
    for rel,h in prep['data_files'].items():assert sha(ROOT/rel)==h,rel
    atomic(path,{'at_utc':now(),'request':'Add the exact nonhuman humanV11 and TUnA seed-47 releases to both frozen v5 human tests',
        'addendum':record(ROOT/'HUMAN-RELEASES-ADDENDUM.md'),'previous_completion':record(dest/'completed.json'),
        'previous_collection':record(dest/'results/collection.json'),'previous_manifest':record(dest/'artifact-manifest.json'),
        'models':{'native-human':native,'tuna-human':{'checkpoint':record(tuna['path']),'revision':tuna['revision'],'url':tuna['url'],
            'seed':47,'hid_dim':256,'ff_dim':1024}},'inference_code':{rel:sha(ROOT/rel) for rel in code},
        'frozen_data':prep['data_files'],'prepared':record(ROOT/'provenance/prepared.json'),
        'images':{'native':record(PROJECT/'images/plm-interact/plm-interact-native-arm64-v1.sif'),
                  'tuna':record(EXTERNAL/'benchmark/containers/images/tuna-arm64-v1.sif')},
        'copied_adapters':inherited,'selection_uses_these_tests':False,'training':False})
    atomic(ROOT/'human-releases-status.json',{'at_utc':now(),'complete':False,'phase':'qualification','previous_nine_model_benchmark_complete':True})
    print('Archived nine-model artifacts and froze both requested releases',flush=True)
if __name__=='__main__':main()
