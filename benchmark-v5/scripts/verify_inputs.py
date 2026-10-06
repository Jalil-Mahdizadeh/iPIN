"""Host-side verification of pinned benchmark runtimes and released weights."""
import concurrent.futures,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parent;E=P.parent/'iPIN-OpenPPI'
items={
 'native-runtime':(P/'images/plm-interact/plm-interact-native-arm64-v1.sif','e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'),
 'esmc-runtime':(P/'images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif','ed6aeac781502090632bc2a705300b161b6a65ccfbcf6c7cf64d8d61d8e5b5ef'),
 'tuna-runtime':(E/'benchmark/containers/images/tuna-arm64-v1.sif','98070c7c2d206d8421817849e11ffce308016dc982938a7d9a3ef3141ab1dec1'),
 'rapppid-runtime':(E/'benchmark/containers/images/rapppid-native-arm64-v1.sif','e853a89768c5351927f90fd0ccfb2ad899a15a6fc239a9014726af0c34442d48'),
 'sprint-runtime':(E/'benchmark/containers/images/sprint-native-arm64-v1.sif','699112807a7829b1b4268fa83d358a206d62de5a5a0f7c7d199beb18f9f47f09'),
 'xpair-runtime':(E/'benchmark/containers/images/plm-interact-native-arm64-v1.sif','e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'),
 'tuna-released':(E/'benchmark/tuna/weights/bernett_original.pt','bf2dd42af75d98324798b76837ca738e4ed230b096ee538ccdbaa001dcc88500'),
 'tuna-encoder':(E/'.private/frozen_pair_models_v1/bundle/encoder/model.safetensors','c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566'),
 'xpair-bernett':(E/'experiments/x_pair_test2_v1/sources/X-PAIR/pretrained_models/interaction_bernett.ckpt','17bf35197982c28fe25de1a45cf4cae67b94ae2edacc3db7967cd4f4c6da0fc6'),
 'xpair-default':(E/'experiments/x_pair_test2_v1/sources/X-PAIR/pretrained_models/multitask_xfair.ckpt','d8c50d03cd6107d2b411d7095a34243ec1b31e6b7cc1d590329c9328c1e8df72'),
 'xpair-encoder':(E/'experiments/x_pair_test2_v1/sources/ankh-large/pytorch_model.bin','517b6e8b279dedcb477af240b35c46bd6eb3307723eb281e60d4b2c8a87b889b')}
def check(item):
 name,(path,expected)=item
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 assert h.hexdigest()==expected,(name,h.hexdigest(),expected)
 result={'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()};print(name,'verified',flush=True)
 return name,result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:verified=dict(pool.map(check,items.items()))
path=ROOT/'provenance/runtime-inputs.json';temp=path.with_suffix('.tmp')
temp.write_text(json.dumps({'verified':True,'items':verified},indent=2)+'\n');os.replace(temp,path)
