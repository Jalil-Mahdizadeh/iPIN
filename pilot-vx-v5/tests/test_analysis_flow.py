"""Synthetic all-arm analysis/audit, exact reuse, corruption and fit isolation."""
import copy,hashlib,io,sys,tempfile,unittest
from pathlib import Path
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import study,inputs,analyze,verify,strata as strata_module
from msa import pair,shuffled,quality_features
from common import text_hash
from heads import evidence
from diagnostics import describe
from statistics_v5 import bootstrap


def fixture(root,cfg):
    n=640;rng=np.random.default_rng(92);y=np.arange(n)%2;base=rng.normal(size=n)+.3*y
    for name in ('data','results','features','scripts'):(root/name).mkdir()
    (root/'scripts/verify.py').write_bytes(Path(verify.__file__).read_bytes());study.atomic(root/'config.json',cfg)
    oc=study.read(study.BASE/'config.json');oc['maximum_paired_rows_including_query']=33
    study.atomic(root/'data/original_config.json',oc);templates={};monomers={}
    def mono():
        q='ARNDCEQGHILKMFPS';alphabet='ARNDCEQGHILKMFPSTWYV'
        return {'query':q,'mask':'*'*12+'-'*4,'query_sha256':hashlib.sha256(q.encode()).hexdigest(),
                'rows':[(f'k{j}',''.join(alphabet[v] for v in rng.integers(0,20,len(q))),f'g:f{j%7}:o:c:p') for j in range(80)]}
    for kind,count in [('extended',80),('unchanged',20)]:
        ma,mb=mono(),mono();ma['rows']=ma['rows'][:count];mb['rows']=mb['rows'][:count]
        t,m=pair(ma,mb,oc);tt,mm=pair(ma,mb,{**oc,'maximum_paired_rows_including_query':65})
        templates[kind]=(ma,mb,t,m,tt,mm)
    rows=[];pairs=[];oldmeta=[];audits={};gate=[];quality=[]
    for j in range(n):
        role='train' if j<320 else 'calibration' if j<440 else 'assessment' if j<600 else 'crossing'
        r={'uid':str(j),'a':2*j,'b':2*j+1,'label':int(y[j]),'split':'train' if j<320 else 'val','role':role,'length':32}
        available=j%9!=0;ma,mb,t,m,tt,mm=templates['unchanged' if j%7==0 else 'extended']
        p={'uid':str(j),'a':2*j,'b':2*j+1,'available':available,'gate':m['gate'] if available else 0.,'reason':'available' if available else 'missing_monomer','length':32,'breakpoint':16}
        q=quality_features(ma,mb,m) if available else np.zeros(68);quality.append(q);gate.append(p['gate'])
        if available:
            monomers[str(2*j)]=ma;monomers[str(2*j+1)]=mb
            seed=int(text_hash(f"{cfg['seed']}:{j}")[:16],16);s,nn=shuffled(tt,16,mm['taxonomy'],seed);_,oldnull=shuffled(t,16,m['taxonomy'],seed)
            qq=inputs.quality_at_depth(q,m,mm);path=root/'data/depth_inputs'/(str(j)+'.npz');study.atomic_npz(path,T256=tt,S256=s,P256=qq)
            p.update({'paired_depth':len(t),'depth_256':len(tt),'depth_increased':len(tt)>len(t),'paired_tokens_sha256':study.token_hash(t),
                'q_tokens_sha256':study.token_hash(t[:1]),'T256_tokens_sha256':study.token_hash(tt),'S256_tokens_sha256':study.token_hash(s),
                'P256_sha256':inputs.vector_hash(qq),'msa256':mm,'null256':nn,'depth_input_file':str(path.relative_to(root)),'depth_input_sha256':study.sha(path)})
            audits[p['uid']]=describe(tt,s,p,m,mm,nn);oldmeta.append({**p,'msa':m,'null':oldnull})
        else:oldmeta.append({**p,'msa':None})
        rows.append(r);pairs.append(p)
    gate=np.array(gate);train=np.arange(n)<320;cal=(np.arange(n)>=320)&(np.arange(n)<440);ass=(np.arange(n)>=440)&(np.arange(n)<600);dev=~train
    for name,value in [('sample',rows),('pairs',pairs),('source_metadata',oldmeta),('input-audits',audits),('diagnostics',[None]*n),
                       ('dev_family_witnesses',{}),('v2_metrics',{'diagnostic_stratum_cuts_from_eligible_train':{}}),('test_monomers',monomers)]:study.atomic(root/'data'/(name+'.json'),value)
    study.atomic(root/'data/baseline.json',{r['uid']:{**r,'score':float(base[j])} for j,r in enumerate(rows) if dev[j]})
    def head(dim,c):return {'standard_mean':[0.]*dim,'standard_scale':[1.]*dim,'pca_mean':[0.]*dim,
        'pca_components':np.eye(dim)[:32].tolist(),'coef':[[c]*32],'intercept':[.01],'alpha':1}
    hh={'true':head(128,.04),'shuffled':head(128,.03),'quality':head(68,.02)}
    old={'true':rng.normal(size=(n,128)),'shuffled':rng.normal(size=(n,128)),'quality':np.array(quality)}
    for v in old.values():v[gate==0]=0
    study.atomic(root/'data/source_heads.json',hh);study.atomic_npz(root/'data/source.npz',uids=np.array([r['uid'] for r in rows]),gate=gate,**old)
    scores={'baseline':base,**{a:study.fuse(base,evidence(hh[a],v),gate,1) for a,v in old.items()}}
    previous={'results':{},'fit_counts':{k:[int(np.sum(m&(y==c))) for c in (0,1)] for k,m in [('fit',train&(gate>0)),('calibration',cal),('assessment',ass)]}}
    for pop,m in [('assessment',ass),('fixed_dev_sample',dev),('calibration',cal)]:
        previous['results'][pop]={'metrics':{a:{'ap':float(average_precision_score(y[m],v[m])),'auroc':float(roc_auc_score(y[m],v[m]))} for a,v in scores.items()}}
        if pop!='calibration':
            ci=bootstrap([r for r,k in zip(rows,m) if k],y[m],{**{a:v[m] for a,v in scores.items()},'T256':base[m],'S256':base[m],'P256':base[m]},
                {pop:np.ones(m.sum(),dtype=bool)},cfg['fusion'],cfg['seed'])[pop]
            previous['results'][pop]['ap_intervals']={oldname:ci[new] for new,oldname in [('true_minus_baseline','true-minus-baseline'),('true_minus_shuffled','true-minus-shuffled')]}
    study.atomic(root/'data/source_metrics.json',previous)
    study.atomic_npz(root/'data/vy_dev_predictions.npz',uids=np.array([r['uid'] for r in rows if r['split']=='val']),labels=y[dev],baseline=base[dev],**{'vx_'+a:v[dev] for a,v in scores.items() if a!='baseline'})
    x={a:rng.normal(size=(n,dim)) for a,dim in study.DIMENSIONS.items()}
    with patch.object(study,'ROOT',root):
        for j,p in enumerate(pairs):
            if p['available']:
                with np.load(root/p['depth_input_file']) as f:x['P256'][j]=f['P256']
                if p['depth_increased']:
                    timing={'kind':'encoded',**{a:{'orientation_depths':[p['depth_256']]*2,'orientation_layers':list(range(16))*2,'length':32,'breakpoint':16} for a in ('T256','S256')}}
                else:
                    timing={'kind':'reused_unchanged_depth'}
                    for a,b in [('T256','true'),('S256','shuffled'),('P256','quality')]:x[a][j]=old[b][j]
            else:
                timing=None
                for v in x.values():v[j]=0
            study.save_record(p['uid'],{a:v[j] for a,v in x.items()},'fixture',p,timing)
    return rows,pairs,x,base,gate


class AnalysisFlow(unittest.TestCase):
    def cfg(self):
        c=copy.deepcopy(study.config());c['expected']['pairs']=640;c['fusion']['bootstrap_replicates']=20;c['preparation_workers']=2
        c['depth_sampling'].update(original_cap=33,new_cap=65);c['encoder']['maximum_rows_including_query']=65
        return c
    def patches(self,root,cfg):
        stack=ExitStack()
        for module in (study,inputs,analyze,verify,strata_module):stack.enter_context(patch.object(module,'ROOT',root))
        for module in (study,inputs,analyze,verify):stack.enter_context(patch.object(module,'config',return_value=cfg))
        for module in (analyze,verify):stack.enter_context(patch.object(module,'verify_freeze',return_value='fixture'))
        stack.enter_context(patch.object(analyze,'check_budget'));monomers=study.read(root/'data/test_monomers.json')
        stack.enter_context(patch.object(inputs,'monomer_reader',return_value=lambda pid:monomers[str(pid)]));stack.enter_context(redirect_stdout(io.StringIO()))
        return stack
    def test_full_analysis_audit_all_arms_reuse_and_tampering(self):
        cfg=self.cfg()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows,pairs,x,base,gate=fixture(root,cfg)
            with self.patches(root,cfg):
                analyze.main();verify.main();v=study.read(root/'results/verification.json');self.assertTrue(v['passed'])
                self.assertEqual(v['pairs_verified'],640);self.assertEqual(v['depth_input_audit']['eligible_inputs_verified'],sum(p['available'] for p in pairs))
                fit=np.array([r['split']=='train' and p['available'] for r,p in zip(rows,pairs)]);heads=study.read(root/'results/heads.json')
                for a in study.DIMENSIONS:np.testing.assert_allclose(heads[a]['standard_mean'],x[a][fit].mean(0))
                with np.load(root/'results/dev_predictions.npz') as f:saved={k:f[k].copy() for k in f.files}
                for arm in study.DIMENSIONS:np.testing.assert_array_equal(saved[arm][gate[320:]==0],base[320:][gate[320:]==0])
                saved['S256'][0]+=.1;study.atomic_npz(root/'results/dev_predictions.npz',**saved)
                with self.assertRaises(AssertionError):verify.main()
                with self.assertRaisesRegex(RuntimeError,'already started'):analyze.main()
    def test_missing_arm_record_prevents_any_fit(self):
        cfg=self.cfg()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root,cfg);(root/'features/1.json').unlink();(root/'features/1.sha.json').unlink()
            with self.patches(root,cfg),patch.object(analyze,'fit_head') as fit:
                with self.assertRaisesRegex(RuntimeError,'Incomplete computational coverage'):analyze.main()
                fit.assert_not_called()
            self.assertFalse((root/'results/fit-start.json').exists())


if __name__=='__main__':unittest.main()
