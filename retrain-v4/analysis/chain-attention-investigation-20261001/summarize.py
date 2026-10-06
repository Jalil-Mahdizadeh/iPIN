"""Summarize completed diagnostics and draw a standalone scientific figure."""
import datetime
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
summary={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backbones':{}}
for b in ['esm2','esmc']:
    d=json.loads((OUT/f'{b}-pretrained-diagnosis.json').read_text())
    t=json.loads((OUT/f'{b}-pretrained-attention-traces.json').read_text())
    p=json.loads((OUT/f'{b}-training-probes.json').read_text())
    mass={}
    for activation,score in [('standard','standard'),('standard','chain_aware'),('chain_aware','chain_aware')]:
        rows=[r for r in t if r['activation_mode']==activation and r['score_mode']==score]
        mass[activation+'_activations__'+score+'_scores']={k:statistics.mean(r[k] for r in rows) for k in
            ['residue_cross_mass_mean','cls_cross_mass_mean','cross_score_mean','within_score_mean','normalized_entropy_mean']}
    checks={}
    for precision in ['fp32','bf16']:
        for mode in ['standard','chain_aware']:
            rows=[r for r in d['kernel_checks'] if r['mode']==mode and r['precision']==precision]
            checks[precision+'_'+mode]={k:max(r[k] for r in rows) for k in ['forward_relative_l2','input_gradient_relative_l2','parameter_gradient_relative_l2_max']}
    summary['backbones'][b]={'attention_mass':mass,'mean_embedding_cosine_chain_vs_standard':statistics.mean(r['chain_vs_standard_hidden_cosine_mean'] for r in d['representations']),
        'kernel_checks_max_relative_l2':checks,'fp32_all_one_chain_control':p['fp32_all_one_chain_control'],
        'masked_residue':{s:{m:{k:v[k] for k in ['mean_ce','accuracy','tokens']} for m,v in r['masked_residue'].items()} for s,r in p['stages'].items()},
        'trained_checkpoint':p['checkpoint'],'trained_model_gradients':p['stages']['trained_chain_best1000']['ppi_gradient']}
pred=json.loads((OUT/'prediction-audit.json').read_text())
summary['full_validation_snapshot_utc']=pred['checked_at_utc']
summary['latest_validation']={a:r['validations'][-1] for a,r in pred['runs'].items()}
summary['no_production_changes']=True
summary['no_optimizer_steps']=True
summary['test_data_used']=False
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(15.5,4.7),layout='constrained')
colors={'S0':'#245b91','S1':'#789fc5','C0':'#b64b25','C1':'#dc9b77'}
labels={'S0':'ESM2 standard (v3)','S1':'ESM2 chain-aware','C0':'ESM C standard','C1':'ESM C chain-aware'}
for arm,r in pred['runs'].items():
    vs=r['validations']
    axes[0].plot([v['update'] for v in vs],[v['all']['ap'] for v in vs],marker='o',markersize=4,
                 color=colors[arm],linestyle='--' if arm in ['S1','C1'] else '-',label=labels[arm])
axes[0].axhline(29628/59258,color='#777777',linewidth=.8,linestyle=':')
axes[0].set(xlabel='Optimizer update',ylabel='Pooled validation AP',ylim=(.49,.68),title='A  Full validation: 59,258 pairs')
axes[0].legend(fontsize=8,frameon=False,loc='upper left')
x=np.arange(2);width=.34
for offset,mode,color,label in [(-width/2,'standard','#245b91','Standard'),(width/2,'chain_aware','#b64b25','Chain-aware')]:
    values=[100*summary['backbones'][b]['attention_mass']['standard_activations__'+mode+'_scores']['residue_cross_mass_mean'] for b in ['esm2','esmc']]
    bars=axes[1].bar(x+offset,values,width,color=color,label=label)
    axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
axes[1].set(xticks=x,xticklabels=['ESM2','ESM C'],ylabel='Attention mass to the other chain (%)',ylim=(0,78),title='B  Same pretrained activations')
axes[1].legend(fontsize=9,frameon=False)
for offset,mode,color,label in [(-width/2,'standard','#245b91','Standard'),(width/2,'chain_aware','#b64b25','Chain-aware')]:
    values=[100*summary['backbones'][b]['masked_residue']['pretrained'][mode]['accuracy'] for b in ['esm2','esmc']]
    bars=axes[2].bar(x+offset,values,width,color=color,label=label)
    axes[2].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
axes[2].set(xticks=x,xticklabels=['ESM2','ESM C'],ylabel='Masked-residue top-1 accuracy (%)',ylim=(0,65),title='C  Same pretrained weights and inputs')
axes[2].legend(fontsize=9,frameon=False)
fig.suptitle('V4 chain-aware attention investigation — 1 October 2026',fontsize=14)
fig.supxlabel('B–C: eight training pairs, both orientations; C: 2,354 masked residues. Diagnostic interventions, no optimizer steps.',fontsize=9)
fig.savefig(OUT/'diagnosis.png',dpi=180)
fig.savefig(OUT/'diagnosis.pdf')
plt.close(fig)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ['.py','.json','.png','.pdf'] and p.name!='evidence-manifest.json'}
(OUT/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
