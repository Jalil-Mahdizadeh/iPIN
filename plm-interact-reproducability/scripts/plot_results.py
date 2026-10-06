"""Publication-style standalone diagnostic figures, generated from saved results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score,roc_auc_score
from datasets_eval import ROOT,SPECIES

OUT=ROOT/'results';PLOTS=OUT/'plots';PLOTS.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(12,7.3),layout='constrained')
for ax,(task,title,source) in zip(axes.flat,[(f'cross_{s}',s.capitalize(),f'figure2_PLM-interact_{s}.csv') for s in SPECIES]+[('bernett_full','Bernett — full sequences','figure4_PLM-interact.csv')]):
    d=pd.read_csv(OUT/f'{task}.csv');ref=pd.read_csv(ROOT/'data/published'/source)
    for frame,color,style,label in [(ref,'#666666','--','Deposited PLM-interact'),(d,'#0072B2','-','Fresh PLM-interact')]:
        p,r,_=precision_recall_curve(frame.label,frame.score)
        ap=average_precision_score(frame.label,frame.score)
        ax.plot(r,p,color=color,ls=style,lw=1.7,label=f'{label}: AP {ap:.4f}')
    baseline=source.replace('PLM-interact','TUnA')
    b=pd.read_csv(ROOT/'data/published'/baseline)
    p,r,_=precision_recall_curve(b.label,b.score)
    ax.plot(r,p,color='#D55E00',lw=1.1,alpha=.8,label=f'Deposited TUnA: AP {average_precision_score(b.label,b.score):.4f}')
    ax.set(title=title,xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
    ax.legend(loc='lower left',fontsize=8,frameon=False)
fig.suptitle('PLM-interact checkpoint reproduction · frozen models, supplied SIF',fontsize=14)
for ext in ['png','pdf']:fig.savefig(PLOTS/f'pr-curves-main.{ext}',dpi=200)
plt.close(fig)

mapping=pd.read_csv(ROOT/'data/mutation-figure5-mapping.csv')
ft=pd.read_csv(OUT/'mutation_ft_full.csv').set_index('row_id').loc[mapping.row_id]
zero=pd.read_csv(OUT/'mutation_zero_full.csv').set_index('row_id').loc[mapping.row_id]
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for d,color,model in [(ft,'#0072B2','Mutation checkpoint'),(zero,'#D55E00','Zero-shot checkpoint')]:
    for score,style,definition in [('score','-','released code'),('probability_log_ratio','--','paper equation')]:
        p,r,_=precision_recall_curve(d.label,d[score]);ap=average_precision_score(d.label,d[score])
        axes[0].plot(r,p,c=color,ls=style,lw=1.8,label=f'{model}, {definition}: {ap:.3f}')
        fpr,tpr,_=roc_curve(d.label,d[score]);roc=roc_auc_score(d.label,d[score])
        axes[1].plot(fpr,tpr,c=color,ls=style,lw=1.8,label=f'{model}, {definition}: {roc:.3f}')
axes[0].axhline(ft.label.mean(),color='#999999',ls=':',lw=1)
axes[1].plot([0,1],[0,1],c='#999999',ls=':',lw=1)
axes[0].set(xlabel='Recall',ylabel='Precision',title='Average precision')
axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='AUROC')
for ax in axes:
    ax.set(xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8,loc='lower left' if ax==axes[0] else 'lower right',frameon=False)
fig.suptitle('Mutation scoring definitions · the same 598 published cases',fontsize=14)
for ext in ['png','pdf']:fig.savefig(PLOTS/f'mutation-score-definitions.{ext}',dpi=200)
plt.close(fig)
print('Saved four standalone PNG/PDF figures to',PLOTS)
