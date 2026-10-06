"""Compare published summary tables with the independently recomputed scores."""
import json
import pandas as pd
from datasets_eval import ROOT,SPECIES
small=json.loads((ROOT.parent/'literature/plm-interact/source-data-small-tables.json').read_text())
scores=pd.read_csv(ROOT/'results/published-score-metrics.csv')
rows=[]
for row in small['Figure2']:
    r={''.join(c for c in k if c.isalpha()):v for k,v in row.items()}
    if r.get('G')!='AUPR':continue
    for col,s in zip('BCDEF',SPECIES):
        match=scores[(scores.figure=='2')&(scores.dataset==s)&(scores.model==r['A'])]
        reported=float(r[col]);actual=float(match.iloc[0].average_precision) if len(match) else None
        rows.append(dict(model=r['A'],species=s,bar_aupr=reported,deposited_curve_average_precision=actual,
                         difference=actual-reported if actual is not None else None,
                         agrees_at_3_decimals=abs(actual-reported)<.0005 if actual is not None else None))
pd.DataFrame(rows).to_csv(ROOT/'results/baseline-bars-versus-curves.csv',index=False)

mask=pd.read_csv(ROOT/'results/published-mask-ablation-metrics.csv')
mask_checks=[]
for row in small['Supplemtrary Figure2']:
    r={''.join(c for c in k if c.isalpha()):v for k,v in row.items()}
    if r.get('A') not in SPECIES:continue
    key='0%(binary)' if r['C']=='mask_0% (binary)' else r['C'].replace('%','')
    actual=float(mask[(mask.species==r['A'])&(mask['mask']==key)].iloc[0].average_precision)
    mask_checks.append(dict(species=r['A'],mask=key,figure_axis_label=r['D'],reported_ap=float(r['B']),recomputed_ap=actual,absolute_delta=abs(float(r['B'])-actual)))
pd.DataFrame(mask_checks).to_csv(ROOT/'results/mask-summary-agreement.csv',index=False)

bins=pd.read_csv(ROOT/'results/published-identity-bin-metrics.csv',dtype={'identity_bin':str})
bin_checks=[]
for row in small['Supplemtrary Figure7']:
    r={''.join(c for c in k if c.isalpha()):v for k,v in row.items()}
    if r.get('A') not in SPECIES or 'C' not in r or 'D' not in r:continue
    for model,col in [('PLM-interact','C'),('TUnA','D')]:
        sel=bins[(bins.species==r['A'])&(bins.identity_bin==r['B'])&(bins.model==model)]
        if not len(sel):continue
        actual=float(sel.iloc[0].average_precision)
        bin_checks.append(dict(species=r['A'],identity_bin=r['B'],model=model,n=int(sel.iloc[0]['n']),reported_ap=float(r[col]),recomputed_ap=actual,absolute_delta=abs(float(r[col])-actual)))
pd.DataFrame(bin_checks).to_csv(ROOT/'results/identity-summary-agreement.csv',index=False)
print('Mask AUPR maximum delta:',max(r['absolute_delta'] for r in mask_checks))
print('Identity-bin AUPR maximum delta:',max(r['absolute_delta'] for r in bin_checks))
print(pd.DataFrame(rows).dropna().query('agrees_at_3_decimals == False').to_string(index=False))
