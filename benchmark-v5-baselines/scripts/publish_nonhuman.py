"""Update the existing report, tables and all image formats from verified scores."""
import csv
from nonhuman_common import *
from evaluate import write_csv,table as human_table

def csvrows(path):return list(csv.DictReader(Path(path).open()))

def plots(human,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,stem):
        for ext in ['png','pdf','svg']:fig.savefig(ROOT/'results'/(stem+'.'+ext),dpi=210,bbox_inches='tight')
        plt.close(fig)
    def panel(ax,summary,datasets,names,metric,labels,title):
        a=np.array([[summary['datasets'][d]['models'].get(n,{}).get(metric,np.nan) for d in datasets] for n in names])
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
        image=ax.imshow(a,vmin=0,vmax=1,aspect='auto',cmap=cmap)
        for i in range(len(names)):
            for j in range(len(datasets)):
                v=a[i,j];ax.text(j,i,'N/A' if np.isnan(v) else f'{v:.4f}',ha='center',va='center',fontsize=9,
                                 color='white' if v>.64 else '#111111')
        ax.set_yticks(range(len(names)),[labels[n] for n in names],fontsize=9)
        ax.set_xticks(range(len(datasets)),[SPECIES_LABELS.get(d,{'validation':'v5 validation','original':'Original test','ilp':'ILP test'}.get(d,d)) for d in datasets],fontsize=9)
        ax.set_title(title+' | '+('AP' if metric=='ap' else 'AUROC'),fontsize=12,pad=9)
        return image
    fig,axes=plt.subplots(3,2,figsize=(18,14.6))
    rownames=[NAMES,['v5:'+n for n in BASELINES]+['ipin-esm2','ipin-esmc'],
                    ['bernett:'+n for n in BASELINES]+['v2-capped','v2-clean-bce','native-plm']]
    for row in range(3):
        summary=human if row==0 else new;datasets=DATASETS if row==0 else SPECIES
        title=['Existing human evaluations','Five species: v5 TRAIN baselines','Five species: Bernett TRAIN baselines'][row]
        for col,metric in enumerate(['ap','auroc']):
            panel(axes[row,col],summary,datasets,rownames[row],metric,NLABELS,title)
            axes[row,col].axhline(3.5,color='white',lw=2)
    fig.suptitle('Fixed cheap baselines | v5 fits reused, original Bernett fits added',fontsize=17,y=.985)
    fig.text(.04,.012,'All original observations retained. Human reference AP = 0.500; five-species reference AP = 0.0909; reference AUROC = 0.500.\n'
             'Original human values are unchanged. Neural predictions are reused. Frozen score directions; no test-based inversion or tuning.',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.96),h_pad=3,w_pad=3);save(fig,'comparison')
    fig,axes=plt.subplots(1,2,figsize=(20,9.7))
    for ax,metric in zip(axes,['ap','auroc']):
        im=panel(ax,new,SPECIES,ALL_MODELS,metric,NLABELS,'Five-species test')
        for boundary in [3.5,7.5]:ax.axhline(boundary,color='white',lw=3)
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('Same five-species observations | both TRAIN references and frozen neural models',fontsize=16,y=.99)
    fig.text(.03,.008,'AP reference = 1/11; AUROC reference = 0.5. Below-chance scores retain their frozen direction.\n'
             'Paired 95% protein-bootstrap intervals and common-subset checks accompany the tables; cells are point estimates.',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.955),w_pad=2);save(fig,'nonhuman-comparison')
    diffs=csvrows(ROOT/'results/nonhuman-paired-differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(15,8.5),sharey=True)
    methods=BASELINES[1:];colors=['#267A8C','#9468AA','#BB793E']
    labels=[SPECIES_LABELS[s]+' · '+{'homology-degree':'homology degree','sequence-propensity':'sequence propensity','interolog':'interolog'}[m] for s in SPECIES for m in methods]
    for ax,metric in zip(axes,['ap','auroc']):
        selected=[]
        for i,(species,method) in enumerate((s,m) for s in SPECIES for m in methods):
            r=next(r for r in diffs if r['dataset']==species and r['model']=='bernett:'+method and r['reference']=='v5:'+method and r['metric']==metric)
            value,lo,hi=[float(r[k]) for k in ['difference','low','high']];selected.append((value,lo,hi))
            ax.plot([lo,hi],[i,i],color=colors[i%3],lw=2);ax.plot(value,i,'o',color=colors[i%3],ms=5)
        ax.axvline(0,color='#555555',ls='--',lw=1);ax.set_yticks(range(len(labels)),labels,fontsize=9)
        ax.set_title('Bernett minus v5: '+('AP' if metric=='ap' else 'AUROC'))
        ax.set_xlabel('Difference (95% paired protein-bootstrap interval)');ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        lo=min(r[1] for r in selected);hi=max(r[2] for r in selected);pad=.13*(hi-lo)
        ax.set_xlim(lo-pad,hi+pad)
        for y in [2.5,5.5,8.5,11.5]:ax.axhline(y,color='#dddddd',lw=.7)
    axes[0].invert_yaxis()
    fig.suptitle('Training-reference comparison for each fixed baseline',fontsize=16,y=.99)
    fig.text(.04,.012,'500 shared protein-endpoint bootstrap replicates per species; descriptive, unadjusted intervals.\n'
             'Positive differences favor the Bernett-trained baseline in its fixed scoring direction; they do not identify the neural model’s mechanism.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.945));save(fig,'nonhuman-paired-differences')

def species_table(new,metric):
    out=['| Model / TRAIN reference | Mouse | Fly | Worm | Yeast | E. coli | Mean¹ |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ALL_MODELS:
        values=[new['datasets'][s]['models'][name][metric] for s in SPECIES]
        out.append('| '+' | '.join([NLABELS[name]]+[f'{v:.4f}' for v in values]+[f'{np.mean(values):.4f}'])+' |')
    return out

def report(human,new):
    verification=read(ROOT/'results/nonhuman-verification.json')
    fallback=csvrows(ROOT/'results/nonhuman-v5-fallback-diagnostic.csv')
    agreement=[float(r['fraction_matching_fallback_ordering']) for r in fallback]
    means={name:{metric:float(np.mean([new['datasets'][s]['models'][name][metric] for s in SPECIES]))
                 for metric in ['ap','auroc']} for name in ALL_MODELS}
    lines=['# Fixed cheap baselines: v5 and Bernett TRAIN, human and five-species tests','',
      'Updated 2026-10-07. The existing v5 model/propensities/positive graph were reused unchanged. '
      'Corresponding original-Bernett baselines were fitted with the identical fixed configuration. '
      'All 242,000 nonhuman source observations were evaluated. Earlier human results remain unchanged.','',
      '## Findings','',
      '**These three Bernett-trained baselines recover modest cross-species signal, but do not reproduce the much stronger Bernett-trained v2 performance. '
      'The result does not establish the hypothesized shortcut as the explanation of the neural gap.**','',
      '- Bernett mean AUROC: homology-transferred degree **0.5584**, sequence-only propensity **0.5430**, interolog **0.5295**. '
      'Each is below both v2 controls in AP and AUROC on every species; all 60 corresponding paired baseline-minus-v2 intervals lie below zero '
      'under the declared descriptive bootstrap. This does not rule out other shortcuts or prove what the neural models learned.',
      '- **The reused v5 interolog baseline exceeds both v5 neural models in AP on all five species.** '
      f"Its mean AP is **{means['v5:interolog']['ap']:.4f}**, versus **{means['ipin-esm2']['ap']:.4f}** for v5 ESM2 and **{means['ipin-esmc']['ap']:.4f}** for v5 ESMC. "
      'It also exceeds the Bernett interolog baseline in AP and AUROC on every species. This is evidence of conserved-positive-graph signal '
      'available to this simple lookup; conservation is not automatically an illegitimate shortcut.',
      '- v5 homology-degree AUROC is **0.4569, 0.3243, 0.2542, 0.3495, 0.4097**. '
      '**Below chance is inverse association, not absence of signal.** Its TRAIN target is almost constant: 99.82% of proteins have equal positive/negative degrees. '
      'The fixed no-hit fallback is slightly positive (0.000101364), whereas many matched proteins contribute zero. '
      f"Between {100*min(agreement):.2f}% and {100*max(agreement):.2f}% of pairs within each species/class have exactly this zero-for-hit plus fallback-for-no-hit score. "
      'Test positives tend to have more matches, so this inherited convention tends to rank them lower. The score direction and fallback were not changed after seeing results.',
      '- v5 sequence-propensity mean AUROC is **0.4844**; its fixed degree-ratio target is not a general test of all sequence-only classifiers.',
      '- The v5 interolog AP advantage over both v5 neural models persists descriptively after excluding exact endpoints from either TRAIN, '
      'and after deduplication/removal of conflicting-label pairs. These secondary subsets have different prevalence and are not directly comparable to the full AP values.',
      '- The previous V11 study recovered much stronger endpoint-transfer signal (mean homology-degree AUROC 0.8591), '
      'but those V11-trained results are not Bernett-trained evidence. See [that separate study](../literature/string-v11-baselines/REPORT.md).','',
      '## Five-species average precision','',*species_table(new,'ap'),'',
      '## Five-species AUROC','',*species_table(new,'auroc'),'',
      '¹ Mean is the unweighted mean of five separate species metrics, not a pooled test or selected ensemble. '
      'Reference AP = 1/11 = 0.0909; reference AUROC = 0.5. Exact-degree lookup is the inherited control, not a fourth proposed method.','',
      '![Five-species primary scores](results/nonhuman-comparison.png)','',
      '![Paired baseline differences](results/nonhuman-paired-differences.png)','',
      '## Original human evaluations — unchanged','',
      'All three v5-trained baselines remain near chance on v5 validation and both human tests. '
      'There was no v5 refitting, rescoring of those human rows, or replacement of their intervals. '
      'The original report/figures/metrics and completion manifest are retained in `archive/before-nonhuman/`.','',
      '### Average precision','',*human_table(human,'ap'),'',
      '### AUROC','',*human_table(human,'auroc'),'',
      'Human prevalence is 1/2. The two human tests share all positives. Validation neural scores were used for checkpoint selection. '
      'Native Bernett has no v5-validation predictions; its own validation was not substituted.','',
      '![Combined human and nonhuman comparison](results/comparison.png)','',
      '## Class-specific homology/interolog coverage','',
      'Each cell shows **positive percentage / negative percentage**. This is label association in the fixed evaluation data, '
      'not a predictor fitted to those labels. An exact match counts as a homolog. All missing-hit pairs remain in the denominator.','',
      '| Species | TRAIN | Both endpoints have a retained match (%) | Positive-graph interolog support (%) |','|---|---|---:|---:|']
    coverage=csvrows(ROOT/'results/nonhuman-coverage.csv')
    for species in SPECIES:
        for ref in TRAINING:
            pair={int(r['label']):r for r in coverage if r['dataset']==species and r['training_reference']==ref}
            a=' / '.join(f"{100*float(pair[y]['fraction_with_both_homologs']):.2f}" for y in [1,0])
            b=' / '.join(f"{100*float(pair[y]['fraction_with_interolog_support']):.2f}" for y in [1,0])
            lines.append(f'| {SPECIES_LABELS[species]} | {ref} | {a} | {b} |')
    lines+=['','These class-conditional differences are information that FADI’s distinct-protein separation score does not quantify. '
      'Neither a total overlap count nor a training-size comparison can determine the mechanism responsible for a neural performance difference.','',
      '## Common-row sensitivity analyses','',
      '| Species | Subset | Rows | Positives | Negatives |','|---|---|---:|---:|---:|']
    for r in csvrows(ROOT/'results/nonhuman-subset-counts.csv'):
        lines.append(f"| {SPECIES_LABELS[r['dataset']]} | {r['subset']} | {int(r['rows']):,} | {int(r['positives']):,} | {int(r['negatives']):,} |")
    lines+=['','Both masks apply identically to all 13 predictors. The exact-exposure mask concerns these two TRAIN references only; '
      'it is not the broader all-competitor exposure mask in the neural benchmark. The primary results preserve duplicates/conflicts. '
      'Subset metrics are point estimates in [nonhuman-subsets.csv](results/nonhuman-subsets.csv).','',
      '## Methods, uncertainty and limits','',
      '- [Original frozen protocol](PROTOCOL.md) and [five-species addendum](NONHUMAN-ADDENDUM.md). '
      'New numerical baseline results were unknown when the addendum was frozen; historical neural results and FADI/TRIQ were already known.',
      '- v5 TRAIN: 700,764 rows, 350,382 positives, 13,110 distinct sequences. Original Bernett TRAIN: '
      '163,192 rows, 81,596 positives, 4,286 distinct sequences. No validation examples enter counts, graph construction or fitting. '
      'Bernett is the full source release; the historical neural models used their recorded cleaned/capped variants and different training trajectories.',
      '- Fixed log((positive occurrences + 1)/(negative occurrences + 1)); self-pairs count once. '
      'The same 422 features and 300-iteration histogram gradient boosting regressor are used. '
      'Graph lookup uses up to five matches per endpoint and maximizes the product of their weights over positive TRAIN edges.',
      '- CPU MMseqs2 uses the original 25%-identity/50%-bilateral-coverage search settings, E≤1e-5, sensitivity 7.5 and 1,000 candidates. '
      'Searches are against each TRAIN separately. The inherited reported-identity/coverage weights and tie-breaking are unchanged. '
      'FADI’s different search was not substituted.',
      '- 500 paired protein-endpoint bootstrap replicates per species, seed 20261006, shared across all 13 predictors. '
      'Intervals are descriptive and unadjusted for multiple comparisons; they omit training-seed, family/phylogenetic and data-construction uncertainty. '
      'No scores were inverted, recalibrated or selected using the test labels.',
      '- Three fixed probes cannot exhaust every shortcut. Protein-propensity models do not learn partner-specific compatibility; '
      'interolog lookup may exploit legitimate conserved biology. Baseline performance does not identify a neural model’s causal mechanism.','',
      '## Verification and computational audit','',
      f"- **{len(verification['checks'])} independent checks passed**, including all 242,000 exported rows with 13 scores per row. "
      'Point metrics, shared bootstrap samples, intervals, sources and neural score mappings were verified.',
      '- Original human study: 83 artifact hashes verified before extension and archived. The v5 regressor, degree targets, graph and original prediction arrays remain unchanged. '
      'The new homolog parser reproduces every original v5 index/weight exactly; reloaded model predictions reproduce saved values exactly.',
      '- An auxiliary export issue was diagnosed: MMseqs `nident` was unavailable without backtraces. Backtrace searches reproduced '
      'all nine baseline-relevant fields for every alignment, so **no score changed**. The raw initial integer diagnostics in '
      '`predictions-frozen.json` are superseded by `alignment-precision-{v5,bernett}.json`. '
      'One apparent boundary discrepancy in the integer export was checked by counting aligned residues directly: 211/841 = 25.0892%, '
      'which passes the requested identity cutoff. See [identity-boundary-check.json](provenance/nonhuman/identity-boundary-check.json).',
      '- [Fallback diagnostic](results/nonhuman-v5-fallback-diagnostic.csv) was added after seeing below-chance v5 homology-degree results. '
      'It measures agreement with the existing fallback convention; no alternative predictor or flipped score was evaluated.',
      '- CPU only on the existing Arrhenius allocation; two 16-thread searches run concurrently. Initial search times: v5 125.34 s, '
      'Bernett 87.03 s. Auxiliary backtrace checks add 126.38 s and 87.44 s. The new Bernett regressor fit took about 1.04 s; '
      'v5 fit time is zero. These exclude preparation, qualification, export and plotting and are machine-specific timings.','',
      '## Artifacts and reproduction','',
      '- [Combined metrics](results/metrics.csv), [intervals](results/confidence-intervals.csv), '
      '[paired differences](results/paired-differences.csv), and [summary](results/summary.json).',
      '- `results/nonhuman-*` contains extension-only tables, coverage, subset analyses, mean metrics and verification. '
      '`results/{species}-predictions.csv.gz` preserves all row-level scores, labels, sequence hashes and support counts.',
      '- All three figure groups are exported as PNG/PDF/SVG. `models/nonhuman/` stores the Bernett fit and both references’ mapped protein values/graphs/hits. '
      '`provenance/nonhuman/` stores frozen settings, source mappings, checksums, reuse proofs and corrections.',
      '- Existing heavy inputs, models and predictions remain local; lightweight reports, tables, figures and code are versionable.','',
      '```bash','bash benchmark-v5-baselines/scripts/run.sh','```','',
      'The runner verifies and reuses completed stages. It never refits the saved v5 model. '
      'Final completion is recorded in `results/COMPLETE.json` after the combined outputs pass integrity checks.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))

def main():
    freeze_extension();assert read(ROOT/'results/nonhuman-verification.json')['status']=='passed'
    human=read(ARCHIVE/'results/summary.json');new=read(ROOT/'results/nonhuman-summary.json')
    for target,extension in [('metrics.csv','nonhuman-metrics.csv'),('confidence-intervals.csv','nonhuman-confidence-intervals.csv'),('paired-differences.csv','nonhuman-paired-differences.csv')]:
        old=csvrows(ARCHIVE/'results'/target);added=csvrows(ROOT/'results'/extension)
        write_csv(ROOT/'results'/target,old+added)
    means=[{'model':name,'mean_ap':float(np.mean([new['datasets'][s]['models'][name]['ap'] for s in SPECIES])),
                       'mean_auroc':float(np.mean([new['datasets'][s]['models'][name]['auroc'] for s in SPECIES]))} for name in ALL_MODELS]
    write_csv(ROOT/'results/nonhuman-means.csv',means)
    combined={'complete':True,'at_utc':now(),'config':CONFIG,'datasets':{**human['datasets'],**new['datasets']},
        'human_checks':human['checks'],'nonhuman_checks':new['checks'],'training_references':new['training'],
        'shared_human_test_pairs':human['shared_test_pairs'],'original_human_summary':'archive/before-nonhuman/results/summary.json',
        'nonhuman_summary':'results/nonhuman-summary.json','original_v5_fitted_models_reused_unchanged':True}
    atomic(ROOT/'results/summary.json',combined)
    plots(human,new);report(human,new)
    print('Combined report, tables and all PNG/PDF/SVG figures updated',flush=True)

if __name__=='__main__':main()
