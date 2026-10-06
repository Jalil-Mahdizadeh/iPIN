"""Build the evidence tables and report only after all requested tasks finish."""
import json
from pathlib import Path
import pandas as pd
from datasets_eval import ROOT,SPECIES

R=ROOT/'results'
coverage=json.loads((R/'coverage.json').read_text())
assert coverage['complete'] and not coverage['missing'], 'Final report requires complete coverage.'
fresh=pd.read_csv(R/'fresh-metrics.csv');source=pd.read_csv(R/'published-score-metrics.csv')
concord=pd.read_csv(R/'score-concordance.csv')
audit=json.loads((R/'data-audit.json').read_text())
seq=json.loads((R/'sequence-identity-audit.json').read_text())
split=json.loads((R/'mutation-split-audit.json').read_text())
diagnostic=json.loads((R/'bernett-cap-diagnostic.json').read_text())
assert diagnostic['status']=='score_concordance_explained'
def metric(task,subset='all',score='score'):
    d=fresh[(fresh.task==task)&(fresh.subset==subset)&(fresh.score_definition==score)]
    assert len(d)==1,(task,subset,score)
    return d.iloc[0]
def deposited(figure,dataset,model):
    d=source[(source.figure==str(figure))&(source.dataset==dataset)&(source.model==model)]
    assert len(d)==1
    return d.iloc[0]
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def f(value):return f'{value:.6f}'

cross=[]
for s,reported in zip(SPECIES,[.904,.913,.888,.706,.722]):
    d=metric('cross_'+s);p=deposited('2',s,'PLM-interact')
    cross.append([s.capitalize(),f'{int(d.n):,}',f'{reported:.3f}',f(p.average_precision),f(d.average_precision),f(d.average_precision_sigmoid_float64)])
cross_table=table(['Test','Rows','Paper AUPR','Deposited AP','Fresh FP32 AP','Fresh AP, source precision'],cross)

bernett=[]
cols=['average_precision','auroc','precision','recall','f1']
bernett.append(['Paper']+[f'{x:.2f}' for x in [.69,.70,.63,.71,.66]])
bernett.append(['Deposited scores']+[f(deposited('4','Bernett','PLM-interact')[c]) for c in cols])
bernett.append(['TUnA deposited scores']+[f(deposited('4','Bernett','TUnA')[c]) for c in cols])
for task,label in [('bernett_full','Fresh, full sequences'),('bernett_1603','Fresh, cap 1,603 tokens'),('bernett_2196','Fresh, cap 2,196 tokens')]:
    bernett.append([label]+[f(metric(task)[c]) for c in cols])
bernett.append([f'Fresh, inferred {diagnostic["inferred_token_cap"]:,}-token cap (post hoc)']+[f(diagnostic['metrics'][c]) for c in cols])
bernett_table=table(['Evaluation','AP','AUROC','Precision','Recall','F1'],bernett)

mutation=[]
for prefix,label,source_model in [('mutation_ft','Mutation checkpoint','PLM-interact-FT_all_layers'),('mutation_zero','Zero-shot humanV11','PLM-interact-zero_shot')]:
    p=deposited('5','mutation_published598',source_model)
    d=metric(prefix+'_full','published598');written=metric(prefix+'_full','published598','probability_log_ratio')
    mutation.extend([[label,'Deposited scores',f(p.average_precision),f(p.auroc)],
                     [label,'Fresh, released-code formula',f(d.average_precision),f(d.auroc)],
                     [label,'Fresh, paper equation',f(written.average_precision),f(written.auroc)]])
mutation_table=table(['Checkpoint','598-case scoring','AP','AUROC'],mutation)

mut_sens=[]
for prefix,label in [('mutation_ft','Mutation checkpoint'),('mutation_zero','Zero-shot humanV11')]:
    for suffix,length in [('full','Full sequences'),('1603','1,603-token cap'),('2196','2,196-token cap')]:
        d=metric(prefix+'_'+suffix,'published598')
        mut_sens.append([label,length,f(d.average_precision),f(d.auroc)])
mut_sens_table=table(['Checkpoint','Sequence handling','598-case AP','598-case AUROC'],mut_sens)
ft841=metric('mutation_ft_full','all841');zero841=metric('mutation_zero_full','all841')
bc=concord[concord.task.isin(['bernett_full','bernett_1603','bernett_2196'])]
bernett_best=bc.sort_values('mae').iloc[0]
bernett_concordance_table=table(['Sequence handling','Mean absolute score difference','Maximum difference','Classifications differing at 0.5'],
    [[r.task,f'{r.mae:.3g}',f'{r.max_abs:.3g}',int(r.classification_flips_at_0_5)] for r in bc.itertuples()])

supp=[]
for s in SPECIES:
    a=audit[s]['published_reverse_concordance']
    c=concord[concord.task==f'reverse_{s}_sample_vs_fresh_original'].iloc[0]
    supp.append([s.capitalize(),f"{a['classification_flips_at_0_5']:,} / {audit[s]['n']:,}",f"{int(c.classification_flips_at_0_5)} / {int(c.n)}"])
reverse_table=table(['Species','Published full-test order flips','Fresh sampled order flips'],supp)
gpu_hours=sum(t['summed_gpu_worker_seconds'] for t in coverage['tasks'])/3600
cross_precision_delta=concord[concord.task.isin(['cross_'+s for s in SPECIES])].ap_float64_sigmoid_minus_source.abs().max()
main=[metric('cross_'+s) for s in SPECIES]
paper_matching_cross=all(abs(d.average_precision_sigmoid_float64-p)<.0005 for d,p in zip(main,[.904,.913,.888,.706,.722]))
mutation_match=abs(metric('mutation_ft_full','published598').average_precision-.612)<.0005 and abs(metric('mutation_ft_full','published598').auroc-.794)<.0005
verdict=dict(cross_species_matches_reported_precision_with_source_score_precision=paper_matching_cross,
             mutation_checkpoint_matches_figure5_with_code_formula=bool(mutation_match),
             bernett_full_matches_reported_rounded_metrics=all(abs(metric('bernett_full')[k]-v)<.005 for k,v in zip(cols,[.69,.70,.63,.71,.66])),
             bernett_score_closest_variant=bernett_best.task,bernett_closest_mae=float(bernett_best.mae),
             bernett_post_hoc_inferred_token_cap=diagnostic['inferred_token_cap'],bernett_inferred_cap_mae=diagnostic['mean_absolute_score_error'],
             no_retraining=True,complete_test_coverage=True,not_a_reproduction_of_every_paper_claim=True)
(R/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')

report=f'''**PLM-interact reproducibility check — frozen released checkpoints, supplied SIF**

The released checkpoints reproduce the main cross-species, Bernett, and mutation benchmark metrics when the evaluation inputs and actual scoring convention are respected. Two implementation details matter: mutation prediction uses a different formula from the paper's equation, and Bernett's deposited predictions are strongly consistent with an undocumented 3,570-token cap. This is a checkpoint-inference reproduction, with **no retraining**. It is not a claim that every result in the paper has been independently reproduced.

The evaluation covers all **294,889 released test rows**: 242,000 cross-species pairs, 52,048 Bernett pairs, and 841 mutation cases. The humanV11, Bernett, and mutation checkpoints were loaded strictly and kept frozen. The same supplied SIF ran every fresh prediction. All expected rows and labels passed the final coverage checks, and output/checkpoint hashes are retained. The paper and supplements are [available locally](../literature/plm-interact/READING_NOTES.md); the detailed procedure is in [PROTOCOL.md](PROTOCOL.md).

**Figure 2: all five cross-species benchmarks agree.** AUPR here means average precision (AP), as in the authors' code. Trapezoidal PR-AUC is retained separately in the metric files.

{cross_table}

The workbook's probabilities exactly equal a float64 sigmoid applied to the deposited logits, within about 1e-15. Applying that same numeric representation to fresh logits gives a maximum AP disagreement of {cross_precision_delta:.3g} across species. Direct FP32 sigmoid outputs create more tied scores. The small yeast difference crosses the three-decimal rounding boundary: direct FP32 AP rounds to 0.705, while the source-consistent representation rounds to 0.706. Both results are retained; no checkpoint, threshold, or model parameter was selected to improve agreement. Fresh and deposited original-order cross-species predictions make identical binary decisions at 0.5.

The baseline curve files use the same sequence/label multisets, despite accession aliases and different row orders. PLM-interact remains ahead of their deposited APs on all five species. However, the paper's baseline bars and its deposited curves are not always numerically the same experiment: the workbook explicitly says the bars were taken from prior publications or correspondence, whereas the curves came from checkpoint reruns. For example, E. coli TUnA is 0.677 in the bar but 0.688506 from the deposited curve. Thus, some quoted percentage improvements depend on which baseline numbers are used. See [baseline-bars-versus-curves.csv](results/baseline-bars-versus-curves.csv). Baseline models were not rerun in this task.

**Figure 4: Bernett, all 52,048 test pairs.** The threshold is fixed at 0.5. Full sequences were the prespecified primary evaluation; the two explicit truncation settings are sensitivity checks.

{bernett_table}

{bernett_concordance_table}

Among the prespecified evaluations, the closest score-level agreement is `{bernett_best.task}` (mean absolute difference {bernett_best.mae:.3g}). There are 8,200 pairs longer than 1,603 tokens and 3,392 longer than 2,196 tokens; the maximum is 7,426. Full-sequence inference reproduces every Figure 4 metric at the displayed two-decimal precision, but differs from the deposited classification for 39 pairs. The paper's AP/AUROC tie with TUnA is a rounded tie: the unrounded deposited TUnA ranking metrics are slightly higher, while PLM-interact has higher recall and F1. No rows were dropped for length.

A separately labeled **post-hoc diagnostic** localized almost all Bernett score errors to the longest 1% of inputs. Caps 3,569–3,573 were probed against individual deposited scores, without selecting on labels or AP. Only **3,570 tokens** matched the diagnostic cases closely. Fresh inference with that cap was then checked on **all {diagnostic['affected_rows']} affected rows**, retaining the full-sequence results elsewhere. It reduced the full-test mean absolute score error to **{diagnostic['mean_absolute_score_error']:.3g}**, maximum error to **{diagnostic['max_absolute_score_error']:.3g}**, and classification disagreements to **{diagnostic['classification_flips_at_0_5']}**. This strongly supports a 3,570-token inference cap as the missing preprocessing detail; that number was not found in the paper or released example commands. The original primary results remain unchanged. All probe results, the criterion, and the additional predictions are retained in [bernett-cap-diagnostic.json](results/bernett-cap-diagnostic.json) and its accompanying CSVs.

**Figure 5: mutation results reproduce with the released code's scoring formula.** The 598 published samples were matched to the released 841-case test set using mutation/interaction metadata. They are exactly the cases with each individual protein at most 2,000 residues. Nine workbook mutation ranges had been converted into Excel dates; the documented metadata/date checks resolve those joins without editing the inference sequences.

{mutation_table}

The implementation scores `sigmoid(mutant_logit - wild_logit)`. The paper instead writes `log(p_mutant / p_wild)`, where `p = sigmoid(logit)`. These formulas share the classification direction but can rank mutations differently. The paper-equation rows above use a stable float64 calculation from the same fresh logits. Consequently, the numerical Figure 5 result is reproducible from the released code, but the written equation does not fully specify the implementation that generated it. The all-layer mutation checkpoint is the authors' already fine-tuned release; this run performed no fine-tuning.

On all 841 released cases, the mutation checkpoint obtains AP **{ft841.average_precision:.6f}**, AUROC **{ft841.auroc:.6f}**; zero-shot humanV11 obtains AP **{zero841.average_precision:.6f}**, AUROC **{zero841.auroc:.6f}**. The 598-case comparison is therefore not representative of the full test result. The classifier-only fine-tuned result was checked from deposited scores (AP 0.244588, AUROC 0.584039), but the authors' public model catalog does not provide that separate checkpoint, so its inference was not independently reproduced.

Sequence handling also materially affects mutation results:

{mut_sens_table}

At the public CLI default of 1,603 tokens, 169 of the 841 cases (72 of the published 598) have identical tokenized wild and mutant inputs because truncation removes the mutation. At 2,196 tokens the counts are 93 and 36. Full inputs retain every mutation. The released mutation split has no exact sequence-triplet or wild-type interaction-pair overlap between training and test, although 271 test cases have both individual proteins seen in training and 643 have at least one seen. These overlap checks only read the split files; they did not fit or select a model.

**Supplementary results require narrower interpretations.** All deposited masking-ablation APs and sequence-identity-bin APs were independently recomputed. Masking does not improve every reported AP: mouse AP is 0.907840 without masking and 0.903963 at 15% masking. PLM-interact is also below TUnA in four deposited identity bins, including mouse identity (40,60] (0.622185 vs 0.714071). The fly/worm display labels are swapped relative to their hostname fields in the masking table. The reported McNemar p-values agree with exact binomial calculations from the reported discordant counts, but those counts do not reproduce when the deposited scores are classified at a common probability threshold of 0.5. The classification threshold/procedure for that test is insufficiently specified for this audit to verify the counts. These findings are preserved in the supplementary CSVs, without retraining any ablation.

Near-equal AP after reversing proteins does not mean individual predictions are symmetric. The deposited full-test scores, and fresh inference on a fixed every-53rd-row subset, show classification changes:

{reverse_table}

The E. coli deposited predictions flip for 11.4% of pairs even though the two aggregate APs are close. The fresh reversed subset is compared with both fresh original-order scores and the matching deposited reversed scores in [score-concordance.csv](results/score-concordance.csv). All primary runs preserve the released pair order. Also, E. coli contains 3,770 duplicate unordered sequence/label rows among 22,000 rows; original multiplicities were retained to reproduce the paper, and should not be mistaken for independent observations.

Fresh inference and all prespecified sensitivity runs used approximately **{gpu_hours:.2f} GPU-hours of measured worker evaluation time**, excluding loading, queueing, the qualification pilot, and the short post-hoc Bernett diagnostic. SLURM job 3110591 used eight GH200 GPUs on two nodes; mutation inference used the existing one-GPU interactive allocation 3085063. Each worker records its GPU UUID. Qualification produced bitwise-identical logits against the upstream model class, and singleton/padding comparisons passed. The publication archive and current inference model computations agree; the relevant source change only strengthens checkpoint loading from non-strict to strict.

The main limits are explicit: there was no retraining; no fresh baseline-model inference; no independent classifier-only mutation checkpoint evaluation; no rerun of structure-generation examples or MMseqs alignments; and no fresh virus-host inference. Figure 7's workbook provides a summary table and training sequences, rather than held-out prediction scores, so its reported AP/F1/MCC are recorded but not independently verified here. The [published paper](https://www.nature.com/articles/s41467-025-64512-w), [publication code archive](https://doi.org/10.5281/zenodo.16643324), and pinned release provenance identify the claims and artifacts under review.

The numerical evidence is in [fresh-metrics.csv](results/fresh-metrics.csv), [published-score-metrics.csv](results/published-score-metrics.csv), [score-concordance.csv](results/score-concordance.csv), and [coverage.json](results/coverage.json). The raw logits, probabilities, labels, and row IDs are saved in `results/predictions/` and the merged result CSVs. The [main PR curves](results/plots/pr-curves-main.pdf) and [mutation scoring comparison](results/plots/mutation-score-definitions.pdf) provide standalone figures; PNG versions are alongside them. Run settings, hashes, commands, and exact scope are documented in [PROTOCOL.md](PROTOCOL.md) and [README.md](README.md).
'''
(ROOT/'REPORT.md').write_text(report)
print('Wrote REPORT.md and results/verdict.json')
