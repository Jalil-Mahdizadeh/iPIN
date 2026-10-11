# V8: evaluate the useful signal before further model development

Authorized 11 October 2026. Execution lives in `benchmark-v8/`.

## Recommendation and evidence

Pause new model training. First determine whether frozen Vx improves partner
recognition in existing validation data; then assess whether the historical
human experimental panel can support a useful frozen-model comparison.

Vx-128 improved original Bernett TEST AP from 0.690319 to 0.698620, a modest
gain. Its pilots did not establish homolog-pairing-specific co-evolution, and
V7 contact summaries added only 0.000105 AP on the pilot assessment. Preserve
these completed results; V8 does not reopen TEST or select another model.

The historical `../iPIN-OpenPPI` repository supplies relevant evaluation methods:

- A matched-data pooled head reached C3 concordance 0.7794 versus TUnA-31k's
  0.7867; the paired difference was unresolved. The pooled head retrieved more
  known partners in the top ten (17.33% versus 13.96%).
- Within-anchor ranking and endpoint-balanced swaps challenge endpoint
  propensity. Simple sequence similarities must remain controls: they also
  performed well on the historical swaps.
- Its curated Lambourne Y2H panel retained 77 positives and 1,863 confirmed
  assay non-detections after historical exposure exclusions. TUnA-31k achieved
  AUROC 0.6115 and AP 0.0556, against positive prevalence 0.0397. Our frozen
  Bernett PLM-interact and Vx were not included in that comparison.

These are different benchmarks and estimands. Historical PLM-interact used
humanV11, one chain order and a 1,603-token cap; its numbers cannot establish
superiority over our full-length, symmetric Bernett baseline. Relevant records:
[final historical assessment](../iPIN-OpenPPI/experiments/study_finalization_v1/REPORT.md),
[partner diagnostics](../iPIN-OpenPPI/experiments/expanded_controls_v1/results/INTERPRETATION.md),
[assay source contract](../iPIN-OpenPPI/experiments/independent_human_binary_v1/SOURCE_ADDENDUM.md).

## A. Existing VAL scores: partner recognition

Reuse the verified `retrain-vx` native, true-Vx and quality/profile logits, heads,
fusion weights, protein identities and component assignments. No encoder calls,
fits, calibration, hyperparameter search or homolog shuffling. Add fixed
amino-acid-composition cosine, ordered 3-mer cosine and length-ratio controls;
these require sequences only and are not fitted to labels.

Primary population: the original `assessment_new_rows` (assessment components,
excluding the original pilot rows). All assessment rows are a secondary
sensitivity. Calibration and crossing rows never enter these comparisons. The
data were previously examined; every V8 result is a retrospective diagnostic.

1. **Within-anchor concordance:** for each protein with at least one positive
   and one negative-labelled candidate, compare its positive partners with its
   negative-labelled partners. Average equally across anchors, retaining ties.
2. **Early retrieval:** expected-tie Recall and NDCG at 1/5/10, restricted to
   anchors with at least 20 negative-labelled candidates. Keep the same candidate
   lists for every model and report their size distribution.
3. **Endpoint-balanced swaps:** compare `s(A,B)+s(C,D)` with
   `s(A,D)+s(C,B)`, allowing both alternative matchings, requiring four distinct
   proteins, two existing positive edges and two existing negative-labelled
   edges in the same population. Never invent or score a missing edge. Use
   deterministic, score-independent sampling, maximum 2,000 quartets, maximum
   20 appearances per positive edge and 40 per protein. Retain native logit
   scales; probability transformations can change summed-score preferences.
   An additive score `g(A)+g(B)` cancels exactly. This tests query-partner
   preference, not homolog-row pairing or physical contact mechanisms.

Use 2,000 paired component-multiplier bootstrap replicates (seed 20261011).
Count a shared component once per edge/quartet. For within-anchor concordance,
resample anchor and partner components and recompute the ratios. For retrieval,
resample anchor components conditional on the fixed candidate lists; disclose
remaining dependence through shared partners. Components are the inherited
heuristic sequence components, not certified independent biological families.

Interpret intervals only with >=50 anchors and >=20 anchor components (anchor
and retrieval metrics), or >=200 quartets and >=20 participating components
(swaps), and >=95% finite bootstrap replicates. Preserve unsupported points.
Report Vx minus native and Vx minus quality plus every fixed sequence control;
never select the easiest comparator after seeing results.

Evidence warranting further partner-specific development requires BOTH primary
within-anchor and swap improvements >=0.010 over native with positive paired
95% lower bounds, positive lower bounds over quality on both, and positive swap
lower bounds over both composition and 3-mer controls. This conjunction is a
screen for further research, not independent confirmation. Other contrasts and
retrieval results are descriptive pointwise intervals. Failure to clear the
screen, including insufficient support, does not establish equivalence.

## B. External assay: exposure and support before predictions

Validate the historical source freezes and reuse the curated source/mappings,
including their technical exclusions and discordant-record exclusions. Do not
reinterpret non-detections as universal noninteraction, or import historical
model scores. Begin with the historical primary exposure-excluded population;
also report attrition from the full eligible final panel for transparency.

Audit exact unordered sequence pairs against every original Bernett TRAIN and
VAL row. This is conservative relative to the native cleaned TRAIN subset and
the Vx eligible fitting subset. Use the already materialized Bernett TEST
sequence-hash exclusion file, without opening TEST rows, sequences, labels or
predictions. Exclude pairs touching those hashes from the primary external
comparison: this conservative endpoint exclusion removes possible previously
examined TEST pairs without reopening TEST. Report its attrition separately;
it is stricter than exact-pair exclusion and changes the target population.
Report TRAIN/VAL endpoint familiarity separately from exact-pair exposure.

Before new alignments or GPU work, calculate an optimistic MSA-support ceiling
using existing TRAIN/VAL monomer identities and the frozen 1,536-residue limit.
Absence from these caches is **unknown**, not proven biological MSA absence.
Audit the archive's published/local query inventory if this is needed to resolve
support; no new MSA search or input rescue is allowed. An unresolved cache-only
ceiling cannot by itself prove the complete archive lacks support.

Minimum eligibility for an external comparison: >=40 positives, >=100 confirmed
non-detections, >=20 sequence components, at least 40 positives with usable Vx
input, and no one component accounting for >25% of retained positives. Build or
verify components and training homology/interolog exposure before any scoring
if the earlier class-support gates pass. These are feasibility floors, not a
claim of power to resolve the small Bernett gain. A surviving panel could test
large transfer effects; report its precision limitations explicitly.

If an earlier required floor fails, finish the audit and report a scientific
no-go for this panel. Do not relax exclusions, choose favorable subgroups, replace
the source or run expensive encoding to rescue it. Continue stage A regardless.

## C. Conditional frozen comparison

Only if B passes and exact input/scoring qualification succeeds: compare native
Bernett PLM-interact, native+Vx-128 and native+quality using the exact frozen
`retrain-vx` heads and fusion weights, full native sequences and mean AB/BA
logits. Preserve the accession-matched MSA recipe, projections, pooling, gates,
and native fallback for true biological unavailability. For external homolog
context exclude exact TRAIN/VAL query sequences using the existing held-out
convention. Incomplete/corrupt computation aborts, never becomes fallback.

Freeze eligible rows, code, hashes and scoring configuration before inference;
keep assay outcomes unavailable to the scorer. Primary readout is paired AP
gain over native, with AUROC and known-positive recovery at budgets 10/50/100,
paired component intervals, quality comparison and MSA-coverage counts. A
practical external signal requires >=0.010 AP gain, positive lower bounds over
native and quality, and no AUROC loss greater than 0.005. This is an exploratory
source-transfer comparison, not a freshly blind confirmatory benchmark.

## Integrity, resources and stopping

Write new artifacts only in `benchmark-v8/` and this proposal. Pin source inputs
and use sequence hashes, never implicit embedding row order. Verify saved VAL
predictions against their completion manifest and independently replay metric
arithmetic, ties, quartet memberships, additive cancellation and paired draws.
No model training, TEST evaluation, source modification or automatic new study.

Stages A/B use the existing allocation, CPU-only inside a pinned container,
<=8 threads, <=4 hours and <=5 GB additional storage. Do not request a GPU node
for work that can use cached predictions. If C qualifies, use at most one GPU,
<=8 GPU-hours and <=20 GB, after a label-free timing check. A resource overrun or
failed scientific gate stops with an explicit report. No automatic retry of a
scientific readout, outcome-driven amendment, commit or push.

The final report must distinguish evidence against useful gain, insufficient
statistical support, unavailable biological input and execution failure. Resume
model development only if this work identifies a reproducible, specific gap.
