# V7: frozen MSA Pairformer contact features for PLM-interact

**Status:** agreed proposal, saved 11 October 2026. Implementation and experiment launch are not part of this document-saving task. The review was read-only and did not access TEST.

**Recommendation:** conduct one small, tightly specified pilot. The hypothesis is that the pretrained contact head identifies useful residue-level signals before Vx's global compression loses them. Expectations remain cautious given the modest full TRAIN/VAL gain and the unsuccessful Vx-v2 pooling experiment.

## 1. Which pretrained output to use

The released code contains two relevant outputs:

| Output | Input representation | Recommendation |
|---|---|---|
| `predicted_cb_contacts` | Zero-based layer 15, matching Vx | **Use first:** Cβ-distance contact predictions |
| `predicted_confind_contacts` | Zero-based layer 18 | Defer: different contact definition and representation layer |

Both are residue-contact heads; the A×B block supplies inter-chain predictions. Neither output is a calibrated probability that two proteins interact. The head applies **LayerNorm → learned linear projection → sigmoid → probability symmetrization**. Applying it to Vx's already averaged or compressed features would not reproduce that computation. Use the existing pinned source and weights. [Released model](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py), [head implementation](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/regression.py).

## 2. Why this is worth a bounded pilot

There is encouraging evidence already in the repository: on the example complex **1B70**, the contact head recovered **9/10 top-ranked contacts with true pairing versus 0/10 after global shuffling**. This is an implementation sanity check involving one known complex. Its global shuffle differs from Vx's restricted taxonomic shuffle, and it does not establish Bernett interaction discrimination or independent generalization. [Existing diagnostic](pilot-vx-series/pilot-vx/audit-20261008/positive-control.json).

The complementary information would come from the head's **structurally supervised interpretation of the existing tensor**, rather than new input information. That could outperform random projections. However, full retraining improved primary assessment AP only from **0.647962 to 0.654981**, and Vx's advantage over quality/profile was uncertain: **+0.003078**, with a 95% protein interval of **[-0.001289, +0.007732]**. Generic local pooling also failed to improve Vx. Beating native PLM-interact alone would therefore be insufficient evidence for this proposal. [Full results](retrain-vx/results/REPORT.md), [Vx-v2 results](pilot-vx-series/pilot-vx-v2/results/REPORT.md).

## 3. Minimal experiment

Reuse the original **4,000 TRAIN / 4,000 VAL pilot pairs**. Preserve depth 128, accession matching, eligibility, masks, length limits, both chain orientations and exact native fallback. Use the frozen Cβ head at layer 15, with no encoder or contact-head training. Extract the contact probabilities before summary compression, align the AB/BA outputs to the same A×B coordinates, and average their probability maps before computing features.

Start with **six features**, specified before examining outcomes:

| Feature | Intended evidence |
|---|---|
| Mean of the top 20 valid contact probabilities, capped by the number of valid cells | Strength of the strongest contact evidence |
| Fraction of valid cells with probability ≥0.5 | Contact density |
| Smaller and larger A/B participating-residue fractions — two features; participation means at least one valid contact probability ≥0.5 | Whether support involves multiple residues on both chains |
| Average normalized entropy of row/column contact mass | Concentrated versus diffuse support |
| Highest supported 5×5-window mean, using original residue coordinates | Local coherence in sequence coordinates |

Fix the threshold, valid-window support rule, degenerate-case handling and `k` before outcomes; do not search over alternatives. Treat 0.5 as an operational cutoff, not established calibration. Record maximum probability diagnostically, but exclude it from the initial fitted feature set: one spurious contact can dominate it. Sequence-grid patches must not be described as verified physical interface patches.

Fit a **small, regularized logistic readout**. Keep PLM-interact frozen. Use TRAIN only for new fitting, standardization, calibration and fusion, with out-of-fold predictions for trainable heads used in stacking. Preserve the existing VAL component assignments and assessment population; do not select weights or features on VAL. Existing Vx predictions remain frozen references. Because native PLM-interact was itself trained on TRAIN, TRAIN calibration is not independent validation.

Compare:

- Native PLM-interact.
- Native + quality/length control.
- Native + Vx.
- Native + contact features.
- Native + Vx + contact features.

The **primary comparison** is native + Vx + contacts versus native + Vx: do contact features contribute information beyond the existing readout? The contact-only fusion is a secondary comparison testing whether the contact readout is a stronger replacement. Freeze these comparisons and the readout/calibration specification before outcomes; do not select a winning specification on VAL.

## 4. Essential controls and safeguards

- **Original Vx homolog-row shuffle:** use identical contact extraction and fitting, and also evaluate the true-trained head on shuffled inputs. Similar performance would permit predictive usefulness but would not establish pairing-specific evidence.
- **Quality and opportunity controls:** report fixed strata for paired depth, Neff80, effective joint coverage, valid map size and chain lengths, plus protein/family influence. Determine whether contact features add value beyond quality/length controls. Preserve the limits of available family annotations.
- **Map-coordinate permutation diagnostic:** permuting rows and columns preserves confidence distributions and participating-residue counts while disrupting sequence-local windows. This provides a cheap diagnostic of whether the window feature contributes useful locality, without another encoder pass.
- **Boundary sensitivity:** upstream code assigns adjacent-token bond features across the concatenated A/B boundary, although positional encoding accounts for the chain break. This is a potential terminal-contact artifact, not evidence that existing results are wrong. Require a predefined boundary-exclusion sensitivity check without modifying the pretrained model. [Relevant code](images/msa-pairformer/sources/upstream/MSA_Pairformer/model.py).
- **Extraction correctness:** verify valid-residue masking, chain-order symmetry and agreement with the official contact API. Masked cells must not contribute as zero-valued evidence or inflate denominators. Preserve fail-closed handling and exact baseline fallback; do not silently discard failed pairs or change the eligible population.
- **Exposure audit:** the released head introduces structural supervision that Vx deliberately avoided. Audit and document head-training membership and TRAIN/VAL overlap before interpreting a gain. Unresolved exposure prevents a claim of exposure-free generalization. [Original provenance discussion](improvment-proposal-vx.md).
- **No TEST access:** freeze the proposal, features, fitting policy, comparisons and diagnostics before evaluation. Do not modify the existing Vx pipeline or the running benchmark as part of this experiment.

## 5. Success, failure and interpretation

The proposed success criterion is **at least +0.005 primary assessment AP over Vx**, with a positive matched protein-bootstrap 95% interval, separation from quality controls, and no AUROC loss exceeding 0.005.

- A contact-only fusion outperforming Vx supports a stronger replacement readout.
- Improvement only when combined with Vx supports complementarity.
- An interval spanning zero is inconclusive; a small positive point estimate alone does not justify escalation.
- Improvement confined to long maps, deep MSAs, boundary contacts or a few families weakens the interface interpretation.
- True/shuffled similarity does not negate predictive utility, but it prevents a pairing-specific claim.

This repeatedly inspected VAL remains exploratory. Neither contact-score confidence nor improved Bernett AP alone establishes correctly localized physical interfaces.

The largest biological risk is **confident contacts for non-interacting pairs**: a structural contact head has not necessarily learned when to predict no interaction. Conversely, transient, disordered or indirect associations may receive weak contact evidence. MSA quality and family composition may still explain apparent gains.

## 6. Cost, caching and a possible follow-up

The head is cheap, but **Vx's cached summaries cannot reconstruct contact maps**, so trunk inference must be repeated. The comparable Vx-v2 true/shuffled run used about **8 allocated GH200 GPU-hours**; that is a starting estimate before qualification, calibration-score inference and map I/O, not a measured runtime for V7. Measure representative lengths and establish a resource ceiling before launch. [Resource record](pilot-vx-series/pilot-vx-v2/results/resources.json).

Reuse the existing MSAs, pinned image/weights and available baseline scores. Cache inter-chain maps, masks and provenance so later diagnostics require no new backbone passes; avoid storing the full 256-channel pair tensor. Keep the new extraction code separate from the frozen Vx implementation.

If these summaries fail despite convincing localized contact evidence, the next focused alternative would be **contact-weighted pooling of the existing projected Pairformer features**, with a shuffled-weight control. Reserve that for a separate hypothesis rather than expanding this pilot into a pooling search.
