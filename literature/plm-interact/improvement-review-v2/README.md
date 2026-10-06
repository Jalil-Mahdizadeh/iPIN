**Evidence supporting the second PLM-interact improvement proposal**

Prepared 29 September 2026 for [improvment-proposal-v2.md](../../../improvment-proposal-v2.md).

`audit_existing_results.py` describes the prepared train/validation/test data and recomputes length-stratified metrics from frozen benchmark predictions. It verifies prediction row coverage, labels and finiteness before calculating metrics. It performs no model fitting, GPU inference, ensembling or checkpoint selection. Test-stratum findings are exploratory because the historical test has already been examined.

Run from the workspace root:

```bash
bash retrain-v1/scripts/container.sh python literature/plm-interact/improvement-review-v2/audit_existing_results.py
```

`existing-data-audit.json` records the results, input hashes and a timestamp. The two `*-events-snapshot.jsonl` files preserve the live training observations used for runtime estimates and validation status. Rerunning the script overwrites these review artifacts and captures a newer training snapshot; it does not alter the training runs or benchmark.

`arrhenius-snapshot.json` captures read-only scheduler, project-usage and shared-storage queries. The usage tool explicitly warns that Arrhenius GPU accounting may be unreliable and does not establish a remaining allocation.

`sources/` contains downloaded primary research documents and text extractions. Download manifests record successful retrievals and failures; `source-manifest.json` indexes the retained files and their hashes. TUnA was read through the web PDF interface despite a failed direct download. RaftPPI's author repository was reviewed; its submission PDF was browser-blocked, so no numerical performance comparison from that PDF is asserted. External citations and their implications appear next to the relevant claims in the proposal.

The prior [native reproduction](../../../plm-interact-reproducability/REPORT.md), [benchmark](../../../benchmark-v1/REPORT.md), and [v1 proposal](../../../improvment-proposal-v1.md) supply the earlier experimental evidence.
