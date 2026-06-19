# Session 5 Final Frozen-Test Evaluation

This is the one-time final evaluation on frozen labels for February-April 2025. It is distinct from all development results and was not used for tuning.

## Frozen-Test Cohort

- Rows: `974`
- Elevators: `326`
- Positive rows: `389`
- Prevalence: `0.399384`

## Final Frozen-Test Performance

- PR-AUC: `0.648004`
- ROC-AUC: `0.716806`
- Brier score: `0.209976`
- Precision at `0.4433219097353501`: `0.526316`
- Recall at `0.4433219097353501`: `0.694087`
- Confusion matrix: TN `342`, FP `243`, FN `119`, TP `270`
- Calibration ECE: `0.075761`
- Mean predicted probability: `0.474748`

## Metrics By Frozen-Test Month

| Month | Rows | Prevalence | PR-AUC | ROC-AUC | Brier | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-02-01 | 324 | 0.425926 | 0.689900 | 0.722553 | 0.208113 | 0.540698 | 0.673913 | 79 | 45 |
| 2025-03-01 | 324 | 0.413580 | 0.655915 | 0.734368 | 0.204615 | 0.556213 | 0.701493 | 75 | 40 |
| 2025-04-01 | 326 | 0.358896 | 0.601887 | 0.696602 | 0.217156 | 0.482558 | 0.709402 | 89 | 34 |

## Development Comparison

- Development four-fold mean PR-AUC: `0.679572`
- Development latest-fold PR-AUC: `0.715758`
- Development latest-fold precision: `0.626223`
- Development latest-fold recall: `0.706402`
- Frozen-test results do not alter any locked decision.

## Runtime And Guardrails

- Runtime seconds: `2.231`
- Model fit count: `1`
- Fit rows: `13202` ending `2025-01-01`
- Frozen labels accessed after model fit: `True`
- Audit log: `outputs/metrics/frozen_test_access.log`

## Limitations

- The source is a fixed cleaned snapshot with unknown historical cleaning steps.
- X-suffixed equipment is excluded because target reporting is structurally incomplete.
- Six consecutive observed months are required, so true zero-history elevator performance is not measured.
- Time Since Major Improvement semantics remain unresolved and the field is excluded.
- The frozen test covers only three months and may not represent longer-term temporal drift.
