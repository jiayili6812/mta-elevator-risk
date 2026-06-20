# Session 6 Prospective External Evaluation

This evaluates the locked Session 5 Random Forest on target-known official
months after `2025-05-01`. No model, feature, threshold, calibration, target,
or eligibility rule was changed.

## Source And Canonical Data

- Latest official source: `/content/mta-elevator-pipeline/data/external/MTA_NYCT_Subway_Elevator_and_Escalator_Availability__Beginning_2015.csv`
- Source rows: `80316` from `2015-01-01` through `2026-04-01`
- Saved canonical file: `data/processed/session6_latest_canonical_availability.csv`
- Canonical rows: `23239` from `2021-01-01` through `2026-04-01`
- Canonical schema matches locked contract: `True`

## Prospective Cohort

- Prediction months: `2025-06-01, 2025-07-01, 2025-08-01, 2025-09-01, 2025-10-01, 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01`
- Rows: `3425`
- Elevators: `346`
- Positive rows: `1499`
- Prevalence: `0.437664`

## Locked Model Performance

- PR-AUC: `0.653008`
- ROC-AUC: `0.711994`
- Brier score: `0.213711`
- Precision at `0.4433219097353501`: `0.564278`
- Recall at `0.4433219097353501`: `0.714476`
- Confusion matrix: TN `1099`, FP `827`, FN `428`, TP `1071`
- Calibration ECE: `0.043823`

## Metrics By Month

| Month | Rows | Elevators | Prevalence | PR-AUC | ROC-AUC | Brier | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-06-01 | 340 | 340 | 0.402941 | 0.626351 | 0.722016 | 0.211645 | 0.556150 | 0.759124 | 83 | 33 |
| 2025-07-01 | 340 | 340 | 0.373529 | 0.582517 | 0.704373 | 0.214610 | 0.481081 | 0.700787 | 96 | 38 |
| 2025-08-01 | 341 | 341 | 0.422287 | 0.615591 | 0.696983 | 0.216635 | 0.567901 | 0.638889 | 70 | 52 |
| 2025-09-01 | 341 | 341 | 0.463343 | 0.650556 | 0.708134 | 0.215523 | 0.604396 | 0.696203 | 72 | 48 |
| 2025-10-01 | 341 | 341 | 0.451613 | 0.646559 | 0.709251 | 0.215139 | 0.578947 | 0.714286 | 80 | 44 |
| 2025-11-01 | 341 | 341 | 0.466276 | 0.695730 | 0.723201 | 0.210364 | 0.583756 | 0.723270 | 82 | 44 |
| 2025-12-01 | 343 | 343 | 0.489796 | 0.759180 | 0.743333 | 0.203449 | 0.623711 | 0.720238 | 73 | 47 |
| 2026-01-01 | 346 | 346 | 0.485549 | 0.704036 | 0.724886 | 0.209602 | 0.625641 | 0.726190 | 73 | 46 |
| 2026-02-01 | 346 | 346 | 0.427746 | 0.666795 | 0.714544 | 0.214364 | 0.555556 | 0.777027 | 92 | 33 |
| 2026-03-01 | 346 | 346 | 0.393064 | 0.612019 | 0.677416 | 0.225710 | 0.467337 | 0.683824 | 106 | 43 |

## Frozen-Test Comparison

- Frozen-test PR-AUC: `0.648004` versus prospective `0.653008`
- Frozen-test ROC-AUC: `0.716806` versus prospective `0.711994`
- Frozen-test Brier: `0.209976` versus prospective `0.213711`
- Frozen-test precision: `0.526316` versus prospective `0.564278`
- Frozen-test recall: `0.694087` versus prospective `0.714476`

Plain-language assessment: Prospective performance is broadly consistent with the frozen-test result: PR-AUC is slightly higher, ROC-AUC is slightly lower, Brier score is slightly worse, and recall remains near the locked approximately-70% policy. Precision is higher than the frozen-test estimate, though the longer prospective window naturally contains more absolute false positives and false negatives.

## Guardrails

- Frozen-test evaluation was not rerun.
- The saved Session 5 model artifact was loaded unchanged.
- Feature set: `full_history_without_age`.
- Threshold: `0.4433219097353501` unchanged.
- Calibration: none.
- Prospective results were not used for tuning or retraining.
