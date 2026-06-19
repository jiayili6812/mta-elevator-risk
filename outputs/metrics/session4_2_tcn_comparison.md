# Session 4.2 Development-Only TCN Comparison

Sequence length and architecture selected using folds 1-3 only. Selected configuration evaluated once on fold 4 across three predeclared seeds.

| Model | Folds 1-3 mean PR-AUC | Four-fold mean | PR-AUC std | Worst fold | Fold 4 PR-AUC | Mean ROC-AUC | Mean Brier | Fold 4 precision | Fold 4 recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tcn_seed_42 | 0.680844 | 0.687796 | 0.028627 | 0.638681 | 0.708650 | 0.741655 | 0.205439 | 0.658354 | 0.602740 | 137 | 174 |
| random_forest | 0.669710 | 0.682456 | 0.032807 | 0.633389 | 0.720693 | 0.738513 | 0.204708 | 0.613861 | 0.707763 | 195 | 128 |
| hist_gradient_boosting | 0.663043 | 0.675375 | 0.035789 | 0.617363 | 0.712371 | 0.737653 | 0.203219 | 0.621677 | 0.694064 | 185 | 134 |

Selected sequence length: `12` months.
TCN selection-fold PR-AUC difference versus strongest matching tabular model: `0.011134`.
Retain TCN under the predeclared materiality rule: `False`.

Frozen-test labels and metrics were not accessed. No final production model decision was made.
