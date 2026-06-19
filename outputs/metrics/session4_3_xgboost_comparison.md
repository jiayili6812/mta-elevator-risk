# Session 4.3 Development-Only XGBoost Comparison

Configure XGBoost and select its approximately-70%-recall threshold using folds 1-3 only; evaluate the selected configuration unchanged on fold 4.

| Model | Folds 1-3 mean PR-AUC | Four-fold mean | PR-AUC std | Worst fold | Fold 4 PR-AUC | Mean ROC-AUC | Mean Brier | Fold 4 precision | Fold 4 recall | FP | FN | Runtime seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.665060 | 0.676464 | 0.030117 | 0.630528 | 0.710675 | 0.735403 | 0.203411 | 0.625490 | 0.704194 | 191 | 134 | 8.034 |
| random_forest | 0.667510 | 0.679572 | 0.032174 | 0.630940 | 0.715758 | 0.737848 | 0.205023 | 0.626223 | 0.706402 | 191 | 133 | 14.691 |

XGBoost folds-1-3 mean PR-AUC improvement over Random Forest: `-0.002450`.
Material improvement threshold: `0.01`.
Important operational benefit established: `False`.

**Decision: Random Forest remains the selected development-only production candidate.**

XGBoost remains optional unless selected. Frozen-test labels and metrics were not accessed, and `approved_for_final_test` remains `false`.
