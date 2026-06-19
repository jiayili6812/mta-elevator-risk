# Focused Development-Only Boosted-Tree Comparison

Select parameters and approximately-70%-recall threshold using folds 1-3 only; evaluate unchanged on fold 4.

| Model | Folds 1-3 mean PR-AUC | Four-fold mean PR-AUC | Worst-fold PR-AUC | Fold 4 PR-AUC | Fold 4 precision | Fold 4 recall | Fold 4 FP | Fold 4 FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 0.667510 | 0.679572 | 0.630940 | 0.715758 | 0.626223 | 0.706402 | 191 | 133 |
| hist_gradient_boosting | 0.664433 | 0.676516 | 0.634391 | 0.712766 | 0.646939 | 0.699779 | 173 | 136 |
| logistic_regression | 0.663685 | 0.673873 | 0.624136 | 0.704437 | 0.623274 | 0.697572 | 191 | 137 |

External challenger: XGBoost is not installed in the current environment; optional external evaluation was skipped without changing core dependencies.

No final model decision made. Comparison is provided for review.

Frozen-test labels and metrics were not accessed.
