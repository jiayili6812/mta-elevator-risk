# Final Development-Only Model Decision

## Decision

Select **Random Forest** as the production candidate:

- Feature set: full history without age
- Parameters: 600 estimators, depth 20, minimum leaf size 15, square-root
  feature sampling, balanced-subsample class weighting, random seed 42
- Calibration: none
- Model-specific threshold: `0.4433219097353501`
- Threshold policy: approximately 70% recall, selected using folds 1-3 only

The selection is recorded but remains unapproved for frozen-test access.
Frozen-test evaluation is deferred to Session 5.

## Final Development Comparison

| Model | Folds 1-3 mean PR-AUC | Four-fold mean PR-AUC | Worst-fold PR-AUC | Fold 4 PR-AUC | Fold 4 precision | Fold 4 recall | Fold 4 FP | Fold 4 FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.667510 | 0.679572 | 0.630940 | 0.715758 | 0.626223 | 0.706402 | 191 | 133 |
| HistGradientBoosting | 0.664433 | 0.676516 | 0.634391 | 0.712766 | 0.646939 | 0.699779 | 173 | 136 |
| Logistic Regression | 0.663685 | 0.673873 | 0.624136 | 0.704437 | 0.623274 | 0.697572 | 191 | 137 |

Random Forest leads selection-fold, four-fold, and fold-4 PR-AUC and has the
fewest fold-4 false negatives. HistGradientBoosting produces 18 fewer false
positives but three more false negatives. Logistic regression is simpler but
trails both tree models after the completed development comparison.

The TCN is rejected as non-material evidence. Its matching-cohort gain is below
the retention rule and its fold-4 recall tradeoff is weaker.

## XGBoost Decision

Skip optional XGBoost. The bounded sklearn boosted-tree evaluation already
establishes that boosting does not surpass Random Forest on development PR-AUC.
Running another optional implementation would extend model search after enough
development evidence exists to make the decision.

## Guardrails

- No frozen-test labels or metrics were accessed.
- No further model, feature, calibration, or threshold tuning is permitted.
- `config/final_model_selection.yaml` remains unapproved.
