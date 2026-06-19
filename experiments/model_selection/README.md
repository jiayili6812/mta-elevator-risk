# Tabular Model Selection

Initial candidates:

- Constant and rule-based baselines.
- Regularized logistic regression.
- Random forest.
- Histogram gradient boosting.

Fine-tuning limits:

- Logistic regression: approximately 5-10 intentional configurations.
- Random forest: approximately 20-40 configurations.
- Gradient boosting: approximately 20-50 configurations.

Spend more effort on target quality, temporal validity, calibration, and
threshold selection than on large hyperparameter searches.

