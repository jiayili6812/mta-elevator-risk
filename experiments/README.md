# Experiment Policy

Experiments must answer a stated question and use development data only.

Before running an experiment, record:

- Hypothesis.
- Target definition.
- Feature set.
- Training and validation periods.
- Models and parameter ranges.
- Primary decision criterion.

After running it, record:

- Metrics for every temporal validation period.
- Mean and variation, not only the best fold.
- Calibration and threshold behavior.
- Decision and reason.

Do not access the frozen February-April 2025 labels from experiment code.

## Experiment Order

1. Confirm target and prevalence.
2. Establish naive baselines.
3. Compare logistic regression, random forest, and gradient boosting.
4. Compare a small number of feature sets.
5. Perform focused tuning on promising models.
6. Apply the baseline-readiness gate.
7. Optionally run the TCN challenger.

