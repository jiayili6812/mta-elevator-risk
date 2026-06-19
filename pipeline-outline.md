# Pipeline Outline

## Objective

Build a readable, testable, repeatable pipeline that estimates next-month
elevator failure risk from monthly operational history.

The pipeline must work with newly supplied data, run locally during Codex
development sessions, and run in a clean Google Colab environment.

## Prediction Contract

For an elevator observed through month `T`, predict whether it experiences a
qualifying event in month `T+1`.

Primary failure definition:

```text
next_month_entrapments > 0 OR next_month_unscheduled_outages >= 2
```

Secondary target experiment:

```text
next_month_entrapments > 0 OR next_month_unscheduled_outages >= 1
```

Rows without an observed next month must have an unknown target and must never
be converted to negatives.

## Pipeline Stages

### 1. Configuration

- Load relative input and output paths.
- Load target definition and date boundaries.
- Set random seeds.
- Validate that the frozen-test configuration has not changed.

### 2. Data Loading And Validation

- Validate required columns.
- Parse `Month`.
- Assert unique `Equipment Code` and `Month` pairs.
- Validate availability values are between zero and one.
- Validate counts are nonnegative.
- Validate `Total Outages = Scheduled Outages + Unscheduled Outages`.
- Report missing values and equipment-history lengths.
- Detect non-consecutive monthly records.
- Retain X-suffixed equipment in the audit, but flag it as ineligible for
  supervised training under the current snapshot.

### 3. Target Construction

- Sort by elevator and month.
- Shift outcome columns backward one row within each elevator.
- Verify the next observed row is exactly one calendar month later.
- Mark rows without a known next month as unlabeled.
- Build primary and secondary target variants from configuration.
- Construct target prevalence and modeling datasets only from eligible
  equipment.

### 4. Feature Engineering

Initial candidate features:

- Current-month operational values available at the end of month `T`.
- One-, two-, and three-month lags.
- Three- and six-month rolling statistics using data through month `T`.
- Time since major improvement.
- Month or season indicators.
- Missingness indicators where informative.
- Preserve missing age rows, add `age_missing`, and median-impute age using
  training-fold data only.

Every feature must pass:

> Could this value be known at the end of month T?

Current-month operational features therefore assume prediction generation
occurs after complete month-T reporting is available.

`Time Since Major Improvement` was documented as days, but its semantic unit is
unresolved because the fixed snapshot increments by exactly one per consecutive
month.

Feature sets are compared intentionally rather than through unrestricted search.

### 5. Temporal Splitting

- Frozen test: February 2025 through April 2025.
- Development data: all labeled rows before February 2025.
- Training and validation: rolling temporal validation windows within
  development data.
- Initial rolling validation uses four non-overlapping three-month windows:
  February-April 2024, May-July 2024, August-October 2024, and November
  2024-January 2025, with expanding training data before each window.
- Test labels are unavailable to normal training and experiment commands.
- Development operating thresholds are selected on the first three rolling
  folds and evaluated at that fixed value on the latest development fold.
- Final-test access requires a separate command and is logged.

### 6. Baseline Models

Evaluate in this order:

1. Constant-prevalence and simple rule-based baselines.
2. Regularized logistic regression.
3. Random forest.
4. Histogram gradient boosting.

All models use identical development folds and evaluation functions.

### 7. Focused Experiments

Target experiments:

- Primary `>=2` unscheduled outages or entrapment.
- Secondary `>=1` unscheduled outage or entrapment.

Feature experiments:

- Recent operational features only.
- Operational features plus asset age.
- Short versus longer rolling histories.

Avoid large combinatorial searches. Record each experiment hypothesis before
running it.

### 8. Baseline Readiness Gate

A tabular baseline is ready for the TCN challenger when:

- Target, leakage, split, and frozen-test guardrail tests pass.
- It consistently beats naive baselines across temporal validation windows.
- Mean development PR-AUC is at least approximately `0.45`, or the results
  clearly establish that the available data cannot meet that provisional goal.
- Performance is not dependent on one unusually strong validation window.
- Calibration and operational threshold behavior have been reviewed.

The approximate positive prevalence is expected to be `0.35-0.38`; therefore,
a random ranking has expected PR-AUC near that range.

### 9. TCN Challenger

Run only after the readiness gate.

- Compare 6-, 12-, and 24-month sequences only if sufficient history remains.
- Use the same development periods as tabular models.
- Evaluate multiple seeds.
- Keep the TCN only if it improves validation PR-AUC by approximately
  `0.02-0.03`, improves an important operational tradeoff, and remains stable.
- TCN dependencies and GPU use remain optional.

### 10. Final Model Selection

Use development validation only to choose:

- Target variant.
- Features.
- Model family and hyperparameters.
- Ensemble weights, if any.
- Decision threshold.

The frozen test is accessed once after these decisions are recorded.

### 11. Final Evaluation And Artifacts

Export:

- Frozen-test metrics.
- Predictions by elevator and month.
- Chosen threshold.
- Fitted preprocessing and model artifact.
- Configuration snapshot.
- Data-validation report.
- Model card and limitations.

### 12. Colab Verification

- Install core dependencies from `requirements.txt`.
- Run all tests.
- Run the primary pipeline on CPU.
- Run optional TCN work only after installing `requirements-deep.txt`.
