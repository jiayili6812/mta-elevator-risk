# Feature-Set Experiments

Initial controlled comparisons:

1. Current-month operational values only.
2. Current values plus 1-3 month lags.
3. Current values, lags, and 3/6 month rolling summaries.
4. Operational features without asset age.
5. Operational features with asset age.

Reject any feature unavailable at the end of prediction month `T`.

## Session 2 Controlled Results

Command:

```bash
python -m mta_elevator_pipeline.run_pipeline session2
```

The command compares all six planned logistic-regression feature variants on
the four locked development-only rolling folds. It also evaluates constant
prevalence, previous-month failure persistence, and a simple current-month
outage rule on each matching feature-history cohort.

The operating threshold is selected once from pooled development
rolling-validation predictions by maximizing precision subject to recall of at
least `0.70`, with an F1 fallback. This is a development-selection estimate,
not an unbiased final-performance estimate.

| Feature variant | Mean PR-AUC | PR-AUC std | Mean ROC-AUC | Mean Brier |
|---|---:|---:|---:|---:|
| Current only, without age | 0.6460 | 0.0318 | 0.7094 | 0.2179 |
| Current only, with age | 0.6493 | 0.0310 | 0.7141 | 0.2173 |
| Short history, without age | 0.6678 | 0.0292 | 0.7308 | 0.2106 |
| Short history, with age | 0.6696 | 0.0294 | 0.7333 | 0.2100 |
| Full history, without age | 0.6733 | 0.0314 | 0.7354 | 0.2091 |
| Full history, with age | **0.6746** | 0.0312 | **0.7375** | **0.2086** |

For full history with age, fold PR-AUC values were `0.6737`, `0.6934`,
`0.6245`, and `0.7066`. At the selected threshold `0.4478`, pooled precision
was `0.6008`, recall was `0.7024`, false positives were `754`, and false
negatives were `481`.

Matching-cohort mean PR-AUC baselines for full history were:

- Constant prevalence: `0.4275`.
- Previous-month failure persistence: `0.5392`.
- Simple outage rule: `0.5110`.

Recommendation: carry short-history and full-history feature sets, both with
and without age, into controlled tree-based modeling. Do not carry current-only
variants. Retain the age comparison because its lift is small and may not
persist for nonlinear models.

Machine-readable metrics and experiment records:

- `outputs/metrics/session2_development_validation.json`
- `experiments/feature_sets/session2_experiment_record.json`
