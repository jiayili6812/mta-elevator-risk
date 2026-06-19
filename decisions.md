# Modeling Decisions

This file records agreed decisions. Change a locked decision only after
documenting the reason and expected methodological effect.

## Locked

| Decision | Value |
|---|---|
| Scope | Backend predictive pipeline only |
| Clustering | Excluded |
| Frontend and map generation | Excluded |
| Prediction horizon | Next calendar month |
| Primary target | Entrapment `> 0` or unscheduled outages `>= 2` |
| Secondary target | Entrapment `> 0` or unscheduled outages `>= 1` |
| Frozen test period | 2025-02-01 through 2025-04-01 |
| Primary metric | PR-AUC |
| Supporting metrics | Precision, recall, false negatives, ROC-AUC, calibration |
| Primary development environment | Codex sessions |
| Final execution environment | Google Colab |
| Core runtime | Clean Colab CPU |
| Optional deep learning | TCN challenger after tabular readiness gate |
| Development dataset | Preserve the current cleaned snapshot; do not replace it mid-development |
| Newly introduced elevators | Include, subject to feature-history availability |
| Unscheduled outage unit | Each count represents a distinct incident |
| Time Since Major Improvement unit | Unresolved: documented as days, but the fixed snapshot increments by exactly 1 per consecutive month |
| X-suffixed equipment | Retain in audit; exclude from supervised training for the current snapshot |
| Missing Time Since Major Improvement | Preserve rows; median-impute within model training and include an `age_missing` indicator |
| Current-only feature history | 1 consecutive observed month |
| Short-history feature history | 4 consecutive observed months for lags through month 3 and a 3-month rolling window |
| Full-history feature history | 6 consecutive observed months for lags through month 3 and 3/6-month rolling windows |
| Rolling validation folds | Four non-overlapping 3-month validation windows with expanding prior-month training data |
| Development threshold evaluation | Select on the first three rolling folds and evaluate the fixed threshold on the latest development fold |

## Provisional Goals

- Minimum-useful development PR-AUC: approximately `0.45`.
- Strong development PR-AUC: approximately `0.55`.
- Recall goal at chosen validation threshold: at least approximately `0.70`.
- Precision goal at chosen validation threshold: at least approximately `0.45`.

These are planning targets, not claims or hard requirements. Reassess after
running naive and logistic-regression baselines.

## Frozen-Test Policy

- Do not use frozen-test labels for feature selection, target selection,
  hyperparameter tuning, model selection, ensemble selection, or threshold
  selection.
- Normal commands must not return frozen-test metrics.
- Frozen-test access requires explicit acknowledgment and creates an audit log.
- Frozen-test access also requires a completed and approved
  `config/final_model_selection.yaml`.
- Record all final modeling decisions before frozen-test evaluation.

## Open Decisions

- Final operational threshold and false-negative cost.
- Whether a TCN provides enough incremental value to retain.

## Real-Data Decisions

- Equipment eligibility excludes X-suffixed records but does not globally
  exclude newly introduced elevators.
- Feature-set eligibility is row-level. A newly introduced elevator becomes
  eligible once it has the selected feature set's required consecutive history.
- The selected initial full-history feature set requires six consecutive
  observed months.
- The 53 missing non-X `Time Since Major Improvement` values are all on the
  affected elevator's first observed record. Excluding them would selectively
  remove new-equipment observations, so the pipeline preserves them, adds an
  `age_missing` indicator, and median-imputes age inside each model-training
  fold.
- The four rolling validation windows are February-April 2024, May-July 2024,
  August-October 2024, and November 2024-January 2025.
- The earlier approximate `0.367` primary prevalence is superseded by the
  audited development-only full-history prevalence of `0.428344`. No saved
  earlier calculation exists; including X-suffixed zero-outcome records with
  the correct target yields `0.368198` and is the strongest explanation.
- Session 2 controlled logistic-regression comparisons recommend carrying
  short-history and full-history feature sets, each with and without asset age,
  into tree-based modeling. Current-only variants are not recommended. Full
  history with age led at mean development PR-AUC `0.674565`, but its advantage
  over full history without age was only `0.001267`.
- Session 3 confirms that imputation, scaling, model fitting, and other learned
  preprocessing are fit separately on training rows inside each fold. Matching
  eligible cohorts are used for model and baseline comparisons.
- Current-month operational features assume predictions are generated only
  after complete month-T reporting is available.
- Threshold-independent metrics are reported across all development folds.
  Threshold-dependent metrics use a threshold selected on the first three
  folds and fixed before evaluation on the November 2024-January 2025 fold.
- The August-October 2024 fold remains weaker, but performance rebounds in the
  November 2024-January 2025 fold; there is no monotonic late-period decline.
- All compared models improve Brier score over the matching fold-specific
  constant-prevalence baseline.
- `Time Since Major Improvement` is not behaving like elapsed days in the fixed
  snapshot: all `14,464` observed consecutive-month transitions increment by
  exactly `1`, with no apparent resets. Treat its semantic unit as unresolved
  and do not prefer age-bearing variants until provenance is resolved.
- Missing-age validation rows are absent from the full-history cohort and rare
  in the current-history diagnostic, so observed-versus-imputed age performance
  is not estimable with useful precision.
- Grouped-by-equipment validation is retained only as a secondary diagnostic.
  It does not replace rolling temporal validation and does not measure
  zero-history performance for brand-new elevators.
- Controlled Session 3 tree searches found no material PR-AUC improvement.
  Full-history random forest with age ranked first at mean PR-AUC `0.678555`,
  only about `0.0040` above full-history logistic regression with age, while
  adding complexity. Histogram gradient boosting did not improve PR-AUC.
- Carry full-history-without-age logistic regression and random forest into
  focused Session 4 work. Age-bearing models remain diagnostic candidates only
  until the age-field semantics are resolved.
- Session 4 focused tuning selects full-history-without-age logistic regression
  with an L1 penalty, `C=0.1`, and the `liblinear` solver. Its four-fold mean
  PR-AUC is `0.673873`; the latest-fold PR-AUC is `0.704437`.
- Session 4 focused tuning selects a random forest with `600` estimators,
  maximum depth `20`, minimum leaf size `15`, and square-root feature sampling.
  Its four-fold mean PR-AUC is `0.679572`; the latest-fold PR-AUC is `0.715758`.
- Random Forest's mean PR-AUC advantage over tuned logistic regression is
  `0.005699`, below the pre-declared `0.01` materiality rule for accepting its
  added complexity. Across five seeds, forest mean PR-AUC is `0.680911` with
  standard deviation `0.000989`; the small advantage survives seed variation
  but remains non-material. Propose logistic regression as the final tabular
  model, pending review.
- Use the earlier-fold-selected approximately-70%-recall policy as the proposed
  operating policy. For tuned logistic regression the threshold is `0.446530`;
  on the latest development fold it yields precision `0.623274`, recall
  `0.697572`, `191` false positives, and `137` false negatives.
- A three-to-one false-negative-versus-false-positive cost is retained only as
  a documented high-recall alternative. It raises latest-fold logistic recall
  to `0.922737` but creates `357` false positives.
- Calibration selection must preserve ranking performance within `0.005`
  PR-AUC of the uncalibrated model and improve mean Brier score by at least
  `0.005` to justify added final-fit and threshold complexity. Isotonic
  calibration is rejected because it materially reduces mean logistic PR-AUC.
  Platt calibration preserves PR-AUC and improves mean Brier score by about
  `0.0033`, but does not clear the materiality rule. Propose no calibration.
- The August-October 2024 weakness is not driven by one month or a small set of
  elevators. Its prevalence is `0.402299` versus `0.436370` in the other
  validation folds, feature-distribution shifts are modest, missingness is not
  elevated, and the ten highest-error elevators account for less than `9%` of
  errors. Treat it as a broad temporal weakness and do not tune specifically
  to it.
- The tabular-readiness gate passes. The optional TCN challenger is rejected
  for Session 4 because TensorFlow is not installed in the current optional
  dependency environment; no unverifiable deep-learning result will be used
  for model selection.
- A focused development-only boosted-tree comparison evaluates twelve
  intentional `HistGradientBoostingClassifier` configurations on the
  full-history-without-age cohort. Parameters and the approximately-70%-recall
  threshold are selected using folds 1-3 only and evaluated unchanged on fold
  4.
- The selected histogram-gradient-boosting configuration uses `500`
  iterations, learning rate `0.03`, `7` maximum leaf nodes, minimum leaf size
  `40`, L2 regularization `5.0`, and `255` bins. Its folds-1-3 mean PR-AUC is
  `0.664433`, four-fold mean PR-AUC is `0.676516`, worst-fold PR-AUC is
  `0.634391`, and fold-4 PR-AUC is `0.712766`.
- At its folds-1-3-selected threshold `0.394978`, histogram gradient boosting
  achieves fold-4 precision `0.646939`, recall `0.699779`, `173` false
  positives, and `136` false negatives. This is a potentially useful
  precision/false-positive tradeoff compared with the other tabular models.
- XGBoost is not installed in the current environment, so the optional
  external boosted-tree challenger is skipped without adding a new core
  dependency. No final model decision is made from the boosted-tree
  comparison; the comparison remains under review.
- Session 4.2 evaluates an optional compact causal TCN using only the seven raw
  current-month operational channels, consecutive within-equipment sequences,
  and fold-local median imputation plus robust scaling. Sequence lengths of 6,
  12, and 24 months and two limited architectures are selected using folds 1-3
  only; fold 4 remains excluded from sequence-length, architecture, and
  threshold selection.
- The selected development-only TCN uses 12-month sequences, 32 filters,
  dropout `0.2`, and dilations `[1, 2, 4]`. On the exact matching cohort, its
  folds-1-3 mean PR-AUC is `0.680844`, only `0.011134` above the matching
  Random Forest, below the predeclared approximately `0.02-0.03` materiality
  rule. Its seed-42 fold-4 recall is `0.602740`, versus `0.707763` for the
  matching Random Forest, so it does not establish an important operational
  improvement at comparable recall.
- Across TCN seeds `21`, `42`, and `84`, four-fold mean PR-AUC ranges from
  `0.681210` to `0.687796`. This variation, sequence-cohort loss, added
  TensorFlow dependency, and non-material matching-cohort gain do not justify
  retaining the TCN challenger. This is methodological evidence only and does
  not approve or select a final production model.
- Final development-only model decision: select the tuned
  full-history-without-age Random Forest as the production candidate. This
  supersedes the earlier provisional logistic-regression recommendation after
  the completed boosted-tree and TCN comparisons.
- The selected Random Forest uses `600` estimators, maximum depth `20`, minimum
  leaf size `15`, square-root feature sampling, balanced-subsample class
  weighting, and random seed `42`. Use no calibration.
- Lock the Random Forest model-specific approximately-70%-recall threshold at
  `0.4433219097353501`. It was selected using folds 1-3 only and achieves
  fold-4 PR-AUC `0.715758`, precision `0.626223`, recall `0.706402`, `191`
  false positives, and `133` false negatives.
- Random Forest is selected because it leads folds-1-3 mean PR-AUC, four-fold
  mean PR-AUC, and fold-4 PR-AUC and has the fewest fold-4 false negatives.
  Histogram gradient boosting's cleaner intervention list is retained as an
  important tradeoff finding, but its 18 fewer false positives come with three
  additional false negatives and it does not lead PR-AUC.
- Skip optional XGBoost. The bounded sklearn boosted-tree comparison already
  establishes that boosting does not surpass Random Forest on development
  PR-AUC, and extending optional model search is no longer justified.
- Session 4.3 supersedes the earlier decision to skip optional XGBoost and
  evaluates eight intentional XGBoost `2.1.4` configurations using the exact
  full-history-without-age cohort. XGBoost remains isolated in
  `requirements-xgboost.txt` and executes successfully on CPU.
- The selected XGBoost configuration uses depth `2`, learning rate `0.05`,
  `700` estimators, minimum child weight `10`, full row and column sampling,
  no L1 regularization, and L2 regularization `5.0`.
- Selected XGBoost achieves folds-1-3 mean PR-AUC `0.665060`, four-fold mean
  PR-AUC `0.676464`, worst-fold PR-AUC `0.630528`, and fold-4 PR-AUC
  `0.710675`. At its own folds-1-3-selected approximately-70%-recall threshold
  `0.389102`, fold-4 precision is `0.625490`, recall is `0.704194`, with `191`
  false positives and `134` false negatives.
- XGBoost trails the matching Random Forest by `0.002450` folds-1-3 mean
  PR-AUC, has lower fold-4 PR-AUC, and produces one additional false negative
  with the same false-positive count. It establishes neither a material
  ranking improvement nor an important operational benefit.
- Random Forest remains the selected production candidate after Session 4.3.
  Do not add XGBoost to core `requirements.txt`; keep
  `config/final_model_selection.yaml` unapproved and unchanged.
- The TCN remains rejected as non-material evidence. Do not rerun TCN, tabular
  tuning, feature selection, calibration selection, or threshold selection.
  Frozen-test evaluation was reserved for Session 5.
- `config/final_model_selection.yaml` was populated with the development-only
  selection and remained unapproved until the successful Session 5 preflight.
- Session 5 preflight passed before authorization. The complete suite passed
  (`42` tests), the frozen access log was absent, approval was `false`, the
  locked selection matched its development evidence, and the exact final-fit
  plan used `13,202` eligible development rows ending January 2025 with no
  learned preprocessing from frozen rows.
- The frozen February-April 2025 labels were accessed exactly once through the
  configured acknowledgment and audit log after fitting the locked pipeline
  once. No model, feature, target, preprocessing, calibration, eligibility, or
  threshold decision was changed after access.
- Final frozen-test performance on `974` rows across `326` elevators is:
  prevalence `0.399384`, PR-AUC `0.648004`, ROC-AUC `0.716806`, Brier score
  `0.209976`, precision `0.526316`, recall `0.694087`, `243` false positives,
  and `119` false negatives at threshold `0.4433219097353501`.
- Frozen-test PR-AUC is below the four-fold development mean (`0.679572`) and
  latest development fold (`0.715758`). Precision is also below the latest
  development fold (`0.626223`), while recall remains close to the locked
  approximately-70%-recall policy. April 2025 is the weakest frozen month by
  PR-AUC (`0.601887`) and calibration ECE (`0.116008`).
- Treat the Session 5 result as the final untouched-test estimate. Do not
  retune or rerun the final evaluation; future assessment must use a separately
  defined prospective dataset.
- Session 5.5 reconstructs the fixed `df3_availability.csv` lineage using the
  local July 2025 official export. The canonical snapshot is exactly the
  official 22-column availability export filtered to elevators and
  `Month >= 2021-01-01`, then sorted by month descending with stable source
  order within month. No aggregation, renaming, availability scaling, or
  missing-value imputation is part of source-to-canonical conversion.
- The regenerated canonical DataFrame matches the fixed training snapshot
  exactly (`18,780` rows, `22` columns, SHA-256
  `ba428c133e42ec3f9d3bddaff01d434d8bad2631587e602f3509a22dfe47c1a2`).
- Add `src/mta_elevator_pipeline/source_to_canonical.py` as the tested
  source-to-canonical path. It must write regenerated/prospective canonical
  files separately and must not overwrite `data/raw/df3_availability.csv`.
- Session 6 may proceed only through this tested transformation path and must
  not rerun the frozen-test evaluation.

## Dataset-Version Policy

- The current cleaned snapshot is the fixed development dataset.
- The Session 5.5 lineage audit supersedes the earlier assumption that the
  raw-to-`df3` cleaning history was unknown for the local July 2025 source.
  `df2` remains undocumented because no local `df2` file or transformation
  record was found.
- Do not replace the development dataset with a newly downloaded version during
  feature or model selection.
- After the model, features, and threshold are locked, use newly available
  months as a prospective external evaluation dataset.
- After prospective evaluation is reported, a separate deployment model may be
  retrained on all eligible historical data.

## X-Suffixed Equipment Policy

In the current snapshot, X-suffixed equipment has structurally incomplete
reporting: all rows have zero unscheduled outages and zero entrapments, and all
`Time Since Major Improvement` values are missing. These rows cannot provide
trustworthy supervised target labels.

- Keep X-suffixed rows in source-data audits.
- Exclude them from target prevalence, model fitting, validation, and final
  test metrics for this snapshot.
- Re-evaluate inclusion using a future source version only if the relevant
  fields are populated and their reporting meaning is confirmed.
