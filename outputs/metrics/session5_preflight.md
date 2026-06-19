# Session 5 Frozen-Test Preflight

Status: **passed**

Created at UTC: `2026-06-15T17:18:00.591891+00:00`

Complete test suite: `42 passed in 1.28s`

## Checks

- selected_target_matches_lock: **True**
- selected_model_matches_lock: **True**
- selected_feature_set_matches_lock: **True**
- selected_threshold_matches_lock: **True**
- selected_parameters_matches_lock: **True**
- calibration_matches_lock: **True**
- development_record_exists: **True**
- development_selected_model_matches: **True**
- development_feature_set_matches: **True**
- development_parameters_match: **True**
- development_threshold_matches: **True**
- development_calibration_matches: **True**
- development_scope_excludes_frozen: **True**
- development_guardrail_says_not_accessed: **True**
- development_guardrail_forbids_tuning: **True**
- frozen_period_lock_matches: **True**
- approved_for_final_test_is_false: **True**
- frozen_test_access_log_absent: **True**
- final_outputs_absent: **True**
- complete_test_suite_passed: **True**
- development_rows_precede_february_2025: **True**
- frozen_features_exclude_target: **True**
- preprocessing_is_inside_model_pipeline: **True**
- preprocessing_receives_development_rows_only_during_fit: **True**
- final_fit_contract_is_exactly_once_on_all_development_rows: **True**
- threshold_will_be_used_unchanged: **True**
- no_age_features: **True**
- model_parameters_match_lock: **True**

## Locked Final Fit Plan

- Fit exactly once before frozen-label access.
- Development rows: `13202` across `324` elevators.
- Development period: `2021-06-01` through `2025-01-01`.
- Frozen feature rows confirmed without labels: `974`.
- Feature set: `full_history_without_age` with `42` features and no age fields.
- Threshold: `0.4433219097353501` unchanged.
- Preprocessing: `median imputation inside the fitted sklearn Pipeline` fitted only on development rows.

All preflight checks passed. Approval may now be changed before the one-time final evaluation.
