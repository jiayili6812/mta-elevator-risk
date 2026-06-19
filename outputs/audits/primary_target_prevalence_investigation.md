# Primary Target Prevalence Investigation

**Status:** passed

## Scope And Contract

- Development-only. February 2025 is used only as the outcome source for January 2025 labels; no February-April 2025 prediction-row target labels are constructed, accessed, or reported.
- Target: `next-month Entrapments > 0 OR next-month Unscheduled Outages >= 2`.
- Target construction uses next-month outcomes and requires an exact consecutive next calendar month.
- Unknown targets remain unlabeled and are excluded from prevalence.
- Unlabeled rows in the development-safe source: 340; converted to negatives: 0.

## Population Decomposition

| Population | Labeled | Positive | Negative | Prevalence | Elevators |
|---|---:|---:|---:|---:|---:|
| `all_non_x_target_eligible_rows` | 14851 | 6340 | 8511 | 0.426907 | 340 |
| `after_newly_introduced_and_equipment_history_rules` | 14851 | 6340 | 8511 | 0.426907 | 340 |
| `current_only_feature_eligibility` | 14851 | 6340 | 8511 | 0.426907 | 340 |
| `short_history_feature_eligibility` | 13850 | 5927 | 7923 | 0.427942 | 324 |
| `full_history_feature_eligibility` | 13202 | 5655 | 7547 | 0.428344 | 324 |
| `development_only_excluding_frozen_period` | 13202 | 5655 | 7547 | 0.428344 | 324 |

Population findings:

- `new_equipment_policy_effect`: No additional development rows are removed after X exclusion because the equipment-level minimum-history rule is null and newly introduced equipment remains eligible.
- `feature_history_effect`: Moving from current-only to full-history eligibility removes 1,649 early-history labeled rows and changes prevalence from 0.426907 to 0.428344.
- `development_date_filter_effect`: The full-history and development-only populations are identical because every prevalence population in this audit excludes February-April 2025 prediction rows by construction.

## Rolling Temporal Validation

| Fold | Train Range | Train Rows / Pos / Prev | Validation Range | Validation Rows / Pos / Prev | Overlap | Frozen Rows |
|---|---|---|---|---|---|---|
| `2024-02_to_2024-04` | 2021-06-01 to 2024-01-01 | 9424 / 4039 / 0.428587 | 2024-02-01 to 2024-04-01 | 918 / 381 / 0.415033 | False | False |
| `2024-05_to_2024-07` | 2021-06-01 to 2024-04-01 | 10342 / 4420 / 0.427383 | 2024-05-01 to 2024-07-01 | 937 / 397 / 0.423693 | False | False |
| `2024-08_to_2024-10` | 2021-06-01 to 2024-07-01 | 11279 / 4817 / 0.427077 | 2024-08-01 to 2024-10-01 | 957 / 385 / 0.402299 | False | False |
| `2024-11_to_2025-01` | 2021-06-01 to 2024-10-01 | 12236 / 5202 / 0.425139 | 2024-11-01 to 2025-01-01 | 966 / 453 / 0.468944 | False | False |

## Earlier Estimate Investigation

- `correct_next_month_target_including_x_rows`: 0.368198 (6340 / 17219).
- `current_month_target_including_x_rows`: 0.366688 (6314 / 17219).
- `non_x_next_month_unscheduled_outages_only`: 0.369201 (5483 / 14851).

Finding by hypothesis:

- `x_suffixed_equipment`: Strongly supported as the main explanation: including X rows with the correct next-month OR target produces 0.368198.
- `different_target_definition`: Possible additional explanation: omitting the entrapment OR clause produces 0.369201 on non-X development rows.
- `different_eligibility_filtering`: Not a material explanation: non-X prevalence is 0.426907 before feature-history filtering and 0.428344 after it.
- `another_calculation_difference`: Current-month targeting while including X rows produces 0.366688, but tests confirm the implemented target does not use current-month outcomes.

The earlier estimate cannot be uniquely attributed because no saved calculation was found. Including X-suffixed zero-outcome records is the strongest quantitative explanation. The implemented 0.428344 prevalence is not caused by target leakage, current-month targeting, or the feature-history eligibility rules.

## Session 1 Assessment

Session 1 can safely be considered complete for target construction, eligibility, prevalence, and temporal-fold validation. The earlier estimate remains a documented provenance limitation rather than an unresolved implementation defect.

Frozen-test labels accessed: **False**.
