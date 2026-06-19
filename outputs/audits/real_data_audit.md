# Standalone Pipeline Real-Data Audit

## Result

- Validation status: **passed**
- Source rows: 18780
- Equipment: 391
- Date range: 2021-01-01 through 2025-05-01
- Nonconsecutive transitions: 0

## Eligibility

- X-suffixed rows retained in audit: 2566
- X-suffixed equipment excluded from modeling: 50
- Eligible non-X equipment: 341
- Newly introduced eligible equipment: 53

## Feature History

- `current_only`: requires 1 consecutive month(s); 16214 rows and 341 equipment qualify.
- `short_history`: requires 4 consecutive month(s); 15191 rows and 340 equipment qualify.
- `full_history`: requires 6 consecutive month(s); 14511 rows and 335 equipment qualify.

## Missing Age

- Missing non-X rows: 53
- Affected equipment: 53
- All occur on first observed record: True
- Strategy: median_imputation_with_missingness_indicator

## Rolling Validation

- `2024-02_to_2024-04`: train rows 9424; validation rows 918.
- `2024-05_to_2024-07`: train rows 10342; validation rows 937.
- `2024-08_to_2024-10`: train rows 11279; validation rows 957.
- `2024-11_to_2025-01`: train rows 12236; validation rows 966.

## Frozen-Test Guardrail

- Target-eligible frozen rows before feature-history filtering: 1022
- Full-feature frozen rows: 974
- Counts match locked expectations: True
- Frozen labels exposed by audit command: False

## Target Reports

- `primary` development-only prevalence: 0.428344 (5655 / 13202).
- `secondary` development-only prevalence: 0.619982 (8185 / 13202).
