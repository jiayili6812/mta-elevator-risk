# Session 5.5 Data Lineage Reconstruction

Status: **passed**

Scope: local source-to-canonical lineage only. No prospective evaluation,
frozen-test evaluation, retraining, or latest-data download was performed.

## Local Files Inventoried

| File | Role | Rows | Columns |
|---|---:|---:|---:|
| `data/raw/df1_station_master.csv` | Station master / lookup table with GTFS station identifiers, ADA fields, borough, line, latitude, longitude, and georeference. Not used by the locked training pipeline. | 496 | 19 |
| `data/raw/MTA_NYCT_Subway_Elevator_and_Escalator_Availability_accessed_July_2025 copy.csv` | Official availability export accessed in July 2025. Contains elevator and escalator monthly equipment records from January 2015 through May 2025. | 72,797 | 22 |
| `data/raw/df3_availability.csv` | Fixed canonical cleaned snapshot used for all Sessions 1-5 training and evaluation. | 18,780 | 22 |

No local `df2` file was found. No notebooks or scripts defining `df1`,
`df2`, or `df3` transformations were found in the repository.

## Dataset Meanings

- `df1`: station master data. It is a station-level lookup source and is not a
  row-level elevator availability table.
- `df2`: not present locally. Its content and role remain unresolved unless
  the project owner has an external notebook or intermediate file.
- Official July 2025 availability export: monthly equipment-level official
  source containing both elevators and escalators, already at one
  equipment-month per row with no duplicate `Equipment Code` and `Month` pairs.
- `df3_availability.csv`: canonical elevator-only subset used by the modeling
  pipeline.

## Reconstructed Transformation

The fixed `df3_availability.csv` is exactly regenerated from the local July
2025 official export by:

1. Keeping the official 22 columns unchanged.
2. Filtering to `Equipment Type == "Elevator"`.
3. Filtering to `Month >= 2021-01-01`.
4. Sorting by parsed `Month` descending while preserving the official source
   row order within each month.

No aggregation is required. No columns are renamed. No availability scale
conversion is required because availability fields are already proportions in
`[0, 1]`. No missing values are imputed during canonicalization.

X-suffixed equipment is retained in the canonical snapshot. It remains excluded
later from supervised modeling eligibility under the existing Session 1-5
policy because target and age reporting are structurally incomplete for those
rows.

## Schema Comparison

The official July 2025 availability export and canonical `df3` snapshot have
identical 22 columns in identical order:

- `Month`
- `Borough`
- `Equipment Type`
- `Equipment Code`
- `Total Outages`
- `Scheduled Outages`
- `Unscheduled Outages`
- `Entrapments`
- `Time Since Major Improvement`
- `AM Peak Availability`
- `AM Peak Hours Available`
- `AM Peak Total Hours`
- `PM Peak Availability`
- `PM Peak Hours Available`
- `PM Peak Total Hours`
- `24-Hour Availability`
- `24-Hour Hours Available`
- `24-Hour Total Hours`
- `Station Name`
- `Station MRN`
- `Station Complex Name`
- `Station Complex MRN`

The detailed machine-readable comparison is saved at
`outputs/audits/session5_5_schema_comparison.json`.

## Regeneration Result

Using the reconstructed transformation, the regenerated canonical DataFrame
matches the fixed training snapshot exactly:

- Regenerated rows: `18,780`
- Regenerated columns: `22`
- Fixed snapshot rows: `18,780`
- Fixed snapshot SHA-256:
  `ba428c133e42ec3f9d3bddaff01d434d8bad2631587e602f3509a22dfe47c1a2`
- Exact DataFrame match: `true`

## Implemented Guardrails

Added `src/mta_elevator_pipeline/source_to_canonical.py` with:

- canonical column contract;
- deterministic official-source-to-canonical transformation;
- validation through the existing availability validator;
- refusal to write to `data/raw/df3_availability.csv`.

Added tests covering:

- canonical schema preservation;
- elevator and month filtering;
- rejection of missing official columns;
- refusal to overwrite the fixed training snapshot;
- refusal to overwrite any existing output unless explicitly allowed.

## Required Versus Exploratory Steps

Required for canonicalization:

- preserve the official 22-column schema;
- parse `Month`;
- filter to elevators;
- filter to January 2021 and later;
- preserve X-suffixed rows for audit compatibility;
- validate unique equipment-month rows, outage identity, nonnegative counts,
  and availability ranges.

Exploratory or downstream-only, not part of canonicalization:

- excluding X-suffixed equipment from supervised modeling;
- target construction;
- feature-history filtering;
- lag and rolling feature generation;
- missing-age imputation inside model folds;
- station master joins;
- model evaluation.

## Remaining Gaps

- `df2` is still unknown because no local `df2` file or transformation record
  exists.
- The formal business meaning of X-suffixed equipment still needs owner or MTA
  confirmation.
- `Time Since Major Improvement` is still semantically unresolved: the source
  describes it as days, but the fixed snapshot behaves like a monthly
  incrementing counter.
- This session proves the July 2025 official export can reproduce the fixed
  snapshot. Future official downloads should still be schema-checked before
  use.

## Session 6 Gate

Session 6 can proceed to source-to-canonical preparation safely, provided it
uses the tested transformation and writes regenerated or prospective canonical
files separately from `data/raw/df3_availability.csv`.

Session 6 must not rerun the frozen-test evaluation and must not use the
February-April 2025 frozen-test result for any new tuning decision.
