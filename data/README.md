# Required Data Sources

The standalone pipeline must not depend on files outside this repository.

## Required Now

Place this file in `data/raw/`:

### `df3_availability.csv`

Monthly elevator operational records. Required columns:

- `Month`
- `Equipment Code`
- `Total Outages`
- `Scheduled Outages`
- `Unscheduled Outages`
- `Entrapments`
- `Time Since Major Improvement`
- `AM Peak Availability`
- `PM Peak Availability`
- `24-Hour Availability`

Strongly preferred contextual columns:

- `Equipment Type`
- `Borough`
- `Station Name`
- `Station MRN`
- `Station Complex Name`
- `Station Complex MRN`

## Known Source And Provenance

- Source organization: Metropolitan Transportation Authority via data.ny.gov.
- Source page:
  <https://data.ny.gov/Transportation/MTA-NYCT-Subway-Elevator-and-Escalator-Availabilit/rc78-7x78/about_data>
- Local source export:
  `data/raw/MTA_NYCT_Subway_Elevator_and_Escalator_Availability_accessed_July_2025 copy.csv`.
- Session 6 latest official export:
  `data/external/MTA_NYCT_Subway_Elevator_and_Escalator_Availability__Beginning_2015.csv`.
- Session 6 canonical latest file:
  `data/processed/session6_latest_canonical_availability.csv`.
- Session 7 latest unlabeled prediction output:
  `outputs/predictions/latest_unlabeled_risk_scores.csv`.
- The current `df3_availability.csv` is an exact canonical subset of that
  July 2025 official export.
- Reconstructed Session 5.5 lineage: keep the official 22 columns unchanged,
  filter to `Equipment Type == "Elevator"`, filter to
  `Month >= 2021-01-01`, and sort by month descending while preserving source
  order within each month.
- No aggregation, column renaming, availability scale conversion, or
  missing-value imputation is required to regenerate `df3_availability.csv`.
- The June 16, 2026 latest official export stores availability fields as
  percent strings such as `100%`; the canonical converter normalizes those
  fields to proportions in `[0, 1]` for pipeline compatibility.
- `Time Since Major Improvement` is measured in days.
- Unscheduled outage counts represent distinct incidents.
- Newly introduced elevators should remain eligible.

The fixed cleaned development snapshot must still be preserved unchanged. Use
`src/mta_elevator_pipeline/source_to_canonical.py` to create regenerated or
prospective canonical files at separate paths; it refuses to overwrite
`data/raw/df3_availability.csv`.

## Still Needed From The Project Owner

Provide or confirm:

1. Definitions for scheduled outage, entrapment, and each
   availability field.
2. Confirmation that each row represents one complete calendar month.
3. How missing `Time Since Major Improvement` values should be interpreted.
4. What X suffixes formally mean in the source system.
5. Whether future data will preserve the same schema and column names.
6. Whether any external `df2` intermediate existed and what it represented.

Complete `data/provenance-template.yaml` when supplying the source file.

## Development Versus New Data

Use the existing cleaned snapshot for pipeline development and model selection.
Do not switch to newly downloaded data midway, because changing the dataset
would invalidate comparisons and the frozen-test contract.

Once the model and threshold are locked, download the latest official data and
use genuinely later months as a prospective external evaluation. After that
evaluation is recorded, retraining a deployment model on all eligible data is
appropriate.

Session 6 evaluates only target-known prediction months after `2025-05-01`.
The fixed training snapshot remains `data/raw/df3_availability.csv`; never
replace it with the latest processed canonical file.

Session 7 uses April 2026 rows from the latest canonical file to predict May
2026 risk with the locked research model. Those predictions are unlabeled and
must not be evaluated or used to change the locked model because May 2026
outcomes are not present in the current source data.

## Useful Optional Sources

These are not required for the first prediction pipeline but could improve
interpretation or future modeling:

- Equipment installation and major-improvement dates.
- Maintenance ownership or contractor records.
- Elevator manufacturer and model.
- Inspection records and defects.
- Work-order or repair-duration records.
- Ridership or estimated elevator usage.
- Weather data by month.
- Planned capital-work schedules.
- Station accessibility and transfer-complex metadata.

## Not Required

- Station coordinates.
- Subway-line geometries.
- Cluster assignments.
- Map or frontend files.
