# Colab Compatibility

Sessions 6 and 7 are packaged for a clean Google Colab CPU runtime through `notebooks/colab_runner.ipynb`.

Session 6 is the prospective external evaluation: it applies the locked Session 5 model to later target-known MTA months after the frozen-test period. Session 7 generates final backend reports and latest unlabeled risk scores for the most recent month in the external export.

The notebook installs only core dependencies from `requirements.txt`, installs the package, runs the test suite, validates the fixed training snapshot, converts the committed reference official export through `source_to_canonical.py`, runs the prospective external evaluation command, and generates the final backend
reports plus latest unlabeled predictions.

## Included Files

The repository already includes the files needed for the standard Colab run:

- `data/raw/df3_availability.csv`
- `data/raw/df1_station_master.csv`
- `data/external/MTA_NYCT_Subway_Elevator_and_Escalator_Availability__Beginning_2015.csv`
- `outputs/metrics/session5_frozen_test_metrics.json`

The included external MTA export was downloaded on June 16, 2026. It contains 80,316 rows of monthly elevator/escalator availability data from January 1, 2015 through April 1, 2026.

With the included example data, the locked historical workflow uses February 1, 2025 through April 1, 2025 as the frozen final test period. Session 6 then evaluates later target-known prediction months from June 1, 2025 through March 1, 2026. Session 7 uses the latest available feature month, April 1, 2026, to generate unlabeled risk scores for the next target month, May 1, 2026.

## Model Artifact

The trained model artifact is not stored directly in Git:

- `outputs/models/final_random_forest.joblib`

The Colab notebook downloads it from this project's GitHub Release assets into:

- `outputs/models/final_random_forest.joblib`

The release asset must exist before the notebook can run end to end.

## Optional Newer MTA Export

To run the workflow with a newer official MTA export:

1. Visit the MTA/data.ny.gov source page:
   <https://data.ny.gov/Transportation/MTA-NYCT-Subway-Elevator-and-Escalator-Availabilit/rc78-7x78>
2. Use the export/download button to download the CSV.
3. Rename the downloaded file to remove the appended download date so the path is exactly:
   `data/external/MTA_NYCT_Subway_Elevator_and_Escalator_Availability__Beginning_2015.csv`
4. Replace the existing file at that path before running Session 6 and Session 7.

The pipeline currently expects that exact external CSV path.

## CPU Workflow

The notebook runs the equivalent of:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m mta_elevator_pipeline.run_pipeline validate
python -m mta_elevator_pipeline.run_pipeline session6
python -m mta_elevator_pipeline.run_pipeline session7
```

Do not run `final-evaluate` in Colab. That command is the one-time frozen-test
path and is not part of Session 6 or Session 7 reproducibility.

## Expected Outputs

Expected Session 6 outputs:

- `data/processed/session6_latest_canonical_availability.csv`
- `outputs/metrics/session6_prospective_evaluation.md`
- `outputs/metrics/session6_prospective_metrics.json`
- `outputs/predictions/session6_prospective_predictions.csv`

Expected Session 7 outputs:

- `outputs/reports/final_model_card.md`
- `outputs/reports/final_backend_summary.md`
- `outputs/reports/final_metrics_comparison.csv`
- `outputs/reports/frontend_prediction_schema.md`
- `outputs/predictions/latest_unlabeled_risk_scores.csv`
- `outputs/predictions/latest_unlabeled_risk_scores_metadata.json`

Optional TensorFlow and XGBoost dependencies remain outside the core Colab workflow.

Current status: `notebooks/colab_runner.ipynb` has been run successfully in a clean Google Colab CPU runtime after downloading the locked model artifact from the GitHub Release asset.
