# Colab Compatibility

Sessions 6 and 7 are packaged for a clean Google Colab CPU runtime. The notebook
`notebooks/colab_runner.ipynb` installs only core dependencies from
`requirements.txt`, installs the package, runs the test suite, validates the
fixed training snapshot, converts the latest official export through
`source_to_canonical.py`, runs the prospective external evaluation command,
and generates the final backend reports plus latest unlabeled predictions.

## Required Files

Upload or place these files in the cloned repository:

- `data/raw/df3_availability.csv`
- `data/external/MTA_NYCT_Subway_Elevator_and_Escalator_Availability__Beginning_2015_20260616.csv`
- `outputs/models/final_random_forest.joblib`
- `outputs/metrics/session5_frozen_test_metrics.json`

Do not run `final-evaluate` in Colab. That command is the one-time frozen-test
path and is not part of Session 6 or Session 7 reproducibility.

## CPU Workflow

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m mta_elevator_pipeline.run_pipeline validate
python -m mta_elevator_pipeline.run_pipeline session6
python -m mta_elevator_pipeline.run_pipeline session7
```

Expected Session 6 outputs:

- `data/processed/session6_latest_canonical_availability.csv`
- `outputs/metrics/session6_prospective_evaluation.md`
- `outputs/metrics/session6_prospective_metrics.json`
- `outputs/predictions/session6_prospective_predictions.csv`
- `outputs/reports/final_model_card.md`
- `outputs/reports/final_backend_summary.md`
- `outputs/reports/final_metrics_comparison.csv`
- `outputs/reports/frontend_prediction_schema.md`
- `outputs/predictions/latest_unlabeled_risk_scores.csv`
- `outputs/predictions/latest_unlabeled_risk_scores_metadata.json`

Optional TensorFlow and XGBoost dependencies remain outside the core Colab
workflow.

Current status: the notebook exists and has valid JSON. A live clean Colab CPU
runtime execution is still pending unless run manually.
