# Notebook Policy

Notebooks are optional presentation and diagnostic layers. They must call the
tested functions in `src/mta_elevator_pipeline/` rather than duplicate pipeline
logic.

Planned notebooks:

- `01_data_audit.ipynb`
- `02_model_comparison.ipynb`
- `03_final_results.ipynb`
- `colab_runner.ipynb`

Create these only when the corresponding pipeline stages are stable.

`colab_runner.ipynb` now contains the Session 6 CPU workflow: install core
dependencies, run tests, validate source data, and execute prospective
evaluation without invoking frozen-test evaluation.
