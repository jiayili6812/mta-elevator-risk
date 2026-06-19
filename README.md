# MTA Elevator Next-Month Failure Pipeline

Standalone backend project for building and evaluating a reproducible next-month
elevator failure prediction pipeline. The project is designed for development in
Codex sessions and final execution in a clean Google Colab runtime.

Clustering, map generation, and frontend presentation are intentionally out of
scope.

## Locked Modeling Contract

- Prediction unit: one elevator-month.
- Prediction question: using information available through month `T`, estimate
  whether an elevator will experience a qualifying failure in month `T+1`.
- Primary target: next-month `Entrapments > 0` or
  `Unscheduled Outages >= 2`.
- Secondary target experiment: next-month `Entrapments > 0` or
  `Unscheduled Outages >= 1`.
- Frozen final test period: February 1, 2025 through April 1, 2025.
- Primary model-selection metric: PR-AUC.
- Supporting metrics: precision, recall, false negatives, ROC-AUC, and
  calibration.
- Primary candidates: naive baseline, logistic regression, random forest, and
  gradient-boosted trees.
- Deep-learning challenger: one TCN study after a trustworthy tabular baseline
  is established.

See [pipeline-outline.md](pipeline-outline.md) and [decisions.md](decisions.md)
for the complete contract.

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m mta_elevator_pipeline.run_pipeline validate
python -m mta_elevator_pipeline.run_pipeline targets
python -m mta_elevator_pipeline.run_pipeline prevalence-audit
python -m mta_elevator_pipeline.run_pipeline train
python -m mta_elevator_pipeline.run_pipeline session2
python -m mta_elevator_pipeline.run_pipeline session3
python -m mta_elevator_pipeline.run_pipeline session4
python -m pip install -r requirements-deep.txt
python -m mta_elevator_pipeline.run_pipeline session4-2
python -m pip install -r requirements-xgboost.txt
python -m mta_elevator_pipeline.run_pipeline session4-3
python -m mta_elevator_pipeline.run_pipeline boosted-trees
python -m mta_elevator_pipeline.run_pipeline session6
python -m mta_elevator_pipeline.run_pipeline session7
```

`validate` saves formal real-data audit reports under `outputs/audits/`.
`targets` saves development-only primary and secondary prevalence reports under
`outputs/metrics/`; it does not expose frozen-test labels.

Frozen-test evaluation is deliberately separate:

```bash
python -m mta_elevator_pipeline.run_pipeline final-evaluate \
  --acknowledge I_UNDERSTAND_THIS_ACCESSES_THE_FROZEN_TEST_SET
```

Do not run final evaluation during model or feature development.

Session 6 prospective evaluation uses the saved Session 5 model artifact and a
separate latest official export under `data/external/`. It writes canonical
latest data to `data/processed/` and does not rerun frozen-test evaluation.

Session 7 final backend reporting uses the unchanged Session 5 model artifact
to generate final reports and latest unlabeled April 2026 risk scores for May
2026. It does not evaluate those scores because May 2026 outcomes are
unavailable.

## Data Required

Place the required source files in `data/raw/`. Exact requirements and
provenance fields are documented in [data/README.md](data/README.md).

## Colab Requirement

All primary pipeline code and core dependencies must run in a clean Google
Colab CPU runtime. GPU use is optional only for the TCN challenger.

Use [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) or
[COLAB_COMPATIBILITY.md](COLAB_COMPATIBILITY.md) for the CPU Colab workflow.
The notebook exists and has valid JSON; a live clean Colab CPU runtime run is
still pending unless performed manually.
