# TCN Challenger

Do not begin until the tabular baseline-readiness gate in
`pipeline-outline.md` is satisfied.

The experiment should:

- Use optional dependencies from `requirements-deep.txt`.
- Compare sequence lengths only where sufficient history remains.
- Use development temporal validation periods identical to tabular models.
- Run multiple random seeds.
- Compare PR-AUC, precision/recall tradeoffs, calibration, and stability.

Retain the TCN only if its incremental value justifies its additional
complexity. A PR-AUC improvement of approximately `0.02-0.03` is the initial
materiality threshold.

