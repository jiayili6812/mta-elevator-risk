# Notebooks

This directory contains notebook entry points for running or presenting the
tested pipeline. Notebooks should call package code in
`src/mta_elevator_pipeline/` rather than duplicate pipeline logic.

## Colab Runner

`colab_runner.ipynb` is the supported clean-Colab reproducibility notebook. It
clones the GitHub repo, downloads the locked model artifact from GitHub
Releases, installs core dependencies, runs tests, validates the fixed data
snapshot, runs Session 6 prospective external evaluation, and runs Session 7
final report and latest-score generation.

It does not run `final-evaluate`.

See `../COLAB_COMPATIBILITY.md` for the full Colab workflow, included data
files, model artifact requirements, and expected outputs.