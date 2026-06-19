from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from mta_elevator_pipeline.config import load_config, load_yaml, project_path
from mta_elevator_pipeline.run_pipeline import final_evaluate_command
from mta_elevator_pipeline.session5 import LOCKED_SELECTION, build_locked_model


def test_final_evaluation_requires_approved_selection_record(tmp_path):
    config = deepcopy(load_config())
    selection = load_yaml(project_path(config["guardrails"]["final_selection_record"]))
    selection["approved_for_final_test"] = False
    path = tmp_path / "unapproved_selection.yaml"
    path.write_text(yaml.safe_dump(selection), encoding="utf-8")
    config["guardrails"]["final_selection_record"] = str(path)
    with pytest.raises(PermissionError, match="not approved"):
        final_evaluate_command(config, acknowledgment="")


def test_locked_random_forest_preserves_all_selected_parameters():
    selection = {"selected_parameters": LOCKED_SELECTION["selected_parameters"]}
    model = build_locked_model(["Total Outages"], selection).named_steps["model"]
    for name, expected in LOCKED_SELECTION["selected_parameters"].items():
        assert model.get_params()[name] == expected
