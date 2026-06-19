from __future__ import annotations

import numpy as np

from mta_elevator_pipeline.session4 import _threshold_candidates, threshold_policy_report


def test_session4_thresholds_use_only_earlier_folds():
    earlier = [
        ("one", np.array([0, 1]), np.array([0.2, 0.8])),
        ("two", np.array([0, 1]), np.array([0.3, 0.7])),
        ("three", np.array([0, 1]), np.array([0.1, 0.9])),
    ]
    first = threshold_policy_report(earlier + [("latest", np.array([0, 1]), np.array([0.1, 0.9]))])
    second = threshold_policy_report(earlier + [("latest", np.array([0, 1]), np.array([0.99, 1.0]))])
    assert [item["selected_threshold"] for item in first["policies"]] == [
        item["selected_threshold"] for item in second["policies"]
    ]
    assert first["latest_fold_excluded_from_threshold_selection"]


def test_session4_threshold_report_includes_required_policies():
    policies = _threshold_candidates(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert {item["policy"] for item in policies} == {
        "approximately_70_percent_recall",
        "approximately_80_percent_recall",
        "best_validation_f1",
        "fn_cost_3x_fp",
    }
