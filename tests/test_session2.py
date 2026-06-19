from __future__ import annotations

import numpy as np

from mta_elevator_pipeline.session2 import (
    baseline_probabilities,
    select_operating_threshold,
    summarize_fold_metrics,
)
from mta_elevator_pipeline.targets import TARGET_COLUMN, build_next_month_target


def test_baselines_use_only_training_prevalence_and_current_month_values(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    train = targeted.iloc[:8].dropna(subset=[TARGET_COLUMN]).copy()
    validation = targeted.iloc[8:14].dropna(subset=[TARGET_COLUMN]).copy()

    constant = baseline_probabilities("constant_prevalence", train, validation, 2, 0)
    previous_failure = baseline_probabilities(
        "previous_month_failure", train, validation, 2, 0
    )
    outage_rule = baseline_probabilities("simple_outage_rule", train, validation, 2, 0)

    assert np.all(constant == train[TARGET_COLUMN].mean())
    np.testing.assert_array_equal(
        previous_failure,
        (
            (validation["Entrapments"] > 0)
            | (validation["Unscheduled Outages"] >= 2)
        ).astype(float),
    )
    np.testing.assert_array_equal(
        outage_rule, (validation["Unscheduled Outages"] >= 1).astype(float)
    )


def test_threshold_selection_meets_recall_goal_and_is_deterministic():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])
    threshold = select_operating_threshold(y_true, probabilities, minimum_recall=0.66)
    assert threshold == 0.8


def test_fold_summary_reports_mean_and_variation():
    folds = {
        "one": {
            "pr_auc": 0.4,
            "roc_auc": 0.6,
            "brier_score": 0.2,
            "precision": 0.5,
            "recall": 0.7,
            "false_positives": 3,
            "false_negatives": 2,
        },
        "two": {
            "pr_auc": 0.6,
            "roc_auc": 0.8,
            "brier_score": 0.1,
            "precision": 0.7,
            "recall": 0.9,
            "false_positives": 1,
            "false_negatives": 1,
        },
    }
    summary = summarize_fold_metrics(folds)
    assert summary["pr_auc"]["mean"] == 0.5
    assert summary["pr_auc"]["std"] > 0
