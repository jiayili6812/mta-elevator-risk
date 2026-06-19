from __future__ import annotations

import numpy as np
import pandas as pd

from mta_elevator_pipeline.features import CURRENT_OPERATIONAL_FEATURES
from mta_elevator_pipeline.tcn import build_sequences
from mta_elevator_pipeline.session4_2 import select_tcn_candidate
from mta_elevator_pipeline.targets import TARGET_COLUMN, build_next_month_target


def _sequence_frame():
    rows = []
    for equipment, months in {
        "A": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "B": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "C": ["2024-01-01", "2024-03-01", "2024-04-01"],
    }.items():
        for index, month in enumerate(months):
            row = {"Equipment Code": equipment, "Month": month}
            row.update(
                {
                    column: index + (100 if equipment == "B" else 200 if equipment == "C" else 0)
                    for column in CURRENT_OPERATIONAL_FEATURES
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequences_never_cross_equipment_and_require_consecutive_months():
    targeted = build_next_month_target(_sequence_frame(), unscheduled_threshold=2)
    sequences = build_sequences(targeted, 2)
    assert set(sequences.endpoints["Equipment Code"]) == {"A", "B"}
    assert not (
        (sequences.endpoints["Equipment Code"] == "C")
        & (sequences.endpoints["sequence_end"] == pd.Timestamp("2024-03-01"))
    ).any()
    assert (sequences.endpoints["sequence_end"] - sequences.endpoints["sequence_start"]).dt.days.between(28, 31).all()
    assert all((window < 100).all() or (window >= 100).all() for window in sequences.values)


def test_sequence_ending_at_t_predicts_t_plus_one():
    frame = _sequence_frame()
    frame.loc[(frame["Equipment Code"] == "A") & (frame["Month"] == "2024-04-01"), "Entrapments"] = 1
    targeted = build_next_month_target(frame, unscheduled_threshold=2)
    sequences = build_sequences(targeted, 2)
    march = sequences.endpoints[
        (sequences.endpoints["Equipment Code"] == "A")
        & (sequences.endpoints["Month"] == pd.Timestamp("2024-03-01"))
    ]
    assert march[TARGET_COLUMN].iloc[0] == 1


def test_future_values_do_not_change_earlier_sequence_features():
    targeted = build_next_month_target(_sequence_frame(), unscheduled_threshold=2)
    first = build_sequences(targeted, 2)
    changed = targeted.copy()
    changed.loc[changed["Month"] == pd.Timestamp("2024-04-01"), CURRENT_OPERATIONAL_FEATURES] = 999
    second = build_sequences(changed, 2)
    earlier = first.endpoints["Month"] < pd.Timestamp("2024-04-01")
    np.testing.assert_array_equal(first.values[earlier], second.values[earlier])


def test_fold_4_cannot_affect_tcn_candidate_selection():
    candidates = [
        {"selection_folds_mean_pr_auc": 0.6, "selection_folds_pr_auc_std": 0.02, "fold_4": 0.1},
        {"selection_folds_mean_pr_auc": 0.59, "selection_folds_pr_auc_std": 0.01, "fold_4": 0.99},
    ]
    first = select_tcn_candidate(candidates)
    candidates[0]["fold_4"] = 0.0
    candidates[1]["fold_4"] = 1.0
    assert select_tcn_candidate(candidates) is first
