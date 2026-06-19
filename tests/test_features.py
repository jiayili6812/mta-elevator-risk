from __future__ import annotations

import pandas as pd

from mta_elevator_pipeline.features import build_features, minimum_history_for_feature_set


def test_lags_do_not_cross_equipment_boundaries(monthly_data):
    featured, _ = build_features(monthly_data, [1], [3])
    first_rows = featured.groupby("Equipment Code").head(1)
    assert first_rows["unscheduled_outages_lag_1"].isna().all()


def test_rolling_feature_uses_current_and_prior_history(monthly_data):
    featured, _ = build_features(monthly_data, [1], [3])
    equipment = featured[featured["Equipment Code"] == "EL001"].reset_index(drop=True)
    expected = equipment.loc[:2, "Unscheduled Outages"].mean()
    assert equipment.loc[2, "unscheduled_outages_roll_3_mean"] == expected


def test_missing_age_is_preserved_with_indicator(monthly_data):
    data = monthly_data.copy()
    data.loc[0, "Time Since Major Improvement"] = None
    featured, columns = build_features(data, [1], [3])
    assert "age_missing" in columns
    assert featured.loc[featured["age_missing"] == 1, "Time Since Major Improvement"].isna().all()


def test_minimum_history_matches_feature_observation_requirements():
    assert minimum_history_for_feature_set([], []) == 1
    assert minimum_history_for_feature_set([1, 2, 3], [3]) == 4
    assert minimum_history_for_feature_set([1, 2, 3], [3, 6]) == 6


def test_future_observations_cannot_change_features_available_at_month_t(monthly_data):
    original, columns = build_features(monthly_data, [1, 2, 3], [3, 6])
    changed = monthly_data.copy()
    future = changed["Month"] > pd.Timestamp("2025-01-01")
    changed.loc[future, "Unscheduled Outages"] = 999
    changed.loc[future, "24-Hour Availability"] = 0.01
    changed_features, _ = build_features(changed, [1, 2, 3], [3, 6])

    through_t = original["Month"] <= pd.Timestamp("2025-01-01")
    pd.testing.assert_frame_equal(
        original.loc[through_t, columns].reset_index(drop=True),
        changed_features.loc[through_t, columns].reset_index(drop=True),
    )


def test_lags_and_rolls_use_only_same_equipment_history(monthly_data):
    changed = monthly_data.copy()
    changed.loc[changed["Equipment Code"] == "EL001", "Unscheduled Outages"] = 999
    original_features, columns = build_features(monthly_data, [1, 2, 3], [3, 6])
    changed_features, _ = build_features(changed, [1, 2, 3], [3, 6])
    temporal_columns = [
        column
        for column in columns
        if column.startswith("unscheduled_outages_lag_")
        or column.startswith("unscheduled_outages_roll_")
    ]
    original_el002 = original_features["Equipment Code"] == "EL002"
    changed_el002 = changed_features["Equipment Code"] == "EL002"
    pd.testing.assert_frame_equal(
        original_features.loc[original_el002, temporal_columns].reset_index(drop=True),
        changed_features.loc[changed_el002, temporal_columns].reset_index(drop=True),
    )


def test_operational_only_feature_set_excludes_age_and_seasonality(monthly_data):
    _, columns = build_features(
        monthly_data,
        lag_months=[],
        rolling_windows=[],
        include_seasonality=False,
        include_age=False,
    )
    assert "Time Since Major Improvement" not in columns
    assert "age_missing" not in columns
    assert "month_of_year" not in columns
