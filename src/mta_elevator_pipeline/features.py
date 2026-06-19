"""Time-safe feature engineering."""

from __future__ import annotations

import pandas as pd


CURRENT_OPERATIONAL_FEATURES = [
    "Total Outages",
    "Unscheduled Outages",
    "Scheduled Outages",
    "Entrapments",
    "AM Peak Availability",
    "PM Peak Availability",
    "24-Hour Availability",
]

TEMPORAL_SOURCE_COLUMNS = CURRENT_OPERATIONAL_FEATURES

AGE_FEATURES = [
    "Time Since Major Improvement",
    "age_missing",
]

BASE_FEATURES = CURRENT_OPERATIONAL_FEATURES + AGE_FEATURES


def build_features(
    df: pd.DataFrame,
    lag_months: list[int],
    rolling_windows: list[int],
    include_seasonality: bool = True,
    include_age: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    result["Month"] = pd.to_datetime(result["Month"])
    result = result.sort_values(["Equipment Code", "Month"]).reset_index(drop=True)
    grouped = result.groupby("Equipment Code", sort=False)
    result["age_missing"] = result["Time Since Major Improvement"].isna().astype(int)

    feature_columns = list(CURRENT_OPERATIONAL_FEATURES)
    if include_age:
        feature_columns.extend(AGE_FEATURES)
    for column in TEMPORAL_SOURCE_COLUMNS:
        prefix = column.lower().replace("-", "").replace(" ", "_")
        for lag in lag_months:
            feature_name = f"{prefix}_lag_{lag}"
            result[feature_name] = grouped[column].shift(lag)
            feature_columns.append(feature_name)

        for window in rolling_windows:
            feature_name = f"{prefix}_roll_{window}_mean"
            result[feature_name] = (
                grouped[column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            feature_columns.append(feature_name)

    if include_seasonality:
        result["month_of_year"] = result["Month"].dt.month
        result["is_winter"] = result["Month"].dt.month.isin([12, 1, 2]).astype(int)
        feature_columns.extend(["month_of_year", "is_winter"])

    return result, feature_columns


def minimum_history_for_feature_set(
    lag_months: list[int],
    rolling_windows: list[int],
) -> int:
    """Return the minimum consecutive months needed for fully observed features."""
    lag_requirement = max(lag_months, default=0) + 1
    rolling_requirement = max(rolling_windows, default=1)
    return max(1, lag_requirement, rolling_requirement)
