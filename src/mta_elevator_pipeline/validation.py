"""Input schema and data-quality validation."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "Month",
    "Equipment Code",
    "Total Outages",
    "Scheduled Outages",
    "Unscheduled Outages",
    "Entrapments",
    "Time Since Major Improvement",
    "AM Peak Availability",
    "PM Peak Availability",
    "24-Hour Availability",
}

COUNT_COLUMNS = [
    "Total Outages",
    "Scheduled Outages",
    "Unscheduled Outages",
    "Entrapments",
]

AVAILABILITY_COLUMNS = [
    "AM Peak Availability",
    "PM Peak Availability",
    "24-Hour Availability",
]


class DataValidationError(ValueError):
    """Raised when source data violates a required pipeline invariant."""


def validate_availability_data(df: pd.DataFrame) -> dict[str, object]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    checked = df.copy()
    checked["Month"] = pd.to_datetime(checked["Month"], errors="coerce")
    if checked["Month"].isna().any():
        raise DataValidationError("Month contains unparseable values.")

    duplicate_count = int(checked.duplicated(["Equipment Code", "Month"]).sum())
    if duplicate_count:
        raise DataValidationError(
            f"Found {duplicate_count} duplicate Equipment Code and Month rows."
        )

    for column in COUNT_COLUMNS:
        if (checked[column].dropna() < 0).any():
            raise DataValidationError(f"{column} contains negative values.")

    for column in AVAILABILITY_COLUMNS:
        values = checked[column].dropna()
        if ((values < 0) | (values > 1)).any():
            raise DataValidationError(f"{column} contains values outside [0, 1].")

    mismatch = (
        checked["Total Outages"]
        != checked["Scheduled Outages"] + checked["Unscheduled Outages"]
    )
    if mismatch.any():
        raise DataValidationError(
            f"Found {int(mismatch.sum())} inconsistent Total Outages rows."
        )

    ordered = checked.sort_values(["Equipment Code", "Month"])
    next_month = ordered.groupby("Equipment Code")["Month"].shift(-1)
    gap_months = (
        next_month.dt.to_period("M") - ordered["Month"].dt.to_period("M")
    ).dropna()
    nonconsecutive = sum(gap.n != 1 for gap in gap_months)
    history_lengths = checked.groupby("Equipment Code").size()

    return {
        "rows": len(checked),
        "equipment": int(checked["Equipment Code"].nunique()),
        "start_month": checked["Month"].min().date().isoformat(),
        "end_month": checked["Month"].max().date().isoformat(),
        "nonconsecutive_transitions": int(nonconsecutive),
        "missing_values": checked.isna().sum().to_dict(),
        "equipment_history_months": {
            "minimum": int(history_lengths.min()),
            "median": float(history_lengths.median()),
            "maximum": int(history_lengths.max()),
        },
    }
