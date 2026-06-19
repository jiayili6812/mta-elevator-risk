"""Explicit equipment eligibility rules for supervised modeling."""

from __future__ import annotations

import pandas as pd


def add_consecutive_history_months(df: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive observed months ending at each equipment-month row."""
    result = df.copy()
    result["Month"] = pd.to_datetime(result["Month"])
    result = result.sort_values(["Equipment Code", "Month"]).reset_index(drop=True)

    previous = result.groupby("Equipment Code", sort=False)["Month"].shift(1)
    month_delta = result["Month"].dt.to_period("M") - previous.dt.to_period("M")
    starts_sequence = month_delta.map(
        lambda value: value.n != 1 if pd.notna(value) else True
    )
    sequence_id = starts_sequence.groupby(result["Equipment Code"]).cumsum()
    result["consecutive_history_months"] = (
        result.groupby(["Equipment Code", sequence_id]).cumcount() + 1
    )
    return result


def add_eligibility_flags(
    df: pd.DataFrame,
    excluded_equipment_code_suffixes: list[str],
    minimum_history_months: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    equipment = result["Equipment Code"].astype(str)
    excluded_suffix = equipment.str.endswith(
        tuple(excluded_equipment_code_suffixes), na=False
    )

    history = result.groupby("Equipment Code")["Month"].transform("count")
    insufficient_history = (
        history < minimum_history_months
        if minimum_history_months is not None
        else pd.Series(False, index=result.index)
    )

    result["eligible_for_modeling"] = ~(excluded_suffix | insufficient_history)
    result["eligibility_reason"] = "eligible"
    result.loc[excluded_suffix, "eligibility_reason"] = "excluded_equipment_suffix"
    result.loc[
        insufficient_history & ~excluded_suffix, "eligibility_reason"
    ] = "insufficient_history"
    return result


def add_feature_set_eligibility(
    df: pd.DataFrame,
    minimum_history_months: int,
) -> pd.DataFrame:
    result = (
        df.copy()
        if "consecutive_history_months" in df
        else add_consecutive_history_months(df)
    )
    result["eligible_for_feature_set"] = (
        result["eligible_for_modeling"]
        & (result["consecutive_history_months"] >= minimum_history_months)
    )
    return result
