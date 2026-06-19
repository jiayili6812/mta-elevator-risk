"""Leakage-safe next-calendar-month target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "failure_next_month"
SECONDARY_TARGET_COLUMN = "failure_next_month_secondary"


def build_next_month_target(
    df: pd.DataFrame,
    unscheduled_threshold: int,
    entrapment_threshold: int = 0,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    result = df.copy()
    result["Month"] = pd.to_datetime(result["Month"])
    result = result.sort_values(["Equipment Code", "Month"]).reset_index(drop=True)

    grouped = result.groupby("Equipment Code", sort=False)
    next_month = grouped["Month"].shift(-1)
    next_unscheduled = grouped["Unscheduled Outages"].shift(-1)
    next_entrapments = grouped["Entrapments"].shift(-1)

    month_delta = next_month.dt.to_period("M") - result["Month"].dt.to_period("M")
    is_consecutive = month_delta.map(
        lambda value: value.n == 1 if pd.notna(value) else False
    )

    known = is_consecutive & next_unscheduled.notna() & next_entrapments.notna()
    target = pd.Series(np.nan, index=result.index, dtype="float64")
    target.loc[known] = (
        (next_unscheduled.loc[known] >= unscheduled_threshold)
        | (next_entrapments.loc[known] > entrapment_threshold)
    ).astype(int)

    result[target_column] = target
    return result


def build_target_variants(
    df: pd.DataFrame,
    primary_unscheduled_threshold: int,
    secondary_unscheduled_threshold: int,
    entrapment_threshold: int = 0,
) -> pd.DataFrame:
    primary = build_next_month_target(
        df,
        unscheduled_threshold=primary_unscheduled_threshold,
        entrapment_threshold=entrapment_threshold,
        target_column=TARGET_COLUMN,
    )
    return build_next_month_target(
        primary,
        unscheduled_threshold=secondary_unscheduled_threshold,
        entrapment_threshold=entrapment_threshold,
        target_column=SECONDARY_TARGET_COLUMN,
    )
