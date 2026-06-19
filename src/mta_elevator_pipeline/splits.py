"""Temporal splitting and frozen-test access guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DevelopmentSplit:
    development: pd.DataFrame
    frozen_features: pd.DataFrame
    _frozen_labeled: pd.DataFrame


@dataclass(frozen=True)
class TemporalValidationFold:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame


def create_development_split(
    labeled: pd.DataFrame,
    frozen_test_start: str,
    frozen_test_end: str,
    target_column: str,
) -> DevelopmentSplit:
    data = labeled.dropna(subset=[target_column]).copy()
    data["Month"] = pd.to_datetime(data["Month"])
    start = pd.Timestamp(frozen_test_start)
    end = pd.Timestamp(frozen_test_end)

    development = data[data["Month"] < start].copy()
    frozen = data[data["Month"].between(start, end)].copy()
    if development.empty or frozen.empty:
        raise ValueError("Development or frozen-test split is empty.")
    if development["Month"].max() >= frozen["Month"].min():
        raise ValueError("Development data overlaps the frozen test period.")

    frozen_features = frozen.drop(columns=[target_column])
    return DevelopmentSplit(development, frozen_features, frozen)


def access_frozen_test_labels(
    split: DevelopmentSplit,
    acknowledgment: str,
    required_acknowledgment: str,
    audit_log: Path,
) -> pd.DataFrame:
    if acknowledgment != required_acknowledgment:
        raise PermissionError(
            "Frozen-test labels require the exact configured acknowledgment."
        )
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with audit_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\tfrozen test labels accessed\n")
    return split._frozen_labeled.copy()


def latest_temporal_validation_split(
    development: pd.DataFrame,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = development["Month"].max()
    validation_start = end - pd.DateOffset(months=validation_months - 1)
    train = development[development["Month"] < validation_start].copy()
    validation = development[development["Month"] >= validation_start].copy()
    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")
    return train, validation


def rolling_temporal_validation_splits(
    development: pd.DataFrame,
    validation_months: int,
    fold_count: int,
) -> list[TemporalValidationFold]:
    """Create expanding-training, non-overlapping validation folds."""
    data = development.copy()
    data["Month"] = pd.to_datetime(data["Month"])
    latest_month = data["Month"].max()
    folds = []

    for offset in reversed(range(fold_count)):
        validation_end = latest_month - pd.DateOffset(months=offset * validation_months)
        validation_start = validation_end - pd.DateOffset(months=validation_months - 1)
        train = data[data["Month"] < validation_start].copy()
        validation = data[data["Month"].between(validation_start, validation_end)].copy()
        if train.empty or validation.empty:
            raise ValueError(
                f"Training or validation split is empty for fold ending {validation_end:%Y-%m}."
            )
        folds.append(
            TemporalValidationFold(
                name=f"{validation_start:%Y-%m}_to_{validation_end:%Y-%m}",
                train=train,
                validation=validation,
            )
        )
    return folds


def assert_frozen_feature_counts(
    frozen_features: pd.DataFrame,
    expected_rows: int,
    expected_rows_by_month: dict[str, int],
) -> dict[str, object]:
    """Verify frozen-set feature counts without accessing or returning labels."""
    actual_by_month = {
        month.strftime("%Y-%m-%d"): int(count)
        for month, count in frozen_features.groupby("Month").size().items()
    }
    if len(frozen_features) != expected_rows or actual_by_month != expected_rows_by_month:
        raise ValueError(
            "Frozen-test feature counts differ from the locked expectations."
        )
    return {
        "rows": int(len(frozen_features)),
        "rows_by_month": actual_by_month,
        "labels_exposed": False,
        "matches_locked_expectations": True,
    }
