from __future__ import annotations

import pandas as pd
import pytest

from mta_elevator_pipeline.splits import (
    access_frozen_test_labels,
    assert_frozen_feature_counts,
    create_development_split,
    rolling_temporal_validation_splits,
)
from mta_elevator_pipeline.targets import TARGET_COLUMN, build_next_month_target


def test_development_split_never_contains_frozen_months(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    split = create_development_split(
        targeted,
        frozen_test_start="2025-02-01",
        frozen_test_end="2025-04-01",
        target_column=TARGET_COLUMN,
    )
    assert split.development["Month"].max() < pd.Timestamp("2025-02-01")
    assert TARGET_COLUMN not in split.frozen_features.columns


def test_frozen_labels_require_exact_acknowledgment(monthly_data, tmp_path):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    split = create_development_split(
        targeted,
        frozen_test_start="2025-02-01",
        frozen_test_end="2025-04-01",
        target_column=TARGET_COLUMN,
    )
    with pytest.raises(PermissionError):
        access_frozen_test_labels(
            split,
            acknowledgment="",
            required_acknowledgment="EXACT_ACK",
            audit_log=tmp_path / "audit.log",
        )


def test_authorized_frozen_access_is_logged(monthly_data, tmp_path):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    split = create_development_split(
        targeted,
        frozen_test_start="2025-02-01",
        frozen_test_end="2025-04-01",
        target_column=TARGET_COLUMN,
    )
    audit = tmp_path / "audit.log"
    frozen = access_frozen_test_labels(
        split,
        acknowledgment="EXACT_ACK",
        required_acknowledgment="EXACT_ACK",
        audit_log=audit,
    )
    assert TARGET_COLUMN in frozen.columns
    assert "frozen test labels accessed" in audit.read_text(encoding="utf-8")


def test_rolling_validation_folds_are_temporally_ordered(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    split = create_development_split(
        targeted,
        frozen_test_start="2025-02-01",
        frozen_test_end="2025-04-01",
        target_column=TARGET_COLUMN,
    )
    folds = rolling_temporal_validation_splits(
        split.development, validation_months=1, fold_count=3
    )
    assert len(folds) == 3
    for fold in folds:
        assert fold.train["Month"].max() < fold.validation["Month"].min()
        assert not fold.train["Month"].between("2025-02-01", "2025-04-01").any()
        assert not fold.validation["Month"].between("2025-02-01", "2025-04-01").any()
    validation_months = [set(fold.validation["Month"]) for fold in folds]
    assert validation_months[0].isdisjoint(validation_months[1])
    assert validation_months[1].isdisjoint(validation_months[2])


def test_frozen_count_confirmation_does_not_expose_labels(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    split = create_development_split(
        targeted,
        frozen_test_start="2025-02-01",
        frozen_test_end="2025-04-01",
        target_column=TARGET_COLUMN,
    )
    counts = {
        month.strftime("%Y-%m-%d"): int(count)
        for month, count in split.frozen_features.groupby("Month").size().items()
    }
    report = assert_frozen_feature_counts(
        split.frozen_features,
        expected_rows=len(split.frozen_features),
        expected_rows_by_month=counts,
    )
    assert report["labels_exposed"] is False
    assert TARGET_COLUMN not in split.frozen_features
