from __future__ import annotations

import pytest

from mta_elevator_pipeline.validation import (
    DataValidationError,
    validate_availability_data,
)


def test_valid_data_returns_report(monthly_data):
    report = validate_availability_data(monthly_data)
    assert report["rows"] == len(monthly_data)
    assert report["equipment"] == 2
    assert report["nonconsecutive_transitions"] == 0


def test_duplicate_equipment_month_is_rejected(monthly_data):
    duplicated = monthly_data._append(monthly_data.iloc[0], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_availability_data(duplicated)


def test_outage_identity_is_enforced(monthly_data):
    invalid = monthly_data.copy()
    invalid.loc[0, "Total Outages"] = 999
    with pytest.raises(DataValidationError, match="inconsistent"):
        validate_availability_data(invalid)

