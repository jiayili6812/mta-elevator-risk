from __future__ import annotations

import pandas as pd

from mta_elevator_pipeline.eligibility import (
    add_consecutive_history_months,
    add_eligibility_flags,
    add_feature_set_eligibility,
)


def test_excluded_suffix_is_retained_but_not_model_eligible(monthly_data):
    data = monthly_data.copy()
    data.loc[data["Equipment Code"] == "EL002", "Equipment Code"] = "EL002X"
    audited = add_eligibility_flags(data, excluded_equipment_code_suffixes=["X"])

    assert len(audited) == len(data)
    x_rows = audited[audited["Equipment Code"] == "EL002X"]
    assert not x_rows["eligible_for_modeling"].any()
    assert set(x_rows["eligibility_reason"]) == {"excluded_equipment_suffix"}


def test_new_equipment_remains_eligible_without_minimum_history(monthly_data):
    one_month = monthly_data.groupby("Equipment Code").head(1)
    audited = add_eligibility_flags(
        one_month,
        excluded_equipment_code_suffixes=["X"],
        minimum_history_months=None,
    )
    assert audited["eligible_for_modeling"].all()


def test_new_equipment_becomes_feature_eligible_after_required_history(monthly_data):
    new_equipment = monthly_data[monthly_data["Equipment Code"] == "EL001"].head(6)
    audited = add_eligibility_flags(new_equipment, ["X"])
    audited = add_consecutive_history_months(audited)
    audited = add_feature_set_eligibility(audited, minimum_history_months=4)

    assert audited["consecutive_history_months"].tolist() == [1, 2, 3, 4, 5, 6]
    assert audited["eligible_for_feature_set"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_consecutive_history_resets_after_gap(monthly_data):
    data = monthly_data[monthly_data["Equipment Code"] == "EL001"].copy()
    data = data[data["Month"] != pd.Timestamp("2024-12-01")]
    audited = add_consecutive_history_months(data)
    january = audited[audited["Month"] == pd.Timestamp("2025-01-01")]
    assert january["consecutive_history_months"].item() == 1
