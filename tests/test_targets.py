from __future__ import annotations

import pandas as pd

from mta_elevator_pipeline.targets import TARGET_COLUMN, build_next_month_target


def test_last_row_per_equipment_has_unknown_target(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    last_rows = targeted.groupby("Equipment Code").tail(1)
    assert last_rows[TARGET_COLUMN].isna().all()


def test_target_uses_next_month_not_current_month(monthly_data):
    targeted = build_next_month_target(monthly_data, unscheduled_threshold=2)
    equipment = targeted[targeted["Equipment Code"] == "EL001"].reset_index(drop=True)
    # EL001 has two unscheduled outages at source index 4, so source index 3 is positive.
    assert equipment.loc[3, TARGET_COLUMN] == 1
    assert equipment.loc[4, TARGET_COLUMN] == 0


def test_nonconsecutive_next_record_is_not_treated_as_next_month(monthly_data):
    missing_month = monthly_data[
        ~(
            (monthly_data["Equipment Code"] == "EL001")
            & (pd.to_datetime(monthly_data["Month"]) == pd.Timestamp("2024-12-01"))
        )
    ]
    targeted = build_next_month_target(missing_month, unscheduled_threshold=2)
    november = targeted[
        (targeted["Equipment Code"] == "EL001")
        & (targeted["Month"] == pd.Timestamp("2024-11-01"))
    ]
    assert november[TARGET_COLUMN].isna().all()


def test_primary_target_is_next_month_entrapment_or_two_unscheduled(monthly_data):
    data = monthly_data.copy()
    equipment = data["Equipment Code"] == "EL001"
    data.loc[equipment, ["Unscheduled Outages", "Entrapments"]] = 0
    months = sorted(data.loc[equipment, "Month"].unique())
    data.loc[equipment & (data["Month"] == months[1]), "Entrapments"] = 1
    data.loc[equipment & (data["Month"] == months[3]), "Unscheduled Outages"] = 2
    targeted = build_next_month_target(data, unscheduled_threshold=2)
    rows = targeted[targeted["Equipment Code"] == "EL001"].reset_index(drop=True)

    assert rows.loc[0, TARGET_COLUMN] == 1  # Next month has an entrapment.
    assert rows.loc[1, TARGET_COLUMN] == 0  # Current entrapment does not label itself.
    assert rows.loc[2, TARGET_COLUMN] == 1  # Next month has two unscheduled outages.
    assert rows.loc[3, TARGET_COLUMN] == 0  # Current outages do not label themselves.
