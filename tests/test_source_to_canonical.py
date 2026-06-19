from __future__ import annotations

import pandas as pd
import pytest

from mta_elevator_pipeline.config import PROJECT_ROOT
from mta_elevator_pipeline.source_to_canonical import (
    CANONICAL_COLUMNS,
    SourceToCanonicalError,
    canonicalize_official_availability,
    write_canonical_availability,
)


def official_rows() -> pd.DataFrame:
    rows = []
    for equipment_type, code in [("Elevator", "EL001"), ("Escalator", "ES001")]:
        for month in ["12/01/2020", "01/01/2021"]:
            rows.append(
                {
                    "Month": month,
                    "Borough": "Manhattan",
                    "Equipment Type": equipment_type,
                    "Equipment Code": code,
                    "Total Outages": 1,
                    "Scheduled Outages": 1,
                    "Unscheduled Outages": 0,
                    "Entrapments": 0,
                    "Time Since Major Improvement": 10.0,
                    "AM Peak Availability": 1.0,
                    "AM Peak Hours Available": 124.0,
                    "AM Peak Total Hours": 124.0,
                    "PM Peak Availability": 1.0,
                    "PM Peak Hours Available": 124.0,
                    "PM Peak Total Hours": 124.0,
                    "24-Hour Availability": 1.0,
                    "24-Hour Hours Available": 744.0,
                    "24-Hour Total Hours": 744.0,
                    "Station Name": "TEST",
                    "Station MRN": 1,
                    "Station Complex Name": "Test Station",
                    "Station Complex MRN": 1,
                }
            )
    return pd.DataFrame(rows)


def test_canonicalize_filters_to_elevator_months_since_2021():
    canonical = canonicalize_official_availability(official_rows())

    assert list(canonical.columns) == CANONICAL_COLUMNS
    assert len(canonical) == 1
    assert canonical.loc[0, "Equipment Type"] == "Elevator"
    assert canonical.loc[0, "Equipment Code"] == "EL001"
    assert canonical.loc[0, "Month"] == "01/01/2021"


def test_canonicalize_rejects_missing_source_columns():
    source = official_rows().drop(columns=["24-Hour Availability"])

    with pytest.raises(SourceToCanonicalError, match="Missing official source columns"):
        canonicalize_official_availability(source)


def test_canonicalize_normalizes_percent_string_availability():
    source = official_rows()
    source["AM Peak Availability"] = "100%"
    source["PM Peak Availability"] = "50%"
    source["24-Hour Availability"] = "98.60215053763%"

    canonical = canonicalize_official_availability(source)

    assert canonical.loc[0, "AM Peak Availability"] == 1.0
    assert canonical.loc[0, "PM Peak Availability"] == 0.5
    assert canonical.loc[0, "24-Hour Availability"] == pytest.approx(0.9860215053763)


def test_write_refuses_to_overwrite_fixed_training_snapshot(tmp_path):
    source_csv = tmp_path / "official.csv"
    official_rows().to_csv(source_csv, index=False)
    fixed_snapshot = PROJECT_ROOT / "data/raw/df3_availability.csv"

    with pytest.raises(SourceToCanonicalError, match="Refusing to overwrite"):
        write_canonical_availability(source_csv, fixed_snapshot)


def test_write_refuses_existing_output_without_overwrite(tmp_path):
    source_csv = tmp_path / "official.csv"
    output_csv = tmp_path / "canonical.csv"
    official_rows().to_csv(source_csv, index=False)
    output_csv.write_text("already here", encoding="utf-8")

    with pytest.raises(SourceToCanonicalError, match="Output already exists"):
        write_canonical_availability(source_csv, output_csv)
