from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def monthly_data() -> pd.DataFrame:
    rows = []
    for equipment in ["EL001", "EL002"]:
        for index, month in enumerate(pd.date_range("2024-09-01", periods=9, freq="MS")):
            scheduled = index % 2
            unscheduled = 2 if equipment == "EL001" and index in [4, 7] else index % 2
            rows.append(
                {
                    "Month": month,
                    "Equipment Code": equipment,
                    "Total Outages": scheduled + unscheduled,
                    "Scheduled Outages": scheduled,
                    "Unscheduled Outages": unscheduled,
                    "Entrapments": int(equipment == "EL002" and index == 5),
                    "Time Since Major Improvement": 24 + index,
                    "AM Peak Availability": 0.98,
                    "PM Peak Availability": 0.97,
                    "24-Hour Availability": 0.975,
                }
            )
    return pd.DataFrame(rows)

