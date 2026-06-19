"""Convert official MTA availability exports to the canonical training schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .validation import AVAILABILITY_COLUMNS, COUNT_COLUMNS, validate_availability_data


CANONICAL_COLUMNS = [
    "Month",
    "Borough",
    "Equipment Type",
    "Equipment Code",
    "Total Outages",
    "Scheduled Outages",
    "Unscheduled Outages",
    "Entrapments",
    "Time Since Major Improvement",
    "AM Peak Availability",
    "AM Peak Hours Available",
    "AM Peak Total Hours",
    "PM Peak Availability",
    "PM Peak Hours Available",
    "PM Peak Total Hours",
    "24-Hour Availability",
    "24-Hour Hours Available",
    "24-Hour Total Hours",
    "Station Name",
    "Station MRN",
    "Station Complex Name",
    "Station Complex MRN",
]

FIXED_TRAINING_SNAPSHOT = (PROJECT_ROOT / "data/raw/df3_availability.csv").resolve()


class SourceToCanonicalError(ValueError):
    """Raised when a source export cannot be safely converted."""


def _coerce_availability_column(series: pd.Series) -> pd.Series:
    """Normalize official availability values to proportions in [0, 1]."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        text = series.astype("string").str.strip()
        has_percent = text.str.endswith("%", na=False)
        values = pd.to_numeric(text.str.rstrip("%"), errors="coerce").astype(float)
        values.loc[has_percent] = values.loc[has_percent] / 100.0

    nonmissing = values.dropna()
    if not nonmissing.empty and nonmissing.max() > 1.0:
        if nonmissing.max() <= 100.0:
            values = values / 100.0
        else:
            raise SourceToCanonicalError(
                f"Availability values exceed expected percent scale in {series.name}."
            )
    return values


def canonicalize_official_availability(
    source: pd.DataFrame,
    *,
    start_month: str = "2021-01-01",
    equipment_type: str = "Elevator",
) -> pd.DataFrame:
    """Return the canonical elevator-month snapshot from an official export.

    The known July 2025 lineage is a row-preserving subset: keep official rows
    where `Equipment Type` is `Elevator` and `Month` is on or after 2021-01-01.
    No aggregation, renaming, or missing-value imputation is performed.
    Availability values are normalized to proportions because later official
    exports may encode the same fields as percent strings.
    """

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in source.columns]
    if missing_columns:
        raise SourceToCanonicalError(f"Missing official source columns: {missing_columns}")

    canonical = source.loc[:, CANONICAL_COLUMNS].copy()
    parsed_month = pd.to_datetime(canonical["Month"], errors="coerce")
    if parsed_month.isna().any():
        raise SourceToCanonicalError("Month contains unparseable values.")

    for column in COUNT_COLUMNS:
        canonical[column] = pd.to_numeric(canonical[column], errors="raise")
    for column in AVAILABILITY_COLUMNS:
        canonical[column] = _coerce_availability_column(canonical[column])

    canonical = canonical[
        (canonical["Equipment Type"] == equipment_type)
        & (parsed_month >= pd.Timestamp(start_month))
    ].copy()
    canonical["_parsed_month"] = pd.to_datetime(canonical["Month"])
    canonical = (
        canonical.sort_values("_parsed_month", ascending=False, kind="mergesort")
        .drop(columns=["_parsed_month"])
        .reset_index(drop=True)
    )
    validate_availability_data(canonical)
    return canonical


def write_canonical_availability(
    source_csv: str | Path,
    output_csv: str | Path,
    *,
    start_month: str = "2021-01-01",
    equipment_type: str = "Elevator",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Canonicalize an official export and write it somewhere other than df3."""

    output_path = Path(output_csv)
    resolved_output = output_path.resolve()
    if resolved_output == FIXED_TRAINING_SNAPSHOT:
        raise SourceToCanonicalError(
            "Refusing to overwrite fixed training snapshot data/raw/df3_availability.csv."
        )
    if output_path.exists() and not overwrite:
        raise SourceToCanonicalError(f"Output already exists: {output_path}")

    source = pd.read_csv(source_csv)
    canonical = canonicalize_official_availability(
        source,
        start_month=start_month,
        equipment_type=equipment_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_path, index=False)
    return canonical
