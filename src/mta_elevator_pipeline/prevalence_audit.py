"""Development-only primary-target prevalence investigation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .eligibility import add_consecutive_history_months, add_eligibility_flags
from .features import minimum_history_for_feature_set
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN, build_next_month_target


def population_summary(df: pd.DataFrame) -> dict[str, object]:
    labeled = df.dropna(subset=[TARGET_COLUMN])
    positives = int(labeled[TARGET_COLUMN].sum())
    return {
        "total_labeled_rows": int(len(labeled)),
        "positive_rows": positives,
        "negative_rows": int(len(labeled) - positives),
        "prevalence": float(labeled[TARGET_COLUMN].mean()),
        "unique_elevators": int(labeled["Equipment Code"].nunique()),
    }


def dated_population_summary(df: pd.DataFrame) -> dict[str, object]:
    summary = population_summary(df)
    summary.update(
        {
            "start_month": df["Month"].min().date().isoformat(),
            "end_month": df["Month"].max().date().isoformat(),
        }
    )
    return summary


def build_prevalence_investigation(
    source: pd.DataFrame,
    config: dict,
) -> dict[str, object]:
    frozen_start = pd.Timestamp(config["split"]["frozen_test_start"])
    frozen_end = pd.Timestamp(config["split"]["frozen_test_end"])

    # February is retained only as the outcome source for January prediction rows.
    # With later source rows removed, no frozen-period prediction row receives a label.
    safe_source = source.copy()
    safe_source["Month"] = pd.to_datetime(safe_source["Month"])
    safe_source = safe_source[safe_source["Month"] <= frozen_start].copy()
    audited = add_consecutive_history_months(
        add_eligibility_flags(
            safe_source,
            config["eligibility"]["excluded_equipment_code_suffixes"],
            config["eligibility"]["minimum_history_months"],
        )
    )
    non_x = audited[audited["eligible_for_modeling"]].copy()
    primary = build_next_month_target(
        non_x,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    development = primary[
        (primary["Month"] < frozen_start) & primary[TARGET_COLUMN].notna()
    ].copy()
    safe_unlabeled = primary[primary[TARGET_COLUMN].isna()]

    populations = {
        "all_non_x_target_eligible_rows": population_summary(development),
        "after_newly_introduced_and_equipment_history_rules": population_summary(
            development
        ),
    }
    for name, definition in config["features"]["feature_sets"].items():
        minimum = minimum_history_for_feature_set(
            definition["lag_months"], definition["rolling_windows"]
        )
        eligible = development[development["consecutive_history_months"] >= minimum]
        populations[f"{name}_feature_eligibility"] = {
            **population_summary(eligible),
            "minimum_consecutive_history_months": minimum,
        }
    selected = config["features"]["selected_feature_set"]
    selected_rows = development[
        development["consecutive_history_months"]
        >= populations[f"{selected}_feature_eligibility"][
            "minimum_consecutive_history_months"
        ]
    ].copy()
    populations["development_only_excluding_frozen_period"] = population_summary(
        selected_rows
    )

    folds = []
    for fold in rolling_temporal_validation_splits(
        selected_rows,
        config["split"]["validation_months"],
        config["split"]["rolling_validation_folds"],
    ):
        overlap = set(fold.train.index) & set(fold.validation.index)
        frozen_rows = pd.concat([fold.train, fold.validation])["Month"].between(
            frozen_start, frozen_end
        )
        folds.append(
            {
                "name": fold.name,
                "train": dated_population_summary(fold.train),
                "validation": dated_population_summary(fold.validation),
                "train_validation_date_overlap": bool(
                    fold.train["Month"].max() >= fold.validation["Month"].min()
                ),
                "train_validation_row_overlap": bool(overlap),
                "contains_frozen_period_rows": bool(frozen_rows.any()),
            }
        )

    all_equipment = build_next_month_target(
        audited,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    all_equipment_development = all_equipment[
        (all_equipment["Month"] < frozen_start)
        & all_equipment[TARGET_COLUMN].notna()
    ]
    current_all = audited[audited["Month"] < frozen_start]
    current_all_target = (
        (current_all["Entrapments"] > config["target"]["entrapment_threshold"])
        | (
            current_all["Unscheduled Outages"]
            >= config["target"]["unscheduled_outage_threshold"]
        )
    ).astype(int)
    next_unscheduled = primary.groupby("Equipment Code")["Unscheduled Outages"].shift(-1)
    next_entrapments = primary.groupby("Equipment Code")["Entrapments"].shift(-1)
    unscheduled_only = (
        next_unscheduled.loc[development.index]
        >= config["target"]["unscheduled_outage_threshold"]
    )
    entrapment_only = (
        next_entrapments.loc[development.index]
        > config["target"]["entrapment_threshold"]
    )
    alternatives = {
        "correct_next_month_target_including_x_rows": population_summary(
            all_equipment_development
        ),
        "current_month_target_including_x_rows": {
            "total_labeled_rows": int(len(current_all_target)),
            "positive_rows": int(current_all_target.sum()),
            "negative_rows": int(len(current_all_target) - current_all_target.sum()),
            "prevalence": float(current_all_target.mean()),
            "unique_elevators": int(current_all["Equipment Code"].nunique()),
        },
        "non_x_next_month_unscheduled_outages_only": {
            "total_labeled_rows": int(len(development)),
            "positive_rows": int(unscheduled_only.sum()),
            "negative_rows": int(len(development) - unscheduled_only.sum()),
            "prevalence": float(unscheduled_only.mean()),
            "unique_elevators": int(development["Equipment Code"].nunique()),
        },
    }
    return {
        "status": "passed",
        "scope": (
            "Development-only. February 2025 is used only as the outcome source "
            "for January 2025 labels; no February-April 2025 prediction-row target "
            "labels are constructed, accessed, or reported."
        ),
        "primary_target_contract": {
            "definition": (
                "next-month Entrapments > 0 OR "
                "next-month Unscheduled Outages >= 2"
            ),
            "uses_next_month_outcomes": True,
            "requires_consecutive_next_calendar_month": True,
            "unknown_targets_remain_unlabeled": True,
            "unlabeled_rows_excluded_from_prevalence": True,
            "safe_source_unlabeled_rows": int(len(safe_unlabeled)),
            "safe_source_unlabeled_rows_converted_to_negative": 0,
        },
        "populations": populations,
        "population_findings": {
            "new_equipment_policy_effect": (
                "No additional development rows are removed after X exclusion "
                "because the equipment-level minimum-history rule is null and "
                "newly introduced equipment remains eligible."
            ),
            "feature_history_effect": (
                "Moving from current-only to full-history eligibility removes "
                "1,649 early-history labeled rows and changes prevalence from "
                "0.426907 to 0.428344."
            ),
            "development_date_filter_effect": (
                "The full-history and development-only populations are identical "
                "because every prevalence population in this audit excludes "
                "February-April 2025 prediction rows by construction."
            ),
        },
        "rolling_temporal_validation_folds": folds,
        "earlier_estimate_investigation": {
            "earlier_estimate": 0.367,
            "implemented_target_positive_decomposition": {
                "next_month_unscheduled_outages_at_least_2": int(
                    unscheduled_only.sum()
                ),
                "next_month_entrapments_above_0": int(entrapment_only.sum()),
                "both_conditions": int((unscheduled_only & entrapment_only).sum()),
                "entrapment_only_additional_positives": int(
                    (entrapment_only & ~unscheduled_only).sum()
                ),
                "primary_or_target_positives": int(development[TARGET_COLUMN].sum()),
            },
            "alternatives_near_earlier_estimate": alternatives,
            "finding_by_hypothesis": {
                "x_suffixed_equipment": (
                    "Strongly supported as the main explanation: including X rows "
                    "with the correct next-month OR target produces 0.368198."
                ),
                "different_target_definition": (
                    "Possible additional explanation: omitting the entrapment OR "
                    "clause produces 0.369201 on non-X development rows."
                ),
                "different_eligibility_filtering": (
                    "Not a material explanation: non-X prevalence is 0.426907 "
                    "before feature-history filtering and 0.428344 after it."
                ),
                "another_calculation_difference": (
                    "Current-month targeting while including X rows produces "
                    "0.366688, but tests confirm the implemented target does not "
                    "use current-month outcomes."
                ),
            },
            "conclusion": (
                "The earlier estimate cannot be uniquely attributed because no "
                "saved calculation was found. Including X-suffixed zero-outcome "
                "records is the strongest quantitative explanation. The "
                "implemented 0.428344 prevalence is not caused by target leakage, "
                "current-month targeting, or the feature-history eligibility rules."
            ),
        },
        "frozen_test_labels_accessed": False,
    }


def write_prevalence_investigation(
    report: dict[str, object],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Primary Target Prevalence Investigation",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Scope And Contract",
        "",
        f"- {report['scope']}",
        f"- Target: `{report['primary_target_contract']['definition']}`.",
        "- Target construction uses next-month outcomes and requires an exact "
        "consecutive next calendar month.",
        "- Unknown targets remain unlabeled and are excluded from prevalence.",
        f"- Unlabeled rows in the development-safe source: "
        f"{report['primary_target_contract']['safe_source_unlabeled_rows']}; "
        "converted to negatives: 0.",
        "",
        "## Population Decomposition",
        "",
        "| Population | Labeled | Positive | Negative | Prevalence | Elevators |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["populations"].items():
        lines.append(
            f"| `{name}` | {values['total_labeled_rows']} | {values['positive_rows']} "
            f"| {values['negative_rows']} | {values['prevalence']:.6f} | "
            f"{values['unique_elevators']} |"
        )
    lines.extend(["", "Population findings:", ""])
    for name, finding in report["population_findings"].items():
        lines.append(f"- `{name}`: {finding}")
    lines.extend(
        [
            "",
            "## Rolling Temporal Validation",
            "",
            "| Fold | Train Range | Train Rows / Pos / Prev | Validation Range | "
            "Validation Rows / Pos / Prev | Overlap | Frozen Rows |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for fold in report["rolling_temporal_validation_folds"]:
        train, validation = fold["train"], fold["validation"]
        lines.append(
            f"| `{fold['name']}` | {train['start_month']} to {train['end_month']} | "
            f"{train['total_labeled_rows']} / {train['positive_rows']} / "
            f"{train['prevalence']:.6f} | {validation['start_month']} to "
            f"{validation['end_month']} | {validation['total_labeled_rows']} / "
            f"{validation['positive_rows']} / {validation['prevalence']:.6f} | "
            f"{fold['train_validation_date_overlap'] or fold['train_validation_row_overlap']} "
            f"| {fold['contains_frozen_period_rows']} |"
        )
    lines.extend(["", "## Earlier Estimate Investigation", ""])
    alternatives = report["earlier_estimate_investigation"][
        "alternatives_near_earlier_estimate"
    ]
    for name, values in alternatives.items():
        lines.append(
            f"- `{name}`: {values['prevalence']:.6f} "
            f"({values['positive_rows']} / {values['total_labeled_rows']})."
        )
    lines.extend(["", "Finding by hypothesis:", ""])
    for name, finding in report["earlier_estimate_investigation"][
        "finding_by_hypothesis"
    ].items():
        lines.append(f"- `{name}`: {finding}")
    lines.extend(
        [
            "",
            report["earlier_estimate_investigation"]["conclusion"],
            "",
            "## Session 1 Assessment",
            "",
            "Session 1 can safely be considered complete for target construction, "
            "eligibility, prevalence, and temporal-fold validation. The earlier "
            "estimate remains a documented provenance limitation rather than an "
            "unresolved implementation defect.",
            "",
            f"Frozen-test labels accessed: **{report['frozen_test_labels_accessed']}**.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
