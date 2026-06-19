"""Command-line entry point for validation and development model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import assert_frozen_test_config, load_config, load_yaml, project_path
from .boosted_trees import run_boosted_tree_evaluation, write_boosted_tree_records
from .eligibility import (
    add_consecutive_history_months,
    add_eligibility_flags,
    add_feature_set_eligibility,
)
from .evaluate import evaluate_probabilities
from .features import build_features, minimum_history_for_feature_set
from .models import build_tabular_models
from .prevalence_audit import (
    build_prevalence_investigation,
    write_prevalence_investigation,
)
from .reports import write_audit_markdown, write_json_report
from .session2 import run_session2_experiments, write_session2_records
from .session3 import run_session3_experiments, write_session3_records
from .session4 import run_session4_experiments, write_session4_records
from .session4_2 import run_session4_2_experiment, write_session4_2_records
from .session4_3 import run_session4_3_experiment, write_session4_3_records
from .session5 import final_evaluate_once, preflight_report, write_preflight_markdown
from .session6 import run_session6_prospective_evaluation
from .session7 import run_session7_final_backend_reporting
from .splits import (
    access_frozen_test_labels,
    assert_frozen_feature_counts,
    create_development_split,
    rolling_temporal_validation_splits,
)
from .targets import TARGET_COLUMN, build_next_month_target
from .validation import validate_availability_data


def load_source(config: dict) -> pd.DataFrame:
    source = project_path(config["paths"]["availability_csv"])
    if not source.exists():
        raise FileNotFoundError(
            f"Required source file not found: {source}. See data/README.md."
        )
    return pd.read_csv(source)


def selected_feature_definition(config: dict) -> tuple[list[int], list[int], int]:
    selected = config["features"]["selected_feature_set"]
    definition = config["features"]["feature_sets"][selected]
    lag_months = definition["lag_months"]
    rolling_windows = definition["rolling_windows"]
    return (
        lag_months,
        rolling_windows,
        minimum_history_for_feature_set(lag_months, rolling_windows),
    )


def prepare_target(
    config: dict,
    unscheduled_threshold: int,
) -> tuple[object, list[str]]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"][
            "excluded_equipment_code_suffixes"
        ],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    audited = add_consecutive_history_months(audited)
    eligible = audited[audited["eligible_for_modeling"]].copy()
    targeted = build_next_month_target(
        eligible,
        unscheduled_threshold=unscheduled_threshold,
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    lag_months, rolling_windows, minimum_history = selected_feature_definition(config)
    featured, feature_columns = build_features(
        targeted,
        lag_months=lag_months,
        rolling_windows=rolling_windows,
        include_seasonality=config["features"]["include_seasonality"],
    )
    featured = add_feature_set_eligibility(featured, minimum_history)
    featured = featured[featured["eligible_for_feature_set"]].copy()
    split = create_development_split(
        featured,
        frozen_test_start=config["split"]["frozen_test_start"],
        frozen_test_end=config["split"]["frozen_test_end"],
        target_column=TARGET_COLUMN,
    )
    return split, feature_columns


def prepare(config: dict) -> tuple[object, list[str]]:
    return prepare_target(config, config["target"]["unscheduled_outage_threshold"])


def target_prevalence_report(config: dict, name: str, threshold: int) -> dict[str, object]:
    split, _ = prepare_target(config, threshold)
    development = split.development
    positives = int(development[TARGET_COLUMN].sum())
    return {
        "target": name,
        "unscheduled_outage_threshold": threshold,
        "entrapment_threshold": config["target"]["entrapment_threshold"],
        "scope": "development rows eligible for selected feature set; frozen labels excluded",
        "labeled_rows": int(len(development)),
        "positive_rows": positives,
        "negative_rows": int(len(development) - positives),
        "prevalence": float(development[TARGET_COLUMN].mean()),
    }


def targets_command(config: dict) -> dict[str, object]:
    reports = {}
    output = project_path(config["paths"]["outputs_dir"]) / "metrics"
    for name, section in [
        ("primary", config["target"]),
        ("secondary", config["secondary_target"]),
    ]:
        report = target_prevalence_report(
            config, name, section["unscheduled_outage_threshold"]
        )
        reports[name] = report
        write_json_report(report, output / f"{name}_target_prevalence.json")
    print(json.dumps(reports, indent=2))
    return reports


def prevalence_audit_command(config: dict) -> dict[str, object]:
    report = build_prevalence_investigation(load_source(config), config)
    output = project_path(config["paths"]["outputs_dir"]) / "audits"
    write_prevalence_investigation(
        report,
        output / "primary_target_prevalence_investigation.json",
        output / "primary_target_prevalence_investigation.md",
    )
    print(json.dumps(report, indent=2))
    print(f"Saved development-only prevalence investigation to {output}")
    return report


def validate_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    source_report = validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"][
            "excluded_equipment_code_suffixes"
        ],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    audited = add_consecutive_history_months(audited)
    equipment = audited["Equipment Code"].astype(str)
    x_suffix = equipment.str.endswith(
        tuple(config["eligibility"]["excluded_equipment_code_suffixes"]), na=False
    )
    non_x = audited[~x_suffix].copy()
    first_observation = non_x.groupby("Equipment Code").cumcount() == 0
    missing_age = non_x["Time Since Major Improvement"].isna()

    feature_history = {}
    for name, definition in config["features"]["feature_sets"].items():
        minimum = minimum_history_for_feature_set(
            definition["lag_months"], definition["rolling_windows"]
        )
        eligible = non_x["consecutive_history_months"] >= minimum
        feature_history[name] = {
            "minimum_consecutive_months": minimum,
            "eligible_rows": int(eligible.sum()),
            "eligible_equipment": int(non_x.loc[eligible, "Equipment Code"].nunique()),
        }

    target_eligible = build_next_month_target(
        non_x,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    target_eligible_split = create_development_split(
        target_eligible,
        frozen_test_start=config["split"]["frozen_test_start"],
        frozen_test_end=config["split"]["frozen_test_end"],
        target_column=TARGET_COLUMN,
    )
    target_eligible_confirmation = assert_frozen_feature_counts(
        target_eligible_split.frozen_features,
        expected_rows=config["guardrails"]["expected_frozen_target_eligible_rows"],
        expected_rows_by_month=config["guardrails"][
            "expected_frozen_target_eligible_rows_by_month"
        ],
    )
    split, _ = prepare(config)
    frozen_confirmation = assert_frozen_feature_counts(
        split.frozen_features,
        expected_rows=config["guardrails"]["expected_frozen_feature_rows"],
        expected_rows_by_month=config["guardrails"][
            "expected_frozen_feature_rows_by_month"
        ],
    )
    folds = rolling_temporal_validation_splits(
        split.development,
        validation_months=config["split"]["validation_months"],
        fold_count=config["split"]["rolling_validation_folds"],
    )
    target_reports = targets_command(config)
    report = {
        "status": "passed",
        "source_validation": source_report,
        "eligibility": {
            "x_suffix_rows": int(x_suffix.sum()),
            "x_suffix_equipment": int(audited.loc[x_suffix, "Equipment Code"].nunique()),
            "eligible_rows": int(audited["eligible_for_modeling"].sum()),
            "eligible_equipment": int(
                audited.loc[audited["eligible_for_modeling"], "Equipment Code"].nunique()
            ),
            "newly_introduced_equipment": int(
                (
                    non_x.groupby("Equipment Code")["Month"].min()
                    > pd.to_datetime(non_x["Month"]).min()
                ).sum()
            ),
        },
        "feature_set_history": feature_history,
        "missing_age": {
            "missing_non_x_rows": int(missing_age.sum()),
            "affected_equipment": int(non_x.loc[missing_age, "Equipment Code"].nunique()),
            "all_on_first_observed_record": bool((~missing_age | first_observation).all()),
            "strategy": config["features"]["age_missing_strategy"],
        },
        "rolling_validation_folds": [
            {
                "name": fold.name,
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.validation)),
            }
            for fold in folds
        ],
        "frozen_test_count_confirmation": {
            **frozen_confirmation,
            "target_eligible_rows_before_feature_history_filter": (
                target_eligible_confirmation["rows"]
            ),
            "target_eligible_rows_by_month": target_eligible_confirmation[
                "rows_by_month"
            ],
        },
        "development_target_prevalence": target_reports,
    }
    output = project_path(config["paths"]["outputs_dir"]) / "audits"
    write_json_report(report, output / "real_data_audit.json")
    write_audit_markdown(report, output / "real_data_audit.md")
    print(json.dumps(report, indent=2))
    print(f"Saved formal audit reports to {output}")
    return report


def train_command(config: dict) -> None:
    split, features = prepare(config)
    folds = rolling_temporal_validation_splits(
        split.development,
        validation_months=config["split"]["validation_months"],
        fold_count=config["split"]["rolling_validation_folds"],
    )
    categorical = ["month_of_year", "is_winter"]
    numeric = [column for column in features if column not in categorical]
    models = build_tabular_models(
        numeric, categorical, config["project"]["random_seed"]
    )

    results = {}
    for name, model in models.items():
        results[name] = {}
        for fold in folds:
            model.fit(fold.train[features], fold.train[TARGET_COLUMN].astype(int))
            probabilities = model.predict_proba(fold.validation[features])[:, 1]
            results[name][fold.name] = evaluate_probabilities(
                fold.validation[TARGET_COLUMN].astype(int).to_numpy(), probabilities
            )

    output = project_path(config["paths"]["outputs_dir"]) / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rolling_development_validation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved development-only metrics to {path}")


def session2_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"][
            "excluded_equipment_code_suffixes"
        ],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    results = run_session2_experiments(config, prepared)
    metrics_path = (
        project_path(config["paths"]["outputs_dir"])
        / "metrics"
        / "session2_development_validation.json"
    )
    record_path = (
        project_path("experiments")
        / "feature_sets"
        / "session2_experiment_record.json"
    )
    write_session2_records(results, metrics_path, record_path)
    print(json.dumps(results["recommendation"] if "recommendation" in results else {}, indent=2))
    print(f"Saved development-only Session 2 metrics to {metrics_path}")
    print(f"Saved controlled experiment record to {record_path}")
    return results


def session3_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    results = run_session3_experiments(config, prepared)
    metrics_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session3_development_validation.json"
    record_path = project_path("experiments") / "model_selection" / "session3_experiment_record.json"
    write_session3_records(results, metrics_path, record_path)
    print(json.dumps(results["recommendation"], indent=2))
    print(f"Saved development-only Session 3 metrics to {metrics_path}")
    print(f"Saved controlled experiment record to {record_path}")
    return results


def session4_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    audit_log = project_path(config["guardrails"]["audit_log"])
    results = run_session4_experiments(config, prepared, frozen_log_exists=audit_log.exists())
    metrics_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_development_validation.json"
    record_path = project_path("experiments") / "model_selection" / "session4_experiment_record.json"
    write_session4_records(results, metrics_path, record_path)
    print(json.dumps(results["recommendation"], indent=2))
    print(f"Saved development-only Session 4 metrics to {metrics_path}")
    print(f"Saved controlled experiment record to {record_path}")
    return results


def boosted_trees_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    session4_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_development_validation.json"
    session4_results = json.loads(session4_path.read_text(encoding="utf-8"))
    audit_log = project_path(config["guardrails"]["audit_log"])
    results = run_boosted_tree_evaluation(
        config, prepared, session4_results, frozen_log_exists=audit_log.exists()
    )
    metrics_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "boosted_tree_development_validation.json"
    record_path = project_path("experiments") / "model_selection" / "boosted_tree_experiment_record.json"
    table_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "boosted_tree_comparison.md"
    write_boosted_tree_records(results, metrics_path, record_path, table_path)
    print(table_path.read_text(encoding="utf-8"))
    return results


def session4_2_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    audit_log = project_path(config["guardrails"]["audit_log"])
    results = run_session4_2_experiment(config, prepared, frozen_log_exists=audit_log.exists())
    metrics_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_2_tcn_development_validation.json"
    comparison_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_2_tcn_comparison.md"
    record_path = project_path("experiments") / "deep_learning" / "session4_2_tcn_experiment_record.json"
    write_session4_2_records(results, metrics_path, record_path, comparison_path)
    print(comparison_path.read_text(encoding="utf-8"))
    return results


def session4_3_command(config: dict) -> dict[str, object]:
    source = load_source(config)
    validate_availability_data(source)
    audited = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(audited)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    audit_log = project_path(config["guardrails"]["audit_log"])
    results = run_session4_3_experiment(
        config, prepared, frozen_log_exists=audit_log.exists()
    )
    metrics_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_3_xgboost_development_validation.json"
    comparison_path = project_path(config["paths"]["outputs_dir"]) / "metrics" / "session4_3_xgboost_comparison.md"
    record_path = project_path("experiments") / "model_selection" / "session4_3_xgboost_experiment_record.json"
    write_session4_3_records(results, metrics_path, record_path, comparison_path)
    print(comparison_path.read_text(encoding="utf-8"))
    return results


def session5_preflight_command(config: dict, tests_summary: str) -> dict[str, object]:
    report = preflight_report(config, tests_summary)
    path = project_path("outputs/metrics/session5_preflight.md")
    write_preflight_markdown(report, path)
    print(path.read_text(encoding="utf-8"))
    return report


def final_evaluate_command(config: dict, acknowledgment: str) -> dict[str, object]:
    report = final_evaluate_once(config, acknowledgment)
    print(json.dumps(report["overall_metrics"], indent=2))
    return report


def session6_command(config: dict) -> dict[str, object]:
    report = run_session6_prospective_evaluation(config)
    if report["prospective_data_available"]:
        print(json.dumps(report["overall_metrics"], indent=2))
    else:
        print("No target-known prospective rows after 2025-05-01.")
    print("Saved Session 6 prospective outputs.")
    return report


def session7_command(config: dict) -> dict[str, object]:
    report = run_session7_final_backend_reporting(config)
    print(json.dumps(report["latest_predictions"], indent=2))
    print("Saved Session 7 final backend reports and latest predictions.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "targets",
            "prevalence-audit",
            "train",
            "session2",
            "session3",
            "session4",
            "session4-2",
            "session4-3",
            "boosted-trees",
            "session5-preflight",
            "final-evaluate",
            "session6",
            "session7",
        ],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--acknowledge", default="")
    parser.add_argument("--tests-summary", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    assert_frozen_test_config(config)

    if args.command == "validate":
        validate_command(config)
    elif args.command == "targets":
        targets_command(config)
    elif args.command == "prevalence-audit":
        prevalence_audit_command(config)
    elif args.command == "train":
        train_command(config)
    elif args.command == "session2":
        session2_command(config)
    elif args.command == "session3":
        session3_command(config)
    elif args.command == "session4":
        session4_command(config)
    elif args.command == "session4-2":
        session4_2_command(config)
    elif args.command == "session4-3":
        session4_3_command(config)
    elif args.command == "boosted-trees":
        boosted_trees_command(config)
    elif args.command == "session5-preflight":
        session5_preflight_command(config, args.tests_summary)
    elif args.command == "final-evaluate":
        final_evaluate_command(config, args.acknowledge)
    elif args.command == "session6":
        session6_command(config)
    else:
        session7_command(config)


if __name__ == "__main__":
    main()
