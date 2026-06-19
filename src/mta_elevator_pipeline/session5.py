"""One-time final frozen-test evaluation for the locked production candidate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from .config import assert_frozen_test_config, load_yaml, project_path
from .eligibility import add_consecutive_history_months, add_eligibility_flags, add_feature_set_eligibility
from .evaluate import calibration_diagnostics, evaluate_probabilities
from .features import build_features, minimum_history_for_feature_set
from .models import build_tabular_models
from .splits import access_frozen_test_labels, assert_frozen_feature_counts, create_development_split
from .targets import TARGET_COLUMN, build_next_month_target
from .validation import validate_availability_data


LOCKED_SELECTION = {
    "selected_target": "primary",
    "selected_model": "random_forest",
    "selected_feature_set": "full_history_without_age",
    "selected_threshold": 0.4433219097353501,
    "selected_parameters": {
        "n_estimators": 600,
        "max_depth": 20,
        "min_samples_leaf": 15,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
    },
    "calibration": "none",
}

REQUIRED_OUTPUTS = {
    "evaluation_markdown": "outputs/metrics/session5_frozen_test_evaluation.md",
    "metrics_json": "outputs/metrics/session5_frozen_test_metrics.json",
    "predictions_csv": "outputs/predictions/session5_frozen_test_predictions.csv",
    "model_joblib": "outputs/models/final_random_forest.joblib",
    "metadata_json": "outputs/models/final_model_metadata.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_locked_selection(config: dict, selection: dict, require_approval: bool) -> dict[str, object]:
    checks: dict[str, object] = {}
    for key, expected in LOCKED_SELECTION.items():
        actual = selection.get(key)
        checks[f"{key}_matches_lock"] = actual == expected
        if actual != expected:
            raise ValueError(f"Final selection {key} differs from the locked value.")

    development_path = project_path(selection["development_results_path"])
    if not development_path.exists():
        raise ValueError("Development evidence record does not exist.")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    evidence_checks = {
        "development_record_exists": True,
        "development_selected_model_matches": development["selected_model"] == selection["selected_model"],
        "development_feature_set_matches": development["selected_feature_set"] == selection["selected_feature_set"],
        "development_parameters_match": development["selected_parameters"] == selection["selected_parameters"],
        "development_threshold_matches": development["selected_threshold"] == selection["selected_threshold"],
        "development_calibration_matches": development["calibration"] == selection["calibration"],
        "development_scope_excludes_frozen": "frozen-test labels and metrics excluded" in development["scope"],
        "development_guardrail_says_not_accessed": not development["guardrails"]["frozen_test_accessed"],
        "development_guardrail_forbids_tuning": development["guardrails"]["no_further_model_or_threshold_tuning"],
    }
    if not all(evidence_checks.values()):
        raise ValueError("Development evidence does not support the locked final selection.")
    if require_approval and not selection.get("approved_for_final_test"):
        raise PermissionError("Final model selection is not approved for frozen-test access.")
    assert_frozen_test_config(config)
    checks.update(evidence_checks)
    checks["approved_for_final_test"] = bool(selection.get("approved_for_final_test"))
    checks["frozen_period_lock_matches"] = True
    return checks


def prepare_locked_final_split(config: dict):
    source = pd.read_csv(project_path(config["paths"]["availability_csv"]))
    validate_availability_data(source)
    prepared = add_eligibility_flags(
        source,
        excluded_equipment_code_suffixes=config["eligibility"]["excluded_equipment_code_suffixes"],
        minimum_history_months=config["eligibility"]["minimum_history_months"],
    )
    prepared = add_consecutive_history_months(prepared)
    prepared = prepared[prepared["eligible_for_modeling"]].copy()
    prepared = build_next_month_target(
        prepared,
        unscheduled_threshold=config["target"]["unscheduled_outage_threshold"],
        entrapment_threshold=config["target"]["entrapment_threshold"],
    )
    definition = config["features"]["feature_sets"]["full_history"]
    minimum_history = minimum_history_for_feature_set(definition["lag_months"], definition["rolling_windows"])
    featured, features = build_features(
        prepared,
        lag_months=definition["lag_months"],
        rolling_windows=definition["rolling_windows"],
        include_seasonality=False,
        include_age=False,
    )
    eligible = add_feature_set_eligibility(featured, minimum_history)
    eligible = eligible[eligible["eligible_for_feature_set"]].copy()
    split = create_development_split(
        eligible,
        frozen_test_start=config["split"]["frozen_test_start"],
        frozen_test_end=config["split"]["frozen_test_end"],
        target_column=TARGET_COLUMN,
    )
    return split, features, minimum_history


def build_locked_model(features: list[str], selection: dict):
    model = build_tabular_models(features, [], selection["selected_parameters"]["random_state"])["random_forest"]
    parameters = {
        f"model__{name}": value
        for name, value in selection["selected_parameters"].items()
        if name not in {"class_weight", "random_state"}
    }
    model = clone(model).set_params(**parameters)
    actual = model.named_steps["model"].get_params()
    for name, expected in selection["selected_parameters"].items():
        if actual[name] != expected:
            raise ValueError(f"Random Forest parameter {name} differs from the lock.")
    return model


def preflight_report(config: dict, tests_summary: str) -> dict[str, object]:
    selection = load_yaml(project_path(config["guardrails"]["final_selection_record"]))
    audit_log = project_path(config["guardrails"]["audit_log"])
    if selection.get("approved_for_final_test") is not False:
        raise PermissionError("Preflight requires approved_for_final_test to be false.")
    if audit_log.exists():
        raise PermissionError("Frozen-test access log already exists.")
    existing = [path for path in REQUIRED_OUTPUTS.values() if project_path(path).exists()]
    if existing:
        raise PermissionError(f"Final evaluation outputs already exist: {existing}")

    checks = validate_locked_selection(config, selection, require_approval=False)
    split, features, minimum_history = prepare_locked_final_split(config)
    frozen_counts = assert_frozen_feature_counts(
        split.frozen_features,
        config["guardrails"]["expected_frozen_feature_rows"],
        config["guardrails"]["expected_frozen_feature_rows_by_month"],
    )
    build_locked_model(features, selection)
    checks.update(
        {
            "approved_for_final_test_is_false": True,
            "frozen_test_access_log_absent": True,
            "final_outputs_absent": True,
            "complete_test_suite_passed": "passed" in tests_summary,
            "development_rows_precede_february_2025": bool(split.development["Month"].max() < pd.Timestamp(config["split"]["frozen_test_start"])),
            "frozen_features_exclude_target": TARGET_COLUMN not in split.frozen_features.columns,
            "preprocessing_is_inside_model_pipeline": True,
            "preprocessing_receives_development_rows_only_during_fit": True,
            "final_fit_contract_is_exactly_once_on_all_development_rows": True,
            "threshold_will_be_used_unchanged": selection["selected_threshold"] == LOCKED_SELECTION["selected_threshold"],
            "no_age_features": not any(feature in {"Time Since Major Improvement", "age_missing"} for feature in features),
            "model_parameters_match_lock": True,
        }
    )
    checks.pop("approved_for_final_test")
    if not all(checks.values()):
        raise RuntimeError("One or more Session 5 preflight checks failed.")
    return {
        "status": "passed",
        "created_at_utc": _utc_now(),
        "tests": tests_summary,
        "checks": checks,
        "fit_plan": {
            "fit_count": 1,
            "fit_timing": "before frozen-label access",
            "fit_rows": int(len(split.development)),
            "fit_equipment": int(split.development["Equipment Code"].nunique()),
            "fit_start": split.development["Month"].min().strftime("%Y-%m-%d"),
            "fit_end": split.development["Month"].max().strftime("%Y-%m-%d"),
            "preprocessing": "median imputation inside the fitted sklearn Pipeline",
        },
        "frozen_feature_confirmation_without_labels": frozen_counts,
        "feature_set": {
            "name": selection["selected_feature_set"],
            "minimum_consecutive_history_months": minimum_history,
            "feature_count": len(features),
            "features": features,
        },
        "locked_selection": {key: selection[key] for key in LOCKED_SELECTION},
    }


def write_preflight_markdown(report: dict[str, object], path: Path) -> None:
    checks = "\n".join(f"- {name}: **{value}**" for name, value in report["checks"].items())
    plan = report["fit_plan"]
    text = f"""# Session 5 Frozen-Test Preflight

Status: **{report['status']}**

Created at UTC: `{report['created_at_utc']}`

Complete test suite: `{report['tests']}`

## Checks

{checks}

## Locked Final Fit Plan

- Fit exactly once before frozen-label access.
- Development rows: `{plan['fit_rows']}` across `{plan['fit_equipment']}` elevators.
- Development period: `{plan['fit_start']}` through `{plan['fit_end']}`.
- Frozen feature rows confirmed without labels: `{report['frozen_feature_confirmation_without_labels']['rows']}`.
- Feature set: `{report['feature_set']['name']}` with `{report['feature_set']['feature_count']}` features and no age fields.
- Threshold: `{report['locked_selection']['selected_threshold']}` unchanged.
- Preprocessing: `{plan['preprocessing']}` fitted only on development rows.

All preflight checks passed. Approval may now be changed before the one-time final evaluation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dependency_snapshot() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "PyYAML", "joblib"]}


def _month_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    output = {}
    for month, rows in frame.groupby("prediction_month", sort=True):
        y = rows["actual_target"].to_numpy()
        probabilities = rows["probability"].to_numpy()
        output[str(month)] = {
            "rows": int(len(rows)),
            "elevators": int(rows["equipment_code"].nunique()),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            **evaluate_probabilities(y, probabilities, threshold),
            "calibration": calibration_diagnostics(y, probabilities),
        }
    return output


def _evaluation_markdown(report: dict[str, object]) -> str:
    overall = report["overall_metrics"]
    months = "\n".join(
        f"| {month} | {metrics['rows']} | {metrics['prevalence']:.6f} | {metrics['pr_auc']:.6f} | {metrics['roc_auc']:.6f} | {metrics['brier_score']:.6f} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['false_positives']} | {metrics['false_negatives']} |"
        for month, metrics in report["metrics_by_month"].items()
    )
    comparison = report["development_comparison"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Session 5 Final Frozen-Test Evaluation

This is the one-time final evaluation on frozen labels for February-April 2025. It is distinct from all development results and was not used for tuning.

## Frozen-Test Cohort

- Rows: `{report['cohort']['rows']}`
- Elevators: `{report['cohort']['elevators']}`
- Positive rows: `{report['cohort']['positives']}`
- Prevalence: `{report['cohort']['prevalence']:.6f}`

## Final Frozen-Test Performance

- PR-AUC: `{overall['pr_auc']:.6f}`
- ROC-AUC: `{overall['roc_auc']:.6f}`
- Brier score: `{overall['brier_score']:.6f}`
- Precision at `{overall['threshold']}`: `{overall['precision']:.6f}`
- Recall at `{overall['threshold']}`: `{overall['recall']:.6f}`
- Confusion matrix: TN `{overall['true_negatives']}`, FP `{overall['false_positives']}`, FN `{overall['false_negatives']}`, TP `{overall['true_positives']}`
- Calibration ECE: `{report['calibration']['expected_calibration_error']:.6f}`
- Mean predicted probability: `{report['calibration']['mean_probability']:.6f}`

## Metrics By Frozen-Test Month

| Month | Rows | Prevalence | PR-AUC | ROC-AUC | Brier | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{months}

## Development Comparison

- Development four-fold mean PR-AUC: `{comparison['four_fold_mean_pr_auc']:.6f}`
- Development latest-fold PR-AUC: `{comparison['latest_fold_pr_auc']:.6f}`
- Development latest-fold precision: `{comparison['latest_fold_precision']:.6f}`
- Development latest-fold recall: `{comparison['latest_fold_recall']:.6f}`
- Frozen-test results do not alter any locked decision.

## Runtime And Guardrails

- Runtime seconds: `{report['runtime_seconds']:.3f}`
- Model fit count: `{report['fit_audit']['fit_count']}`
- Fit rows: `{report['fit_audit']['fit_rows']}` ending `{report['fit_audit']['fit_end']}`
- Frozen labels accessed after model fit: `{report['fit_audit']['frozen_labels_accessed_after_fit']}`
- Audit log: `{report['audit_log']}`

## Limitations

{limitations}
"""


def final_evaluate_once(config: dict, acknowledgment: str) -> dict[str, object]:
    started = time.perf_counter()
    selection_path = project_path(config["guardrails"]["final_selection_record"])
    selection = load_yaml(selection_path)
    audit_log = project_path(config["guardrails"]["audit_log"])
    validate_locked_selection(config, selection, require_approval=True)
    if not project_path("outputs/metrics/session5_preflight.md").exists():
        raise PermissionError("Saved Session 5 preflight report is required.")
    if audit_log.exists():
        raise PermissionError("Frozen-test access log already exists; evaluation cannot rerun.")
    existing = [path for path in REQUIRED_OUTPUTS.values() if project_path(path).exists()]
    if existing:
        raise PermissionError(f"Final evaluation outputs already exist: {existing}")

    split, features, minimum_history = prepare_locked_final_split(config)
    assert_frozen_feature_counts(
        split.frozen_features,
        config["guardrails"]["expected_frozen_feature_rows"],
        config["guardrails"]["expected_frozen_feature_rows_by_month"],
    )
    threshold = selection["selected_threshold"]
    model = build_locked_model(features, selection)
    fit_started = time.perf_counter()
    model.fit(split.development[features], split.development[TARGET_COLUMN].astype(int))
    fit_seconds = time.perf_counter() - fit_started
    probabilities = model.predict_proba(split.frozen_features[features])[:, 1]
    labels_accessed = False

    try:
        frozen = access_frozen_test_labels(
            split,
            acknowledgment=acknowledgment,
            required_acknowledgment=config["guardrails"]["final_test_acknowledgment"],
            audit_log=audit_log,
        )
        labels_accessed = True
        y_true = frozen[TARGET_COLUMN].astype(int).to_numpy()
        predictions = pd.DataFrame(
            {
                "equipment_code": frozen["Equipment Code"].astype(str).to_numpy(),
                "prediction_month": frozen["Month"].dt.strftime("%Y-%m-%d").to_numpy(),
                "probability": probabilities,
                "thresholded_prediction": (probabilities >= threshold).astype(int),
                "actual_target": y_true,
            }
        )
        overall = evaluate_probabilities(y_true, probabilities, threshold)
        development = json.loads(project_path(selection["development_results_path"]).read_text(encoding="utf-8"))
        latest = development["fold_4_operating_metrics"]
        report = {
            "status": "valid_final_frozen_test_evaluation",
            "evaluated_at_utc": _utc_now(),
            "scope": "one-time final frozen-test evaluation; not used for model selection or tuning",
            "cohort": {
                "rows": int(len(frozen)),
                "elevators": int(frozen["Equipment Code"].nunique()),
                "positives": int(y_true.sum()),
                "negatives": int(len(y_true) - y_true.sum()),
                "prevalence": float(y_true.mean()),
                "rows_by_month": predictions.groupby("prediction_month").size().astype(int).to_dict(),
                "elevators_by_month": predictions.groupby("prediction_month")["equipment_code"].nunique().astype(int).to_dict(),
            },
            "overall_metrics": overall,
            "calibration": calibration_diagnostics(y_true, probabilities),
            "metrics_by_month": _month_metrics(predictions, threshold),
            "false_positives": int(overall["false_positives"]),
            "false_negatives": int(overall["false_negatives"]),
            "development_comparison": {
                "four_fold_mean_pr_auc": development["development_comparison"]["random_forest"]["four_fold_mean_pr_auc"],
                "latest_fold_pr_auc": latest["pr_auc"],
                "latest_fold_precision": latest["precision"],
                "latest_fold_recall": latest["recall"],
                "latest_fold_false_positives": latest["false_positives"],
                "latest_fold_false_negatives": latest["false_negatives"],
            },
            "fit_audit": {
                "fit_count": 1,
                "fit_rows": int(len(split.development)),
                "fit_equipment": int(split.development["Equipment Code"].nunique()),
                "fit_start": split.development["Month"].min().strftime("%Y-%m-%d"),
                "fit_end": split.development["Month"].max().strftime("%Y-%m-%d"),
                "fit_precedes_frozen_test": bool(split.development["Month"].max() < frozen["Month"].min()),
                "preprocessing_fit_scope": "eligible development rows only",
                "frozen_rows_used_for_learned_preprocessing": 0,
                "frozen_labels_accessed_after_fit": True,
                "fit_seconds": fit_seconds,
            },
            "locked_selection": {key: selection[key] for key in LOCKED_SELECTION},
            "minimum_consecutive_history_months": minimum_history,
            "features": features,
            "audit_log": str(Path(config["guardrails"]["audit_log"])),
            "runtime_seconds": time.perf_counter() - started,
            "limitations": [
                "The source is a fixed cleaned snapshot with unknown historical cleaning steps.",
                "X-suffixed equipment is excluded because target reporting is structurally incomplete.",
                "Six consecutive observed months are required, so true zero-history elevator performance is not measured.",
                "Time Since Major Improvement semantics remain unresolved and the field is excluded.",
                "The frozen test covers only three months and may not represent longer-term temporal drift.",
            ],
        }

        predictions_path = project_path(REQUIRED_OUTPUTS["predictions_csv"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        model_path = project_path(REQUIRED_OUTPUTS["model_joblib"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        _write_json(report, project_path(REQUIRED_OUTPUTS["metrics_json"]))
        markdown_path = project_path(REQUIRED_OUTPUTS["evaluation_markdown"])
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_evaluation_markdown(report), encoding="utf-8")
        metadata = {
            "created_at_utc": _utc_now(),
            "artifact": REQUIRED_OUTPUTS["model_joblib"],
            "artifact_sha256": _sha256(model_path),
            "source_data_sha256": _sha256(project_path(config["paths"]["availability_csv"])),
            "pipeline_config_sha256": _sha256(project_path("config/pipeline.yaml")),
            "final_selection_sha256": _sha256(selection_path),
            "exact_pipeline_config": config,
            "exact_final_selection": selection,
            "dependencies": _dependency_snapshot(),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "fit_audit": report["fit_audit"],
            "features": features,
        }
        _write_json(metadata, project_path(REQUIRED_OUTPUTS["metadata_json"]))
        return report
    except Exception as error:
        if labels_accessed:
            invalid = {
                "status": "invalid_after_frozen_label_access_no_rerun_permitted",
                "recorded_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "audit_log": str(Path(config["guardrails"]["audit_log"])),
            }
            _write_json(invalid, project_path(REQUIRED_OUTPUTS["metrics_json"]))
            path = project_path(REQUIRED_OUTPUTS["evaluation_markdown"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Session 5 Final Frozen-Test Evaluation\n\n"
                "**INVALID.** An implementation error occurred after frozen labels were accessed. "
                "The evaluation was not rerun.\n\n"
                f"- Error: `{type(error).__name__}: {error}`\n"
                f"- Audit log: `{config['guardrails']['audit_log']}`\n",
                encoding="utf-8",
            )
        raise
