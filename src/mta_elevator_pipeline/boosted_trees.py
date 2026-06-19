"""Focused development-only boosted-tree evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .models import build_tabular_models
from .session3 import _fit_predict, _variant_data
from .session4 import _focused_tuning, threshold_policy_report
from .splits import rolling_temporal_validation_splits


HIST_GRADIENT_BOOSTING_PARAMETER_SEARCH = [
    {"model__max_iter": 150, "model__learning_rate": 0.10, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 20, "model__l2_regularization": 0.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 20, "model__l2_regularization": 0.0, "model__max_bins": 255},
    {"model__max_iter": 500, "model__learning_rate": 0.03, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 20, "model__l2_regularization": 1.0, "model__max_bins": 255},
    {"model__max_iter": 150, "model__learning_rate": 0.10, "model__max_leaf_nodes": 7, "model__min_samples_leaf": 20, "model__l2_regularization": 1.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 7, "model__min_samples_leaf": 40, "model__l2_regularization": 1.0, "model__max_bins": 255},
    {"model__max_iter": 500, "model__learning_rate": 0.03, "model__max_leaf_nodes": 7, "model__min_samples_leaf": 40, "model__l2_regularization": 5.0, "model__max_bins": 255},
    {"model__max_iter": 150, "model__learning_rate": 0.10, "model__max_leaf_nodes": 31, "model__min_samples_leaf": 20, "model__l2_regularization": 1.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 31, "model__min_samples_leaf": 40, "model__l2_regularization": 5.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 10, "model__l2_regularization": 1.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 40, "model__l2_regularization": 5.0, "model__max_bins": 255},
    {"model__max_iter": 300, "model__learning_rate": 0.05, "model__max_leaf_nodes": 15, "model__min_samples_leaf": 20, "model__l2_regularization": 1.0, "model__max_bins": 63},
    {"model__max_iter": 500, "model__learning_rate": 0.03, "model__max_leaf_nodes": 31, "model__min_samples_leaf": 40, "model__l2_regularization": 5.0, "model__max_bins": 63},
]


def _approximately_70_policy(threshold_report: dict[str, object]) -> dict[str, object]:
    return next(
        item
        for item in threshold_report["policies"]
        if item["policy"] == "approximately_70_percent_recall"
    )


def _comparison_row(model_name: str, parameters: dict[str, object], metrics: dict[str, object], threshold_report: dict[str, object]) -> dict[str, object]:
    policy = _approximately_70_policy(threshold_report)
    latest = policy["latest_fold_metrics"]
    return {
        "model": model_name,
        "selected_parameters": parameters,
        "selection_folds_mean_pr_auc": float(
            sum(metrics["folds"][name]["pr_auc"] for name in metrics["threshold_selection_folds"])
            / len(metrics["threshold_selection_folds"])
        ),
        "four_fold_mean_pr_auc": metrics["summary_all_folds"]["pr_auc"]["mean"],
        "four_fold_std_pr_auc": metrics["summary_all_folds"]["pr_auc"]["std"],
        "worst_fold_pr_auc": metrics["summary_all_folds"]["pr_auc"]["min"],
        "fold_4_pr_auc": metrics["latest_fold"]["pr_auc"],
        "approximately_70_recall_threshold": policy["selected_threshold"],
        "fold_4_precision": latest["precision"],
        "fold_4_recall": latest["recall"],
        "fold_4_false_positives": latest["false_positives"],
        "fold_4_false_negatives": latest["false_negatives"],
        "fold_4_brier_score": metrics["latest_fold"]["brier_score"],
    }


def run_boosted_tree_evaluation(
    config: dict,
    prepared,
    session4_results: dict[str, object],
    frozen_log_exists: bool,
) -> dict[str, object]:
    development, features, minimum_history = _variant_data(
        config, prepared, "full_history", False
    )
    folds = rolling_temporal_validation_splits(
        development,
        config["split"]["validation_months"],
        config["split"]["rolling_validation_folds"],
    )
    base_model = build_tabular_models(
        features, [], config["project"]["random_seed"]
    )["hist_gradient_boosting"]
    tuning = _focused_tuning(
        base_model, HIST_GRADIENT_BOOSTING_PARAMETER_SEARCH, features, folds
    )
    selected_model = base_model.set_params(**tuning["selected_parameters"])
    _, _, predictions = _fit_predict(selected_model, features, folds)
    threshold_report = threshold_policy_report(predictions)

    comparison = []
    for model_name in ["logistic_regression", "random_forest"]:
        prior = session4_results["focused_tuning"][model_name]
        comparison.append(
            _comparison_row(
                model_name,
                prior["selected_parameters"],
                prior["selected_result"]["metrics"],
                session4_results["threshold_tradeoffs"][model_name],
            )
        )
    comparison.append(
        _comparison_row(
            "hist_gradient_boosting",
            tuning["selected_parameters"],
            tuning["selected_result"]["metrics"],
            threshold_report,
        )
    )
    comparison.sort(key=lambda item: item["selection_folds_mean_pr_auc"], reverse=True)

    return {
        "scope": "development-only; frozen-test labels and metrics excluded",
        "feature_set": "full_history_without_age",
        "minimum_consecutive_history_months": minimum_history,
        "cohort_rows": int(len(development)),
        "cohort_equipment": int(development["Equipment Code"].nunique()),
        "folds": [
            {
                "name": fold.name,
                "train_end": fold.train["Month"].max().strftime("%Y-%m-%d"),
                "validation_start": fold.validation["Month"].min().strftime("%Y-%m-%d"),
                "validation_end": fold.validation["Month"].max().strftime("%Y-%m-%d"),
            }
            for fold in folds
        ],
        "selection_policy": "Select parameters and approximately-70%-recall threshold using folds 1-3 only; evaluate unchanged on fold 4.",
        "hist_gradient_boosting_focused_tuning": tuning,
        "hist_gradient_boosting_threshold_tradeoffs": threshold_report,
        "external_challenger": {
            "implementation": "xgboost",
            "evaluated": False,
            "available": importlib.util.find_spec("xgboost") is not None,
            "reason": "XGBoost is not installed in the current environment; optional external evaluation was skipped without changing core dependencies.",
        },
        "tabular_comparison": comparison,
        "decision": "No final model decision made. Comparison is provided for review.",
        "guardrails": {
            "latest_fold_excluded_from_parameter_selection": True,
            "latest_fold_excluded_from_threshold_selection": True,
            "frozen_test_log_absent": not frozen_log_exists,
            "final_selection_record_changed": False,
        },
    }


def comparison_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Focused Development-Only Boosted-Tree Comparison",
        "",
        results["selection_policy"],
        "",
        "| Model | Folds 1-3 mean PR-AUC | Four-fold mean PR-AUC | Worst-fold PR-AUC | Fold 4 PR-AUC | Fold 4 precision | Fold 4 recall | Fold 4 FP | Fold 4 FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["tabular_comparison"]:
        lines.append(
            f"| {row['model']} | {row['selection_folds_mean_pr_auc']:.6f} | "
            f"{row['four_fold_mean_pr_auc']:.6f} | {row['worst_fold_pr_auc']:.6f} | "
            f"{row['fold_4_pr_auc']:.6f} | {row['fold_4_precision']:.6f} | "
            f"{row['fold_4_recall']:.6f} | {row['fold_4_false_positives']} | "
            f"{row['fold_4_false_negatives']} |"
        )
    lines.extend(
        [
            "",
            f"External challenger: {results['external_challenger']['reason']}",
            "",
            results["decision"],
            "",
            "Frozen-test labels and metrics were not accessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_boosted_tree_records(
    results: dict[str, object],
    metrics_path: Path,
    record_path: Path,
    table_path: Path,
) -> None:
    for path in [metrics_path, record_path, table_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
    table_path.write_text(comparison_markdown(results), encoding="utf-8")
