"""Session 4.3 development-only XGBoost challenger experiment."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .evaluate import calibration_diagnostics, evaluate_probabilities
from .models import build_tabular_models
from .session3 import _fit_predict, _fold_metadata, _preprocessing_audit, _variant_data
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN


XGBOOST_PARAMETER_SEARCH = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 400, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.0, "reg_lambda": 5.0},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 700, "min_child_weight": 10, "subsample": 0.90, "colsample_bytree": 0.80, "reg_alpha": 0.0, "reg_lambda": 10.0},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.0, "reg_lambda": 5.0},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 700, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 10.0},
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 400, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 10.0},
    {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 700, "min_child_weight": 20, "subsample": 0.80, "colsample_bytree": 0.70, "reg_alpha": 1.0, "reg_lambda": 15.0},
    {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 700, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 1.0, "reg_lambda": 10.0},
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 700, "min_child_weight": 10, "subsample": 1.00, "colsample_bytree": 1.00, "reg_alpha": 0.0, "reg_lambda": 5.0},
]

RANDOM_FOREST_PARAMETERS = {
    "model__n_estimators": 600,
    "model__max_depth": 20,
    "model__min_samples_leaf": 15,
    "model__max_features": "sqrt",
}
MATERIAL_PR_AUC_IMPROVEMENT = 0.01


def select_xgboost_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Select only from explicitly stored folds-1-3 metrics."""
    return max(
        candidates,
        key=lambda item: (
            item["selection_folds_mean_pr_auc"],
            item["selection_folds_worst_pr_auc"],
            -item["selection_folds_pr_auc_std"],
        ),
    )


def _build_xgboost_pipeline(features: list[str], parameters: dict[str, object], seed: int):
    from xgboost import XGBClassifier

    preprocess = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), features)],
        remainder="drop",
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=seed,
        n_jobs=-1,
        **parameters,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _selection_fold_evaluation(model, features, folds) -> dict[str, object]:
    fold_results = {}
    preprocessing = {}
    started = perf_counter()
    for fold in folds[:3]:
        fitted = clone(model).fit(
            fold.train[features], fold.train[TARGET_COLUMN].astype(int)
        )
        y_true = fold.validation[TARGET_COLUMN].astype(int).to_numpy()
        probabilities = fitted.predict_proba(fold.validation[features])[:, 1]
        fold_results[fold.name] = {
            **_fold_metadata(fold),
            **evaluate_probabilities(y_true, probabilities, threshold=0.5),
            "calibration": calibration_diagnostics(y_true, probabilities),
        }
        preprocessing[fold.name] = _preprocessing_audit(fitted, fold)
    pr_auc = np.asarray([item["pr_auc"] for item in fold_results.values()])
    return {
        "selection_folds": list(fold_results),
        "fold_metrics": fold_results,
        "preprocessing_audit": preprocessing,
        "selection_folds_mean_pr_auc": float(pr_auc.mean()),
        "selection_folds_pr_auc_std": float(pr_auc.std(ddof=0)),
        "selection_folds_worst_pr_auc": float(pr_auc.min()),
        "selection_folds_mean_brier_score": float(
            np.mean([item["brier_score"] for item in fold_results.values()])
        ),
        "selection_folds_mean_expected_calibration_error": float(
            np.mean(
                [
                    item["calibration"]["expected_calibration_error"]
                    for item in fold_results.values()
                ]
            )
        ),
        "runtime_seconds": float(perf_counter() - started),
        "fold_4_evaluated": False,
    }


def _comparison_row(name, parameters, metrics, runtime_seconds):
    folds = metrics["folds"]
    selection_names = metrics["threshold_selection_folds"]
    return {
        "model": name,
        "selected_parameters": parameters,
        "selection_folds_mean_pr_auc": float(
            np.mean([folds[name]["pr_auc"] for name in selection_names])
        ),
        "four_fold_mean_pr_auc": metrics["summary_all_folds"]["pr_auc"]["mean"],
        "pr_auc_standard_deviation": metrics["summary_all_folds"]["pr_auc"]["std"],
        "worst_fold_pr_auc": metrics["summary_all_folds"]["pr_auc"]["min"],
        "fold_4_pr_auc": metrics["latest_fold"]["pr_auc"],
        "mean_roc_auc": metrics["summary_all_folds"]["roc_auc"]["mean"],
        "mean_brier_score": metrics["summary_all_folds"]["brier_score"]["mean"],
        "approximately_70_recall_threshold": metrics["selected_threshold"],
        "fold_4_precision": metrics["latest_fold"]["precision"],
        "fold_4_recall": metrics["latest_fold"]["recall"],
        "fold_4_false_positives": metrics["latest_fold"]["false_positives"],
        "fold_4_false_negatives": metrics["latest_fold"]["false_negatives"],
        "runtime_seconds": float(runtime_seconds),
    }


def run_session4_3_experiment(config: dict, prepared: pd.DataFrame, frozen_log_exists: bool):
    import xgboost

    development, features, minimum_history = _variant_data(
        config, prepared, "full_history", False
    )
    folds = rolling_temporal_validation_splits(
        development,
        config["split"]["validation_months"],
        config["split"]["rolling_validation_folds"],
    )

    candidates = []
    for parameters in XGBOOST_PARAMETER_SEARCH:
        model = _build_xgboost_pipeline(
            features, parameters, config["project"]["random_seed"]
        )
        selection = _selection_fold_evaluation(model, features, folds)
        candidates.append({"parameters": parameters, **selection})
    selected = select_xgboost_candidate(candidates)

    xgboost_model = _build_xgboost_pipeline(
        features, selected["parameters"], config["project"]["random_seed"]
    )
    started = perf_counter()
    xgboost_metrics, xgboost_preprocessing, _ = _fit_predict(
        xgboost_model, features, folds
    )
    xgboost_runtime = perf_counter() - started

    random_forest = build_tabular_models(
        features, [], config["project"]["random_seed"]
    )["random_forest"].set_params(**RANDOM_FOREST_PARAMETERS)
    started = perf_counter()
    random_forest_metrics, random_forest_preprocessing, _ = _fit_predict(
        random_forest, features, folds
    )
    random_forest_runtime = perf_counter() - started

    comparison = [
        _comparison_row(
            "xgboost", selected["parameters"], xgboost_metrics, xgboost_runtime
        ),
        _comparison_row(
            "random_forest",
            RANDOM_FOREST_PARAMETERS,
            random_forest_metrics,
            random_forest_runtime,
        ),
    ]
    xgb_row, rf_row = comparison
    improvement = (
        xgb_row["selection_folds_mean_pr_auc"]
        - rf_row["selection_folds_mean_pr_auc"]
    )
    operational_benefit = bool(
        xgb_row["fold_4_false_negatives"] < rf_row["fold_4_false_negatives"]
        and xgb_row["fold_4_false_positives"] <= rf_row["fold_4_false_positives"]
    )
    replace_random_forest = bool(
        improvement >= MATERIAL_PR_AUC_IMPROVEMENT or operational_benefit
    )
    decision = (
        "XGBoost replaces Random Forest as the selected development-only production candidate."
        if replace_random_forest
        else "Random Forest remains the selected development-only production candidate."
    )
    return {
        "scope": "development-only; frozen-test labels and metrics excluded",
        "target": "primary: next-month Entrapments > 0 OR Unscheduled Outages >= 2",
        "feature_set": "full_history_without_age",
        "features": features,
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
        "selection_policy": "Configure XGBoost and select its approximately-70%-recall threshold using folds 1-3 only; evaluate the selected configuration unchanged on fold 4.",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "execution": "CPU via XGBoost tree_method=hist and device=cpu",
            "optional_dependency_file": "requirements-xgboost.txt",
        },
        "xgboost_focused_tuning": {
            "selection_rule": "Highest folds-1-3 mean PR-AUC, then worst-fold PR-AUC, then lower PR-AUC standard deviation.",
            "tested_configurations": candidates,
            "selected_parameters": selected["parameters"],
            "candidate_search_runtime_seconds": float(
                sum(item["runtime_seconds"] for item in candidates)
            ),
            "fold_4_excluded_from_all_candidate_evaluation": True,
        },
        "selected_xgboost": {
            "parameters": selected["parameters"],
            "metrics": xgboost_metrics,
            "preprocessing_audit": xgboost_preprocessing,
        },
        "matching_random_forest": {
            "parameters": RANDOM_FOREST_PARAMETERS,
            "metrics": random_forest_metrics,
            "preprocessing_audit": random_forest_preprocessing,
        },
        "comparison": comparison,
        "complexity": {
            "xgboost": "Optional pinned external dependency; eight-configuration focused search; CPU histogram tree fitting.",
            "random_forest": "Core scikit-learn dependency; already selected and operationally documented.",
        },
        "materiality": {
            "required_selection_folds_mean_pr_auc_improvement": MATERIAL_PR_AUC_IMPROVEMENT,
            "xgboost_selection_folds_mean_pr_auc_improvement": float(improvement),
            "important_operational_benefit": operational_benefit,
            "operational_benefit_definition": "Fewer fold-4 false negatives without increasing fold-4 false positives at each model's folds-1-3-selected approximately-70%-recall threshold.",
            "replace_random_forest": replace_random_forest,
        },
        "decision": decision,
        "guardrails": {
            "fold_4_excluded_from_configuration_selection": True,
            "fold_4_excluded_from_threshold_selection": True,
            "frozen_test_log_absent": not frozen_log_exists,
            "approved_for_final_test": False,
            "final_selection_record_changed": False,
        },
    }


def comparison_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Session 4.3 Development-Only XGBoost Comparison",
        "",
        results["selection_policy"],
        "",
        "| Model | Folds 1-3 mean PR-AUC | Four-fold mean | PR-AUC std | Worst fold | Fold 4 PR-AUC | Mean ROC-AUC | Mean Brier | Fold 4 precision | Fold 4 recall | FP | FN | Runtime seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["comparison"]:
        lines.append(
            f"| {row['model']} | {row['selection_folds_mean_pr_auc']:.6f} | "
            f"{row['four_fold_mean_pr_auc']:.6f} | {row['pr_auc_standard_deviation']:.6f} | "
            f"{row['worst_fold_pr_auc']:.6f} | {row['fold_4_pr_auc']:.6f} | "
            f"{row['mean_roc_auc']:.6f} | {row['mean_brier_score']:.6f} | "
            f"{row['fold_4_precision']:.6f} | {row['fold_4_recall']:.6f} | "
            f"{row['fold_4_false_positives']} | {row['fold_4_false_negatives']} | "
            f"{row['runtime_seconds']:.3f} |"
        )
    materiality = results["materiality"]
    lines.extend(
        [
            "",
            f"XGBoost folds-1-3 mean PR-AUC improvement over Random Forest: `{materiality['xgboost_selection_folds_mean_pr_auc_improvement']:.6f}`.",
            f"Material improvement threshold: `{materiality['required_selection_folds_mean_pr_auc_improvement']:.2f}`.",
            f"Important operational benefit established: `{materiality['important_operational_benefit']}`.",
            "",
            f"**Decision: {results['decision']}**",
            "",
            "XGBoost remains optional unless selected. Frozen-test labels and metrics were not accessed, and `approved_for_final_test` remains `false`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_session4_3_records(results, metrics_path: Path, record_path: Path, comparison_path: Path):
    for path in [metrics_path, record_path, comparison_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
    comparison_path.write_text(comparison_markdown(results), encoding="utf-8")
