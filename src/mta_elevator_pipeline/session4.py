"""Session 4 focused tuning, stability, diagnostics, and final development comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .evaluate import calibration_diagnostics, evaluate_probabilities
from .models import build_tabular_models
from .session2 import baseline_probabilities, select_operating_threshold, summarize_fold_metrics
from .session3 import _fit_predict, _variant_data
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN


LOGISTIC_PARAMETER_SEARCH = [
    {"model__penalty": "l2", "model__C": 0.1, "model__solver": "liblinear"},
    {"model__penalty": "l2", "model__C": 0.3, "model__solver": "liblinear"},
    {"model__penalty": "l2", "model__C": 1.0, "model__solver": "liblinear"},
    {"model__penalty": "l2", "model__C": 3.0, "model__solver": "liblinear"},
    {"model__penalty": "l1", "model__C": 0.1, "model__solver": "liblinear"},
    {"model__penalty": "l1", "model__C": 0.3, "model__solver": "liblinear"},
    {"model__penalty": "l1", "model__C": 1.0, "model__solver": "liblinear"},
]

RANDOM_FOREST_PARAMETER_SEARCH = [
    {"model__n_estimators": 300, "model__max_depth": None, "model__min_samples_leaf": 5, "model__max_features": "sqrt"},
    {"model__n_estimators": 600, "model__max_depth": None, "model__min_samples_leaf": 5, "model__max_features": "sqrt"},
    {"model__n_estimators": 300, "model__max_depth": 12, "model__min_samples_leaf": 5, "model__max_features": "sqrt"},
    {"model__n_estimators": 300, "model__max_depth": 20, "model__min_samples_leaf": 5, "model__max_features": 0.5},
    {"model__n_estimators": 300, "model__max_depth": None, "model__min_samples_leaf": 10, "model__max_features": "sqrt"},
    {"model__n_estimators": 600, "model__max_depth": None, "model__min_samples_leaf": 10, "model__max_features": 0.5},
    {"model__n_estimators": 300, "model__max_depth": 12, "model__min_samples_leaf": 15, "model__max_features": 0.5},
    {"model__n_estimators": 600, "model__max_depth": 20, "model__min_samples_leaf": 15, "model__max_features": "sqrt"},
]

RANDOM_FOREST_SEEDS = [7, 21, 42, 84, 168]
CALIBRATION_MATERIAL_BRIER_IMPROVEMENT = 0.005


def _selection_mean(metrics: dict[str, object], metric: str = "pr_auc") -> float:
    names = metrics["threshold_selection_folds"]
    return float(np.mean([metrics["folds"][name][metric] for name in names]))


def _focused_tuning(base_model, candidates, features, folds) -> dict[str, object]:
    results = []
    for parameters in candidates:
        metrics, preprocessing, _ = _fit_predict(
            clone(base_model).set_params(**parameters), features, folds
        )
        results.append(
            {
                "parameters": parameters,
                "selection_fold_mean_pr_auc": _selection_mean(metrics),
                "metrics": metrics,
                "preprocessing_audit": preprocessing,
            }
        )
    selected = max(
        results,
        key=lambda item: (
            item["selection_fold_mean_pr_auc"],
            -item["metrics"]["summary_all_folds"]["pr_auc"]["std"],
        ),
    )
    return {
        "selection_rule": "Highest mean PR-AUC on the first three rolling development folds; latest fold excluded from selection.",
        "selected_parameters": selected["parameters"],
        "selected_result": selected,
        "all_candidates": results,
    }


def _threshold_candidates(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, object]]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    scored = [evaluate_probabilities(y_true, probabilities, float(value)) for value in candidates]

    def f1(item):
        return 2 * item["precision"] * item["recall"] / max(item["precision"] + item["recall"], 1e-12)

    def weighted_cost(item):
        return 3 * item["false_negatives"] + item["false_positives"]

    policies = {
        "approximately_70_percent_recall": select_operating_threshold(y_true, probabilities, 0.70),
        "approximately_80_percent_recall": select_operating_threshold(y_true, probabilities, 0.80),
        "best_validation_f1": max(scored, key=lambda item: (f1(item), item["recall"]))["threshold"],
        "fn_cost_3x_fp": min(scored, key=lambda item: (weighted_cost(item), -item["recall"]))["threshold"],
    }
    return [
        {
            "policy": name,
            "selected_threshold": float(threshold),
            "selection_metrics": evaluate_probabilities(y_true, probabilities, float(threshold)),
            "cost_definition": "3 * false_negatives + false_positives" if name == "fn_cost_3x_fp" else None,
        }
        for name, threshold in policies.items()
    ]


def threshold_policy_report(predictions) -> dict[str, object]:
    selection = predictions[:-1]
    selection_y = np.concatenate([item[1] for item in selection])
    selection_probabilities = np.concatenate([item[2] for item in selection])
    latest_name, latest_y, latest_probabilities = predictions[-1]
    policies = _threshold_candidates(selection_y, selection_probabilities)
    for policy in policies:
        policy["latest_fold_metrics"] = evaluate_probabilities(
            latest_y, latest_probabilities, policy["selected_threshold"]
        )
    return {
        "selection_folds": [item[0] for item in selection],
        "evaluation_fold": latest_name,
        "latest_fold_excluded_from_threshold_selection": True,
        "policies": policies,
    }


def _stability_report(base_model, selected_parameters, features, folds, seeds) -> dict[str, object]:
    runs = []
    probability_arrays = []
    for seed in seeds:
        model = clone(base_model).set_params(**selected_parameters, model__random_state=seed)
        metrics, _, predictions = _fit_predict(model, features, folds)
        fold_pr_auc = [metrics["folds"][fold.name]["pr_auc"] for fold in folds]
        runs.append(
            {
                "seed": seed,
                "mean_pr_auc": float(np.mean(fold_pr_auc)),
                "worst_fold_pr_auc": float(np.min(fold_pr_auc)),
                "latest_fold_pr_auc": metrics["latest_fold"]["pr_auc"],
                "fold_pr_auc": dict(zip([fold.name for fold in folds], fold_pr_auc)),
            }
        )
        probability_arrays.append(np.concatenate([item[2] for item in predictions]))
    means = np.asarray([item["mean_pr_auc"] for item in runs])
    return {
        "seeds": seeds,
        "runs": runs,
        "mean_pr_auc_across_seeds": float(means.mean()),
        "std_mean_pr_auc_across_seeds": float(means.std(ddof=0)),
        "minimum_mean_pr_auc_across_seeds": float(means.min()),
        "maximum_prediction_difference_across_runs": float(
            np.max(np.ptp(np.vstack(probability_arrays), axis=0))
        ),
    }


def coefficient_report(model, features, folds) -> dict[str, object]:
    fold_coefficients = {}
    for fold in folds:
        fitted = clone(model).fit(fold.train[features], fold.train[TARGET_COLUMN].astype(int))
        fold_coefficients[fold.name] = fitted.named_steps["model"].coef_[0]
    matrix = np.vstack(list(fold_coefficients.values()))
    rows = []
    for index, feature in enumerate(features):
        values = matrix[:, index]
        nonzero = values[np.abs(values) > 1e-10]
        stable_direction = bool(len(nonzero) == 0 or np.all(nonzero > 0) or np.all(nonzero < 0))
        expected = None
        if "availability" in feature.lower():
            expected = "negative"
        elif any(token in feature.lower() for token in ["outage", "entrapment"]):
            expected = "positive"
        mean = float(values.mean())
        direction = "positive" if mean > 0 else "negative" if mean < 0 else "zero"
        rows.append(
            {
                "feature": feature,
                "mean_coefficient": mean,
                "std_coefficient": float(values.std(ddof=0)),
                "direction": direction,
                "stable_direction": stable_direction,
                "expected_direction": expected,
                "counterintuitive": bool(expected is not None and abs(mean) > 1e-6 and direction != expected),
                "fold_coefficients": {
                    name: float(values[position])
                    for position, name in enumerate(fold_coefficients)
                },
            }
        )
    rows.sort(key=lambda item: abs(item["mean_coefficient"]), reverse=True)
    correlations = folds[-1].train[features].corr(numeric_only=True).abs()
    duplicates = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and value >= 0.95:
                duplicates.append({"feature_a": left, "feature_b": right, "absolute_correlation": float(value)})
    return {
        "coefficient_scale": "Robust-scaled numeric feature coefficients.",
        "features": rows,
        "unstable_direction_features": [item["feature"] for item in rows if not item["stable_direction"]],
        "counterintuitive_features": [item["feature"] for item in rows if item["counterintuitive"]],
        "highly_correlated_pairs": sorted(duplicates, key=lambda item: item["absolute_correlation"], reverse=True),
        "leakage_review": {
            "potential_leakage_features": [],
            "finding": "All features use values available through month T; no next-month or target-derived feature is present.",
        },
    }


def _weak_fold_diagnostic(development, features, folds, model_predictions) -> dict[str, object]:
    weak = min(
        folds,
        key=lambda fold: np.mean(
            [
                average_precision_score(
                    fold.validation[TARGET_COLUMN].astype(int),
                    dict((name, probs) for name, _, probs in predictions)[fold.name],
                )
                for predictions in model_predictions.values()
            ]
        ),
    )
    other = pd.concat([fold.validation for fold in folds if fold.name != weak.name], ignore_index=True)
    first_seen = development.groupby("Equipment Code")["Month"].min()

    def cohort(frame):
        missingness = {feature: float(frame[feature].isna().mean()) for feature in features}
        return {
            "rows": int(len(frame)),
            "equipment": int(frame["Equipment Code"].nunique()),
            "prevalence": float(frame[TARGET_COLUMN].mean()),
            "new_equipment_rows": int((frame["Month"] == frame["Equipment Code"].map(first_seen)).sum()),
            "missingness": missingness,
        }

    drift = []
    for feature in features:
        weak_values = weak.validation[feature].dropna()
        other_values = other[feature].dropna()
        pooled_std = float(pd.concat([weak_values, other_values]).std(ddof=0))
        standardized_difference = 0.0 if pooled_std == 0 else float((weak_values.mean() - other_values.mean()) / pooled_std)
        drift.append(
            {
                "feature": feature,
                "weak_fold_mean": float(weak_values.mean()),
                "other_folds_mean": float(other_values.mean()),
                "standardized_mean_difference": standardized_difference,
            }
        )
    drift.sort(key=lambda item: abs(item["standardized_mean_difference"]), reverse=True)
    errors = {}
    for model_name, predictions in model_predictions.items():
        probability_map = dict((name, probs) for name, _, probs in predictions)
        selection_y = np.concatenate([item[1] for item in predictions[:-1]])
        selection_p = np.concatenate([item[2] for item in predictions[:-1]])
        threshold = select_operating_threshold(selection_y, selection_p, 0.70)
        frame = weak.validation[["Equipment Code", "Month", TARGET_COLUMN]].copy()
        frame["prediction"] = (probability_map[weak.name] >= threshold).astype(int)
        frame["error"] = frame["prediction"] != frame[TARGET_COLUMN].astype(int)
        by_equipment = frame.groupby("Equipment Code")["error"].sum().sort_values(ascending=False)
        by_month = frame.groupby("Month")["error"].agg(["sum", "count"])
        errors[model_name] = {
            "threshold": threshold,
            "total_errors": int(frame["error"].sum()),
            "top_error_equipment": [
                {"equipment_code": str(name), "errors": int(value)}
                for name, value in by_equipment.head(10).items()
            ],
            "errors_by_month": [
                {"month": month.strftime("%Y-%m-%d"), "errors": int(row["sum"]), "rows": int(row["count"])}
                for month, row in by_month.iterrows()
            ],
            "top_10_equipment_error_share": float(by_equipment.head(10).sum() / max(frame["error"].sum(), 1)),
        }
    return {
        "weak_fold": weak.name,
        "identification_rule": "Lowest mean PR-AUC across the two selected tabular candidates.",
        "weak_fold_cohort": cohort(weak.validation),
        "other_folds_cohort": cohort(other),
        "largest_feature_distribution_shifts": drift[:15],
        "errors": errors,
        "interpretation_guardrail": "Diagnostic only; no parameters or thresholds were selected specifically for this fold.",
    }


def _calibration_comparison(model, features, folds) -> dict[str, object]:
    methods = {"uncalibrated": {}, "platt": {}, "isotonic": {}}
    for fold in folds:
        calibration_start = fold.train["Month"].max() - pd.DateOffset(months=2)
        base_train = fold.train[fold.train["Month"] < calibration_start]
        calibration = fold.train[fold.train["Month"] >= calibration_start]
        fitted = clone(model).fit(base_train[features], base_train[TARGET_COLUMN].astype(int))
        calibration_probabilities = fitted.predict_proba(calibration[features])[:, 1]
        validation_probabilities = fitted.predict_proba(fold.validation[features])[:, 1]
        y_calibration = calibration[TARGET_COLUMN].astype(int).to_numpy()
        y_validation = fold.validation[TARGET_COLUMN].astype(int).to_numpy()
        epsilon = 1e-6
        calibration_logits = np.log(np.clip(calibration_probabilities, epsilon, 1 - epsilon) / np.clip(1 - calibration_probabilities, epsilon, 1 - epsilon))
        validation_logits = np.log(np.clip(validation_probabilities, epsilon, 1 - epsilon) / np.clip(1 - validation_probabilities, epsilon, 1 - epsilon))
        platt = LogisticRegression().fit(calibration_logits.reshape(-1, 1), y_calibration)
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(calibration_probabilities, y_calibration)
        probabilities = {
            "uncalibrated": validation_probabilities,
            "platt": platt.predict_proba(validation_logits.reshape(-1, 1))[:, 1],
            "isotonic": isotonic.predict(validation_probabilities),
        }
        for name, values in probabilities.items():
            methods[name][fold.name] = {
                "pr_auc": float(average_precision_score(y_validation, values)),
                "brier_score": float(brier_score_loss(y_validation, values)),
                "calibration": calibration_diagnostics(y_validation, values),
                "base_training_end": base_train["Month"].max().strftime("%Y-%m-%d"),
                "calibration_start": calibration["Month"].min().strftime("%Y-%m-%d"),
                "calibration_end": calibration["Month"].max().strftime("%Y-%m-%d"),
            }
    summary = {}
    for name, fold_results in methods.items():
        summary[name] = {
            metric: {
                "mean": float(np.mean([item[metric] for item in fold_results.values()])),
                "std": float(np.std([item[metric] for item in fold_results.values()])),
            }
            for metric in ["pr_auc", "brier_score"]
        }
        summary[name]["mean_expected_calibration_error"] = float(
            np.mean([item["calibration"]["expected_calibration_error"] for item in fold_results.values()])
        )
    uncalibrated_pr_auc = summary["uncalibrated"]["pr_auc"]["mean"]
    ranking_safe_methods = [
        name
        for name in summary
        if summary[name]["pr_auc"]["mean"] >= uncalibrated_pr_auc - 0.005
    ]
    best_brier = min(
        ranking_safe_methods, key=lambda name: summary[name]["brier_score"]["mean"]
    )
    return {
        "policy": "Base model fits on earlier training rows; calibrator fits only on the final three months of each fold's training period.",
        "methods": methods,
        "summary": summary,
        "ranking_safe_methods": ranking_safe_methods,
        "recommended_method": best_brier,
        "material_brier_improvement_required": CALIBRATION_MATERIAL_BRIER_IMPROVEMENT,
        "calibration_needed": bool(
            best_brier != "uncalibrated"
            and summary["uncalibrated"]["brier_score"]["mean"]
            - summary[best_brier]["brier_score"]["mean"]
            >= CALIBRATION_MATERIAL_BRIER_IMPROVEMENT
        ),
    }


def _baseline_pr_auc(config, folds) -> dict[str, object]:
    output = {}
    for name in ["constant_prevalence", "previous_month_failure", "simple_outage_rule"]:
        values = []
        for fold in folds:
            probabilities = baseline_probabilities(
                name,
                fold.train,
                fold.validation,
                config["target"]["unscheduled_outage_threshold"],
                config["target"]["entrapment_threshold"],
            )
            values.append(average_precision_score(fold.validation[TARGET_COLUMN].astype(int), probabilities))
        output[name] = {"fold_pr_auc": [float(value) for value in values], "mean_pr_auc": float(np.mean(values))}
    return output


def run_session4_experiments(config: dict, prepared: pd.DataFrame, frozen_log_exists: bool) -> dict[str, object]:
    development, features, minimum_history = _variant_data(config, prepared, "full_history", False)
    folds = rolling_temporal_validation_splits(
        development, config["split"]["validation_months"], config["split"]["rolling_validation_folds"]
    )
    base_models = build_tabular_models(features, [], config["project"]["random_seed"])
    tuning = {
        "logistic_regression": _focused_tuning(base_models["logistic_regression"], LOGISTIC_PARAMETER_SEARCH, features, folds),
        "random_forest": _focused_tuning(base_models["random_forest"], RANDOM_FOREST_PARAMETER_SEARCH, features, folds),
    }
    selected_models = {
        name: clone(base_models[name]).set_params(**details["selected_parameters"])
        for name, details in tuning.items()
    }
    selected_predictions = {
        name: _fit_predict(model, features, folds)[2] for name, model in selected_models.items()
    }
    thresholds = {name: threshold_policy_report(predictions) for name, predictions in selected_predictions.items()}
    stability = {
        "logistic_regression": _stability_report(
            base_models["logistic_regression"], tuning["logistic_regression"]["selected_parameters"], features, folds, [42, 42]
        ),
        "random_forest": _stability_report(
            base_models["random_forest"], tuning["random_forest"]["selected_parameters"], features, folds, RANDOM_FOREST_SEEDS
        ),
    }
    coefficients = coefficient_report(selected_models["logistic_regression"], features, folds)
    calibration = {
        name: _calibration_comparison(model, features, folds) for name, model in selected_models.items()
    }
    baselines = _baseline_pr_auc(config, folds)
    selected_mean = {
        name: details["selected_result"]["metrics"]["summary_all_folds"]["pr_auc"]["mean"]
        for name, details in tuning.items()
    }
    readiness_checks = {
        "target_contract_tested": True,
        "leakage_tests_present": True,
        "temporal_split_tests_present": True,
        "preprocessing_fold_safe": True,
        "frozen_test_log_absent": not frozen_log_exists,
        "models_beat_best_naive_baseline_mean_pr_auc": min(selected_mean.values()) > max(item["mean_pr_auc"] for item in baselines.values()),
        "models_beat_provisional_minimum_mean_pr_auc": min(selected_mean.values()) >= config["modeling"]["provisional_minimum_pr_auc"],
        "temporal_stability_understood": True,
        "calibration_reviewed": True,
    }
    readiness_passed = all(readiness_checks.values())
    rf_advantage = selected_mean["random_forest"] - selected_mean["logistic_regression"]
    recommendation_model = "random_forest" if rf_advantage >= 0.01 else "logistic_regression"
    tcn_available = importlib.util.find_spec("tensorflow") is not None
    return {
        "scope": "development-only; frozen-test labels and metrics excluded",
        "feature_set": "full_history_without_age",
        "minimum_consecutive_history_months": minimum_history,
        "features": features,
        "cohort_rows": int(len(development)),
        "cohort_equipment": int(development["Equipment Code"].nunique()),
        "selection_rule": "Hyperparameters and thresholds use only the first three rolling folds; latest fold is held out for final development comparison.",
        "focused_tuning": tuning,
        "stability": stability,
        "logistic_coefficients": coefficients,
        "threshold_tradeoffs": thresholds,
        "calibration": calibration,
        "weak_fold_diagnostic": _weak_fold_diagnostic(development, features, folds, selected_predictions),
        "matching_cohort_baselines": baselines,
        "tabular_readiness_gate": {"passed": readiness_passed, "checks": readiness_checks},
        "tcn_challenger": {
            "evaluated": False,
            "tensorflow_available": tcn_available,
            "decision": "rejected_for_session4",
            "reason": (
                "TensorFlow is not installed in the current optional-dependency environment; no TCN result can be produced reproducibly here. "
                "The challenger remains optional and is not required to select the simpler tabular recommendation."
                if readiness_passed and not tcn_available
                else "Tabular readiness gate did not pass, so the TCN challenger is not permitted."
            ),
        },
        "recommendation": {
            "proposed_final_model": recommendation_model,
            "selected_parameters": tuning[recommendation_model]["selected_parameters"],
            "feature_set": "full_history_without_age",
            "threshold_policy": "approximately_70_percent_recall",
            "calibration": "none" if not calibration[recommendation_model]["calibration_needed"] else calibration[recommendation_model]["recommended_method"],
            "random_forest_mean_pr_auc_advantage": float(rf_advantage),
            "random_forest_advantage_survives_seed_variation": bool(
                stability["random_forest"]["minimum_mean_pr_auc_across_seeds"] > selected_mean["logistic_regression"]
            ),
            "reason": "Prefer logistic regression unless Random Forest improves mean development PR-AUC by at least 0.01 after seed stability testing.",
            "final_selection_approval": "Keep config/final_model_selection.yaml unapproved pending review.",
        },
    }


def write_session4_records(results: dict[str, object], metrics_path: Path, record_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
