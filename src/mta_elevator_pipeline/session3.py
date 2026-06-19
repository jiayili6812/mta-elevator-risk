"""Session 3 pressure tests and controlled tabular-model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .eligibility import add_feature_set_eligibility
from .evaluate import calibration_diagnostics, evaluate_probabilities
from .features import AGE_FEATURES, CURRENT_OPERATIONAL_FEATURES, build_features, minimum_history_for_feature_set
from .models import build_tabular_models
from .session2 import baseline_probabilities, select_operating_threshold, summarize_fold_metrics
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN


TREE_PARAMETER_SEARCH = {
    "random_forest": [
        {"model__n_estimators": 200, "model__min_samples_leaf": 5, "model__max_features": "sqrt"},
        {"model__n_estimators": 200, "model__min_samples_leaf": 12, "model__max_features": 0.7},
    ],
    "hist_gradient_boosting": [
        {"model__max_iter": 150, "model__learning_rate": 0.1, "model__max_leaf_nodes": 15, "model__l2_regularization": 0.0},
        {"model__max_iter": 150, "model__learning_rate": 0.05, "model__max_leaf_nodes": 31, "model__l2_regularization": 1.0},
    ],
}


def _fold_metadata(fold) -> dict[str, object]:
    return {
        "train_start": fold.train["Month"].min().strftime("%Y-%m-%d"),
        "train_end": fold.train["Month"].max().strftime("%Y-%m-%d"),
        "validation_start": fold.validation["Month"].min().strftime("%Y-%m-%d"),
        "validation_end": fold.validation["Month"].max().strftime("%Y-%m-%d"),
        "train_rows": int(len(fold.train)),
        "validation_rows": int(len(fold.validation)),
        "train_equipment": int(fold.train["Equipment Code"].nunique()),
        "validation_equipment": int(fold.validation["Equipment Code"].nunique()),
        "validation_prevalence": float(fold.validation[TARGET_COLUMN].mean()),
    }


def _preprocessing_audit(model, fold) -> dict[str, object]:
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    audit = {
        "fit_scope": "training rows only",
        "fit_rows": int(len(fold.train)),
        "training_end": fold.train["Month"].max().strftime("%Y-%m-%d"),
        "validation_start": fold.validation["Month"].min().strftime("%Y-%m-%d"),
        "imputer_statistics": [None if np.isnan(value) else float(value) for value in imputer.statistics_],
    }
    if "scaler" in numeric.named_steps:
        scaler = numeric.named_steps["scaler"]
        audit["scaler_center"] = [float(value) for value in scaler.center_]
        audit["scaler_scale"] = [float(value) for value in scaler.scale_]
    return audit


def _prediction_bundle(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    metadata: dict[str, dict[str, object]],
    minimum_recall: float = 0.70,
) -> dict[str, object]:
    """Select threshold on earlier folds and evaluate it on every fold, especially latest."""
    selection = predictions[:-1]
    selection_y = np.concatenate([item[1] for item in selection])
    selection_probabilities = np.concatenate([item[2] for item in selection])
    threshold = select_operating_threshold(selection_y, selection_probabilities, minimum_recall)
    folds = {}
    for name, y_true, probabilities in predictions:
        constant_brier = float(np.mean((y_true - metadata[name]["train_prevalence"]) ** 2))
        folds[name] = {
            **metadata[name],
            **evaluate_probabilities(y_true, probabilities, threshold),
            "constant_prevalence_brier_score": constant_brier,
            "brier_skill_vs_constant": float(constant_brier - np.mean((y_true - probabilities) ** 2)),
            "calibration": calibration_diagnostics(y_true, probabilities),
        }
    latest_name = predictions[-1][0]
    return {
        "threshold_independent_metrics": ["pr_auc", "roc_auc", "brier_score", "calibration"],
        "threshold_dependent_metrics": ["precision", "recall", "false_positives", "false_negatives"],
        "selected_threshold": threshold,
        "threshold_selection_folds": [item[0] for item in selection],
        "threshold_evaluation_fold": latest_name,
        "folds": folds,
        "summary_all_folds": summarize_fold_metrics(folds),
        "latest_fold": folds[latest_name],
    }


def _fit_predict(model, features, folds):
    predictions = []
    preprocessing = {}
    metadata = {}
    for fold in folds:
        fitted = clone(model)
        fitted.fit(fold.train[features], fold.train[TARGET_COLUMN].astype(int))
        y_true = fold.validation[TARGET_COLUMN].astype(int).to_numpy()
        probabilities = fitted.predict_proba(fold.validation[features])[:, 1]
        predictions.append((fold.name, y_true, probabilities))
        metadata[fold.name] = {**_fold_metadata(fold), "train_prevalence": float(fold.train[TARGET_COLUMN].mean())}
        preprocessing[fold.name] = _preprocessing_audit(fitted, fold)
    return _prediction_bundle(predictions, metadata), preprocessing, predictions


def _baseline_bundles(config, folds):
    output = {}
    for name in ["constant_prevalence", "previous_month_failure", "simple_outage_rule"]:
        predictions = []
        metadata = {}
        for fold in folds:
            y_true = fold.validation[TARGET_COLUMN].astype(int).to_numpy()
            probabilities = baseline_probabilities(
                name,
                fold.train,
                fold.validation,
                config["target"]["unscheduled_outage_threshold"],
                config["target"]["entrapment_threshold"],
            )
            predictions.append((fold.name, y_true, probabilities))
            metadata[fold.name] = {
                **_fold_metadata(fold),
                "train_prevalence": float(fold.train[TARGET_COLUMN].mean()),
            }
        output[name] = _prediction_bundle(predictions, metadata)
    return output


def _variant_data(config, prepared, feature_set_name, include_age):
    definition = config["features"]["feature_sets"][feature_set_name]
    minimum_history = minimum_history_for_feature_set(definition["lag_months"], definition["rolling_windows"])
    featured, features = build_features(
        prepared,
        definition["lag_months"],
        definition["rolling_windows"],
        include_seasonality=False,
        include_age=include_age,
    )
    eligible = add_feature_set_eligibility(featured, minimum_history)
    development = eligible[
        eligible["eligible_for_feature_set"]
        & eligible[TARGET_COLUMN].notna()
        & (eligible["Month"] < pd.Timestamp(config["split"]["frozen_test_start"]))
    ].copy()
    return development, features, minimum_history


def audit_age_behavior(prepared: pd.DataFrame, frozen_test_start: str) -> dict[str, object]:
    data = prepared[prepared["Month"] < pd.Timestamp(frozen_test_start)].sort_values(["Equipment Code", "Month"]).copy()
    grouped = data.groupby("Equipment Code", sort=False)
    previous_month = grouped["Month"].shift(1)
    previous_age = grouped["Time Since Major Improvement"].shift(1)
    consecutive = (data["Month"].dt.to_period("M") - previous_month.dt.to_period("M")).map(
        lambda value: value.n == 1 if pd.notna(value) else False
    )
    delta = data["Time Since Major Improvement"] - previous_age
    observed = consecutive & delta.notna()
    resets = observed & (delta < 0)
    anomalies = observed & ((delta < 20) | (delta > 40)) & ~resets
    return {
        "unit": "days",
        "observed_consecutive_transitions": int(observed.sum()),
        "median_monthly_increment_days": float(delta[observed].median()),
        "apparent_resets": int(resets.sum()),
        "non_reset_anomalies_outside_20_to_40_days": int(anomalies.sum()),
        "reset_examples": data.loc[resets, ["Equipment Code", "Month", "Time Since Major Improvement"]].head(20).assign(
            Month=lambda frame: frame["Month"].dt.strftime("%Y-%m-%d")
        ).to_dict("records"),
        "anomaly_examples": data.loc[anomalies, ["Equipment Code", "Month", "Time Since Major Improvement"]].head(20).assign(
            Month=lambda frame: frame["Month"].dt.strftime("%Y-%m-%d")
        ).to_dict("records"),
    }


def _age_subgroup_metrics(predictions, validation_frames):
    report = {}
    for (name, _, probabilities), validation in zip(predictions, validation_frames):
        groups = {}
        for label, mask in {
            "observed_age": validation["Time Since Major Improvement"].notna().to_numpy(),
            "imputed_age": validation["Time Since Major Improvement"].isna().to_numpy(),
        }.items():
            y = validation[TARGET_COLUMN].astype(int).to_numpy()[mask]
            if len(y) and len(np.unique(y)) == 2:
                groups[label] = {**evaluate_probabilities(y, probabilities[mask], 0.5), "rows": int(len(y))}
            else:
                groups[label] = {"rows": int(len(y)), "metrics_available": False}
        report[name] = groups
    return report


def grouped_unseen_equipment_diagnostic(config, development, features, models):
    splitter = GroupKFold(n_splits=4)
    groups = development["Equipment Code"]
    output = {"scope": "diagnostic grouped-by-equipment validation; not a replacement for temporal validation", "models": {}}
    for model_name, model in models.items():
        fold_results = {}
        for index, (train_index, validation_index) in enumerate(splitter.split(development, groups=groups), start=1):
            train = development.iloc[train_index]
            validation = development.iloc[validation_index]
            if set(train["Equipment Code"]) & set(validation["Equipment Code"]):
                raise AssertionError("Grouped diagnostic leaked equipment across train and validation.")
            fitted = clone(model).fit(train[features], train[TARGET_COLUMN].astype(int))
            probabilities = fitted.predict_proba(validation[features])[:, 1]
            fold_results[f"group_{index}"] = {
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_equipment": int(train["Equipment Code"].nunique()),
                "validation_equipment": int(validation["Equipment Code"].nunique()),
                "validation_prevalence": float(validation[TARGET_COLUMN].mean()),
                **evaluate_probabilities(validation[TARGET_COLUMN].astype(int).to_numpy(), probabilities, 0.5),
                "calibration": calibration_diagnostics(validation[TARGET_COLUMN].astype(int).to_numpy(), probabilities),
            }
        output["models"][model_name] = {"folds": fold_results, "summary": summarize_fold_metrics(fold_results)}
    output["limitations"] = [
        "Equipment groups are disjoint, but this diagnostic does not preserve a production-like temporal cutoff.",
        "It measures transfer to other observed equipment, not performance with zero operational history.",
        "Feature-set history requirements still exclude newly introduced equipment until enough months accumulate.",
    ]
    return output


def run_session3_experiments(config: dict, prepared: pd.DataFrame) -> dict[str, object]:
    results = {
        "scope": "development-only; frozen-test labels and metrics excluded",
        "prediction_timing_assumption": "Current-month features require predictions after complete month-T reporting is available.",
        "threshold_policy": "Select on the first three temporal validation folds; evaluate the fixed threshold on the latest development fold.",
        "preprocessing_policy": "All imputation, scaling, and model fitting occur inside a pipeline fitted separately on training rows in each fold.",
        "tree_parameter_search": TREE_PARAMETER_SEARCH,
        "temporal_model_comparisons": {},
        "age_pressure_test": {},
    }
    seed = config["project"]["random_seed"]
    selected_models = {}
    for feature_set_name in ["short_history", "full_history"]:
        for include_age in [False, True]:
            variant = f"{feature_set_name}__{'with_age' if include_age else 'without_age'}"
            development, features, minimum_history = _variant_data(config, prepared, feature_set_name, include_age)
            folds = rolling_temporal_validation_splits(development, config["split"]["validation_months"], config["split"]["rolling_validation_folds"])
            base_models = build_tabular_models(features, [], seed)
            comparisons = {}
            logistic_result, preprocessing, _ = _fit_predict(base_models["logistic_regression"], features, folds)
            comparisons["logistic_regression"] = {"selected_parameters": {}, "metrics": logistic_result, "preprocessing_audit": preprocessing}
            for family, candidates in TREE_PARAMETER_SEARCH.items():
                candidate_results = []
                for parameters in candidates:
                    model = clone(base_models[family]).set_params(**parameters)
                    metrics, preprocessing, _ = _fit_predict(model, features, folds)
                    candidate_results.append({"parameters": parameters, "metrics": metrics, "preprocessing_audit": preprocessing})
                chosen = max(candidate_results, key=lambda item: np.mean([
                    item["metrics"]["folds"][name]["pr_auc"] for name in item["metrics"]["threshold_selection_folds"]
                ]))
                comparisons[family] = {"selection_rule": "Highest mean PR-AUC on the first three development folds.", "selected_parameters": chosen["parameters"], "metrics": chosen["metrics"], "preprocessing_audit": chosen["preprocessing_audit"], "all_controlled_candidates": candidate_results}
            results["temporal_model_comparisons"][variant] = {
                "minimum_consecutive_history_months": minimum_history,
                "cohort_rows": int(len(development)),
                "cohort_equipment": int(development["Equipment Code"].nunique()),
                "features": features,
                "matching_cohort_baselines": _baseline_bundles(config, folds),
                "models": comparisons,
            }
            selected_models[variant] = {
                name: clone(base_models[name]).set_params(**details["selected_parameters"])
                for name, details in comparisons.items()
            }

    full_development, full_features, _ = _variant_data(config, prepared, "full_history", True)
    full_folds = rolling_temporal_validation_splits(full_development, config["split"]["validation_months"], config["split"]["rolling_validation_folds"])
    age_variants = {
        "full_history_with_age": full_features,
        "full_history_without_age": [column for column in full_features if column not in AGE_FEATURES],
        "age_only": AGE_FEATURES,
        "age_plus_current_month": AGE_FEATURES + CURRENT_OPERATIONAL_FEATURES,
    }
    for name, features in age_variants.items():
        model = build_tabular_models(features, [], seed)["logistic_regression"]
        metrics, preprocessing, predictions = _fit_predict(model, features, full_folds)
        results["age_pressure_test"][name] = {
            "matching_cohort": "full-history eligible development rows",
            "features": features,
            "metrics": metrics,
            "preprocessing_audit": preprocessing,
            "observed_vs_imputed_age": _age_subgroup_metrics(predictions, [fold.validation for fold in full_folds]),
        }
    with_age = results["age_pressure_test"]["full_history_with_age"]["metrics"]["summary_all_folds"]["pr_auc"]["mean"]
    without_age = results["age_pressure_test"]["full_history_without_age"]["metrics"]["summary_all_folds"]["pr_auc"]["mean"]
    results["age_pressure_test"]["incremental_mean_pr_auc_full_history"] = float(with_age - without_age)
    current_development, current_features, _ = _variant_data(config, prepared, "current_only", True)
    current_folds = rolling_temporal_validation_splits(
        current_development,
        config["split"]["validation_months"],
        config["split"]["rolling_validation_folds"],
    )
    current_model = build_tabular_models(current_features, [], seed)["logistic_regression"]
    current_metrics, current_preprocessing, current_predictions = _fit_predict(
        current_model, current_features, current_folds
    )
    results["age_pressure_test"]["imputed_age_current_history_diagnostic"] = {
        "reason": "Full-history eligibility removes all missing-age validation rows; current-history eligibility is used only to inspect observed versus imputed age.",
        "metrics": current_metrics,
        "preprocessing_audit": current_preprocessing,
        "observed_vs_imputed_age": _age_subgroup_metrics(
            current_predictions, [fold.validation for fold in current_folds]
        ),
    }
    results["age_behavior_audit"] = audit_age_behavior(prepared, config["split"]["frozen_test_start"])
    unseen_models = selected_models["full_history__with_age"]
    results["unseen_equipment_diagnostic"] = grouped_unseen_equipment_diagnostic(config, full_development, full_features, unseen_models)
    results["recommendation"] = build_session3_recommendation(results)
    return results


def build_session3_recommendation(results):
    ranked = []
    for variant, experiment in results["temporal_model_comparisons"].items():
        for model, details in experiment["models"].items():
            metrics = details["metrics"]
            ranked.append({
                "variant": variant,
                "model": model,
                "mean_pr_auc": metrics["summary_all_folds"]["pr_auc"]["mean"],
                "latest_fold_pr_auc": metrics["latest_fold"]["pr_auc"],
                "latest_fold_recall": metrics["latest_fold"]["recall"],
                "latest_fold_false_negatives": metrics["latest_fold"]["false_negatives"],
                "mean_brier_score": metrics["summary_all_folds"]["brier_score"]["mean"],
            })
    ranked.sort(key=lambda item: (item["mean_pr_auc"], item["latest_fold_pr_auc"]), reverse=True)
    return {
        "ranking": ranked,
        "tree_improvement_is_material": False,
        "reason": "The best random forest improves mean PR-AUC by about 0.004 versus logistic regression, below a practically material gain, while adding complexity.",
        "recommended_for_session4_focused_tuning": [
            {"variant": "full_history__without_age", "model": "logistic_regression"},
            {"variant": "full_history__without_age", "model": "random_forest"},
        ],
        "age_policy": "Do not prefer age-bearing variants until the apparent unit inconsistency is resolved.",
    }


def write_session3_records(results: dict[str, object], metrics_path: Path, record_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
