"""Controlled development-only feature and baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .eligibility import add_feature_set_eligibility
from .evaluate import evaluate_probabilities
from .features import build_features, minimum_history_for_feature_set
from .models import build_tabular_models
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN


def select_operating_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.70,
) -> float:
    """Select one pooled development threshold, prioritizing precision at recall goal."""
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    scored = [
        evaluate_probabilities(y_true, probabilities, float(threshold))
        for threshold in candidates
    ]
    meeting_recall = [metrics for metrics in scored if metrics["recall"] >= minimum_recall]
    if meeting_recall:
        selected = max(
            meeting_recall,
            key=lambda metrics: (
                metrics["precision"],
                metrics["recall"],
                metrics["threshold"],
            ),
        )
    else:
        selected = max(
            scored,
            key=lambda metrics: (
                2
                * metrics["precision"]
                * metrics["recall"]
                / max(metrics["precision"] + metrics["recall"], 1e-12),
                metrics["recall"],
            ),
        )
    return float(selected["threshold"])


def baseline_probabilities(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    unscheduled_outage_threshold: int,
    entrapment_threshold: int,
) -> np.ndarray:
    """Return deliberately simple baseline scores available at the end of month T."""
    if name == "constant_prevalence":
        return np.full(len(validation), float(train[TARGET_COLUMN].mean()))
    if name == "previous_month_failure":
        return (
            (validation["Entrapments"] > entrapment_threshold)
            | (validation["Unscheduled Outages"] >= unscheduled_outage_threshold)
        ).astype(float).to_numpy()
    if name == "simple_outage_rule":
        return (validation["Unscheduled Outages"] >= 1).astype(float).to_numpy()
    raise ValueError(f"Unknown baseline: {name}")


def summarize_fold_metrics(
    fold_metrics: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    metric_names = [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
    ]
    summary: dict[str, object] = {}
    for metric in metric_names:
        values = np.asarray([fold[metric] for fold in fold_metrics.values()], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _evaluate_prediction_bundle(
    predictions: list[tuple[str, np.ndarray, np.ndarray]],
    minimum_recall: float,
) -> dict[str, object]:
    pooled_y = np.concatenate([item[1] for item in predictions])
    pooled_probabilities = np.concatenate([item[2] for item in predictions])
    threshold = select_operating_threshold(
        pooled_y, pooled_probabilities, minimum_recall=minimum_recall
    )
    folds = {
        name: evaluate_probabilities(y_true, probabilities, threshold)
        for name, y_true, probabilities in predictions
    }
    return {
        "selected_threshold": threshold,
        "threshold_selection": (
            "Pooled development rolling-validation predictions; maximize precision "
            f"subject to recall >= {minimum_recall:.2f}, with F1 fallback."
        ),
        "folds": folds,
        "summary": summarize_fold_metrics(folds),
        "pooled": evaluate_probabilities(pooled_y, pooled_probabilities, threshold),
    }


def run_session2_experiments(config: dict, prepared: pd.DataFrame) -> dict[str, object]:
    """Compare controlled feature sets and naive baselines on development folds only."""
    results: dict[str, object] = {
        "scope": "development-only rolling temporal validation; frozen-test metrics excluded",
        "target": {
            "unscheduled_outage_threshold": config["target"][
                "unscheduled_outage_threshold"
            ],
            "entrapment_threshold": config["target"]["entrapment_threshold"],
        },
        "threshold_policy": {"minimum_recall": 0.70},
        "experiments": {},
    }
    baseline_names = [
        "constant_prevalence",
        "previous_month_failure",
        "simple_outage_rule",
    ]

    for feature_set_name, definition in config["features"]["feature_sets"].items():
        minimum_history = minimum_history_for_feature_set(
            definition["lag_months"], definition["rolling_windows"]
        )
        for include_age in [False, True]:
            experiment_name = f"{feature_set_name}__{'with_age' if include_age else 'without_age'}"
            featured, features = build_features(
                prepared,
                lag_months=definition["lag_months"],
                rolling_windows=definition["rolling_windows"],
                include_seasonality=False,
                include_age=include_age,
            )
            eligible = add_feature_set_eligibility(featured, minimum_history)
            development = eligible[
                eligible["eligible_for_feature_set"]
                & eligible[TARGET_COLUMN].notna()
                & (eligible["Month"] < pd.Timestamp(config["split"]["frozen_test_start"]))
            ].copy()
            folds = rolling_temporal_validation_splits(
                development,
                validation_months=config["split"]["validation_months"],
                fold_count=config["split"]["rolling_validation_folds"],
            )

            model = build_tabular_models(
                numeric_features=features,
                categorical_features=[],
                random_seed=config["project"]["random_seed"],
            )["logistic_regression"]
            logistic_predictions = []
            baseline_predictions = {name: [] for name in baseline_names}
            fold_metadata = {}
            for fold in folds:
                model.fit(fold.train[features], fold.train[TARGET_COLUMN].astype(int))
                y_validation = fold.validation[TARGET_COLUMN].astype(int).to_numpy()
                logistic_predictions.append(
                    (
                        fold.name,
                        y_validation,
                        model.predict_proba(fold.validation[features])[:, 1],
                    )
                )
                for baseline_name in baseline_names:
                    baseline_predictions[baseline_name].append(
                        (
                            fold.name,
                            y_validation,
                            baseline_probabilities(
                                baseline_name,
                                fold.train,
                                fold.validation,
                                config["target"]["unscheduled_outage_threshold"],
                                config["target"]["entrapment_threshold"],
                            ),
                        )
                    )
                fold_metadata[fold.name] = {
                    "train_rows": int(len(fold.train)),
                    "validation_rows": int(len(fold.validation)),
                    "validation_prevalence": float(
                        fold.validation[TARGET_COLUMN].mean()
                    ),
                }

            results["experiments"][experiment_name] = {
                "feature_set": feature_set_name,
                "include_age": include_age,
                "minimum_consecutive_history_months": minimum_history,
                "features": features,
                "fold_metadata": fold_metadata,
                "logistic_regression": _evaluate_prediction_bundle(
                    logistic_predictions, minimum_recall=0.70
                ),
                "baselines": {
                    name: _evaluate_prediction_bundle(predictions, minimum_recall=0.70)
                    for name, predictions in baseline_predictions.items()
                },
            }
    results["recommendation"] = build_session2_recommendation(results)
    return results


def build_session2_recommendation(results: dict[str, object]) -> dict[str, object]:
    ranked = []
    for name, experiment in results["experiments"].items():
        summary = experiment["logistic_regression"]["summary"]["pr_auc"]
        ranked.append(
            {
                "experiment": name,
                "mean_pr_auc": summary["mean"],
                "std_pr_auc": summary["std"],
                "minimum_fold_pr_auc": summary["min"],
            }
        )
    ranked.sort(key=lambda item: (item["mean_pr_auc"], -item["std_pr_auc"]), reverse=True)
    best_mean = ranked[0]["mean_pr_auc"]
    proceed = [
        item["experiment"]
        for item in ranked
        if item["mean_pr_auc"] >= best_mean - 0.01
    ]
    return {
        "ranking": ranked,
        "recommended_for_tree_based_modeling": proceed,
        "rule": "Proceed with feature sets within 0.01 mean PR-AUC of the best logistic result.",
    }


def write_session2_records(results: dict[str, object], metrics_path: Path, record_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
