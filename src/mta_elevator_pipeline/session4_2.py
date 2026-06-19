"""Session 4.2 development-only TCN challenger experiment."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from .evaluate import calibration_diagnostics, evaluate_probabilities
from .features import build_features
from .models import build_tabular_models
from .session2 import select_operating_threshold, summarize_fold_metrics
from .session3 import _fit_predict
from .splits import rolling_temporal_validation_splits
from .targets import TARGET_COLUMN
from .tcn import SEQUENCE_CHANNELS, SequenceDataset, build_sequences, fit_predict_tcn


SEQUENCE_LENGTHS = [6, 12, 24]
TCN_CONFIGURATIONS = [
    {"filters": 16, "dropout": 0.10, "dilations": [1, 2, 4]},
    {"filters": 32, "dropout": 0.20, "dilations": [1, 2, 4]},
]
TCN_SEEDS = [21, 42, 84]
RANDOM_FOREST_PARAMETERS = {
    "model__n_estimators": 600,
    "model__max_depth": 20,
    "model__min_samples_leaf": 15,
    "model__max_features": "sqrt",
}
HIST_GRADIENT_BOOSTING_PARAMETERS = {
    "model__max_iter": 500,
    "model__learning_rate": 0.03,
    "model__max_leaf_nodes": 7,
    "model__min_samples_leaf": 40,
    "model__l2_regularization": 5.0,
    "model__max_bins": 255,
}


def select_tcn_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Select using only the explicitly stored folds-1-3 metric."""
    return max(
        candidates,
        key=lambda item: (
            item["selection_folds_mean_pr_auc"],
            -item["selection_folds_pr_auc_std"],
        ),
    )


def _cohort_report(
    dataset: SequenceDataset, folds, total_equipment: int, sequence_length: int
) -> dict[str, object]:
    return {
        "sequence_length": sequence_length,
        "rows": int(len(dataset.endpoints)),
        "equipment": int(dataset.endpoints["Equipment Code"].nunique()),
        "equipment_excluded_due_to_insufficient_history": int(
            total_equipment - dataset.endpoints["Equipment Code"].nunique()
        ),
        "validation_folds": [
            {
                "name": fold.name,
                "rows": int(len(fold.validation)),
                "equipment": int(fold.validation["Equipment Code"].nunique()),
                "prevalence": float(fold.validation[TARGET_COLUMN].mean()),
            }
            for fold in folds
        ],
    }


def _fold_sequence_predictions(dataset, folds, configuration, seed, fold_limit=4):
    predictions = []
    audits = {}
    for fold in folds[:fold_limit]:
        train_indices = fold.train.sort_values(
            ["Month", "Equipment Code"]
        )["_sequence_index"].to_numpy(dtype=int)
        validation_indices = fold.validation["_sequence_index"].to_numpy(dtype=int)
        probabilities, audit = fit_predict_tcn(
            dataset.values[train_indices],
            dataset.targets[train_indices],
            dataset.values[validation_indices],
            sequence_length=dataset.values.shape[1],
            filters=configuration["filters"],
            dropout=configuration["dropout"],
            dilations=tuple(configuration["dilations"]),
            seed=seed,
        )
        predictions.append(
            (
                fold.name,
                fold.validation[TARGET_COLUMN].astype(int).to_numpy(),
                probabilities,
            )
        )
        audits[fold.name] = audit
    return predictions, audits


def _metrics_bundle(predictions) -> dict[str, object]:
    selection = predictions[:3]
    y_selection = np.concatenate([item[1] for item in selection])
    p_selection = np.concatenate([item[2] for item in selection])
    threshold = select_operating_threshold(y_selection, p_selection, 0.70)
    folds = {}
    for name, y_true, probabilities in predictions:
        folds[name] = {
            **evaluate_probabilities(y_true, probabilities, threshold),
            "calibration": calibration_diagnostics(y_true, probabilities),
        }
    output = {
        "approximately_70_recall_threshold": threshold,
        "threshold_selection_folds": [item[0] for item in selection],
        "folds": folds,
        "summary_all_available_folds": summarize_fold_metrics(folds),
    }
    if len(predictions) == 4:
        output["fold_4"] = folds[predictions[3][0]]
    return output


def _matching_tabular_comparison(config, prepared, endpoints):
    featured, features = build_features(
        prepared,
        lag_months=[1, 2, 3],
        rolling_windows=[3, 6],
        include_seasonality=False,
        include_age=False,
    )
    keys = endpoints[["Equipment Code", "Month"]].copy()
    cohort = featured.merge(keys, on=["Equipment Code", "Month"], how="inner")
    folds = rolling_temporal_validation_splits(
        cohort, config["split"]["validation_months"], config["split"]["rolling_validation_folds"]
    )
    base = build_tabular_models(features, [], config["project"]["random_seed"])
    models = {
        "random_forest": clone(base["random_forest"]).set_params(**RANDOM_FOREST_PARAMETERS),
        "hist_gradient_boosting": clone(base["hist_gradient_boosting"]).set_params(
            **HIST_GRADIENT_BOOSTING_PARAMETERS
        ),
    }
    output = {}
    for name, model in models.items():
        metrics, preprocessing, _ = _fit_predict(model, features, folds)
        output[name] = {
            "parameters": (
                RANDOM_FOREST_PARAMETERS
                if name == "random_forest"
                else HIST_GRADIENT_BOOSTING_PARAMETERS
            ),
            "metrics": metrics,
            "preprocessing_audit": preprocessing,
        }
    return output


def _comparison_row(name, metrics):
    folds = metrics["folds"]
    names = list(folds)
    return {
        "model": name,
        "selection_folds_mean_pr_auc": float(np.mean([folds[n]["pr_auc"] for n in names[:3]])),
        "four_fold_mean_pr_auc": metrics["summary_all_available_folds"]["pr_auc"]["mean"]
        if "summary_all_available_folds" in metrics
        else metrics["summary_all_folds"]["pr_auc"]["mean"],
        "pr_auc_standard_deviation": metrics["summary_all_available_folds"]["pr_auc"]["std"]
        if "summary_all_available_folds" in metrics
        else metrics["summary_all_folds"]["pr_auc"]["std"],
        "worst_fold_pr_auc": min(folds[n]["pr_auc"] for n in names),
        "fold_4_pr_auc": folds[names[3]]["pr_auc"],
        "mean_roc_auc": float(np.mean([folds[n]["roc_auc"] for n in names])),
        "mean_brier_score": float(np.mean([folds[n]["brier_score"] for n in names])),
        "approximately_70_recall_threshold": metrics.get(
            "approximately_70_recall_threshold", metrics.get("selected_threshold")
        ),
        "fold_4_precision": folds[names[3]]["precision"],
        "fold_4_recall": folds[names[3]]["recall"],
        "fold_4_false_positives": folds[names[3]]["false_positives"],
        "fold_4_false_negatives": folds[names[3]]["false_negatives"],
    }


def run_session4_2_experiment(config: dict, prepared: pd.DataFrame, frozen_log_exists: bool):
    import tensorflow as tf

    development = prepared[
        prepared[TARGET_COLUMN].notna()
        & (prepared["Month"] < pd.Timestamp(config["split"]["frozen_test_start"]))
    ].copy()
    total_equipment = int(development["Equipment Code"].nunique())
    datasets = {}
    cohorts = []
    candidates = []
    for sequence_length in SEQUENCE_LENGTHS:
        dataset = build_sequences(development, sequence_length)
        endpoints = dataset.endpoints.copy()
        endpoints["_sequence_index"] = np.arange(len(endpoints))
        folds = rolling_temporal_validation_splits(
            endpoints,
            config["split"]["validation_months"],
            config["split"]["rolling_validation_folds"],
        )
        datasets[sequence_length] = (dataset, endpoints, folds)
        cohorts.append(_cohort_report(dataset, folds, total_equipment, sequence_length))
        for configuration in TCN_CONFIGURATIONS:
            predictions, audits = _fold_sequence_predictions(
                dataset, folds, configuration, config["project"]["random_seed"], fold_limit=3
            )
            metrics = _metrics_bundle(predictions)
            pr_auc = [item["pr_auc"] for item in metrics["folds"].values()]
            candidates.append(
                {
                    "sequence_length": sequence_length,
                    "configuration": configuration,
                    "seed": config["project"]["random_seed"],
                    "selection_folds_mean_pr_auc": float(np.mean(pr_auc)),
                    "selection_folds_pr_auc_std": float(np.std(pr_auc)),
                    "selection_fold_metrics": metrics,
                    "training_audit": audits,
                    "fold_4_evaluated": False,
                }
            )
    selected = select_tcn_candidate(candidates)
    dataset, endpoints, folds = datasets[selected["sequence_length"]]
    seed_runs = []
    for seed in TCN_SEEDS:
        predictions, audits = _fold_sequence_predictions(
            dataset, folds, selected["configuration"], seed, fold_limit=4
        )
        metrics = _metrics_bundle(predictions)
        seed_runs.append({"seed": seed, "metrics": metrics, "training_audit": audits})

    representative = next(item for item in seed_runs if item["seed"] == config["project"]["random_seed"])
    tabular = _matching_tabular_comparison(config, prepared, endpoints)
    comparison = [_comparison_row("tcn_seed_42", representative["metrics"])]
    comparison.extend(
        _comparison_row(name, details["metrics"]) for name, details in tabular.items()
    )
    best_tabular = max(
        (row for row in comparison if not row["model"].startswith("tcn")),
        key=lambda row: row["selection_folds_mean_pr_auc"],
    )
    tcn_row = comparison[0]
    improvement = tcn_row["selection_folds_mean_pr_auc"] - best_tabular["selection_folds_mean_pr_auc"]
    total_seconds = sum(
        audit["training_seconds"]
        for run in seed_runs
        for audit in run["training_audit"].values()
    )
    candidate_seconds = sum(
        audit["training_seconds"]
        for candidate in candidates
        for audit in candidate["training_audit"].values()
    )
    seed_mean_pr_auc = [
        run["metrics"]["summary_all_available_folds"]["pr_auc"]["mean"]
        for run in seed_runs
    ]
    return {
        "scope": "development-only; frozen-test labels and metrics excluded",
        "target": "next-month Entrapments > 0 OR Unscheduled Outages >= 2",
        "sequence_channels": SEQUENCE_CHANNELS,
        "missing_value_handling": "Per-channel median imputation and RobustScaler fitted on fold training sequences only.",
        "sequence_contract": "Each within-equipment sequence contains consecutive calendar months through T and predicts the existing T+1 target.",
        "selection_policy": "Sequence length and architecture selected using folds 1-3 only. Selected configuration evaluated once on fold 4 across three predeclared seeds.",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow_colab_compatibility": "TensorFlow 2.19.1 supports Python 3.9-3.12 and remains isolated in requirements-deep.txt for clean Colab installation.",
        },
        "cohorts": cohorts,
        "candidate_search": candidates,
        "selected_tcn": {
            "sequence_length": selected["sequence_length"],
            "configuration": selected["configuration"],
            "selection_folds_mean_pr_auc": selected["selection_folds_mean_pr_auc"],
            "seed_runs": seed_runs,
            "mean_pr_auc_across_seeds": float(np.mean(seed_mean_pr_auc)),
            "std_mean_pr_auc_across_seeds": float(np.std(seed_mean_pr_auc)),
            "total_selected_configuration_training_seconds": float(total_seconds),
            "candidate_search_training_seconds": float(candidate_seconds),
            "total_tcn_training_seconds": float(total_seconds + candidate_seconds),
            "cpu_practical": bool(total_seconds < 1800),
        },
        "matching_cohort_tabular": tabular,
        "comparison": comparison,
        "materiality": {
            "best_matching_tabular_model": best_tabular["model"],
            "tcn_selection_pr_auc_improvement": float(improvement),
            "required_stable_improvement": "approximately 0.02-0.03 or important operational gain",
            "retain_tcn": bool(improvement >= 0.02),
        },
        "decision": "Evidence only; no final production model is selected or approved.",
        "guardrails": {
            "fold_4_excluded_from_sequence_length_and_architecture_selection": True,
            "fold_4_excluded_from_threshold_selection": True,
            "frozen_test_log_absent": not frozen_log_exists,
            "final_selection_record_changed": False,
        },
    }


def comparison_markdown(results):
    lines = [
        "# Session 4.2 Development-Only TCN Comparison",
        "",
        results["selection_policy"],
        "",
        "| Model | Folds 1-3 mean PR-AUC | Four-fold mean | PR-AUC std | Worst fold | Fold 4 PR-AUC | Mean ROC-AUC | Mean Brier | Fold 4 precision | Fold 4 recall | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["comparison"]:
        lines.append(
            f"| {row['model']} | {row['selection_folds_mean_pr_auc']:.6f} | "
            f"{row['four_fold_mean_pr_auc']:.6f} | {row['pr_auc_standard_deviation']:.6f} | "
            f"{row['worst_fold_pr_auc']:.6f} | {row['fold_4_pr_auc']:.6f} | "
            f"{row['mean_roc_auc']:.6f} | {row['mean_brier_score']:.6f} | "
            f"{row['fold_4_precision']:.6f} | {row['fold_4_recall']:.6f} | "
            f"{row['fold_4_false_positives']} | {row['fold_4_false_negatives']} |"
        )
    materiality = results["materiality"]
    lines.extend(
        [
            "",
            f"Selected sequence length: `{results['selected_tcn']['sequence_length']}` months.",
            f"TCN selection-fold PR-AUC difference versus strongest matching tabular model: `{materiality['tcn_selection_pr_auc_improvement']:.6f}`.",
            f"Retain TCN under the predeclared materiality rule: `{materiality['retain_tcn']}`.",
            "",
            "Frozen-test labels and metrics were not accessed. No final production model decision was made.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_session4_2_records(results, metrics_path: Path, record_path: Path, comparison_path: Path):
    for path in [metrics_path, record_path, comparison_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    metrics_path.write_text(text, encoding="utf-8")
    record_path.write_text(text, encoding="utf-8")
    comparison_path.write_text(comparison_markdown(results), encoding="utf-8")
