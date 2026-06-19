from __future__ import annotations

from mta_elevator_pipeline.session4_3 import comparison_markdown, select_xgboost_candidate


def test_xgboost_candidate_selection_uses_stored_selection_fold_metrics():
    candidates = [
        {
            "parameters": {"max_depth": 3},
            "selection_folds_mean_pr_auc": 0.67,
            "selection_folds_worst_pr_auc": 0.62,
            "selection_folds_pr_auc_std": 0.03,
            "fold_4_evaluated": False,
        },
        {
            "parameters": {"max_depth": 6},
            "selection_folds_mean_pr_auc": 0.66,
            "selection_folds_worst_pr_auc": 0.64,
            "selection_folds_pr_auc_std": 0.01,
            "fold_4_evaluated": False,
        },
    ]
    assert select_xgboost_candidate(candidates)["parameters"]["max_depth"] == 3


def test_session4_3_markdown_states_decision_and_guardrail():
    results = {
        "selection_policy": "Folds 1-3 only.",
        "comparison": [
            {
                "model": "xgboost",
                "selection_folds_mean_pr_auc": 0.68,
                "four_fold_mean_pr_auc": 0.69,
                "pr_auc_standard_deviation": 0.02,
                "worst_fold_pr_auc": 0.65,
                "fold_4_pr_auc": 0.72,
                "mean_roc_auc": 0.75,
                "mean_brier_score": 0.20,
                "fold_4_precision": 0.64,
                "fold_4_recall": 0.70,
                "fold_4_false_positives": 170,
                "fold_4_false_negatives": 136,
                "runtime_seconds": 2.0,
            }
        ],
        "materiality": {
            "xgboost_selection_folds_mean_pr_auc_improvement": 0.005,
            "required_selection_folds_mean_pr_auc_improvement": 0.01,
            "important_operational_benefit": False,
        },
        "decision": "Random Forest remains the selected development-only production candidate.",
    }
    text = comparison_markdown(results)
    assert "Random Forest remains" in text
    assert "approved_for_final_test` remains `false" in text
