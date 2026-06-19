from __future__ import annotations

from mta_elevator_pipeline.boosted_trees import comparison_markdown


def test_boosted_tree_comparison_table_makes_no_final_decision():
    results = {
        "selection_policy": "Select with folds 1-3 only.",
        "tabular_comparison": [
            {
                "model": "hist_gradient_boosting",
                "selection_folds_mean_pr_auc": 0.6,
                "four_fold_mean_pr_auc": 0.61,
                "worst_fold_pr_auc": 0.5,
                "fold_4_pr_auc": 0.64,
                "fold_4_precision": 0.62,
                "fold_4_recall": 0.70,
                "fold_4_false_positives": 10,
                "fold_4_false_negatives": 5,
            }
        ],
        "external_challenger": {"reason": "Skipped."},
        "decision": "No final model decision made. Comparison is provided for review.",
    }
    table = comparison_markdown(results)
    assert "No final model decision made" in table
    assert "Folds 1-3 mean PR-AUC" in table
