from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from mta_elevator_pipeline.evaluate import calibration_diagnostics
from mta_elevator_pipeline.models import build_tabular_models
from mta_elevator_pipeline.session3 import _prediction_bundle, grouped_unseen_equipment_diagnostic
from mta_elevator_pipeline.targets import TARGET_COLUMN


def test_validation_values_cannot_change_fitted_imputation_or_scaling(monthly_data):
    features = ["Time Since Major Improvement", "Unscheduled Outages"]
    train = monthly_data.iloc[:10].copy()
    train.loc[0, "Time Since Major Improvement"] = np.nan
    validation = monthly_data.iloc[10:].copy()
    y_train = np.array([0, 1] * 5)
    model = build_tabular_models(features, [], 42)["logistic_regression"]

    first = clone(model).fit(train[features], y_train)
    changed_validation = validation.copy()
    changed_validation["Time Since Major Improvement"] = 999999
    changed_validation["Unscheduled Outages"] = 999999
    second = clone(model).fit(train[features], y_train)

    first_numeric = first.named_steps["preprocess"].named_transformers_["numeric"]
    second_numeric = second.named_steps["preprocess"].named_transformers_["numeric"]
    np.testing.assert_array_equal(
        first_numeric.named_steps["imputer"].statistics_,
        second_numeric.named_steps["imputer"].statistics_,
    )
    assert first_numeric.named_steps["imputer"].statistics_[0] == np.nanmedian(
        train["Time Since Major Improvement"]
    )
    assert first_numeric.named_steps["imputer"].statistics_[0] != np.nanmedian(
        pd.concat([train, changed_validation])["Time Since Major Improvement"]
    )
    np.testing.assert_array_equal(
        first_numeric.named_steps["scaler"].center_,
        second_numeric.named_steps["scaler"].center_,
    )
    assert not changed_validation[features].equals(validation[features])


def test_latest_fold_is_not_used_to_select_threshold():
    metadata = {
        name: {
            "train_prevalence": 0.5,
            "validation_prevalence": 0.5,
            "train_rows": 4,
            "validation_rows": 2,
            "train_equipment": 2,
            "validation_equipment": 2,
            "train_start": "2024-01-01",
            "train_end": "2024-01-01",
            "validation_start": "2024-02-01",
            "validation_end": "2024-02-01",
        }
        for name in ["one", "two", "three", "latest"]
    }
    earlier = [
        ("one", np.array([0, 1]), np.array([0.2, 0.8])),
        ("two", np.array([0, 1]), np.array([0.2, 0.8])),
        ("three", np.array([0, 1]), np.array([0.2, 0.8])),
    ]
    first = _prediction_bundle(
        earlier + [("latest", np.array([0, 1]), np.array([0.1, 0.9]))],
        metadata,
    )
    second = _prediction_bundle(
        earlier + [("latest", np.array([0, 1]), np.array([0.95, 0.99]))],
        metadata,
    )
    assert first["selected_threshold"] == second["selected_threshold"]
    assert first["threshold_selection_folds"] == ["one", "two", "three"]
    assert first["threshold_evaluation_fold"] == "latest"


def test_calibration_diagnostics_account_for_every_row():
    diagnostics = calibration_diagnostics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), bins=4
    )
    assert sum(item["rows"] for item in diagnostics["bins"]) == 4
    assert 0 <= diagnostics["expected_calibration_error"] <= 1


def test_grouped_diagnostic_has_disjoint_equipment(monthly_data):
    data = monthly_data.copy()
    data[TARGET_COLUMN] = (
        (data["Unscheduled Outages"] >= 1) | (data["Entrapments"] > 0)
    ).astype(int)
    # GroupKFold requires at least four equipment groups.
    copies = []
    for suffix in range(4):
        copied = data.copy()
        copied["Equipment Code"] = copied["Equipment Code"] + str(suffix)
        copies.append(copied)
    development = pd.concat(copies, ignore_index=True)
    features = ["Unscheduled Outages", "Time Since Major Improvement"]
    model = build_tabular_models(features, [], 42)["logistic_regression"]
    report = grouped_unseen_equipment_diagnostic(
        {}, development, features, {"logistic_regression": model}
    )
    assert len(report["models"]["logistic_regression"]["folds"]) == 4
    assert "not a replacement" in report["scope"]
