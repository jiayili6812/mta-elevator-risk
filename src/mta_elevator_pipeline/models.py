"""Initial tabular baseline model registry."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_tabular_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int,
) -> dict[str, Pipeline]:
    numeric_scaled = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    numeric_unscaled = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    def preprocessor(numeric_pipeline: Pipeline) -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical, categorical_features),
            ],
            remainder="drop",
        )

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessor(numeric_scaled)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor(numeric_unscaled)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", preprocessor(numeric_unscaled)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }

