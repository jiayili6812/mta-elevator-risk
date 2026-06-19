"""Optional, development-only Temporal Convolutional Network utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .features import CURRENT_OPERATIONAL_FEATURES
from .targets import TARGET_COLUMN


SEQUENCE_CHANNELS = list(CURRENT_OPERATIONAL_FEATURES)


@dataclass(frozen=True)
class SequenceDataset:
    values: np.ndarray
    targets: np.ndarray
    endpoints: pd.DataFrame


@dataclass(frozen=True)
class SequencePreprocessor:
    imputer: SimpleImputer
    scaler: RobustScaler


def build_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    channels: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> SequenceDataset:
    """Build consecutive, within-equipment sequences ending at prediction month T."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    channels = channels or SEQUENCE_CHANNELS
    ordered = data.copy()
    ordered["Month"] = pd.to_datetime(ordered["Month"])
    ordered = ordered.sort_values(["Equipment Code", "Month"]).reset_index(drop=True)

    values: list[np.ndarray] = []
    targets: list[int] = []
    endpoints: list[dict[str, object]] = []
    for equipment, group in ordered.groupby("Equipment Code", sort=False):
        group = group.reset_index(drop=True)
        periods = group["Month"].dt.to_period("M")
        for end in range(sequence_length - 1, len(group)):
            row = group.iloc[end]
            if pd.isna(row[target_column]):
                continue
            start = end - sequence_length + 1
            window_periods = periods.iloc[start : end + 1]
            expected = pd.period_range(
                window_periods.iloc[0], periods=sequence_length, freq="M"
            )
            if not np.array_equal(window_periods.to_numpy(), expected.to_numpy()):
                continue
            values.append(group.iloc[start : end + 1][channels].to_numpy(dtype=float))
            targets.append(int(row[target_column]))
            endpoints.append(
                {
                    "Equipment Code": equipment,
                    "Month": row["Month"],
                    target_column: int(row[target_column]),
                    "sequence_start": group.iloc[start]["Month"],
                    "sequence_end": row["Month"],
                }
            )

    shape = (0, sequence_length, len(channels))
    array = np.stack(values) if values else np.empty(shape, dtype=float)
    return SequenceDataset(
        values=array,
        targets=np.asarray(targets, dtype=int),
        endpoints=pd.DataFrame(endpoints),
    )


def fit_sequence_preprocessor(values: np.ndarray) -> SequencePreprocessor:
    """Fit channel-wise imputation and scaling using training sequences only."""
    flattened = values.reshape(-1, values.shape[-1])
    imputer = SimpleImputer(strategy="median").fit(flattened)
    imputed = imputer.transform(flattened)
    scaler = RobustScaler().fit(imputed)
    return SequencePreprocessor(imputer=imputer, scaler=scaler)


def transform_sequences(values: np.ndarray, preprocessor: SequencePreprocessor) -> np.ndarray:
    shape = values.shape
    flattened = values.reshape(-1, shape[-1])
    transformed = preprocessor.scaler.transform(preprocessor.imputer.transform(flattened))
    return transformed.reshape(shape).astype("float32")


def set_random_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def build_compact_tcn(
    sequence_length: int,
    channel_count: int,
    filters: int,
    dropout: float,
    dilations: tuple[int, ...] = (1, 2, 4),
):
    """Build one compact causal residual TCN architecture."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(sequence_length, channel_count))
    x = inputs
    for dilation in dilations:
        residual = x
        x = tf.keras.layers.Conv1D(
            filters, 3, padding="causal", dilation_rate=dilation, activation="relu"
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(
            filters, 3, padding="causal", dilation_rate=dilation, activation="relu"
        )(x)
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def fit_predict_tcn(
    train_values: np.ndarray,
    train_targets: np.ndarray,
    validation_values: np.ndarray,
    *,
    sequence_length: int,
    filters: int,
    dropout: float,
    dilations: tuple[int, ...],
    seed: int,
    max_epochs: int = 40,
    patience: int = 5,
    batch_size: int = 128,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a fold-local TCN and return validation probabilities and training audit."""
    import tensorflow as tf

    set_random_seeds(seed)
    preprocessor = fit_sequence_preprocessor(train_values)
    x_train = transform_sequences(train_values, preprocessor)
    x_validation = transform_sequences(validation_values, preprocessor)
    split_at = max(1, int(len(x_train) * 0.85))
    if split_at >= len(x_train):
        split_at = len(x_train) - 1
    y_train = train_targets[:split_at]
    positives = max(int(y_train.sum()), 1)
    negatives = max(int(len(y_train) - positives), 1)
    class_weight = {0: len(y_train) / (2 * negatives), 1: len(y_train) / (2 * positives)}
    model = build_compact_tcn(
        sequence_length, x_train.shape[-1], filters, dropout, dilations
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    started = time.perf_counter()
    history = model.fit(
        x_train[:split_at],
        y_train,
        validation_data=(x_train[split_at:], train_targets[split_at:]),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[callback],
        verbose=0,
        shuffle=False,
    )
    probabilities = model.predict(x_validation, batch_size=batch_size, verbose=0).ravel()
    elapsed = time.perf_counter() - started
    tf.keras.backend.clear_session()
    return probabilities, {
        "seed": seed,
        "training_rows": int(len(train_values)),
        "early_stopping_training_rows": int(split_at),
        "early_stopping_validation_rows": int(len(train_values) - split_at),
        "epochs_run": int(len(history.history["loss"])),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "training_seconds": float(elapsed),
        "preprocessing_fit_scope": "fold training sequences only",
        "imputer_statistics": [float(value) for value in preprocessor.imputer.statistics_],
        "scaler_center": [float(value) for value in preprocessor.scaler.center_],
        "scaler_scale": [float(value) for value in preprocessor.scaler.scale_],
    }
