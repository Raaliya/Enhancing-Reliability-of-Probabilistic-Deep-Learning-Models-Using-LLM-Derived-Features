"""Experiment 01: TensorFlow Deep Ensemble baseline for skincare ratings.

The baseline uses only the eight original structured skincare features. The
same enriched CSV may be supplied as Experiment 02; every LLM-derived column is
deliberately ignored so both experiments can use identical rows and splits.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Set before importing TensorFlow for more deterministic GPU behaviour.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf  # noqa: E402

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

N_ENSEMBLES = 5
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.10
EARLY_STOPPING_PATIENCE = 12

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = Path(
    os.environ.get(
        "SKINCARE_CSV",
        SCRIPT_DIR / "reviews_1000_1500_with_llm_features_2000.csv",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "SKINCARE_DE_EXP01_OUTPUT",
        SCRIPT_DIR / "outputs_exp01_deep_ensembles_skincare",
    )
)

TARGET_COLUMN = "rating"
BASE_FEATURES = [
    "loves_count",
    "reviews",
    "price_usd",
    "value_price_usd",
    "sale_price_usd",
    "child_count",
    "child_max_price",
    "child_min_price",
]

PREDICTIONS_CSV = OUTPUT_DIR / "de_exp01_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "de_exp01_metrics.json"
SCALER_PATH = OUTPUT_DIR / "de_exp01_scaler.joblib"
IMPUTER_PATH = OUTPUT_DIR / "de_exp01_imputer.joblib"
FEATURES_TXT = OUTPUT_DIR / "de_exp01_features.txt"
SPLIT_INDICES_NPZ = OUTPUT_DIR / "de_exp01_split_indices.npz"
MODELS_DIR = OUTPUT_DIR / "ensemble_models"


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_CSV}\n"
            "Place the CSV beside this script or set the SKINCARE_CSV "
            "environment variable."
        )

    data = pd.read_csv(INPUT_CSV, low_memory=False).copy()
    required = BASE_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing from the dataset: {missing}")

    # Preserve the original CSV position for traceable test predictions.
    data["source_row_index"] = np.arange(len(data))
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Missing targets are removed, never imputed.
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("At least two rows with valid target values are required.")

    features = data[BASE_FEATURES].copy()
    target = data[TARGET_COLUMN].astype(float).copy()
    return data, features, target


def build_de_model(input_dim: int) -> tf.keras.Model:
    """Build one independently initialized ensemble member."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    set_all_seeds(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading skincare dataset for Experiment 01...")
    data, features, target = load_data()
    print(f"Rows with valid target: {len(data)}")
    print("Baseline features:")
    for feature in BASE_FEATURES:
        print(f" - {feature}")

    # Split explicit row indices so the same split can be reused in Exp-02.
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    x_train = features.iloc[train_indices].copy()
    x_test = features.iloc[test_indices].copy()
    y_train = target.iloc[train_indices].to_numpy(dtype=np.float32)
    y_test = target.iloc[test_indices].to_numpy(dtype=np.float32)

    print(f"Train rows: {len(x_train)}")
    print(f"Test rows : {len(x_test)}")

    # Fit preprocessing only on training data.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
    x_train_scaled = scaler.fit_transform(x_train_imputed).astype(np.float32)
    x_test_scaled = scaler.transform(x_test_imputed).astype(np.float32)

    input_dim = x_train_scaled.shape[1]
    member_predictions: list[np.ndarray] = []
    model_paths: list[str] = []
    training_epochs: list[int] = []

    print(f"\nTraining Deep Ensemble with {N_ENSEMBLES} members...")
    for member_number in range(1, N_ENSEMBLES + 1):
        member_seed = RANDOM_STATE + member_number
        print("\n" + "=" * 72)
        print(
            f"Training ensemble member {member_number}/{N_ENSEMBLES} "
            f"| seed={member_seed}"
        )

        tf.keras.backend.clear_session()
        set_all_seeds(member_seed)
        model = build_de_model(input_dim)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
        history = model.fit(
            x_train_scaled,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping],
        )
        training_epochs.append(len(history.history["loss"]))

        predictions = model.predict(x_test_scaled, verbose=0).reshape(-1)
        member_predictions.append(predictions)

        model_path = MODELS_DIR / f"de_member_{member_number}.keras"
        model.save(model_path)
        model_paths.append(str(model_path))

    prediction_matrix = np.stack(member_predictions, axis=0)
    prediction_mean = prediction_matrix.mean(axis=0)
    prediction_std = prediction_matrix.std(axis=0)

    mae = float(mean_absolute_error(y_test, prediction_mean))
    rmse = float(np.sqrt(mean_squared_error(y_test, prediction_mean)))
    r2 = float(r2_score(y_test, prediction_mean))
    mean_predictive_std = float(prediction_std.mean())

    print("\n===== DEEP ENSEMBLE EXPERIMENT 01 RESULTS =====")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"Mean ensemble std: {mean_predictive_std:.6f}")

    prediction_data = {
        "dataset_row_index": test_indices,
        "source_row_index": data.iloc[test_indices]["source_row_index"].to_numpy(),
        "y_true": y_test,
        "y_pred_mean": prediction_mean,
        "y_pred_std": prediction_std,
    }
    for member_index in range(N_ENSEMBLES):
        prediction_data[f"member_{member_index + 1}_pred"] = prediction_matrix[
            member_index
        ]
    pd.DataFrame(prediction_data).to_csv(
        PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "experiment": "Experiment 01 Deep Ensemble baseline",
        "dataset": "skincare",
        "input_file": str(INPUT_CSV),
        "target": TARGET_COLUMN,
        "rows_total": int(len(data)),
        "rows_train": int(len(train_indices)),
        "rows_test": int(len(test_indices)),
        "num_features": len(BASE_FEATURES),
        "features_used": BASE_FEATURES,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_ensemble_std": mean_predictive_std,
        "n_ensembles": N_ENSEMBLES,
        "epochs_max": EPOCHS,
        "epochs_trained_per_member": training_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "validation_split": VALIDATION_SPLIT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "random_state": RANDOM_STATE,
        "member_seeds": [RANDOM_STATE + i for i in range(1, N_ENSEMBLES + 1)],
        "test_size": TEST_SIZE,
        "framework": "TensorFlow/Keras",
        "model": "Deep Ensemble neural network",
        "member_model_paths": model_paths,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.savez(
        SPLIT_INDICES_NPZ,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    FEATURES_TXT.write_text(
        "Deep Ensemble Experiment 01 baseline features\n"
        + "=" * 46
        + "\n"
        + "\n".join(BASE_FEATURES)
        + "\n",
        encoding="utf-8",
    )

    print("\nSaved files:")
    print(f"- Predictions : {PREDICTIONS_CSV}")
    print(f"- Metrics     : {METRICS_JSON}")
    print(f"- Imputer     : {IMPUTER_PATH}")
    print(f"- Scaler      : {SCALER_PATH}")
    print(f"- Features    : {FEATURES_TXT}")
    print(f"- Split       : {SPLIT_INDICES_NPZ}")
    print(f"- Models      : {MODELS_DIR}")


if __name__ == "__main__":
    main()
