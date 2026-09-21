"""Experiment 01: PyTorch BNN baseline for the skincare dataset.

This script uses only the original structured features. It does not use or
filter on any llm_* columns, even if they are present in the input CSV.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# Keep these model settings identical in Experiments 01 and 02.
# ============================================================
RANDOM_STATE = 42
EXPECTED_ROWS = 2000
TEST_SIZE = 0.20

EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

PRIOR_SIGMA = 1.0
KL_WEIGHT = 1e-3
N_PRED_SAMPLES = 50

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(
    os.environ.get(
        "SKINCARE_CSV",
        SCRIPT_DIR / "skincare_2000_seed42_llm.csv",
    )
)
OUTPUT_DIR = Path(
    os.environ.get("SKINCARE_BNN_OUTPUT_DIR", SCRIPT_DIR / "outputs_exp01_skincare_bnn")
)

PREDICTIONS_FILE = OUTPUT_DIR / "skincare_exp01_bnn_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "skincare_exp01_bnn_metrics.txt"

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


def set_seed(seed: int) -> None:
    """Set the available random-number generators for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BayesianLinear(nn.Module):
    """Linear layer with a factorized Gaussian posterior over its parameters."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be greater than zero.")

        self.prior_sigma = float(prior_sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(-3, 0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(inputs, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL(q || p) for Gaussian posterior q and prior p."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        prior_variance = self.prior_sigma**2

        weight_kl = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma.square() + self.weight_mu.square()) / (2 * prior_variance)
            - 0.5
        ).sum()
        bias_kl = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma.square() + self.bias_mu.square()) / (2 * prior_variance)
            - 0.5
        ).sum()
        return weight_kl + bias_kl


class BNNRegressor(nn.Module):
    """Bayesian neural-network regressor matching the Women experiment design."""

    def __init__(self, input_dim: int, prior_sigma: float = 1.0):
        super().__init__()
        self.b1 = BayesianLinear(input_dim, 64, prior_sigma)
        self.b2 = BayesianLinear(64, 32, prior_sigma)
        self.b3 = BayesianLinear(32, 1, prior_sigma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.b1(inputs))
        hidden = torch.relu(self.b2(hidden))
        return self.b3(hidden)

    def kl_divergence(self) -> torch.Tensor:
        return sum(layer.kl_divergence() for layer in (self.b1, self.b2, self.b3))


def load_data(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            "Place skincare_2000_seed42_llm.csv beside this script, or set "
            "the SKINCARE_CSV environment variable to the dataset path."
        )

    data = pd.read_csv(data_file)
    required_columns = BASE_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing from the CSV: {missing}")

    data = data.copy()
    data["source_row_index"] = np.arange(len(data))
    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Targets must be observed; filling a missing target would invent a label.
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(data) != EXPECTED_ROWS:
        print(
            f"Warning: expected {EXPECTED_ROWS} valid rows but found {len(data)}. "
            "All valid rows will be used."
        )
    return data


def prepare_train_test(data: pd.DataFrame):
    row_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_features = data.loc[train_indices, BASE_FEATURES].copy()
    test_features = data.loc[test_indices, BASE_FEATURES].copy()

    # Learn imputation and scaling only from training data to avoid leakage.
    training_medians = train_features.median()
    unusable = training_medians[training_medians.isna()].index.tolist()
    if unusable:
        raise ValueError(f"Features contain no usable training values: {unusable}")

    train_features = train_features.fillna(training_medians)
    test_features = test_features.fillna(training_medians)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features).astype(np.float32)
    x_test = scaler.transform(test_features).astype(np.float32)
    y_train = data.loc[train_indices, TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = data.loc[test_indices, TARGET_COLUMN].to_numpy(dtype=np.float32)

    return x_train, x_test, y_train, y_test, test_indices


def main() -> None:
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_data(DATA_FILE)
    x_train, x_test, y_train, y_test, test_indices = prepare_train_test(data)

    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).reshape(-1, 1).to(device)
    x_test_tensor = torch.from_numpy(x_test).to(device)

    model = BNNRegressor(input_dim=x_train.shape[1], prior_sigma=PRIOR_SIGMA).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_train = len(x_train)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(n_train, device=device)
        total_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            inputs = x_train_tensor[batch_indices]
            targets = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            predictions = model(inputs)
            mse = F.mse_loss(predictions, targets)
            normalized_kl = model.kl_divergence() / n_train
            loss = mse + KL_WEIGHT * normalized_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d} | summed batch loss = {total_loss:.6f}")

    # Every forward pass samples new Bayesian weights.
    model.eval()
    with torch.no_grad():
        predictive_samples = np.stack(
            [
                model(x_test_tensor).cpu().numpy().reshape(-1)
                for _ in range(N_PRED_SAMPLES)
            ]
        )

    prediction_mean = predictive_samples.mean(axis=0)
    prediction_std = predictive_samples.std(axis=0)

    mae = mean_absolute_error(y_test, prediction_mean)
    rmse = float(np.sqrt(mean_squared_error(y_test, prediction_mean)))
    r2 = r2_score(y_test, prediction_mean)

    print("\n===== Experiment 01 Results (Skincare | BNN | Baseline) =====")
    print(f"Rows used: {len(data)}")
    print(f"Features used: {BASE_FEATURES}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    prediction_table = pd.DataFrame(
        {
            "dataset_row_index": test_indices,
            "source_row_index": data.loc[test_indices, "source_row_index"].to_numpy(),
            "true_rating": y_test,
            "predicted_rating_mean": prediction_mean,
            "predicted_rating_std": prediction_std,
        }
    ).sort_values("dataset_row_index")
    prediction_table.to_csv(PREDICTIONS_FILE, index=False)

    with METRICS_FILE.open("w", encoding="utf-8") as output:
        output.write("Experiment 01 (Skincare | BNN | Baseline)\n")
        output.write(f"Data file: {DATA_FILE}\n")
        output.write(f"Rows used: {len(data)}\n")
        output.write(f"Features: {', '.join(BASE_FEATURES)}\n")
        output.write(f"MAE: {mae:.6f}\n")
        output.write(f"RMSE: {rmse:.6f}\n")
        output.write(f"R2: {r2:.6f}\n")
        output.write(f"Random seed: {RANDOM_STATE}\n")
        output.write(f"Test size: {TEST_SIZE}\n")
        output.write(f"Epochs: {EPOCHS}\n")
        output.write(f"Batch size: {BATCH_SIZE}\n")
        output.write(f"Learning rate: {LEARNING_RATE}\n")
        output.write(f"KL weight: {KL_WEIGHT}\n")
        output.write(f"Prior sigma: {PRIOR_SIGMA}\n")
        output.write(f"Prediction samples: {N_PRED_SAMPLES}\n")

    print(f"\nSaved predictions to: {PREDICTIONS_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
