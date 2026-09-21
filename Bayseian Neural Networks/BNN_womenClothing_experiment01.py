"""Experiment 01: Bayesian neural-network baseline for Women Clothing reviews.

The baseline uses only the three structured features used by the corresponding
Experiment 02 script. It deliberately contains no LLM-derived features.

Expected input file (by default, in the same directory as this script):
    women_clothing_reviews.csv
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
# CONFIGURATION -- kept aligned with Experiment 02
# ============================================================
RANDOM_STATE = 42
N_ROWS = 2000
TEST_SIZE = 0.20

EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

PRIOR_SIGMA = 1.0
KL_WEIGHT = 1e-3
N_PRED_SAMPLES = 50

# A relative default path makes the script portable for GitHub users.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(
    os.environ.get("WOMEN_CLOTHING_CSV", SCRIPT_DIR / "women_clothing_reviews.csv")
)
OUTPUT_DIR = Path(
    os.environ.get("WOMEN_BNN_OUTPUT_DIR", SCRIPT_DIR / "outputs_exp01_women_bnn")
)

PREDICTIONS_FILE = OUTPUT_DIR / "women_exp01_bnn_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "women_exp01_bnn_metrics.txt"

TARGET_COLUMN = "Rating"
BASE_FEATURES = ["Age", "Recommended IND", "Positive Feedback Count"]


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings improve repeatability on CUDA. Exact results can still
    # differ across PyTorch/CUDA versions and hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# VARIATIONAL BAYESIAN LAYER
# ============================================================
class BayesianLinear(nn.Module):
    """Linear layer with a factorized Gaussian posterior over parameters."""

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
        """Return KL(q || p) for Gaussian posterior q and prior p."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        prior_variance = self.prior_sigma**2

        kl_weight = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma.square() + self.weight_mu.square()) / (2 * prior_variance)
            - 0.5
        ).sum()
        kl_bias = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma.square() + self.bias_mu.square()) / (2 * prior_variance)
            - 0.5
        ).sum()
        return kl_weight + kl_bias


class BNNRegressor(nn.Module):
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


# ============================================================
# DATA
# ============================================================
def load_baseline_data(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            "Place women_clothing_reviews.csv beside this script, or set the "
            "WOMEN_CLOTHING_CSV environment variable to its path."
        )

    data = pd.read_csv(data_file)
    required = BASE_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing from the dataset: {missing}")

    # Preserve the original CSV row number so saved predictions are traceable.
    data = data.copy()
    data["source_row_index"] = np.arange(len(data))

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # A missing target cannot be used for supervised training.
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    if len(data) < N_ROWS:
        raise ValueError(
            f"Only {len(data)} rows have a valid target; {N_ROWS} rows are required."
        )

    # This exactly defines the fixed subset used by this script.
    data = data.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return data


def prepare_train_test(data: pd.DataFrame):
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_features = data.loc[train_indices, BASE_FEATURES].copy()
    test_features = data.loc[test_indices, BASE_FEATURES].copy()

    # Derive imputation values from the training partition only.
    train_medians = train_features.median()
    if train_medians.isna().any():
        bad = train_medians[train_medians.isna()].index.tolist()
        raise ValueError(f"Features contain no usable training values: {bad}")

    train_features = train_features.fillna(train_medians)
    test_features = test_features.fillna(train_medians)

    # Fit preprocessing only on training data to prevent test-set leakage.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features).astype(np.float32)
    x_test = scaler.transform(test_features).astype(np.float32)
    y_train = data.loc[train_indices, TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = data.loc[test_indices, TARGET_COLUMN].to_numpy(dtype=np.float32)

    return x_train, x_test, y_train, y_test, train_indices, test_indices


# ============================================================
# TRAINING AND EVALUATION
# ============================================================
def main() -> None:
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_baseline_data(DATA_FILE)
    x_train, x_test, y_train, y_test, _, test_indices = prepare_train_test(data)

    x_train_tensor = torch.from_numpy(x_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).reshape(-1, 1).to(device)
    x_test_tensor = torch.from_numpy(x_test).to(device)

    model = BNNRegressor(x_train.shape[1], PRIOR_SIGMA).to(device)
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

    # Bayesian layers keep sampling during evaluation. Repeated forward passes
    # therefore approximate the posterior predictive distribution.
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

    print("\n===== Experiment 01 Results (Women | BNN | Baseline) =====")
    print(f"Rows used: {len(data)}")
    print(f"Features: {BASE_FEATURES}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    prediction_table = pd.DataFrame(
        {
            "sampled_row_index": test_indices,
            "source_row_index": data.loc[test_indices, "source_row_index"].to_numpy(),
            "true_rating": y_test,
            "predicted_rating_mean": prediction_mean,
            "predicted_rating_std": prediction_std,
        }
    ).sort_values("sampled_row_index")
    prediction_table.to_csv(PREDICTIONS_FILE, index=False)

    with METRICS_FILE.open("w", encoding="utf-8") as output:
        output.write("Experiment 01 (Women Clothing | BNN | Baseline)\n")
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
