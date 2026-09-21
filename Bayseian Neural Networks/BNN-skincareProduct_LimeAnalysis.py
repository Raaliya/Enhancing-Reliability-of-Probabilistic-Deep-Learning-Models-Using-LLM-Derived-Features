"""LIME analysis for the enriched Skincare Bayesian Neural Network.

The script trains a variational BNN using the eight structured skincare
features plus up to 15 LLM-derived numeric features, evaluates the model, and
creates a single-instance HTML LIME dashboard.

Place ``skincare_2000_seed42_llm.csv`` beside this file, or set the
``SKINCARE_LLM_CSV`` environment variable to the CSV path.
"""

from __future__ import annotations

import html
import os
import random
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 600
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
KL_WEIGHT = 1e-4
PRIOR_SIGMA = 1.0

TOP_N_LLM_FEATURES = 15
TEST_INSTANCE_INDEX = 10
LIME_SAMPLES = 2500
MC_SAMPLES = 100
OPEN_DASHBOARD = True

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(
    os.environ.get(
        "SKINCARE_LLM_CSV",
        SCRIPT_DIR / "skincare_2000_seed42_llm.csv",
    )
)
OUTPUT_HTML = Path(
    os.environ.get(
        "SKINCARE_LIME_HTML",
        SCRIPT_DIR / "bnn_skincare_lime_dashboard.html",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_llm_feature(column: str) -> bool:
    name = str(column).lower()
    return (
        name.startswith("llm_")
        or name.endswith("_present")
        or name.endswith("_polarity")
        or name.endswith("_intensity")
    )


def display_name(column: str) -> str:
    if is_llm_feature(column) and not column.startswith("llm_"):
        return f"llm_{column}"
    return column


class BayesianLinear(nn.Module):
    """Bayesian linear layer with Gaussian prior and posterior."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float):
        super().__init__()
        self.prior_sigma = float(prior_sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(-3, 0.1))

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(inputs, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
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
    def __init__(self, input_dim: int):
        super().__init__()
        self.b1 = BayesianLinear(input_dim, 64, PRIOR_SIGMA)
        self.b2 = BayesianLinear(64, 32, PRIOR_SIGMA)
        self.b3 = BayesianLinear(32, 1, PRIOR_SIGMA)

    def forward(self, inputs: torch.Tensor, sample: bool = True) -> torch.Tensor:
        hidden = F.relu(self.b1(inputs, sample))
        hidden = F.relu(self.b2(hidden, sample))
        return self.b3(hidden, sample)

    def kl_divergence(self) -> torch.Tensor:
        return sum(layer.kl_divergence() for layer in (self.b1, self.b2, self.b3))


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place skincare_2000_seed42_llm.csv beside the script or set "
            "SKINCARE_LLM_CSV."
        )

    data = pd.read_csv(DATA_FILE, low_memory=False)
    required = BASE_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns missing from the CSV: {missing}")

    llm_candidates = [
        column
        for column in data.columns
        if is_llm_feature(column) and column not in required
    ]
    if not llm_candidates:
        raise ValueError("No LLM-derived columns were found in the dataset.")

    for column in required + llm_candidates:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    usable_llm = [column for column in llm_candidates if data[column].notna().any()]
    if not usable_llm:
        raise ValueError("The detected LLM columns contain no numeric values.")

    return data, usable_llm


def select_llm_features(
    train_data: pd.DataFrame,
    llm_candidates: list[str],
) -> list[str]:
    """Select LLM features using training data only."""
    x_mi = train_data[llm_candidates].copy()
    medians = x_mi.median().fillna(0.0)
    x_mi = x_mi.fillna(medians)
    y_mi = train_data[TARGET_COLUMN].to_numpy()

    scores = mutual_info_regression(x_mi, y_mi, random_state=RANDOM_STATE)
    ranking = pd.Series(scores, index=llm_candidates).sort_values(ascending=False)
    number_to_select = min(TOP_N_LLM_FEATURES, len(ranking))
    return ranking.head(number_to_select).index.tolist()


def train_model(
    x_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    device: torch.device,
) -> BNNRegressor:
    x_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = BNNRegressor(x_train_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    n_train = len(x_train_scaled)

    print("\nTraining skincare BNN...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(features, sample=True)
            mse = criterion(predictions, targets)
            normalized_kl = model.kl_divergence() / n_train
            loss = mse + KL_WEIGHT * normalized_kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | loss={epoch_loss / len(loader):.6f}")

    return model


def make_dashboard(
    explanation,
    instance: pd.Series,
    selected_features: list[str],
    actual_value: float,
    mc_predictions: np.ndarray,
    metrics: dict[str, float],
) -> str:
    prediction_mean = float(mc_predictions.mean())
    prediction_std = float(mc_predictions.std(ddof=1))
    lower = float(np.percentile(mc_predictions, 2.5))
    upper = float(np.percentile(mc_predictions, 97.5))

    contribution_by_rule = explanation.as_list()
    table_rows = "".join(
        f"<tr><td>{html.escape(display_name(feature))}</td>"
        f"<td>{float(instance[feature]):.4f}</td></tr>"
        for feature in selected_features
    )

    lime_document = html.escape(explanation.as_html(), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Skincare BNN LIME Dashboard</title>
<style>
body {{font-family:Arial,sans-serif;margin:0;padding:18px;background:#f5f6f8;color:#222}}
h1 {{margin-top:0}} .grid {{display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:14px}}
.card {{background:white;padding:18px;border-radius:12px;box-shadow:0 2px 10px #00000012}}
.metric {{display:flex;justify-content:space-between;padding:8px;background:#f8f8f8;margin:7px 0;border-radius:7px}}
iframe {{width:100%;height:720px;border:0}} table {{width:100%;border-collapse:collapse;font-size:13px}}
th,td {{text-align:left;border-bottom:1px solid #e5e5e5;padding:7px}}
.note {{font-size:12px;color:#666}} @media(max-width:1200px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>BNN Skincare – Single-Instance LIME Explanation</h1>
<p>Test instance: {TEST_INSTANCE_INDEX} | Features displayed: {len(selected_features)}</p>
<div class="grid">
  <section class="card">
    <h2>Prediction Summary</h2>
    <div class="metric"><span>Predicted mean</span><b>{prediction_mean:.3f}</b></div>
    <div class="metric"><span>Actual rating</span><b>{actual_value:.3f}</b></div>
    <div class="metric"><span>Predictive std</span><b>{prediction_std:.3f}</b></div>
    <div class="metric"><span>95% interval</span><b>[{lower:.3f}, {upper:.3f}]</b></div>
    <h2>Test Performance</h2>
    <div class="metric"><span>MAE</span><b>{metrics['mae']:.4f}</b></div>
    <div class="metric"><span>RMSE</span><b>{metrics['rmse']:.4f}</b></div>
    <div class="metric"><span>R²</span><b>{metrics['r2']:.4f}</b></div>
  </section>
  <section class="card"><h2>LIME Explanation</h2><iframe srcdoc="{lime_document}"></iframe></section>
  <section class="card">
    <h2>Feature Values</h2>
    <table><thead><tr><th>Feature</th><th>Raw value</th></tr></thead><tbody>{table_rows}</tbody></table>
    <p class="note">LIME rules and contributions are displayed in the centre panel.</p>
  </section>
</div>
<!-- Number of LIME rules: {len(contribution_by_rule)} -->
</body></html>"""


def main() -> None:
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data, llm_candidates = load_dataset()
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if TEST_INSTANCE_INDEX >= len(test_indices):
        raise ValueError(
            f"TEST_INSTANCE_INDEX={TEST_INSTANCE_INDEX} is out of range for "
            f"the {len(test_indices)} test observations."
        )

    selected_llm = select_llm_features(data.iloc[train_indices], llm_candidates)
    selected_features = BASE_FEATURES + selected_llm
    print("\nSelected LLM features:")
    for feature in selected_llm:
        print(f" - {feature}")

    x_train = data.loc[train_indices, selected_features].copy()
    x_test = data.loc[test_indices, selected_features].copy()
    y_train = data.loc[train_indices, TARGET_COLUMN].to_numpy().reshape(-1, 1)
    y_test = data.loc[test_indices, TARGET_COLUMN].to_numpy().reshape(-1, 1)

    training_medians = x_train.median().fillna(0.0)
    x_train = x_train.fillna(training_medians)
    x_test = x_test.fillna(training_medians)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = train_model(x_train_scaled, y_train_scaled, device)
    model.eval()

    @torch.no_grad()
    def predict_raw(raw_values: np.ndarray, sample: bool = False) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(raw_values), columns=selected_features)
        frame = frame.fillna(training_medians)
        scaled = x_scaler.transform(frame)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
        scaled_predictions = model(tensor, sample=sample).cpu().numpy()
        return y_scaler.inverse_transform(scaled_predictions).ravel()

    deterministic_predictions = predict_raw(x_test.to_numpy(), sample=False)
    metrics = {
        "mae": float(mean_absolute_error(y_test.ravel(), deterministic_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test.ravel(), deterministic_predictions))),
        "r2": float(r2_score(y_test.ravel(), deterministic_predictions)),
    }
    print("\nBNN test performance")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²  : {metrics['r2']:.4f}")

    instance = x_test.iloc[TEST_INSTANCE_INDEX]
    actual_value = float(y_test[TEST_INSTANCE_INDEX, 0])

    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=[display_name(feature) for feature in selected_features],
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        data_row=instance.to_numpy(dtype=float),
        predict_fn=predict_raw,
        num_features=len(selected_features),
        num_samples=LIME_SAMPLES,
    )

    mc_predictions = np.stack(
        [predict_raw(instance.to_numpy().reshape(1, -1), sample=True)[0] for _ in range(MC_SAMPLES)]
    )
    dashboard = make_dashboard(
        explanation,
        instance,
        selected_features,
        actual_value,
        mc_predictions,
        metrics,
    )
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(dashboard, encoding="utf-8")
    print(f"\nDashboard saved to: {OUTPUT_HTML}")

    if OPEN_DASHBOARD:
        webbrowser.open(OUTPUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()
