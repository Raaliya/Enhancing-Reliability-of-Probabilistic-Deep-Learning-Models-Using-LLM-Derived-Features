import os
import math
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\stress_analysis_normalized.csv"
OUTPUT_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\outputs_exp01_stress_bnn"

TARGET_COL = "confidence"

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 0.0007
WEIGHT_DECAY = 1e-6
MAX_KL_WEIGHT = 5e-5
GRAD_CLIP = 3.0
EARLY_STOPPING_PATIENCE = 40

HIDDEN_1 = 32
HIDDEN_2 = 16
MC_SAMPLES_TEST = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# BNN LAYERS
# =========================================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.03))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(-4.5))

        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.03))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(-4.5))

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho) + 1e-6
        bias_sigma = F.softplus(self.bias_rho) + 1e-6

        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho) + 1e-6
        bias_sigma = F.softplus(self.bias_rho) + 1e-6

        prior_var = self.prior_sigma ** 2
        prior_sigma_tensor = torch.tensor(self.prior_sigma, device=weight_sigma.device)

        weight_kl = (
            torch.log(prior_sigma_tensor / weight_sigma)
            + (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * prior_var)
            - 0.5
        ).sum()

        bias_kl = (
            torch.log(prior_sigma_tensor / bias_sigma)
            + (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * prior_var)
            - 0.5
        ).sum()

        return weight_kl + bias_kl


class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, hidden1=32, hidden2=16):
        super().__init__()
        self.b1 = BayesianLinear(input_dim, hidden1)
        self.b2 = BayesianLinear(hidden1, hidden2)
        self.b3 = BayesianLinear(hidden2, 1)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x):
        x = torch.relu(self.b1(x))
        x = self.dropout(x)
        x = torch.relu(self.b2(x))
        x = self.dropout(x)
        x = self.b3(x)

        # bounded output because confidence is in 0-1 range
        x = torch.sigmoid(x)
        return x

    def kl_loss(self):
        return self.b1.kl_loss() + self.b2.kl_loss() + self.b3.kl_loss()


# =========================================================
# DATA PREP
# =========================================================
def load_baseline_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Baseline = original numeric columns only, excluding target
    X = df.select_dtypes(include=[np.number]).copy()

    if TARGET_COL not in X.columns:
        raise ValueError(f"Target '{TARGET_COL}' is not numeric in the dataset.")

    X = X.drop(columns=[TARGET_COL], errors="ignore")

    # remove invalid numeric values
    X = X.replace([np.inf, -np.inf], np.nan)

    # keep only rows with valid target
    valid_mask = y.notna() & np.isfinite(y)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    # drop all-NaN columns
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print("\nDropping all-NaN columns:")
        print(all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric baseline feature columns remain.")

    return df, X, y


def split_scale_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # fill NaNs using train medians only
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0.0)
    X_val = X_val.fillna(train_medians).fillna(0.0)
    X_test = X_test.fillna(train_medians).fillna(0.0)

    # drop zero variance columns from train only
    train_std = X_train.std(ddof=0)
    keep_cols = train_std[train_std > 0].index.tolist()

    X_train = X_train[keep_cols].copy()
    X_val = X_val[keep_cols].copy()
    X_test = X_test[keep_cols].copy()

    if len(keep_cols) == 0:
        raise ValueError("All baseline feature columns were zero-variance after cleaning.")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # scale target into 0-1
    y_train_arr = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_arr = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_arr = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_arr)
    y_val_scaled = y_scaler.transform(y_val_arr)
    y_test_scaled = y_scaler.transform(y_test_arr)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_t, X_val_t, X_test_t,
        y_train_t, y_val_t, y_test_t,
        train_loader, x_scaler, y_scaler, keep_cols
    )


# =========================================================
# TRAINING
# =========================================================
def evaluate_validation(model, X_val_t, y_val, y_scaler, mc_samples=30):
    model.eval()
    preds = []

    with torch.no_grad():
        for _ in range(mc_samples):
            pred = model(X_val_t.to(DEVICE)).cpu().numpy().reshape(-1, 1)
            preds.append(pred)

    preds = np.array(preds)
    pred_mean_scaled = preds.mean(axis=0)
    pred_mean_scaled = np.clip(pred_mean_scaled, 0.0, 1.0)

    pred_mean = y_scaler.inverse_transform(pred_mean_scaled).reshape(-1)
    y_true = y_val.to_numpy(dtype=np.float64)

    rmse = math.sqrt(mean_squared_error(y_true, pred_mean))
    return rmse


def train_model(model, train_loader, X_val_t, y_val, y_scaler, n_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_rmse = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        kl_weight = MAX_KL_WEIGHT * min(1.0, epoch / 100.0)

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            preds = model(xb)
            mse = F.mse_loss(preds, yb)
            kl = model.kl_loss() / n_train
            loss = mse + kl_weight * kl

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            valid_batches += len(xb)

        if valid_batches == 0:
            raise ValueError("Training failed: all batches produced non-finite loss.")

        epoch_loss /= valid_batches
        val_rmse = evaluate_validation(model, X_val_t, y_val, y_scaler, mc_samples=30)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {epoch_loss:.6f} | "
                f"Val RMSE: {val_rmse:.6f} | "
                f"KL Weight: {kl_weight:.8f}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse


# =========================================================
# MC PREDICTION
# =========================================================
def predict_mc(model, X_test_t, y_scaler, mc_samples=100):
    model.eval()
    preds = []

    with torch.no_grad():
        for _ in range(mc_samples):
            pred_scaled = model(X_test_t.to(DEVICE)).cpu().numpy().reshape(-1, 1)
            preds.append(pred_scaled)

    preds = np.array(preds, dtype=np.float64)
    preds = np.clip(preds, 0.0, 1.0)

    pred_mean_scaled = preds.mean(axis=0)
    pred_std_scaled = preds.std(axis=0)

    pred_mean = y_scaler.inverse_transform(pred_mean_scaled).reshape(-1)

    y_min = y_scaler.data_min_[0]
    y_max = y_scaler.data_max_[0]
    scale_range = float(y_max - y_min)
    pred_std = pred_std_scaled.reshape(-1) * scale_range

    return pred_mean, pred_std


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_STATE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Device:", DEVICE)
    print("Loading baseline dataset...")

    original_df, X, y = load_baseline_data()

    print(f"Rows used before split : {len(X)}")
    print(f"Target                 : {TARGET_COL}")
    print(f"Initial feature count  : {X.shape[1]}")

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_t, X_val_t, X_test_t,
        y_train_t, y_val_t, y_test_t,
        train_loader, x_scaler, y_scaler, keep_cols
    ) = split_scale_data(X, y)

    print(f"Train rows             : {len(X_train)}")
    print(f"Validation rows        : {len(X_val)}")
    print(f"Test rows              : {len(X_test)}")
    print(f"Final feature count    : {len(keep_cols)}")

    model = BayesianRegressor(
        input_dim=X_train_t.shape[1],
        hidden1=HIDDEN_1,
        hidden2=HIDDEN_2
    ).to(DEVICE)

    print("\nTraining baseline Bayesian Neural Network...")
    model, best_val_rmse = train_model(
        model=model,
        train_loader=train_loader,
        X_val_t=X_val_t,
        y_val=y_val,
        y_scaler=y_scaler,
        n_train=len(X_train)
    )

    print("\nRunning MC prediction on test set...")
    y_pred_mean, y_pred_std = predict_mc(
        model=model,
        X_test_t=X_test_t,
        y_scaler=y_scaler,
        mc_samples=MC_SAMPLES_TEST
    )

    y_true = y_test.to_numpy(dtype=np.float64)
    y_pred_mean = np.nan_to_num(y_pred_mean, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_std = np.nan_to_num(y_pred_std, nan=0.0, posinf=0.0, neginf=0.0)

    mae = mean_absolute_error(y_true, y_pred_mean)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2 = r2_score(y_true, y_pred_mean)

    print("\n===== BNN BASELINE EXPERIMENT 01 RESULTS =====")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"Mean predictive std: {float(np.mean(y_pred_std)):.6f}")
    print(f"Best validation RMSE: {best_val_rmse:.6f}")

    metrics = {
        "model": "BNN_Exp01_Stress_Baseline",
        "input_csv": INPUT_CSV,
        "target": TARGET_COL,
        "n_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "n_features_final": int(len(keep_cols)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mean_predictive_std": float(np.mean(y_pred_std)),
        "best_validation_rmse": float(best_val_rmse),
        "feature_columns": keep_cols,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_bnn_exp01_stress_baseline.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = X_test.reset_index(drop=True).copy()
    pred_df["y_true"] = y_test.reset_index(drop=True)
    pred_df["y_pred_mean"] = y_pred_mean
    pred_df["y_pred_std"] = y_pred_std

    pred_path = os.path.join(OUTPUT_DIR, "predictions_bnn_exp01_stress_baseline.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    feat_path = os.path.join(OUTPUT_DIR, "feature_columns_bnn_exp01_stress_baseline.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for col in keep_cols:
            f.write(col + "\n")

    print("\nSaved files:")
    print(metrics_path)
    print(pred_path)
    print(feat_path)


if __name__ == "__main__":
    main()
