# deep_ensembles_flipkart_baseline_train.py
# Deep Ensembles baseline for regression (MAE/RMSE/R^2)
# Run example:
# python deep_ensembles_flipkart_baseline_train.py --data_path "D:\...\flipkart_2000_seed42.csv" --target rating

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Model (simple MLP regressor)
# ---------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Data prep
# ---------------------------
def load_and_prepare(
    data_path: str,
    target: str,
    drop_cols: list[str],
    test_size: float,
    seed: int,
    scale_x: bool = True
):
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Drop columns if they exist
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Drop rows with missing target
    df = df.dropna(subset=[target]).copy()

    y = df[target].astype(float).values.reshape(-1, 1)
    X = df.drop(columns=[target]).copy()

    # Encode non-numeric columns safely for baseline
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes

    X = X.fillna(0).astype(np.float32).values
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = None
    if scale_x:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return df, X_train, X_test, y_train, y_test, scaler


# ---------------------------
# Train one model
# ---------------------------
def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

        # Light progress printing
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = loss_fn(val_pred, y_val.to(device)).item()
            model.train()
            print(f"  Epoch {epoch:4d}/{epochs} | Train MSE: {total/n:.6f} | Val MSE: {val_loss:.6f}")

    return model


# ---------------------------
# Main Deep Ensemble Experiment
# ---------------------------
def run_deep_ensemble(
    X_train, y_train, X_test, y_test,
    n_models: int,
    base_seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    device: str
):
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    ensemble_preds = []
    models = []

    # Optional: create a small validation split from training (for sanity loss printing)
    # (Keep simple: use test as "val" printing; you can replace with a proper val split if needed.)
    X_val, y_val = Xte, yte

    for i in range(n_models):
        seed_i = base_seed + i * 100
        set_seed(seed_i)
        print(f"\nTraining model {i+1}/{n_models} (seed={seed_i})")

        model = MLPRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
        model = train_one_model(
            model=model,
            train_loader=train_loader,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            pred = model(Xte.to(device)).cpu().numpy().reshape(-1)  # (N,)
        ensemble_preds.append(pred)
        models.append(model)

    ensemble_preds = np.stack(ensemble_preds, axis=0)  # (M, N)
    pred_mean = ensemble_preds.mean(axis=0)            # (N,)
    pred_std = ensemble_preds.std(axis=0)              # (N,)  (useful for uncertainty plots)

    y_true = y_test.reshape(-1)

    mae = mean_absolute_error(y_true, pred_mean)
    rmse = math.sqrt(mean_squared_error(y_true, pred_mean))
    r2 = r2_score(y_true, pred_mean)

    return models, pred_mean, pred_std, mae, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="Path to Flipkart CSV")
    ap.add_argument("--target", type=str, default="rating", help="Target column name (default: rating)")
    ap.add_argument("--drop_cols", type=str, default="", help="Comma-separated columns to drop (e.g., product_id,review_id)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_models", type=int, default=5, help="Number of ensemble members")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--scale_x", action="store_true", help="Standardize features (recommended)")
    ap.add_argument("--out_pred_csv", type=str, default="flipkart_deep_ensemble_baseline_predictions.csv")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    df, X_train, X_test, y_train, y_test, _ = load_and_prepare(
        data_path=args.data_path,
        target=args.target,
        drop_cols=drop_cols,
        test_size=args.test_size,
        seed=args.seed,
        scale_x=args.scale_x
    )

    print(f"Loaded: {df.shape}")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Features: {X_train.shape[1]} | Target: {args.target}")
    print(f"Ensemble members: {args.n_models}")

    models, pred_mean, pred_std, mae, rmse, r2 = run_deep_ensemble(
        X_train, y_train, X_test, y_test,
        n_models=args.n_models,
        base_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device
    )

    print("\n===== Baseline Results (Flipkart: Deep Ensembles) =====")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")

    # Save predictions (mean + std) for later analysis
    out = pd.DataFrame({
        "y_true": y_test.reshape(-1),
        "y_pred_mean": pred_mean,
        "y_pred_std": pred_std
    })
    out.to_csv(args.out_pred_csv, index=False)
    print(f"\nSaved predictions to: {os.path.abspath(args.out_pred_csv)}")


if __name__ == "__main__":
    main()
    
