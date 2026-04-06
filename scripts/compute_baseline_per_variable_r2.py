#!/usr/bin/env python3
"""
Compute per-variable R² (P, Sw, Sg) for RF and MLP baselines.
Uses coupled_enriched.pt with X[:,:4] (porosity, permeability, depth, time)
to match the paper's evaluation setup.
Runs 3 seeds with 80/20 split to get mean ± std.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
from pathlib import Path

DATA_PATH = Path("/Users/narjisse/Documents/Effat Courses/Microbial/UHS_MultiAgent_IJHE_Final/data/processed/coupled_enriched.pt")
OUTPUT_PATH = Path("/Users/narjisse/Documents/Effat Courses/Microbial/UHS_MultiAgent_IJHE_Final/results")

N_SEEDS = 3


class MLPBaseline(nn.Module):
    """Same MLP architecture as baseline_comparison.py"""
    def __init__(self, input_dim=4, output_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, Y_train, X_val, Y_val, epochs=200, lr=1e-3):
    """Train MLP with same setup as baseline_comparison.py"""
    device = 'cpu'
    model = MLPBaseline(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], hidden=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pred = model(X_v).numpy()
    return y_pred


def main():
    # Load coupled enriched data (same as baseline_comparison.py)
    data = torch.load(DATA_PATH, weights_only=False)
    X = data['X'].numpy()
    Y = data['Y'].numpy()

    # Use only 4 raw flow features (same as baseline_comparison.py line 544)
    X_flow = X[:, :4]  # porosity, permeability, depth, time

    print(f"Data loaded from coupled_enriched.pt:")
    print(f"  X_flow={X_flow.shape}, Y={Y.shape}")
    print(f"  Y columns: P (pressure), Sw (water sat.), Sg (gas sat.)")

    results = {'RandomForest': [], 'MLP': []}

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 80/20 split (same as baseline_comparison.py line 549)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_flow, Y, test_size=0.2, random_state=seed
        )

        # Standardize
        scaler_X = StandardScaler().fit(X_train)
        scaler_Y = StandardScaler().fit(Y_train)
        X_train_n = scaler_X.transform(X_train)
        X_test_n = scaler_X.transform(X_test)
        Y_train_n = scaler_Y.transform(Y_train)
        Y_test_n = scaler_Y.transform(Y_test)

        y_true = scaler_Y.inverse_transform(Y_test_n)

        # --- Random Forest ---
        print("  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed
        )
        rf.fit(X_train_n, Y_train_n)
        y_pred_rf = scaler_Y.inverse_transform(rf.predict(X_test_n))

        r2_overall = r2_score(y_true, y_pred_rf, multioutput='uniform_average')
        r2_P = r2_score(y_true[:, 0], y_pred_rf[:, 0])
        r2_Sw = r2_score(y_true[:, 1], y_pred_rf[:, 1])
        r2_Sg = r2_score(y_true[:, 2], y_pred_rf[:, 2])
        results['RandomForest'].append({
            'seed': seed, 'r2': r2_overall,
            'r2_P': r2_P, 'r2_Sw': r2_Sw, 'r2_Sg': r2_Sg
        })
        print(f"  RF: R²={r2_overall:.4f} | P={r2_P:.4f} Sw={r2_Sw:.4f} Sg={r2_Sg:.4f}")

        # --- MLP (PyTorch, same arch as baseline_comparison.py) ---
        print("  Training MLP...")
        y_pred_mlp_n = train_mlp(X_train_n, Y_train_n, X_test_n, Y_test_n, epochs=200)
        y_pred_mlp = scaler_Y.inverse_transform(y_pred_mlp_n)

        r2_overall_mlp = r2_score(y_true, y_pred_mlp, multioutput='uniform_average')
        r2_P_mlp = r2_score(y_true[:, 0], y_pred_mlp[:, 0])
        r2_Sw_mlp = r2_score(y_true[:, 1], y_pred_mlp[:, 1])
        r2_Sg_mlp = r2_score(y_true[:, 2], y_pred_mlp[:, 2])
        results['MLP'].append({
            'seed': seed, 'r2': r2_overall_mlp,
            'r2_P': r2_P_mlp, 'r2_Sw': r2_Sw_mlp, 'r2_Sg': r2_Sg_mlp
        })
        print(f"  MLP: R²={r2_overall_mlp:.4f} | P={r2_P_mlp:.4f} Sw={r2_Sw_mlp:.4f} Sg={r2_Sg_mlp:.4f}")

    # Aggregate
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std across 3 seeds)")
    print("=" * 60)

    summary = {}
    for model_name, model_results in results.items():
        r2s = np.array([r['r2'] for r in model_results])
        r2_Ps = np.array([r['r2_P'] for r in model_results])
        r2_Sws = np.array([r['r2_Sw'] for r in model_results])
        r2_Sgs = np.array([r['r2_Sg'] for r in model_results])

        summary[model_name] = {
            'r2_mean': float(r2s.mean()), 'r2_std': float(r2s.std()),
            'r2_P_mean': float(r2_Ps.mean()), 'r2_P_std': float(r2_Ps.std()),
            'r2_Sw_mean': float(r2_Sws.mean()), 'r2_Sw_std': float(r2_Sws.std()),
            'r2_Sg_mean': float(r2_Sgs.mean()), 'r2_Sg_std': float(r2_Sgs.std()),
        }

        print(f"\n{model_name}:")
        print(f"  Overall R² = {r2s.mean():.4f} ± {r2s.std():.4f}")
        print(f"  R²(P)      = {r2_Ps.mean():.4f} ± {r2_Ps.std():.4f}")
        print(f"  R²(Sw)     = {r2_Sws.mean():.4f} ± {r2_Sws.std():.4f}")
        print(f"  R²(Sg)     = {r2_Sgs.mean():.4f} ± {r2_Sgs.std():.4f}")

    # Save
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / "baseline_per_variable_r2.json"
    with open(out_file, 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
