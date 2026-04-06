#!/usr/bin/env python3
"""
Compute RF baseline with 30 physics-informed features (same as MA-PINN agents).
This isolates the feature engineering contribution from the multi-agent architecture.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from orchestrator import engineer_hydro_physics_features

DATA_PATH = project_root / "data" / "processed" / "coupled_enriched.pt"
N_SEEDS = 3


def main():
    # Load coupled enriched data
    data = torch.load(DATA_PATH, weights_only=False)
    X = data['X'].numpy()
    Y = data['Y'].numpy()

    # Build the same 30 features as the MA-PINN agents
    X_flow_raw = X[:, 0:4]       # porosity, permeability, depth, time
    X_geochem = X[:, 4:12]       # 8 geochem features
    X_hysteresis = X[:, 12:15]   # 3 hysteresis features

    # Engineer hydro physics features (4 → 19)
    X_hydro_physics = engineer_hydro_physics_features(X_flow_raw)

    # Concatenate all 30 features (19 + 8 + 3)
    X_30 = np.hstack([X_hydro_physics, X_geochem, X_hysteresis])

    print(f"Data: {len(X)} samples")
    print(f"X_flow (4 raw): {X_flow_raw.shape}")
    print(f"X_hydro_physics (19): {X_hydro_physics.shape}")
    print(f"X_geochem (8): {X_geochem.shape}")
    print(f"X_hysteresis (3): {X_hysteresis.shape}")
    print(f"X_30 (all features): {X_30.shape}")

    results_4 = []
    results_30 = []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        X_train_4, X_test_4, Y_train, Y_test = train_test_split(
            X_flow_raw, Y, test_size=0.2, random_state=seed
        )
        X_train_30, X_test_30 = train_test_split(
            X_30, test_size=0.2, random_state=seed
        )[0], train_test_split(X_30, test_size=0.2, random_state=seed)[1]

        # Scale
        scaler_4 = StandardScaler().fit(X_train_4)
        scaler_30 = StandardScaler().fit(X_train_30)
        scaler_Y = StandardScaler().fit(Y_train)

        X_train_4n = scaler_4.transform(X_train_4)
        X_test_4n = scaler_4.transform(X_test_4)
        X_train_30n = scaler_30.transform(X_train_30)
        X_test_30n = scaler_30.transform(X_test_30)
        Y_train_n = scaler_Y.transform(Y_train)
        Y_test_n = scaler_Y.transform(Y_test)

        y_true = scaler_Y.inverse_transform(Y_test_n)

        # RF with 4 raw features
        print("  RF (4 raw features)...")
        rf4 = RandomForestRegressor(n_estimators=200, max_depth=20,
                                     min_samples_split=5, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed)
        rf4.fit(X_train_4n, Y_train_n)
        pred_4 = scaler_Y.inverse_transform(rf4.predict(X_test_4n))
        r2_4 = r2_score(y_true, pred_4, multioutput='uniform_average')
        r2_4_P = r2_score(y_true[:, 0], pred_4[:, 0])
        r2_4_Sw = r2_score(y_true[:, 1], pred_4[:, 1])
        r2_4_Sg = r2_score(y_true[:, 2], pred_4[:, 2])
        results_4.append({'r2': r2_4, 'r2_P': r2_4_P, 'r2_Sw': r2_4_Sw, 'r2_Sg': r2_4_Sg})
        print(f"    R²={r2_4:.4f} | P={r2_4_P:.4f} Sw={r2_4_Sw:.4f} Sg={r2_4_Sg:.4f}")

        # RF with 30 physics features
        print("  RF (30 physics features)...")
        rf30 = RandomForestRegressor(n_estimators=200, max_depth=20,
                                      min_samples_split=5, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
        rf30.fit(X_train_30n, Y_train_n)
        pred_30 = scaler_Y.inverse_transform(rf30.predict(X_test_30n))
        r2_30 = r2_score(y_true, pred_30, multioutput='uniform_average')
        r2_30_P = r2_score(y_true[:, 0], pred_30[:, 0])
        r2_30_Sw = r2_score(y_true[:, 1], pred_30[:, 1])
        r2_30_Sg = r2_score(y_true[:, 2], pred_30[:, 2])
        results_30.append({'r2': r2_30, 'r2_P': r2_30_P, 'r2_Sw': r2_30_Sw, 'r2_Sg': r2_30_Sg})
        print(f"    R²={r2_30:.4f} | P={r2_30_P:.4f} Sw={r2_30_Sw:.4f} Sg={r2_30_Sg:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    r2_4_mean = np.mean([r['r2'] for r in results_4])
    r2_30_mean = np.mean([r['r2'] for r in results_30])
    ma_pinn = 0.976

    print(f"\nRF (4 raw features):    R² = {r2_4_mean:.4f}")
    print(f"RF (30 physics feat.):  R² = {r2_30_mean:.4f}")
    print(f"MA-PINN+MARL:           R² = {ma_pinn:.4f}")
    print(f"\nFeature engineering alone:  ΔR² = {r2_30_mean - r2_4_mean:+.4f} (RF30 - RF4)")
    print(f"Multi-agent architecture:  ΔR² = {ma_pinn - r2_30_mean:+.4f} (MA-PINN - RF30)")

    # Per-variable for RF30
    print(f"\nRF (30 feat.) per-variable:")
    print(f"  R²(P)  = {np.mean([r['r2_P'] for r in results_30]):.4f}")
    print(f"  R²(Sw) = {np.mean([r['r2_Sw'] for r in results_30]):.4f}")
    print(f"  R²(Sg) = {np.mean([r['r2_Sg'] for r in results_30]):.4f}")

    # Save
    out = project_root / "results" / "rf30_baseline.json"
    with open(out, 'w') as f:
        json.dump({
            'rf_4_raw': results_4,
            'rf_30_physics': results_30,
            'summary': {
                'rf4_r2_mean': float(r2_4_mean),
                'rf30_r2_mean': float(r2_30_mean),
                'delta_features': float(r2_30_mean - r2_4_mean),
                'delta_architecture': float(ma_pinn - r2_30_mean),
            }
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
