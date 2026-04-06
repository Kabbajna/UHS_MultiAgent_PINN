#!/usr/bin/env python3
"""
4-Phase MARL Training Pipeline for MA-PINN UHS
===============================================

Phase 1a: Specialist agent pre-training (IL)
Phase 1b: Consensus layer training (IL)
Phase 2:  Per-agent MARL fine-tuning
Phase 3:  Communication fine-tuning
Phase 4:  Recalibration

Usage:
    python run_marl.py [--calib_frac 0.10] [--skip_phase1]
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    PhysicsInformedHydroAgent, GeochemAgent, HysteresisAgent,
    PINNMultiAgentOrchestrator,
    engineer_hydro_physics_features, train_pinn_hydro, train_agent,
    imitation_learning, rl_finetuning, evaluate,
    DEVICE
)
from orchestrator_marl import (
    MARLOrchestrator,
    marl_finetuning, train_communication, recalibrate,
    evaluate_marl
)

project_root = Path(__file__).parent.parent


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load all datasets, engineer features, split, and scale."""
    data_dir = project_root / "data" / "processed"

    # --- Load raw datasets ---
    enriched_data = torch.load(data_dir / "coupled_enriched.pt", weights_only=False)
    X_enriched = enriched_data['X'].numpy()
    Y_coupled = enriched_data['Y'].numpy()

    mrst_data = torch.load(data_dir / "hydro_mrst_only_real.pt", weights_only=False)
    X_mrst_raw = mrst_data['X'].numpy()
    Y_mrst = mrst_data['Y'].numpy()

    phreeqc_data = torch.load(data_dir / "geochem_phreeqc_real.pt", weights_only=False)
    X_phreeqc, Y_phreeqc = phreeqc_data['X'].numpy(), phreeqc_data['Y'].numpy()

    bc_data = torch.load(data_dir / "hysteresis_brooks_corey.pt", weights_only=False)
    X_bc_full, Y_bc = bc_data['X'].numpy(), bc_data['Y'].numpy()
    X_bc = X_bc_full[:, 2:5]  # 3 features: drainage, lambda, history

    print(f"Data: {len(X_enriched)} coupled, {len(X_mrst_raw)} MRST, "
          f"{len(X_phreeqc)} PHREEQC, {len(X_bc)} BC")

    # --- Engineer physics features ---
    X_mrst_physics = engineer_hydro_physics_features(X_mrst_raw)
    X_flow_raw = X_enriched[:, 0:4]
    X_flow_physics = engineer_hydro_physics_features(X_flow_raw)
    print(f"Physics features: MRST {X_mrst_raw.shape[1]}→{X_mrst_physics.shape[1]}, "
          f"Flow {X_flow_raw.shape[1]}→{X_flow_physics.shape[1]}")

    # --- 80/10/10 split on coupled data ---
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_enriched, Y_coupled, test_size=0.10, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.111, random_state=42
    )
    X_flow_temp, X_flow_test = train_test_split(X_flow_physics, test_size=0.10, random_state=42)
    X_flow_train, X_flow_val = train_test_split(X_flow_temp, test_size=0.111, random_state=42)

    print(f"Split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

    # --- Scale ---
    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)
    scaler_X_hydro = StandardScaler().fit(X_mrst_physics)
    scaler_X_flow = StandardScaler().fit(X_flow_train)

    X_train_n = scaler_X.transform(X_train)
    X_val_n = scaler_X.transform(X_val)
    X_test_n = scaler_X.transform(X_test)
    Y_train_n = scaler_Y.transform(Y_train)

    X_flow_train_n = scaler_X_flow.transform(X_flow_train)
    X_flow_val_n = scaler_X_flow.transform(X_flow_val)
    X_flow_test_n = scaler_X_flow.transform(X_flow_test)

    X_mrst_n = scaler_X_hydro.transform(X_mrst_physics)
    Y_mrst_n = scaler_Y.transform(Y_mrst)

    X_geochem_enriched = X_enriched[:, 4:12]
    scaler_X_geochem = StandardScaler().fit(X_geochem_enriched)
    X_geochem_n = scaler_X_geochem.transform(X_geochem_enriched)
    scaler_Y_geochem = StandardScaler().fit(Y_phreeqc)
    Y_phreeqc_n = scaler_Y_geochem.transform(Y_phreeqc)

    scaler_X_bc = StandardScaler().fit(X_bc)
    X_bc_n = scaler_X_bc.transform(X_bc)
    scaler_Y_bc = StandardScaler().fit(Y_bc)
    Y_bc_n = scaler_Y_bc.transform(Y_bc)

    return {
        # Coupled (train/val/test) — normalised
        'X_train_n': X_train_n, 'X_val_n': X_val_n, 'X_test_n': X_test_n,
        'Y_train_n': Y_train_n,
        'X_flow_train_n': X_flow_train_n, 'X_flow_val_n': X_flow_val_n,
        'X_flow_test_n': X_flow_test_n,
        # Coupled (raw for evaluation)
        'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
        # MRST
        'X_mrst_raw': X_mrst_raw, 'X_mrst_n': X_mrst_n, 'X_mrst_physics': X_mrst_physics,
        'Y_mrst': Y_mrst, 'Y_mrst_n': Y_mrst_n,
        # PHREEQC
        'X_geochem_n': X_geochem_n, 'X_phreeqc': X_phreeqc,
        'Y_phreeqc_n': Y_phreeqc_n, 'Y_phreeqc': Y_phreeqc,
        # Brooks-Corey
        'X_bc': X_bc, 'X_bc_n': X_bc_n, 'Y_bc': Y_bc, 'Y_bc_n': Y_bc_n,
        # Scalers
        'scaler_X': scaler_X, 'scaler_Y': scaler_Y,
        'scaler_X_hydro': scaler_X_hydro, 'scaler_X_flow': scaler_X_flow,
        'scaler_Y_geochem': scaler_Y_geochem, 'scaler_Y_bc': scaler_Y_bc,
    }


# =============================================================================
# PHASE 1a: TRAIN SPECIALIST AGENTS
# =============================================================================

def phase_1a(data):
    """Train the three specialist agents on their decoupled datasets."""
    print("\n" + "=" * 70)
    print("PHASE 1a: SPECIALIST AGENT PRE-TRAINING (IL)")
    print("=" * 70)

    # --- Hydro Agent (PINN) ---
    print("\n--- PINN Hydro Agent (19 physics features) ---")
    X_h_tr, X_h_val, Y_h_tr, Y_h_val = train_test_split(
        data['X_mrst_n'], data['Y_mrst_n'], test_size=0.1, random_state=42
    )
    X_h_raw_tr, _ = train_test_split(data['X_mrst_raw'], test_size=0.1, random_state=42)

    hydro_agent = PhysicsInformedHydroAgent(input_dim=19, output_dim=3, hidden=256, n_blocks=4)
    hydro_agent = train_pinn_hydro(hydro_agent, X_h_tr, Y_h_tr, X_h_val, Y_h_val,
                                    X_h_raw_tr, epochs=300)

    hydro_agent.eval()
    with torch.no_grad():
        hydro_pred = hydro_agent(torch.tensor(X_h_val, dtype=torch.float32).to(DEVICE))
    hydro_r2 = r2_score(Y_h_val, hydro_pred.cpu().numpy())
    print(f"  Hydro Agent R²: {hydro_r2:.4f}")

    # --- Geochem Agent ---
    print("\n--- Geochem Agent (8 features) ---")
    n_phreeqc = len(data['Y_phreeqc_n'])
    geochem_indices = np.random.choice(len(data['X_geochem_n']), size=n_phreeqc, replace=False)
    X_geochem_sampled = data['X_geochem_n'][geochem_indices]
    X_g_tr, X_g_val, Y_g_tr, Y_g_val = train_test_split(
        X_geochem_sampled, data['Y_phreeqc_n'], test_size=0.1, random_state=42
    )
    geochem_agent = GeochemAgent(hidden=128, dropout=0.1)
    geochem_agent = train_agent(geochem_agent, X_g_tr, Y_g_tr, X_g_val, Y_g_val, epochs=100)

    # --- Hysteresis Agent ---
    print("\n--- Hysteresis Agent (Brooks-Corey) ---")
    X_b_tr, X_b_val, Y_b_tr, Y_b_val = train_test_split(
        data['X_bc_n'], data['Y_bc_n'], test_size=0.1, random_state=42
    )
    hysteresis_agent = HysteresisAgent(hidden=128, dropout=0.1)
    hysteresis_agent = train_agent(hysteresis_agent, X_b_tr, Y_b_tr, X_b_val, Y_b_val, epochs=100)

    return hydro_agent, geochem_agent, hysteresis_agent, hydro_r2


# =============================================================================
# PHASE 1b: TRAIN CONSENSUS (IL + RL)
# =============================================================================

def phase_1b(hydro_agent, geochem_agent, hysteresis_agent, data, calib_frac=0.10):
    """Train consensus layer via imitation learning + RL fine-tuning."""
    print("\n" + "=" * 70)
    print(f"PHASE 1b: CONSENSUS TRAINING (IL + RL, {calib_frac*100:.0f}% calibration)")
    print("=" * 70)

    n_calib = int(len(data['X_train_n']) * calib_frac)
    idx = np.random.choice(len(data['X_train_n']), n_calib, replace=False)
    X_calib = data['X_train_n'][idx]
    X_hydro_calib = data['X_flow_train_n'][idx]
    Y_calib = data['Y_train_n'][idx]
    print(f"  Calibration samples: {n_calib}")

    # Build the original orchestrator for Phase 1b
    orchestrator = PINNMultiAgentOrchestrator(
        hydro_agent=hydro_agent, geochem_agent=geochem_agent,
        hysteresis_agent=hysteresis_agent, hidden=256
    )

    # IL
    orchestrator = imitation_learning(orchestrator, X_calib, X_hydro_calib, Y_calib, epochs=500)
    metrics_il = evaluate(orchestrator, data['X_val_n'], data['X_flow_val_n'],
                          data['Y_val'], data['scaler_Y'])
    print(f"  After IL (Val): R² = {metrics_il['r2']:.4f}")

    # RL (consensus-only)
    orchestrator = rl_finetuning(orchestrator, X_calib, X_hydro_calib, Y_calib,
                                  data['scaler_Y'], epochs=100)
    metrics_rl = evaluate(orchestrator, data['X_val_n'], data['X_flow_val_n'],
                          data['Y_val'], data['scaler_Y'])
    print(f"  After RL (Val): R² = {metrics_rl['r2']:.4f}")

    return orchestrator, X_calib, X_hydro_calib, Y_calib


# =============================================================================
# BUILD MARL ORCHESTRATOR FROM TRAINED COMPONENTS
# =============================================================================

def build_marl_orchestrator(orchestrator_orig):
    """
    Transfer trained weights from PINNMultiAgentOrchestrator → MARLOrchestrator.
    Communication + summary modules start with fresh (zero-init) weights.
    """
    marl_orch = MARLOrchestrator(
        hydro_agent=orchestrator_orig.hydro,
        geochem_agent=orchestrator_orig.geochem,
        hysteresis_agent=orchestrator_orig.hysteresis,
        hidden=256, message_dim=32, summary_dim=32
    )

    # Copy consensus weights
    marl_orch.reasoning.load_state_dict(orchestrator_orig.reasoning.state_dict())

    print(f"\n  MARL Orchestrator built:")
    print(f"    Communication params: {marl_orch.communication.param_count():,}")
    print(f"    Summary params: {sum(p.numel() for p in marl_orch.summary.parameters()):,}")
    print(f"    Summary inject params: {sum(p.numel() for p in marl_orch.summary_inject.parameters()):,}")
    total = sum(p.numel() for p in marl_orch.parameters())
    print(f"    Total params: {total:,}")

    return marl_orch


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="4-Phase MARL Pipeline for MA-PINN UHS")
    parser.add_argument('--calib_frac', type=float, default=0.10,
                        help='Fraction of training data for calibration (default: 0.10)')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Phase 1 and load from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Phase 1 checkpoint (for --skip_phase1)')
    parser.add_argument('--marl_epochs', type=int, default=100,
                        help='MARL fine-tuning epochs (Phase 2)')
    parser.add_argument('--comm_epochs', type=int, default=100,
                        help='Communication fine-tuning epochs (Phase 3)')
    parser.add_argument('--recalib_epochs', type=int, default=50,
                        help='Recalibration epochs (Phase 4)')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='Collaborative reward weight (Phase 2)')
    args = parser.parse_args()

    print("=" * 70)
    print("MA-PINN UHS — 4-PHASE MARL PIPELINE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Calibration: {args.calib_frac*100:.0f}%")

    # ================================================================
    # LOAD DATA
    # ================================================================
    data = load_and_prepare_data()

    # ================================================================
    # PHASE 1a + 1b
    # ================================================================
    if args.skip_phase1 and args.checkpoint:
        print(f"\nLoading Phase 1 checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=DEVICE)
        hydro_agent = PhysicsInformedHydroAgent(input_dim=19, output_dim=3, hidden=256, n_blocks=4)
        hydro_agent.load_state_dict(ckpt['hydro_agent'])
        geochem_agent = GeochemAgent(hidden=128, dropout=0.1)
        geochem_agent.load_state_dict(ckpt['geochem_agent'])
        hysteresis_agent = HysteresisAgent(hidden=128, dropout=0.1)
        hysteresis_agent.load_state_dict(ckpt['hysteresis_agent'])

        orchestrator_orig = PINNMultiAgentOrchestrator(
            hydro_agent=hydro_agent, geochem_agent=geochem_agent,
            hysteresis_agent=hysteresis_agent, hidden=256
        )
        orchestrator_orig.load_state_dict(ckpt['orchestrator'])
        orchestrator_orig = orchestrator_orig.to(DEVICE)
        hydro_r2 = ckpt.get('hydro_r2', 0.0)

        # Rebuild calibration subset (same random state)
        n_calib = int(len(data['X_train_n']) * args.calib_frac)
        np.random.seed(42)
        idx = np.random.choice(len(data['X_train_n']), n_calib, replace=False)
        X_calib = data['X_train_n'][idx]
        X_hydro_calib = data['X_flow_train_n'][idx]
        Y_calib = data['Y_train_n'][idx]
    else:
        hydro_agent, geochem_agent, hysteresis_agent, hydro_r2 = phase_1a(data)
        orchestrator_orig, X_calib, X_hydro_calib, Y_calib = phase_1b(
            hydro_agent, geochem_agent, hysteresis_agent, data, args.calib_frac
        )

    # Baseline (Phase 1 only)
    metrics_baseline = evaluate(orchestrator_orig, data['X_test_n'], data['X_flow_test_n'],
                                 data['Y_test'], data['scaler_Y'])
    print(f"\n  Phase 1 Baseline (Test): R² = {metrics_baseline['r2']:.4f}")

    # ================================================================
    # BUILD MARL ORCHESTRATOR
    # ================================================================
    marl_orch = build_marl_orchestrator(orchestrator_orig)

    # ================================================================
    # PHASE 2: PER-AGENT MARL
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: PER-AGENT MARL FINE-TUNING")
    print("=" * 70)

    # Scale decoupled data for MARL (agents expect their own scaler domain)
    marl_orch = marl_finetuning(
        marl_orch,
        X_calib, X_hydro_calib, Y_calib,
        X_mrst=data['X_mrst_raw'],
        X_mrst_physics=data['X_mrst_n'],
        Y_mrst=data['Y_mrst_n'],
        scaler_Y_mrst=data['scaler_Y'],
        X_phreeqc=data['X_geochem_n'][:len(data['Y_phreeqc_n'])],
        Y_phreeqc=data['Y_phreeqc_n'],
        scaler_Y_geochem=data['scaler_Y_geochem'],
        X_bc=data['X_bc_n'],
        Y_bc=data['Y_bc_n'],
        scaler_Y_bc=data['scaler_Y_bc'],
        scaler_Y=data['scaler_Y'],
        gamma=args.gamma,
        epochs=args.marl_epochs,
    )

    metrics_p2 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                                data['Y_test'], data['scaler_Y'], use_communication=False)
    print(f"  Phase 2 (Test, no comm): R² = {metrics_p2['r2']:.4f}")

    # ================================================================
    # PHASE 3: COMMUNICATION FINE-TUNING
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: COMMUNICATION FINE-TUNING")
    print("=" * 70)

    marl_orch = train_communication(
        marl_orch, X_calib, X_hydro_calib, Y_calib,
        epochs=args.comm_epochs
    )

    metrics_p3 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                                data['Y_test'], data['scaler_Y'], use_communication=True)
    print(f"  Phase 3 (Test, with comm): R² = {metrics_p3['r2']:.4f}")

    # ================================================================
    # PHASE 4: RECALIBRATION
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: RECALIBRATION")
    print("=" * 70)

    marl_orch = recalibrate(
        marl_orch, X_calib, X_hydro_calib, Y_calib,
        epochs=args.recalib_epochs
    )

    metrics_p4 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                                data['Y_test'], data['scaler_Y'], use_communication=True)
    print(f"  Phase 4 (Test, with comm): R² = {metrics_p4['r2']:.4f}")

    # ================================================================
    # ABLATION: WITH vs WITHOUT COMMUNICATION
    # ================================================================
    print("\n" + "=" * 70)
    print("ABLATION: COMMUNICATION EFFECT")
    print("=" * 70)

    metrics_no_comm = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                                     data['Y_test'], data['scaler_Y'], use_communication=False)
    metrics_with_comm = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                                       data['Y_test'], data['scaler_Y'], use_communication=True)

    print(f"  Without communication: R² = {metrics_no_comm['r2']:.4f} "
          f"(P={metrics_no_comm['r2_P']:.4f}, Sw={metrics_no_comm['r2_Sw']:.4f}, Sg={metrics_no_comm['r2_Sg']:.4f})")
    print(f"  With communication:    R² = {metrics_with_comm['r2']:.4f} "
          f"(P={metrics_with_comm['r2_P']:.4f}, Sw={metrics_with_comm['r2_Sw']:.4f}, Sg={metrics_with_comm['r2_Sg']:.4f})")
    delta = metrics_with_comm['r2'] - metrics_no_comm['r2']
    print(f"  Delta R²: {delta:+.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — 4-PHASE MARL PIPELINE")
    print("=" * 70)

    summary = [
        ("Phase 1 (IL+RL baseline)", metrics_baseline),
        ("Phase 2 (MARL, no comm)", metrics_p2),
        ("Phase 3 (+ communication)", metrics_p3),
        ("Phase 4 (+ recalibration)", metrics_p4),
    ]

    print(f"\n{'Phase':<30} {'R²':>8} {'R²(P)':>8} {'R²(Sw)':>8} {'R²(Sg)':>8}")
    print("-" * 70)
    for name, m in summary:
        r2_p = m.get('r2_P', m.get('r2_P', '-'))
        r2_sw = m.get('r2_Sw', m.get('r2_Sw', '-'))
        r2_sg = m.get('r2_Sg', m.get('r2_Sg', '-'))
        print(f"{name:<30} {m['r2']:>8.4f} {r2_p:>8.4f} {r2_sw:>8.4f} {r2_sg:>8.4f}")

    # ================================================================
    # SAVE
    # ================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / f"marl_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    torch.save({
        'hydro_agent': marl_orch.hydro.state_dict(),
        'geochem_agent': marl_orch.geochem.state_dict(),
        'hysteresis_agent': marl_orch.hysteresis.state_dict(),
        'orchestrator': marl_orch.state_dict(),
        'scaler_X_hydro': data['scaler_X_hydro'],
        'scaler_Y': data['scaler_Y'],
        'hydro_r2': float(hydro_r2),
        'config': {
            'calib_frac': args.calib_frac,
            'marl_epochs': args.marl_epochs,
            'comm_epochs': args.comm_epochs,
            'recalib_epochs': args.recalib_epochs,
            'gamma': args.gamma,
        },
    }, results_dir / "models.pt")

    # Save Phase 1 checkpoint separately (for --skip_phase1 reuse)
    torch.save({
        'hydro_agent': orchestrator_orig.hydro.state_dict(),
        'geochem_agent': orchestrator_orig.geochem.state_dict(),
        'hysteresis_agent': orchestrator_orig.hysteresis.state_dict(),
        'orchestrator': orchestrator_orig.state_dict(),
        'hydro_r2': float(hydro_r2),
    }, results_dir / "phase1_checkpoint.pt")

    # Save results JSON
    results_json = {
        'config': {
            'calib_frac': args.calib_frac,
            'gamma': args.gamma,
            'marl_epochs': args.marl_epochs,
            'comm_epochs': args.comm_epochs,
            'recalib_epochs': args.recalib_epochs,
        },
        'hydro_r2': float(hydro_r2),
        'baseline': metrics_baseline,
        'phase2': metrics_p2,
        'phase3': metrics_p3,
        'phase4': metrics_p4,
        'ablation': {
            'without_comm': metrics_no_comm,
            'with_comm': metrics_with_comm,
            'delta_r2': float(delta),
        },
    }

    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=float)

    print(f"\nResults saved: {results_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
