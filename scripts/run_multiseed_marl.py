#!/usr/bin/env python3
"""
Multi-seed MARL experiment for reproducibility analysis.
Reuses the Phase 1 checkpoint (deterministic) and runs Phases 2-4 with 3 different torch seeds.
Data splits remain fixed (random_state=42) so the test set is identical across seeds.

Usage:
    python scripts/run_multiseed_marl.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from orchestrator import (
    PhysicsInformedHydroAgent, GeochemAgent, HysteresisAgent,
    PINNMultiAgentOrchestrator,
    evaluate, DEVICE
)
from orchestrator_marl import (
    MARLOrchestrator,
    marl_finetuning, train_communication, recalibrate,
    evaluate_marl
)
from run_marl import load_and_prepare_data, build_marl_orchestrator

SEEDS = [42, 123, 456]
CALIB_FRAC = 0.10
PHASE1_CHECKPOINT = project_root / "results" / "marl_20260326_210626" / "phase1_checkpoint.pt"


def run_one_seed(seed, data, phase1_ckpt_path):
    """Run Phases 2-4 with a specific torch seed, reusing Phase 1 checkpoint."""
    print(f"\n{'='*70}")
    print(f"  SEED {seed}")
    print(f"{'='*70}")

    # Set torch seed for model training randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep numpy seed at 42 for calibration subset consistency
    np.random.seed(42)

    # Load Phase 1 checkpoint
    ckpt = torch.load(phase1_ckpt_path, weights_only=False, map_location=DEVICE)
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

    # Rebuild calibration subset (same as original run)
    n_calib = int(len(data['X_train_n']) * CALIB_FRAC)
    idx = np.random.choice(len(data['X_train_n']), n_calib, replace=False)
    X_calib = data['X_train_n'][idx]
    X_hydro_calib = data['X_flow_train_n'][idx]
    Y_calib = data['Y_train_n'][idx]

    # Phase 1 baseline
    metrics_baseline = evaluate(orchestrator_orig, data['X_test_n'], data['X_flow_test_n'],
                                data['Y_test'], data['scaler_Y'])
    print(f"  Phase 1 Baseline: R² = {metrics_baseline['r2']:.4f}")

    # Set torch seed NOW (after data loading) for training randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Build MARL orchestrator (fresh init of comm layers with this seed)
    marl_orch = build_marl_orchestrator(orchestrator_orig)

    # Phase 2: MARL
    print(f"  Phase 2: MARL fine-tuning (seed={seed})...")
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
        gamma=0.3,
        epochs=100,
    )
    metrics_p2 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                               data['Y_test'], data['scaler_Y'], use_communication=False)
    print(f"  Phase 2: R² = {metrics_p2['r2']:.4f} (P={metrics_p2['r2_P']:.4f}, "
          f"Sw={metrics_p2['r2_Sw']:.4f}, Sg={metrics_p2['r2_Sg']:.4f})")

    # Phase 3: Communication
    print(f"  Phase 3: Communication fine-tuning (seed={seed})...")
    marl_orch = train_communication(marl_orch, X_calib, X_hydro_calib, Y_calib, epochs=100)
    metrics_p3 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                               data['Y_test'], data['scaler_Y'], use_communication=True)
    print(f"  Phase 3: R² = {metrics_p3['r2']:.4f}")

    # Phase 4: Recalibration
    print(f"  Phase 4: Recalibration (seed={seed})...")
    marl_orch = recalibrate(marl_orch, X_calib, X_hydro_calib, Y_calib, epochs=50)
    metrics_p4 = evaluate_marl(marl_orch, data['X_test_n'], data['X_flow_test_n'],
                               data['Y_test'], data['scaler_Y'], use_communication=True)
    print(f"  Phase 4 (Final): R² = {metrics_p4['r2']:.4f} (P={metrics_p4['r2_P']:.4f}, "
          f"Sw={metrics_p4['r2_Sw']:.4f}, Sg={metrics_p4['r2_Sg']:.4f})")

    return {
        'seed': seed,
        'phase1': metrics_baseline,
        'phase2': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in metrics_p2.items()},
        'phase3': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in metrics_p3.items()},
        'phase4': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in metrics_p4.items()},
    }


def main():
    print("=" * 70)
    print("MULTI-SEED MARL REPRODUCIBILITY EXPERIMENT")
    print(f"Seeds: {SEEDS}")
    print(f"Phase 1 checkpoint: {PHASE1_CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    if not PHASE1_CHECKPOINT.exists():
        print(f"ERROR: Phase 1 checkpoint not found at {PHASE1_CHECKPOINT}")
        print("Run 'python src/run_marl.py' first to generate Phase 1 checkpoint.")
        sys.exit(1)

    # Load data once (same for all seeds)
    print("\nLoading data...")
    data = load_and_prepare_data()

    # Run each seed
    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, data, PHASE1_CHECKPOINT)
        all_results.append(result)

    # Aggregate
    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    for phase_name in ['phase1', 'phase2', 'phase4']:
        r2s = [r[phase_name]['r2'] for r in all_results]
        r2_Ps = [r[phase_name].get('r2_P', 0) for r in all_results]
        r2_Sws = [r[phase_name].get('r2_Sw', 0) for r in all_results]
        r2_Sgs = [r[phase_name].get('r2_Sg', 0) for r in all_results]

        label = {'phase1': 'Phase 1 (Supervised)', 'phase2': 'Phase 2 (MARL)',
                 'phase4': 'Phase 4 (Final)'}[phase_name]
        print(f"\n{label}:")
        print(f"  R²     = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        print(f"  R²(P)  = {np.mean(r2_Ps):.4f} ± {np.std(r2_Ps):.4f}")
        print(f"  R²(Sw) = {np.mean(r2_Sws):.4f} ± {np.std(r2_Sws):.4f}")
        print(f"  R²(Sg) = {np.mean(r2_Sgs):.4f} ± {np.std(r2_Sgs):.4f}")

    # MARL contribution (Phase 2 - Phase 1)
    delta_marl = [r['phase2']['r2'] - r['phase1']['r2'] for r in all_results]
    print(f"\nΔR² MARL (Phase 2 - Phase 1) = {np.mean(delta_marl):+.4f} ± {np.std(delta_marl):.4f}")

    # Communication contribution (Phase 4 - Phase 2)
    delta_comm = [r['phase4']['r2'] - r['phase2']['r2'] for r in all_results]
    print(f"ΔR² Comm (Phase 4 - Phase 2) = {np.mean(delta_comm):+.4f} ± {np.std(delta_comm):.4f}")

    # Save
    results_dir = project_root / "results"
    out_file = results_dir / "multiseed_marl_results.json"
    with open(out_file, 'w') as f:
        json.dump({
            'seeds': SEEDS,
            'runs': all_results,
            'summary': {
                'phase4_r2_mean': float(np.mean([r['phase4']['r2'] for r in all_results])),
                'phase4_r2_std': float(np.std([r['phase4']['r2'] for r in all_results])),
                'delta_marl_mean': float(np.mean(delta_marl)),
                'delta_marl_std': float(np.std(delta_marl)),
                'delta_comm_mean': float(np.mean(delta_comm)),
                'delta_comm_std': float(np.std(delta_comm)),
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=float)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
