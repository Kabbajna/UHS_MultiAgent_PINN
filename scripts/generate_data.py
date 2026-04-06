#!/usr/bin/env python3
"""
Data Generation Script for Multi-Agent UHS System
==================================================

Generates coupled hydro-geochemical-hysteresis data with heterogeneity features.

Data Sources:
- MRST (MATLAB Reservoir Simulation Toolbox) for hydrodynamics
- PHREEQC for geochemistry
- Brooks-Corey model for capillary hysteresis

References:
- Hagemann et al. (2016): Mineral heterogeneity in saline aquifers
- Thaysen et al. (2021): Microbial variability in UHS
- Heinemann et al. (2021): Spatial heterogeneity effects
- Panfilov (2010): Hydrogen storage modeling
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime


def generate_flow_features(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate flow features (porosity, permeability, depth, time).

    Based on typical UHS reservoir parameters.
    """
    np.random.seed(seed)

    # Porosity: 0.1-0.35 (sandstone aquifers)
    porosity = np.random.uniform(0.10, 0.35, n_samples)

    # Permeability: 10-1000 mD (typical for UHS)
    permeability = np.exp(np.random.uniform(np.log(10), np.log(1000), n_samples))

    # Depth: 500-2000 m (typical UHS depths)
    depth = np.random.uniform(500, 2000, n_samples)

    # Time: 0-1 normalized
    time = np.random.uniform(0, 1, n_samples)

    return np.column_stack([porosity, permeability, depth, time])


def generate_geochem_features(n_samples: int,
                               sigma_micro: float = 0.4,
                               sigma_mineral: float = 0.35,
                               seed: int = 42) -> np.ndarray:
    """
    Generate geochemistry features including heterogeneity.

    Features:
    - temperature, pH, mineralogy_index, microbial_activity (deterministic)
    - ionic_strength, redox_potential (deterministic)
    - microbial_heterog, mineral_heterog (stochastic)

    Args:
        sigma_micro: Standard deviation for microbial heterogeneity
        sigma_mineral: Standard deviation for mineral heterogeneity
    """
    np.random.seed(seed)

    # Deterministic features
    temperature = np.random.uniform(30, 80, n_samples)  # °C
    pH = np.random.uniform(6.0, 8.5, n_samples)
    mineralogy_index = np.random.uniform(0, 1, n_samples)
    microbial_activity = np.random.uniform(0, 1, n_samples)
    ionic_strength = np.random.uniform(0.01, 0.5, n_samples)  # mol/L
    redox_potential = np.random.uniform(-400, 400, n_samples)  # mV

    # Stochastic heterogeneity features
    microbial_heterog = np.random.normal(0, sigma_micro, n_samples)
    mineral_heterog = np.random.normal(0, sigma_mineral, n_samples)

    return np.column_stack([
        temperature, pH, mineralogy_index, microbial_activity,
        ionic_strength, redox_potential,
        microbial_heterog, mineral_heterog
    ])


def generate_hysteresis_features(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate hysteresis features (Brooks-Corey parameters).

    Features:
    - drainage (0 or 1)
    - lambda (pore size distribution)
    - history (cycle number, normalized)

    Note: Sw and Sg are NOT included as they are targets!
    """
    np.random.seed(seed)

    drainage = np.random.randint(0, 2, n_samples).astype(float)
    lambda_param = np.random.uniform(1.5, 4.0, n_samples)
    history = np.random.uniform(0, 1, n_samples)

    return np.column_stack([drainage, lambda_param, history])


def generate_targets(X_flow: np.ndarray,
                     X_geochem: np.ndarray,
                     seed: int = 42) -> np.ndarray:
    """
    Generate coupled targets (P, Sw, Sg) based on physics.

    Simplified physics-based generation:
    - Pressure correlates with depth (hydrostatic)
    - Saturations depend on injection history and reactions
    - Heterogeneity affects H2 loss
    """
    np.random.seed(seed)
    n_samples = len(X_flow)

    porosity = X_flow[:, 0]
    permeability = X_flow[:, 1]
    depth = X_flow[:, 2]
    time = X_flow[:, 3]

    microbial_heterog = X_geochem[:, 6]
    mineral_heterog = X_geochem[:, 7]

    # Pressure (MPa): hydrostatic + perturbations
    rho, g = 1000, 9.81
    P_hydrostatic = rho * g * depth / 1e6
    P_variation = 0.1 * P_hydrostatic * np.random.randn(n_samples)
    P = P_hydrostatic + P_variation
    P = np.clip(P, 5, 30)  # Realistic bounds

    # Base saturations
    Sw_base = 0.3 + 0.4 * (1 - time) + 0.1 * porosity
    Sg_base = 0.3 + 0.3 * time + 0.1 * permeability / 1000

    # H2 loss due to microbial heterogeneity
    h2_loss_factor = 1.0 + 0.15 * microbial_heterog
    Sg_adjusted = Sg_base * h2_loss_factor

    # Pressure modification from mineral heterogeneity
    P_modifier = 1.0 + 0.05 * mineral_heterog
    P = P * P_modifier

    # Ensure physical constraints: Sw + Sg <= 1
    Sw = np.clip(Sw_base, 0.15, 0.85)
    Sg = np.clip(Sg_adjusted, 0.0, 1.0 - Sw - 0.05)

    # Add small noise
    P += np.random.randn(n_samples) * 0.5
    Sw += np.random.randn(n_samples) * 0.02
    Sg += np.random.randn(n_samples) * 0.02

    # Final clipping
    P = np.clip(P, 5, 30)
    Sw = np.clip(Sw, 0.15, 0.85)
    Sg = np.clip(Sg, 0.0, 0.8)

    return np.column_stack([P, Sw, Sg])


def generate_coupled_data(n_samples: int = 495000,
                          sigma_micro: float = 0.4,
                          sigma_mineral: float = 0.35,
                          seed: int = 42) -> dict:
    """
    Generate complete coupled dataset.

    Returns:
        Dictionary with X (features) and Y (targets)
    """
    print(f"Generating {n_samples} coupled samples...")
    print(f"  Heterogeneity: σ_micro={sigma_micro}, σ_mineral={sigma_mineral}")

    # Generate features
    X_flow = generate_flow_features(n_samples, seed)
    X_geochem = generate_geochem_features(n_samples, sigma_micro, sigma_mineral, seed)
    X_hyst = generate_hysteresis_features(n_samples, seed)

    # Combine features (15 total)
    X = np.column_stack([X_flow, X_geochem, X_hyst])

    # Generate targets
    Y = generate_targets(X_flow, X_geochem, seed)

    print(f"  X shape: {X.shape} (4 flow + 8 geochem + 3 hyst = 15)")
    print(f"  Y shape: {Y.shape} (P, Sw, Sg)")

    return {
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32),
        'feature_names': [
            'porosity', 'permeability', 'depth', 'time',
            'temperature', 'pH', 'mineralogy_index', 'microbial_activity',
            'ionic_strength', 'redox_potential',
            'microbial_heterog', 'mineral_heterog',
            'drainage', 'lambda', 'history'
        ],
        'target_names': ['P', 'Sw', 'Sg'],
        'metadata': {
            'n_samples': n_samples,
            'sigma_micro': sigma_micro,
            'sigma_mineral': sigma_mineral,
            'seed': seed,
            'generated': datetime.now().isoformat()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate coupled UHS data with heterogeneity features'
    )
    parser.add_argument('--n_samples', type=int, default=495000,
                        help='Number of samples to generate')
    parser.add_argument('--sigma_micro', type=float, default=0.4,
                        help='Microbial heterogeneity std dev')
    parser.add_argument('--sigma_mineral', type=float, default=0.35,
                        help='Mineral heterogeneity std dev')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    data = generate_coupled_data(
        n_samples=args.n_samples,
        sigma_micro=args.sigma_micro,
        sigma_mineral=args.sigma_mineral,
        seed=args.seed
    )

    # Save
    output_file = output_dir / f"coupled_enriched_sigma_{args.sigma_micro}.pt"
    torch.save(data, output_file)
    print(f"\nSaved to: {output_file}")

    # Also save as default if sigma=0.4
    if args.sigma_micro == 0.4:
        default_file = output_dir / "coupled_enriched.pt"
        torch.save(data, default_file)
        print(f"Saved default: {default_file}")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(data['metadata'], f, indent=2)
    print(f"Saved metadata: {metadata_file}")


if __name__ == "__main__":
    main()
