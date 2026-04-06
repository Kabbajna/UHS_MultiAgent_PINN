# Multi-Agent Physics-Informed Neural Network for Underground Hydrogen Storage

<p align="center">
  <img src="figures/Fig1a_Architecture.pdf" width="90%" alt="MA-PINN Architecture"/>
</p>

> **Paper:** *Multi-Agent Physics-Informed Neural Network for Coupled Multiphase Flow and Geochemistry in Underground Hydrogen Storage*
> **Journal:** Geoenergy Science and Engineering (Elsevier)
> **Authors:** Narjisse Kabbaj — Energy Research Lab, Effat University, Jeddah, Saudi Arabia

---

## Highlights

| Metric | Value |
|--------|-------|
| Overall R² | **0.976 +/- 0.002** (3 seeds) |
| R²(Pressure) | 0.954 |
| R²(Water Saturation) | 0.986 |
| R²(Gas Saturation) | 0.986 |
| Speedup vs coupled simulator | **1.8 x 10^6** |
| Coupled data needed | **Only 10%** of simulation budget |
| Inference time | ~1 ms (single GPU) |

---

## What is this?

Underground Hydrogen Storage (UHS) in saline aquifers requires simulating **three coupled physics**: multiphase flow (Darcy), geochemical reactions (H2 consumption), and capillary hysteresis (drainage/imbibition). A fully coupled MRST-PHREEQC simulation takes ~30 min per scenario --- too slow for parametric screening.

**MA-PINN** replaces the coupled simulator with a multi-agent neural network surrogate that is **6 orders of magnitude faster** while achieving R² = 0.976. The key innovation is a **two-tier data strategy**: specialist agents are pre-trained on cheap decoupled simulators, then coordinated with only 10% of expensive coupled data.

---

## Architecture

<p align="center">
  <img src="figures/Fig2_Physics_Features.pdf" width="85%" alt="Feature Engineering"/>
</p>

Three specialist agents decompose the physics:

| Agent | Input | Output | Data Source | Role |
|-------|-------|--------|-------------|------|
| **Hydro** | 19 physics features | P, Sw, Sg | MRST (625K samples, ~2 min/run) | Multiphase flow dynamics |
| **Geochem** | 8 geochemical features | 5 reaction rates | PHREEQC (1K samples, <1 s/run) | H2 consumption, mineral reactions |
| **Hysteresis** | 3 capillary features | 2 corrections (dk_r, dP_c) | Brooks-Corey (20K, instant) | Drainage/imbibition path dependence |

The 10 intermediate outputs + 4 raw features feed a **confidence-weighted consensus network** (14-dim input) that produces the final predictions. Agents communicate via **gated message passing** (H -> G -> Y -> H), where learned gates control information flow.

**Total parameters:** 1,802,037

---

## Training Pipeline

<p align="center">
  <img src="figures/Fig1b_Training_Pipeline.pdf" width="85%" alt="Training Pipeline"/>
</p>

| Phase | What | Data | Key idea |
|-------|------|------|----------|
| **1a** Supervised pre-training | Each agent trains independently | Decoupled (cheap) | Bulk of predictive capacity from free data |
| **1b** Consensus calibration | Consensus learns to fuse agents | Coupled (10%) | Minimal expensive data needed |
| **2** MARL fine-tuning | Per-agent physics rewards + collaborative reward | Decoupled + Coupled | REINFORCE with gamma=0.3 |
| **3** Communication integration | Gated message passing + recalibration | Coupled (10%) | Agents learn to share information |

**The key result:** Phase 1 alone achieves R² = 0.967 (99% of final accuracy). MARL and communication add the remaining +0.009, disproportionately improving saturations (+2 pp on Sw, Sg).

---

## Results

### Performance Summary

<p align="center">
  <img src="figures/Fig3_MARL_Performance.pdf" width="90%" alt="Performance"/>
</p>

| Method | R² | R²(P) | R²(Sw) | R²(Sg) | Speedup |
|--------|-----|-------|--------|--------|---------|
| MRST-PHREEQC (simulator) | --- | --- | --- | --- | 1x |
| MLP (4 raw features) | 0.107 | --- | --- | --- | 3.6x10^5 |
| Random Forest (4 raw features) | 0.838 | 0.898 | 0.837 | 0.779 | 7.2x10^4 |
| **MA-PINN+MARL (30 features)** | **0.976** | **0.954** | **0.986** | **0.986** | **1.8x10^6** |

### Ablation Hierarchy

```
Features + Architecture:  dR² = +0.129  (dominates)
MARL fine-tuning:         dR² = +0.004
Gated Communication:      dR² = +0.005  (targets Sw, Sg specifically)
```

### Reproducibility (3 seeds)

| Seed | Phase 1 R² | Phase 4 R² |
|------|-----------|-----------|
| 42 | 0.970 | 0.977 |
| 123 | 0.970 | 0.977 |
| 456 | 0.970 | 0.973 |
| **Mean +/- Std** | **0.970 +/- 0.000** | **0.976 +/- 0.002** |

### Agent Interpretability

<p align="center">
  <img src="figures/Fig6_MARL_Interpretability.pdf" width="90%" alt="Interpretability"/>
</p>

- **Gate activations** reveal physically meaningful routing: G->Y channel most open (g=0.70), Y->H most attenuated (g=0.44)
- **Dual-branch weights:** physics branch dominates (w_phys=0.70, w_data=0.30)
- **Pressure-saturation trade-off:** communication improves Sw/Sg by +1.7 pp at the cost of -2.1 pp on P --- discovered autonomously

---

## Repository Structure

```
UHS_MultiAgent_PINN/
|
|-- src/                              # Core framework
|   |-- orchestrator.py               # Agent architectures + consensus + physics losses
|   |-- orchestrator_marl.py          # MARL extension: communication, rewards, training
|   |-- run_marl.py                   # Full 4-phase training pipeline
|
|-- scripts/                          # Experiments & analysis
|   |-- generate_data.py              # Synthetic data generation (MRST, PHREEQC, Brooks-Corey)
|   |-- baseline_comparison.py        # RF and MLP baselines on coupled data
|   |-- compute_baseline_per_variable_r2.py   # Per-variable R² for baselines
|   |-- compute_rf30_baseline.py      # RF with 30 physics features (ablation)
|   |-- run_multiseed_marl.py         # 3-seed reproducibility experiment
|   |-- sensitivity_study.py          # Permeability/depth robustness analysis
|   |-- generate_marl_figures.py      # Generate all paper figures
|   |-- generate_ijhe_figures.py      # Journal-formatted figures
|
|-- data/processed/                   # Preprocessed datasets
|   |-- coupled_enriched.pt           # 495,000 coupled samples (70 MB)
|   |-- hydro_mrst_only_real.pt       # 625,000 MRST flow-only samples (17 MB)
|   |-- geochem_phreeqc_real.pt       # 1,000 PHREEQC equilibrium calculations
|   |-- hysteresis_brooks_corey.pt    # 20,000 Brooks-Corey evaluations
|
|-- results/                          # Experiment outputs
|   |-- checkpoints/                  # Trained model weights
|   |   |-- models.pt                 # Final MA-PINN+MARL model (7 MB)
|   |   |-- phase1_checkpoint.pt      # Phase 1 checkpoint (6 MB)
|   |-- results.json                  # Main training run metrics
|   |-- multiseed_marl_results.json   # 3-seed reproducibility
|   |-- baseline_per_variable_r2.json # Baseline per-variable scores
|   |-- rf30_baseline.json            # RF-30 ablation results
|   |-- sensitivity_results.json      # Robustness analysis
|   |-- gate_activations_summary.json # Communication gate statistics
|
|-- figures/                          # All paper figures (PDF + PNG)
|-- requirements.txt
|-- LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline

```bash
cd src
python run_marl.py --calib_frac 0.10
```

This runs all 4 phases sequentially (~45 min on A100 GPU):
- Phase 1a: Agent pre-training (300 + 100 + 100 epochs)
- Phase 1b: Consensus training (500 epochs)
- Phase 2: MARL fine-tuning (100 + 100 epochs)
- Phase 3: Communication + recalibration (100 + 50 epochs)

### 3. Evaluate baselines

```bash
# Random Forest and MLP on coupled data
python scripts/baseline_comparison.py

# Per-variable R² breakdown
python scripts/compute_baseline_per_variable_r2.py
```

### 4. Reproducibility experiment

```bash
# Run Phases 2-4 with 3 different seeds (reuses Phase 1 checkpoint)
python scripts/run_multiseed_marl.py
```

### 5. Generate figures

```bash
python scripts/generate_marl_figures.py
```

---

## Data Description

All datasets are **synthetic**, generated from established simulators to ensure reproducibility.

| Dataset | Samples | Features | Targets | Source |
|---------|---------|----------|---------|--------|
| `coupled_enriched.pt` | 495,000 | 15 (X) | 3 (P, Sw, Sg) | MRST + PHREEQC + Brooks-Corey |
| `hydro_mrst_only_real.pt` | 625,000 | 4 | 3 (P, Sw, Sg) | MRST flow-only |
| `geochem_phreeqc_real.pt` | 1,000 | 6 | 5 reaction rates | PHREEQC standalone |
| `hysteresis_brooks_corey.pt` | 20,000 | 5 | 2 (dk_r, dP_c) | Brooks-Corey analytical |

**Feature ranges (coupled dataset):**
- Porosity: [0, 1] (normalised)
- Permeability: [2.5, 125] mD
- Depth: [10, 463] m
- Pressure: [67, 562] bar
- Saturations: [0, 1]

---

## Physics-Informed Feature Engineering

The 4 raw inputs (porosity, permeability, depth, time) are expanded to **30 physics-informed features**:

**Hydro Agent (4 -> 19):**
- Hydraulic properties: P_hydro, diffusivity D, transmissivity T_r, conductivity K_h
- Dimensionless groups: Re, Ca, gravity number Gr, Kozeny-Carman k_KC
- Polynomial interactions: phi*k, phi*z, k*z, phi^2, z^2, sqrt(k), phi/k

**Geochem Agent (8 features):** T, pH, mineralogy, microbial activity, ionic strength, redox + 2 stochastic heterogeneity terms

**Hysteresis Agent (3 features):** drainage factor, Brooks-Corey lambda, saturation history

This feature engineering alone accounts for **the largest share of accuracy improvement** (dR² = +0.129 over RF baseline).

---

## Citation

```bibtex
@article{kabbaj2026mapinn,
  title={Multi-Agent Physics-Informed Neural Network for Coupled Multiphase Flow
         and Geochemistry in Underground Hydrogen Storage},
  author={Kabbaj, Narjisse},
  journal={Geoenergy Science and Engineering},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
