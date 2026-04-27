# Physics-Informed Inverse Design of Stable Vanadium Oxide Crystals for Aerospace Applications

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AIAA Journal](https://img.shields.io/badge/Journal-AIAA%20Journal-blue)](https://arc.aiaa.org/journal/aiaaj)

Official implementation for the paper:

> **Physics-Informed Inverse Design of Stable Vanadium Oxide Crystals for Aerospace Applications**  
> Danial Ebrahimzadeh, Sarah S. Sharif, Yaser M. Banad  
> *AIAA Journal* (under review), 2026  
> INQUIRE Lab, School of Electrical and Computer Engineering, University of Oklahoma

---

## Overview

This repository contains the complete code and dataset for a physics-guided inverse design framework that couples a **voxel-based Variational Autoencoder (VAE)** with a **formation-energy-constrained Wasserstein GAN (WGAN)** to discover thermodynamically stable vanadium oxide (V–O) crystal structures for aerospace applications.

The framework:
- Encodes crystal structures as element-resolved 3D voxel grids (64³ for atomic channels, 32³ for lattice geometry)
- Compresses them into a continuous latent space via the VAE
- Samples novel candidates using a WGAN whose generator is steered toward thermodynamically stable regions via an exponential formation-energy penalty predicted by a frozen CNN regressor
- Validates every generated candidate through DFT relaxation and phonon dispersion calculations

**Key results** (from 10,981 training structures):
- 1,622 unique V–O compositions generated spanning 22 distinct stoichiometries
- **81.5% validity**, **96.6% uniqueness**, **99.5% novelty**, **0.973 structural diversity**
- **324 stable** structures (20%) with E_f < 0 and ≤ 300 meV/atom above convex hull
- **243 metastable** structures (15%) within ≤ 500 meV/atom
- Dynamic stability confirmed for selected candidates via phonon calculations


---

## Repository Structure

```
├── prepare/
│   ├── data_transformation.py       # Crystal → 3D voxel grid encoding (σ = 0.26 Å)
│   ├── generate_train.py            # Dataset preparation and train/test splitting
│   ├── constrain_reg.py             # Formation-energy CNN regressor training
│   ├── data_for_constrains.py       # Data pipeline for the CNN regressor
│   ├── Improved_lattice_autoencoder_plot.py   # Lattice VAE training and visualization
│   └── sites_autoencoder_plot.py    # Atomic-site VAE training and visualization
│
├── gan/
│   └── ccdcgan.py                   # WGAN with formation-energy constraint (main generative model)
│
├── outputs/                         # Generated V–O crystal structures organized by stoichiometry
│   ├── V1O1/                        # VO  — rock-salt and related phases
│   ├── V1O2/                        # VO₂ — rutile, monoclinic, and novel polymorphs
│   ├── V1O4/                        # VO₄ — oxygen-rich phases
│   ├── V2O1/                        # V₂O — vanadium-rich phases
│   ├── V2O2/                        # V₂O₂ (≡ VO, larger supercell variants)
│   ├── V2O3/                        # V₂O₃ — corundum-related phases
│   ├── V2O4/                        # V₂O₄ — intermediate oxide phases
│   ├── V2O5/                        # V₂O₅ — layered pentoxide phases
│   ├── V2O6/                        # V₂O₆ — oxygen-rich phases
│   ├── V2O7/                        # V₂O₇
│   ├── V2O8/                        # V₂O₈
│   ├── V2O9/                        # V₂O₉
│   ├── V2O10/                       # V₂O₁₀
│   ├── V2O11/                       # V₂O₁₁
│   ├── V2O12/                       # V₂O₁₂ — most oxygen-rich generated phase
│   ├── V3O1/                        # V₃O — vanadium-rich Magnéli-type phases
│   ├── V3O2/                        # V₃O₂
│   ├── V3O3/                        # V₃O₃ (≡ VO, larger supercell variants)
│   ├── V3O4/                        # V₃O₄ — spinel-related mixed-valence phases
│   ├── V3O5/                        # V₃O₅ — monoclinic intermediate oxide
│   ├── V3O6/                        # V₃O₆
│   ├── V3O7/                        # V₃O₇
│   └── V3O8/                        # V₃O₈
│
├── database/
│   └── database/
│       ├── geometries/              # 10,981 V–O crystal structures in VASP POSCAR format
│       └── properties/
│           └── formation_energy/    # 10,981 DFT formation energies as .npy files
│
├── train_GAN.py                     # Main training entry point
├── convert_cif2vasp.py             # Utility: convert CIF files to VASP POSCAR format
└── requirements.txt                 # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU recommended (training was performed on NVIDIA H100)
- VASP license required for DFT validation steps

### Setup

```bash
# Clone the repository
git clone https://github.com/INQUIRELAB/VAE-WGAN-VanadiumOxides.git
cd VAE-WGAN-VanadiumOxides

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

The `database/` directory contains the full training and validation dataset:

### Training database

| Location | Contents | Count |
|---|---|---|
| `database/database/geometries/` | V–O crystal structures (VASP POSCAR format) | 10,981 files |
| `database/database/properties/formation_energy/` | DFT formation energies (NumPy `.npy` format) | 10,981 files |

**File naming:** Each `.vasp` geometry file and its corresponding `.npy` energy file share the same filename stem (e.g., `mp-12345.vasp` and `mp-12345.npy`).


---

## Methodology Summary

### Crystal Representation
Each crystal is encoded as three volumetric channels inside a 15 Å cubic box:
- **Vanadium channel:** 64 × 64 × 64 grid, Gaussian RBF with σ = 0.26 Å
- **Oxygen channel:** 64 × 64 × 64 grid, Gaussian RBF with σ = 0.26 Å
- **Lattice channel:** 32 × 32 × 32 grid encoding cell dimensions and angles

Voxels below an occupancy threshold of 0.02 are masked to zero.

### VAE Architecture
| Component | Latent dim | Epochs | Optimizer | Learning rate |
|---|---|---|---|---|
| Lattice VAE | 25 | 201 | Adam | 3×10⁻⁴ |
| Atomic-site VAE | 200 | 101 | Adam | 3×10⁻⁴ |

The concatenated 225-dimensional latent code is the input to the WGAN.

### Formation-Energy CNN Regressor
- Input: 225-dimensional latent code, zero-padded to 784 dimensions and reshaped to 28 × 28
- Architecture: 4 Conv2D layers + 5 Dense layers with LeakyReLU activations
- Loss: Mean Squared Error (MSE)
- Training: 50 epochs, 90/10 train–test split

### WGAN Generator Loss
The generator minimizes:

$$\mathcal{L}_G = -\mathbb{E}_{\mathbf{z}}[C(G(\mathbf{z}))] + \frac{\lambda}{B}\sum_{i=1}^{B}\exp(\hat{E}_f(G(\mathbf{z}_i))), \quad \lambda = 0.1$$

where $\hat{E}_f$ is the frozen CNN formation-energy regressor. The exponential penalty is small when $\hat{E}_f < 0$ (stable) and grows rapidly when $\hat{E}_f > 0$ (unstable), biasing generation toward thermodynamically favorable structures.

### Stability Criteria
| Class | Criterion |
|---|---|
| **Stable** | E_f < 0 AND ≤ 300 meV/atom above MP convex hull |
| **Metastable** | E_f < 0 AND ≤ 500 meV/atom above MP convex hull |
<!--
---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{ebrahimzadeh2025aiaa,
  author  = {Ebrahimzadeh, Danial and Sharif, Sarah S. and Banad, Yaser M.},
  title   = {Physics-Informed Inverse Design of Stable Vanadium Oxide Crystals
             for Aerospace Applications},
  journal = {AIAA Journal},
  year    = {2025},
  note    = {Under review}
}
```

Please also cite our earlier related work:

```bibtex
@article{ebrahimzadeh2025mte,
  author  = {Ebrahimzadeh, Danial and Sharif, Sarah S. and Banad, Yaser M.},
  title   = {Accelerated discovery of vanadium oxide compositions:
             A {WGAN-VAE} framework for materials design},
  journal = {Materials Today Electronics},
  volume  = {13},
  pages   = {100155},
  year    = {2025},
  doi     = {10.1016/j.mtelec.2025.100155}
}
```
-->
---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code or dataset, please open a GitHub Issue or contact:

- **Danial Ebrahimzadeh** — danial.ebrahimzadeh@ou.edu  
- **Prof. Yaser M. Banad** — banad@ou.edu  
- **INQUIRE Lab** — https://inquirelab.ai

School of Electrical and Computer Engineering  
University of Oklahoma, Norman, OK 73019
