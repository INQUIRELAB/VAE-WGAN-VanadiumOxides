# Physics-Informed Generative Inverse Design of Stable Vanadium Oxide Crystals with First-Principles Validation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://www.tensorflow.org/)
[![License: Noncommercial](https://img.shields.io/badge/License-Noncommercial-blue.svg)](LICENSE)
<<<<<<< HEAD
[![Data: Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-yellow)](https://huggingface.co/datasets/danial199472/belowhull)
=======
[![AIAA Journal](https://img.shields.io/badge/Journal-AIAA%20Journal-blue)](https://arc.aiaa.org/journal/aiaaj)
>>>>>>> f0932b15bcffb3fb66394c700ab1bdc9989fe3e3

Official code and data for the paper:

> **Physics-Informed Generative Inverse Design of Stable Vanadium Oxide Crystals with First-Principles Validation**
> Danial Ebrahimzadeh, Sarah S. Sharif, Yaser M. Banad
> *Discover Materials* (under review), 2026
> INQUIRE Lab, School of Electrical and Computer Engineering, University of Oklahoma

---

## Overview

This repository contains the code, training database and generated structures for an inverse-design workflow that couples a **voxel-based variational autoencoder (VAE)** with a **formation-energy-constrained Wasserstein GAN (WGAN)** to generate vanadium oxide (V–O) crystal candidates.

- Crystals are encoded as element-resolved 3D voxel grids (64³ for the V and O channels, 32³ for lattice geometry).
- Two VAEs compress the grids into a continuous latent space.
- A WGAN samples new latent codes; its generator is biased toward low formation energy by an exponential penalty from a frozen CNN regressor.
- Every generated candidate is relaxed with DFT (PBE+U) and placed against the Materials Project convex hull; phonon calculations assess lattice-dynamical stability.

**Key results**

| Quantity | Value |
|---|---|
| Training structures (DFT-relaxed) | 10,981 |
| Generated candidates | 6,592 |
| Validity / uniqueness / novelty | 81.5% / 96.6% / 98.7% |
| Candidates with E_f < 0 | 1,846 |
| Within 100 meV/atom of the MP hull | 302 (136 within 50 meV/atom) |
| Below the MP hull | 94 |
| Fraction within 300 meV/atom, composition-matched DFT control (λ = 0 → λ = 0.1) | 15.9% → 25.4% |

---

## Repository Structure

```
├── All_materials_index_with_Ef.csv   # DFT results for all 6,592 generated candidates (see below)
│
├── prepare/
│   ├── data_transformation.py        # Crystal → 3D voxel grid encoding (σ = 0.26 Å)
│   ├── generate_train.py             # Dataset preparation and train/test splitting
│   ├── Improved_lattice_autoencoder_plot.py   # Lattice VAE training and visualisation
│   ├── sites_autoencoder_plot.py     # Atomic-site VAE training and visualisation
│   ├── data_for_constrains.py        # Data pipeline for the formation-energy regressor
│   └── constrain_reg.py              # Formation-energy CNN regressor training
│
├── gan/
│   └── ccdcgan.py                    # WGAN with formation-energy constraint
│
├── outputs/                          # 6,592 generated structures (VASP POSCAR, .vasp)
│   ├── V1O1/                         #   one folder per cell composition VxOy
│   ├── V1O2/
│   ├── ...
│   └── V10O10/
│
├── database/
│   └── database/
│       ├── geometries/               # 10,981 DFT-relaxed V–O structures (VASP POSCAR)
│       └── properties/
│           └── formation_energy/     # 10,981 DFT formation energies (.npy)
│
├── train_GAN.py                      # Main WGAN training entry point
├── convert_cif2vasp.py               # Utility: CIF → VASP POSCAR
├── requirements.txt                  # Python dependencies
├── LICENSE, LICENSE-CODE.txt, LICENSE-MATERIALS.txt, NOTICE
└── README.md
```

---

## Data

### Generated candidates: `All_materials_index_with_Ef.csv`

One row per generated candidate (6,592 rows). The `MaterialID` matches the file name in `outputs/<Composition>/<MaterialID>.vasp`.

| Column | Description |
|---|---|
| `MaterialID` | Candidate identifier (e.g. `epoch_1119`) |
| `Composition` | Cell composition (e.g. `V2O3`) |
| `VAtoms`, `OAtoms` | Number of V and O atoms in the cell |
| `Formation_Energy_eV_per_atom` | DFT (PBE+U) formation energy, eV/atom |

### Generated structures: `outputs/`

The 6,592 generated structures in VASP POSCAR format, grouped into folders by cell composition (`V1O1` … `V10O10`).

### Training database: `database/`

| Location | Contents | Count |
|---|---|---|
| `database/database/geometries/` | DFT-relaxed V–O structures (VASP POSCAR) | 10,981 |
| `database/database/properties/formation_energy/` | DFT formation energies (NumPy `.npy`) | 10,981 |

Each structure and its energy share the same stem, e.g. `mp-1000_copy1_opt.vasp` ↔ `mp-1000_copy1_opt_pack_Ef.npy`.

### Large files (Hugging Face)

| Data | Link |
|---|---|
| Below-hull candidates (DFT calculations) | https://huggingface.co/datasets/danial199472/belowhull |
| Phonon calculations | https://huggingface.co/datasets/danial199472/phonon |

---

## Installation

```bash
git clone https://github.com/INQUIRELAB/VAE-WGAN-VanadiumOxides.git
cd VAE-WGAN-VanadiumOxides

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Requirements:** Python 3.8+; a CUDA-capable GPU is recommended (NVIDIA H100 and H200 GPUs were used in this work). A VASP licence is required only to repeat the DFT validation.

### Workflow

1. `prepare/data_transformation.py`, `prepare/generate_train.py` – voxelise the training database and split it.
2. `prepare/Improved_lattice_autoencoder_plot.py`, `prepare/sites_autoencoder_plot.py` – train the lattice and atomic-site VAEs.
3. `prepare/data_for_constrains.py`, `prepare/constrain_reg.py` – train the formation-energy regressor.
4. `train_GAN.py` (uses `gan/ccdcgan.py`) – train the constrained WGAN and generate candidates.

---

## Methodology Summary

### Crystal representation
Each crystal is placed in a 15 Å cubic box and encoded as:
- **V channel** and **O channel:** 64 × 64 × 64 grids (≈ 0.23 Å spacing), Gaussian functions with σ = 0.26 Å at the atomic positions;
- **Lattice channel:** 32 × 32 × 32 grid encoding the cell lengths and angles.

Voxels below an occupancy of 0.02 are set to zero.

### Model
| Component | Setting |
|---|---|
| Latent dimension – site VAE / lattice VAE | 200 / 25 |
| Generator output and regressor input | 28 × 28 map (425 latent values, 359 zeros) |
| WGAN noise dimension | 200 |
| WGAN training | 5,000 epochs, Adam (lr 2 × 10⁻⁴, β₁ = 0.5), batch 128, n_critic = 5, weight clipping c = 0.01 |
| Formation-energy regressor | 4 Conv2D + 6 dense layers, MSE loss, test MAE 0.064 eV/atom |

The generator minimises

$$\mathcal{L}_G = -\mathbb{E}_{\mathbf{z}}[C(G(\mathbf{z}))] + \frac{\lambda}{B}\sum_{i=1}^{B}\exp\big(\hat{E}_f(G(\mathbf{z}_i))\big), \quad \lambda = 0.1,$$

where $\hat{E}_f$ is the frozen formation-energy regressor. The penalty is small for $\hat{E}_f < 0$ and grows rapidly for $\hat{E}_f > 0$.

### DFT validation
VASP, PBE+U (Dudarev, U_eff = 3.25 eV on V-3d), spin-polarised with ferromagnetic initial moments, 500 eV plane-wave cutoff, Γ-centred k-meshes at 0.5 Å⁻¹ spacing, convergence 10⁻⁵ eV (energy) and 0.05 eV/Å (forces). Energies are corrected with the Materials Project 2020 compatibility scheme before the convex-hull comparison.

### Stability classification
Candidates with E_f < 0 are placed against the Materials Project convex hull: **below-hull** (ΔE_hull < 0), **low-energy** (0–300 meV/atom), **intermediate-energy** (300–500 meV/atom) and **high-energy** (> 500 meV/atom).

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{ebrahimzadeh2026vo,
  author  = {Ebrahimzadeh, Danial and Sharif, Sarah S. and Banad, Yaser M.},
  title   = {Physics-Informed Generative Inverse Design of Stable Vanadium Oxide
             Crystals with First-Principles Validation},
  journal = {Discover Materials},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

<<<<<<< HEAD
Original INQUIRE Lab code is licensed under the PolyForm Noncommercial License 1.0.0 (`LICENSE-CODE.txt`). Original datasets, figures and documentation are licensed under CC BY-NC 4.0 (`LICENSE-MATERIALS.txt`). See [LICENSE](LICENSE) for the scope. Materials from other rights holders retain their original terms.
=======
This project is licensed under the PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE) for details.
>>>>>>> f0932b15bcffb3fb66394c700ab1bdc9989fe3e3

---

## Contact

For questions about the code or data, please open a GitHub Issue or contact:

- **Danial Ebrahimzadeh** — danial.ebrahimzadeh@ou.edu
- **Prof. Yaser M. Banad** — banad@ou.edu
- **INQUIRE Lab** — https://inquirelab.ai

<<<<<<< HEAD
School of Electrical and Computer Engineering, University of Oklahoma, Norman, OK 73019
=======
School of Electrical and Computer Engineering  
University of Oklahoma, Norman, OK 73019

## License scope

Original INQUIRE Lab code is licensed under PolyForm Noncommercial License 1.0.0. Original INQUIRE Lab datasets, figures, and documentation are licensed under CC BY-NC 4.0. See [LICENSE](LICENSE) for scope and the full license texts. Materials from other rights holders retain their original terms.
>>>>>>> f0932b15bcffb3fb66394c700ab1bdc9989fe3e3
