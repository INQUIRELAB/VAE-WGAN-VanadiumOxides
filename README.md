# Physics-Informed Generative Inverse Design of Stable Vanadium Oxide Crystals with First-Principles Validation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code and data for the paper:

> **Physics-Informed Generative Inverse Design of Stable Vanadium Oxide Crystals with First-Principles Validation**
> Danial Ebrahimzadeh, Sarah S. Sharif, Yaser M. Banad
> *Discover Materials* (under review), 2026
> INQUIRE Lab, School of Electrical and Computer Engineering, University of Oklahoma

<p align="center">
  <img src="IMAGE_1_LINK" alt="V–O system" width="750">
</p>

---

## Overview

This repository contains the code, training database and generated structures for an inverse-design workflow that couples a **voxel-based variational autoencoder (VAE)** with a **formation-energy-constrained Wasserstein GAN (WGAN)** to generate vanadium oxide (V–O) crystal candidates.

- Crystals are encoded as element-resolved 3D voxel grids.
- Two VAEs compress the grids into a continuous latent space.
- A WGAN samples new latent codes; its generator is biased toward low formation energy by a penalty from a frozen CNN regressor.
- Every generated candidate is relaxed with DFT and compared with the Materials Project convex hull; phonon calculations assess lattice-dynamical stability.

<p align="center">
  <img src="IMAGE_2_LINK" alt="Inverse design workflow" width="650">
</p>

<p align="center">
  <img src="IMAGE_3_LINK" alt="Formation energy of generated candidates" width="700">
</p>

---

## Repository Structure

```
├── All_materials_index_with_Ef.csv   # Index of the 6,592 generated candidates with formation energies
│
├── prepare/
│   ├── data_transformation.py        # Crystal → 3D voxel grid encoding
│   ├── generate_train.py             # Dataset preparation and train/test splitting
│   ├── Improved_lattice_autoencoder_plot.py   # Lattice VAE training
│   ├── sites_autoencoder_plot.py     # Atomic-site VAE training
│   ├── data_for_constrains.py        # Data pipeline for the formation-energy regressor
│   └── constrain_reg.py              # Formation-energy CNN regressor training
│
├── gan/
│   └── ccdcgan.py                    # WGAN with formation-energy constraint
│
├── outputs/                          # 6,592 generated structures (VASP POSCAR)
│   ├── V1O1/                         #   one folder per composition
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
├── LICENSE
└── README.md
```

---

## Data

- **`All_materials_index_with_Ef.csv`** – composition, number of V and O atoms, and DFT formation energy of the generated candidates.
- **`outputs/`** – the 6,592 generated structures in VASP POSCAR format, grouped by composition.
- **`database/`** – the training database:

| Location | Contents | Count |
|---|---|---|
| `database/database/geometries/` | DFT-relaxed V–O structures (VASP POSCAR) | 10,981 |
| `database/database/properties/formation_energy/` | DFT formation energies (NumPy `.npy`) | 10,981 |

### Large files (Hugging Face)

- DFT calculations of the 94 below-hull candidates: https://huggingface.co/datasets/danial199472/belowhull
- Phonon calculations: https://huggingface.co/datasets/danial199472/phonon

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

**Requirements:** Python 3.8+; a CUDA-capable GPU is recommended (NVIDIA H100 and H200 GPUs were used in this work). A VASP licence is required only to repeat the DFT calculations.

### Workflow

1. `prepare/data_transformation.py`, `prepare/generate_train.py` – voxelise the training database and split it.
2. `prepare/Improved_lattice_autoencoder_plot.py`, `prepare/sites_autoencoder_plot.py` – train the lattice and atomic-site VAEs.
3. `prepare/data_for_constrains.py`, `prepare/constrain_reg.py` – train the formation-energy regressor.
4. `train_GAN.py` (uses `gan/ccdcgan.py`) – train the constrained WGAN and generate candidates.

---

## Methodology Summary

- **Representation:** each crystal is placed in a 15 Å cubic box; V and O atoms are drawn as Gaussian densities on separate 64³ grids, and the cell geometry is encoded on a 32³ grid.
- **Generative model:** two VAEs (atomic sites and lattice) learn a continuous latent space, and a WGAN generates new latent codes.
- **Formation-energy constraint:** a frozen CNN regressor predicts the formation energy of each generated sample, and an exponential penalty in the generator loss steers generation toward low formation energy.
- **Validation:** all candidates are relaxed with DFT (VASP, PBE+U) and compared with the Materials Project convex hull; selected candidates are checked with phonon calculations.

---

## Citation

If you use this code or data, please cite:

D. Ebrahimzadeh, S. S. Sharif, Y. M. Banad, "Physics-Informed Generative Inverse Design of Stable Vanadium Oxide Crystals with First-Principles Validation," *Discover Materials* (under review), 2026.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code or data, please open a GitHub Issue or contact:

- Danial Ebrahimzadeh — [danial.ebrahimzadeh@ou.edu](mailto:danial.ebrahimzadeh@ou.edu)
- Yaser M. Banad — [bana@ou.edu](mailto:bana@ou.edu)
