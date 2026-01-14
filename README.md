# force_matching

For any questions related to this code, please contact: magic.linuxkde@gmail.com

Title: Force Matching with Relativistic Constraints: A Physics-Inspired Approach to Stable and Efficient Generative Modeling

This paper has been accepted at CIKM 2025.

Authors: Yang Cao, Bo Chen, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Mingda Wan


Arxiv Link: https://arxiv.org/abs/2502.08150

## Citation 

```bibtex
@inproceedings{cao2025force,
  title={Force matching with relativistic constraints: A physics-inspired approach to stable and efficient generative modeling},
  author={Cao, Yang and Chen, Bo and Li, Xiaoyu and Liang, Yingyu and Sha, Zhizhou and Shi, Zhenmei and Song, Zhao and Wan, Mingda},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages={179--188},
  year={2025}
}
```






## Repo structure

Each script is self-contained: **generate a toy dataset → train → sample → save figures → print an “Average L2 Norm” metric**.

### Datasets / tasks
- **Onedot:** circular transport
- **Halfmoons:** two-moons transport
- **Spiral:** spiral transport

### Methods (script naming)
- **O1** (first-order / velocity field): `1_*.py`
- **O1+O2** (first + second order / velocity + acceleration): `12_*.py`
- **ForM** (ours / Lorentz force): `form_*.py`

Scripts:
- Onedot: `1_1dot.py`, `12_1dot.py`, `form_1dot.py`
- Halfmoons: `1_halfmoons.py`, `12_halfmoons.py`, `form_halfmoons.py`
- Spiral: `1_spiral.py`, `12_spiral.py`, `form_spiral.py`
- Trajectory demo only: `show_trajectory.py`

---

## Installation

```bash
pip install numpy matplotlib tqdm torch
```
> Note: for CUDA-enabled PyTorch, follow the official selector: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Run Experiments

Create output folders (required; scripts do not create them automatically):

```bash
mkdir -p figures dataset tmp_figures
```

### Run a single experiment

```bash
python form_halfmoons.py
```

### Run all experiments

```bash
# Onedot
python 1_1dot.py
python 12_1dot.py
python form_1dot.py

# Halfmoons
python 1_halfmoons.py
python 12_halfmoons.py
python form_halfmoons.py

# Spiral
python 1_spiral.py
python 12_spiral.py
python form_spiral.py
```

### Trajectory visualization only

```bash
python show_trajectory.py
```

---

## Outputs

* `figures/*.pdf`: dataset plots, generated samples, and sampled trajectories
* `dataset/*.pth`: saved datasets (enabled in `form_*.py`)
* `tmp_figures/*.pdf`: trajectory demo output (`show_trajectory.py`)

Each script prints:

* `Average L2 Norm: ...` (a simple quantitative comparison metric)

---

