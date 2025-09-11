# PRT-DeepONet

## About

PRT-DeepONet is a deep operator network architecture developed for simulating pore scale reactive transport processes with three reaction types:

- Irreversible sorption
- Reversible sorption
- Monod kinetics

Each type ships with (1) a binary file storing trained parameters (.pt), (2) an example geometry (.npz), and (3) a Jupyter notebook (.ipynb) to launch the PRT-DeepONet model.

| Type                 |  Parameter (.pt)                   |  Geometry (.npz)                               |  Model (.ipynb)                                          |
|----------------------|------------------------------------|------------------------------------------------|-------------------------------------------------------------|
| Irreversible_Sorption| `params/Irreversible_Sorption.pt`  | `geometries/Domain_Irreversible_Sorption.npz`  | `models/PRT-DeepONet_Irreversible_Sorption_load.ipynb`   |
| Reversible_Sorption  | `params/Reversible_Sorption.pt`    | `geometries/Domain_Reversible_Sorption.npz`    | `models/PRT-DeepONet_Reversible_Sorption_load.ipynb`     |
| Monod                | `params/Monod.pt`                  | `geometries/Domain_Monod.npz`                  | `models/PRT-DeepONet_Monod_load.ipynb`                   |

---

## How it works

1. Load the parameter & geometry: `params/<TYPE>.pt`, `geometrys/Domain_<TYPE>.npz`  
2. Prepare the GDF: compute the GDF (Geodesic Distance Function) using the code embedded in the notebook files
3. Assemble inputs: organize type-specific conditions 
4. Inference:  
   - Branch CNN ← geometries
   - Branch FNN ← parametric conditions (Pe, Da)
   - Trunk Network ← spatial and temporal coordinates, GDF
   Feed each feature to its appropriate sub-network input to obtain predictions.  

> Run guidance: After downloading this repository from GitHub open the target notebook  
> Click Kernel → Restart & Run All (or Restart the kernel and run all cells).  
> The entire process above should run automatically.

---

## Repository structure

PRT-DeepONet/
├─ README.md
├─ params/
│ ├─ Irreversible_Sorption.pt
│ ├─ Reversible_Sorption.pt
│ └─ Monod.pt
├─ geometries/
│ ├─ Domain_Irreversible_Sorption.npz
│ ├─ Domain_Reversible_Sorption.npz
│ └─ Domain_Monod.npz
├─ models/
│ ├─ PRT-DeepONet_Irreversible_Sorption_load.ipynb
│ ├─ PRT-DeepONet_Reversible_Sorption_load.ipynb
└─└─ PRT-DeepONet_Monod_load.ipynb

---

## Quick Start

1) Clone or download the repository (keep the folder structure).

2) Open one of the notebooks:
   - Irreversible: `models/PRT-DeepONet_Irreversible_Sorption_load.ipynb`
   - Reversible: `models/PRT-DeepONet_Reversible_Sorption_load.ipynb`
   - Monod: `models/PRT-DeepONet_Monod_load.ipynb`

3) Click Kernel → Restart & Run All (or Restart the kernel and run all cells).

That’s it. See [How it works](#how-it-works) for the full pipeline.

Success criterion: a predicted image is displayed.

---

Authors:
Yehoon Kim, Chungnam National University
wnsla7323 <at> naver.com
Heewon Jung, Chungnam National University
hjung <at> cnu.ac.kr

---

License

Copyright (C) 2025 Jung Lab
PRT-DeepONet is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. PRT-DeepONet is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
