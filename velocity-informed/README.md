# velocity-informed

A **velocity-informed** deep operator network (PRT-DeepONet) for pore-scale reactive transport
in heterogeneous porous media. The concentration model is conditioned on a **predicted velocity
field**, so the repository is split into two variants:

- **`flow/`** — predicts the divergence-free velocity field `(ux, uy)` (λ=10) from the pore geometry.
- **`concentration/`** — predicts the concentration field, taking the flow variant's **predicted
  velocity** `(ux, uy)` as an input feature. It covers two reaction types:
  - **Irreversible sorption** — steady-state concentration
  - **Monod kinetics** — transient concentration

> The concentration variant runs the flow model internally, so it loads the flow weights
> (`flow/parameters/Velocity.pt`, `flow/parameters/Pressure_component_UNet.pt`) by relative path.
> **Keep the `flow/` and `concentration/` folders side by side.**

---

## Repository structure

```
velocity-informed/
├── README.md   LICENSE
├── flow/
│   ├── models/        PRT-DeepONet_Velocity.ipynb        (training pipeline)
│   │                  PRT-DeepONet_Velocity_load.ipynb   (load & run)
│   ├── parameters/    Velocity.pt   Pressure_component_UNet.pt
│   └── examples/      Domain_Velocity.npz
└── concentration/
    ├── models/        PRT-DeepONet_Irreversible_Sorption.ipynb        (training pipeline)
    │                  PRT-DeepONet_Irreversible_Sorption_load.ipynb   (load & run)
    │                  PRT-DeepONet_Monod.ipynb                        (training pipeline)
    │                  PRT-DeepONet_Monod_load.ipynb                   (load & run)
    ├── parameters/    Irreversible_Sorption.pt   Monod.pt
    └── examples/      Domain_Irreversible_Sorption.npz   Domain_Monod.npz
```

Each variant folder holds three things: `models/` (notebooks), `parameters/` (trained weights),
and `examples/` (one example pore geometry per type).

**Two kinds of notebook in every `models/` folder:**

- **`*_load.ipynb` — load & run.** Open it, *Kernel → Restart & Run All*. It loads the trained
  weights + the bundled example geometry and renders a predicted field. Self-contained; needs no
  external data. *This is the notebook to run to reproduce a result.*
- **`PRT-DeepONet_<TYPE>.ipynb` — training pipeline.** Documents the full pipeline
  (`Imports → Reproducibility & Paths → Model Definition → Data Loader → Training Utilities →
  Evaluation → Main Entry`) and retrains from scratch. It reads the training dataset from a single
  `DATA_ROOT` (see [Data layout](#data-layout)) and writes weights back into the variant's
  `parameters/`.

---

## Quick start (reproduce a result)

1. Clone or download the repository — **keep the folder structure** (the concentration notebooks
   reference `../../flow/parameters/`).
2. Open one of the load notebooks:
   - Flow: `flow/models/PRT-DeepONet_Velocity_load.ipynb`
   - Irreversible sorption: `concentration/models/PRT-DeepONet_Irreversible_Sorption_load.ipynb`
   - Monod: `concentration/models/PRT-DeepONet_Monod_load.ipynb`
3. **Kernel → Restart & Run All.**

**Success criterion:** a predicted field image is displayed (flow: a velocity vector field;
concentration: the concentration field).

---

## Input features

**Flow variant** — `PRT-DeepONet_Velocity`

- **Branch CNN** ← pore geometry + the **maximum-inscribed-sphere (MIS) map** (local pore size)
  and the **upstream-constrained pore radius map (UPRM)** (non-local, path-dependent connectivity).
- **Branch FNN** ← the Reynolds number (per-domain scalar condition).
- **Trunk** ← `(x, y)`, the **squared wall distance `d_w²`**, and the **harmonic pressure-gradient
  priors `(∂ₓP, ∂ᵧP)`** (predicted by `Pressure_component_UNet`).

**Concentration variant** — `PRT-DeepONet_Irreversible_Sorption`, `PRT-DeepONet_Monod`

- **Branch CNN** ← pore geometry + the flow variant's **predicted velocity `(ux, uy)`** (z-scored).
- **Branch FNN** ← parametric conditions `(Pe, Da)`.
- **Trunk** ← `(x, y)`, the **GDF** (Geodesic Distance Function, inlet distance), and — for Monod —
  the normalized time `t` (integer timesteps `t = 1 … 80`).


---

Authors:

Hyegyeong Jo, Chungnam National University — <hyegyeong6321@naver.com>  
Heewon Jung, Chungnam National University — <hjung@cnu.ac.kr>

---

License

Copyright (C) 2025 Jung Lab. This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. It is distributed in the
hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
