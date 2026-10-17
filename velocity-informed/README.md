# PRT-DeepONet: velocity variant and concentration variant

DeepONet surrogates for 2-D pore geometries (64 × 148). PRT-DeepONet is extended to predict the pore-scale velocity
field (**velocity variant**), and the predicted velocity is used to improve the concentration prediction: the
**concentration variants** receive the velocity predicted by the velocity variant, and each is compared with the
original PRT-DeepONet (geometry and GDF only).

| folder | problem | proposed model | comparison baseline |
|---|---|---|---|
| `velocity/` | steady velocity field (ux, uy) from the pore geometry and Re | velocity variant (with a pressure U-Net) | vanilla DeepONet |
| `irreversible_adsorption/` | steady concentration, (Pe, Da) ∈ {1, 5, 10} × {74, 296, 740} | concentration variant | original PRT-DeepONet |
| `monod/` | transient concentration C(x, y, t), (Pe, Da_M) ∈ {1, 10} × {7.4, 74} | concentration variant | original PRT-DeepONet |

The repository contains the full pipeline of the proposed models (preprocessing, training, evaluation, figures and
tables), their trained weights, and the data behind every figure and table. The comparison baselines are defined in
`<physics>/core.py`; their weights are in `<physics>/weights/baseline/` and `3_extract/` evaluates them into
`<physics>/data/baseline_*.csv`.

**Contents**

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [Data](#3-data)
4. [Models and pipeline](#4-models-and-pipeline)
5. [Training, evaluation, weights and seeds](#5-training-evaluation-weights-and-seeds)
6. [Figures and tables](#6-figures-and-tables)
7. [License](#7-license)

## 1. Quick start

```bash
conda env create -f environment.yml && conda activate prt-deeponet
bash velocity/make_all.sh && bash irreversible_adsorption/make_all.sh && bash monod/make_all.sh   # all figures and tables, CPU, minutes
```

This needs only the repository (no GPU): every figure and table is generated from the shipped `<physics>/data/`.
Each `make_all.sh` takes one stage:

| stage | what it does |
|---|---|
| `figures` (default) | `4_figures/` from the shipped `data/` |
| `extract` | if the LB simulation data are available (`PRT_LB`), recomputes `data/` with the released weights (`1_preprocess` → `3_extract` → `4_figures`) |
| `train` | `1_preprocess` → `2_train`: retrains the proposed models of the physics |
| `all` | `train`, then `extract` with the newly trained weights (`WEIGHTS=$PRT_WORK/<physics>/weights`) |

Intermediate files (geometry features, labels, predictions) go to `PRT_WORK` (default `./work`); they are produced by
running the code and are not shipped. The velocity stages run first, since the concentration models take the
predicted velocity as input (`velocity/3_extract/predicted_velocity.py`).

## 2. Repository layout

```
<physics>/            velocity, irreversible_adsorption, monod
  1_preprocess/       pore geometries -> geometry features, model inputs, labels and normalization statistics ($PRT_WORK)
  2_train/            training scripts
  3_extract/          evaluation of the weights and extraction of the figure data -> data/
  data/               CSV / npz files behind every figure and table, incl. the LB fields shown
                      (baseline_*.csv: comparison baselines)
  4_figures/          one script per figure or table -> 4_figures/output/*.png
  weights/            trained weights of the proposed models; weights/baseline/: comparison baselines
  lb_data/            simulation parameters of every LB sample of the physics
  core.py             code of the physics: LB readers, networks (proposed model and comparison baseline) and
                      model inputs; velocity also the geometry features, harmonic pressure field and losses
  make_all.sh         runs the stages in order
common.py             code shared by the three physics: paths, grid, split, query coordinates, GDF, weight loading,
                      figure helpers
splits/               fixed train / validation / test split and the Monod sample list
```

## 3. Data

### Geometries

3000 base geometries, grid NY × NX = 64 × 148 (rows = y, across the flow; columns = x, along the flow). Column 0 is the
inlet and column 147 the outlet; 10 open columns on each side form the inlet and outlet buffers. The pore geometries
are provided in the `Domains` folder (`Input_domains.zip`) of https://github.com/hjunglab/PRT-DeepONet (main branch):
`domain_<base>.dat`, 9472 values (1 = pore, 0 = solid) with x varying slowest, i.e. `np.loadtxt(f).reshape(148, 64).T`
is the (64, 148) geometry of `base`. The top and bottom rows are treated as walls for every velocity feature, the error
region and Re.

Notation: x is the flow direction and y the cross-stream direction; ux, uy are the velocity components and
px = ∂P/∂x, py = ∂P/∂y the pressure-gradient components. Arrays are stored as (NY, NX) = (64, 148), so the trunk
channels of the velocity variant are ordered [y, x, py, px, d_w²] in the code.

### LB simulations (lattice units)

All LB simulations are D2Q9 lattice-Boltzmann simulations with dx = dt = 1 on the 64 × 148 grid of each geometry.

* **Flow.** Kinematic viscosity ν = 0.25 (relaxation parameter ω = 0.8); the flow is driven by a pressure difference Δp
  imposed between the inlet and the outlet, grains are no-slip walls. Each geometry is simulated at Δp = 0.0025, 0.005
  and 0.01 (Re groups g = 0, 1, 2); Re is varied through Δp only. The two unseen-Re conditions use Δp = 0.00375 (ReA)
  and 0.0075 (ReB).
* **Irreversible adsorption and Monod.** Every simulation of a geometry uses the flow field of that geometry at its
  highest Re (Δp = 0.01, velocity sample `flow_idx = base + 6000`). Pe is set through the solute diffusivity, Da
  through the reaction rate constant and Da_M through the Monod rate constant.

### Samples

* **Velocity.** `idx = base + 3000 · g`, g = 0, 1, 2 = the three Re groups (ascending Re); 9000 samples. The unseen-Re
  conditions (ReA between groups 0 and 1, ReB between groups 1 and 2) are simulated for the 300 test geometries and
  used only for the unseen-Re evaluation.
* **Irreversible adsorption.** `idx = base + 3000 · k`, k = 0..8: Da = 74 (k = 0–2), 296 (3–5), 740 (6–8); Pe = 1, 5, 10
  within each Da. Steady dimensionless concentration (0–1). Four conditions inside the training range,
  (Pe, Da) = (3, 185), (3, 518), (7.5, 185), (7.5, 518), are simulated for the 300 test geometries and used only for
  the interpolation evaluation.
* **Monod.** `idx = base + 3000 · g`, g = 0..3 = (Pe, Da_M) = (1, 7.4), (1, 74), (10, 7.4), (10, 74). Each simulation
  gives the transient concentration at t = 0..100. Training uses t ∈ {1, 4, 8, ..., 80} (21 slices);
  t ∈ {6, 22, 38, 54, 70} of the test geometries is held out for the temporal interpolation.

### Split

* `splits/split.json`: 3000 base geometries → 2400 training / 300 validation / 300 test, shared by every physics,
  model and seed. Porosity-binned (15 bins, 20 test and 20 validation geometries per bin; seeds 27 and 2027). Each
  geometry appears under all its flow / reaction conditions in the same subset. The file also lists the velocity
  samples in training order.
* `splits/monod_samples.npz`: Monod samples (base, condition, t) of each subset in training order.
* Model selection (early stopping, input configurations, λ) uses the validation subset only; the test subset is used
  only for the reported results.

### Shipped data

* `<physics>/data/`: the per-sample errors behind every figure and table, and the LB fields (`gt`, `ux_gt`, `uy_gt`) of
  the samples shown in the field figures, next to the predictions.
* `<physics>/lb_data/` (`flow_conditions.npz` in `velocity/lb_data/`):

| file | content |
|---|---|
| `lattice_parameters_velocity.csv` | one row per velocity sample (15000): `dataset` (`velocity` or `velocity_unseen_re/ReA\|ReB`), `idx`, `base`, `re_group`, `Re`, `delta_p`, `nu`, `dx`, `dt` |
| `lattice_parameters_irreversible_adsorption.csv` | one row per adsorption sample (28200): `dataset` (`irreversible_adsorption` or `irreversible_adsorption_unseen/<condition>`), `idx`, `base`, `Pe`, `Da`, `flow_idx`, `delta_p`, `nu`, `dx`, `dt` |
| `lattice_parameters_monod.csv` | one row per Monod sample (12000): `dataset`, `idx`, `base`, `Pe`, `Da_M`, `flow_idx`, `delta_p`, `nu`, `dx`, `dt` |
| `flow_conditions.npz` | `re`, `re_norm` (9001,), indexed by `idx` (entry 0 unused): Re = mean \|u\| over the pore cells (top/bottom rows blocked) × 148 / ν (0.011 to 1.92 over the 9000 samples), Re_norm = min-max normalized Re (condition of the branch MLP); `re_min`, `re_max`; `MU_UX`, `SD_UX`, `MU_UY`, `SD_UY` = mean and std of ux, uy over the pore cells of the training samples (target standardization) |

## 4. Models and pipeline

| model | branch CNN | branch MLP | trunk | loss |
|---|---|---|---|---|
| **velocity variant** (`VelocityVariant`) | wg, UPRM, MIS | Re | x, y, px, py, d_w² | pore Huber + λ · divergence penalty, λ = 10 |
| vanilla DeepONet (velocity baseline; Table 2 C0) | wg | Re | x, y | – |
| **concentration variant**, adsorption (`ConcentrationVariant`; Table 6 C8) | wg, ux, uy | Da, Pe | x, y, GDF | Huber |
| **concentration variant**, Monod (`ConcentrationVariant`) | wg, ux, uy | Pe, Da_M | x, y, t, GDF | Huber |
| original PRT-DeepONet (concentration baseline, `OriginalPRTDeepONet`; Table 6 C0) | wg | Da, Pe (Monod: Pe, Da_M) | x, y, GDF (Monod: + t) | – |

wg = pore mask with the top/bottom rows blocked; UPRM = flow-neck radius map; MIS = maximal-inscribed-sphere radius
map; d_w² = squared wall distance; GDF = geodesic distance from the inlet. UPRM, MIS and d_w² are normalized with
statistics of the training geometries.

**Divergence penalty.** Mean squared central-difference divergence ∂ux/∂x + ∂uy/∂y of the predicted velocity over the
same pore cells as the Huber loss, the cells next to the solid included; the velocity is zero in the solid (no-slip
walls), so solid neighbours enter the differences with zero velocity (`divergence_penalty` in `velocity/core.py`).

**Pressure U-Net.** px, py of the velocity variant come from a U-Net that maps the pore mask to the pressure gradient of
the harmonic pressure field, ∇²P = 0 in the pore space (`pressure_gradient` in `velocity/core.py`; labels from
`velocity/1_preprocess/pressure_labels.py`).
Its fitting and early stopping use the 2700 non-test geometries (training and validation geometries); it is trained
once and frozen inside the velocity variant.

**Velocity input of the concentration variants.** ux, uy = velocity predicted by the velocity variant (λ = 10) at the
highest Re of the geometry, for every Pe (standardized over the training geometries).

Every physics runs the same stages (`<physics>/make_all.sh`, Section 1):

### Velocity pipeline

| folder | script | output |
|---|---|---|
| `1_preprocess/` | `geometry_features.py` | wg, UPRM, MIS, d_w² and their normalization statistics |
| | `pressure_labels.py` | training labels of the pressure U-Net: px, py of every geometry and their normalization PX_MAX, PY_MAX (training geometries) |
| `2_train/` | `train_pressure_unet.py` | `pressure_unet.pt` |
| | `train_velocity.py` | `velocity_variant_seed<s>.pt` (`LAMBDA=<λ> SEED=<s>`) |
| `3_extract/` | `evaluate.py` | per-sample test NRMSE, divergence and predictions of the velocity variant (`SPLIT=val`: validation samples) and of the vanilla DeepONet (test) |
| | `predicted_velocity.py` | predicted velocity of all 9000 samples (input of the concentration variants) |
| | `interpolation_unseen_re.py` | unseen-Re evaluation (5 seeds) |
| | `physics_diagnostics.py` | Table 4 diagnostics on the validation samples → `data/physics_diagnostics_val.csv` |
| | `extract_figure_data.py` | the other files of `data/` |
| | `timing.py` | inference time of one pore geometry on a single CPU core → `data/timing_measured.csv` |

### Irreversible adsorption pipeline

| folder | script | output |
|---|---|---|
| `1_preprocess/` | `geometry_features.py` | wg and GDF |
| | `velocity_input.py` | predicted velocity input and its statistics |
| `2_train/` | `train_adsorption.py` | `adsorption_variant_seed<s>.pt` (`SEED=<s>`) |
| `3_extract/` | `evaluate.py` | per-sample test RMSE of the concentration variant and the original PRT-DeepONet (2700 seen, 1200 unseen-(Pe, Da) samples); `SPLIT=val`: the concentration variant on the 2700 validation samples |
| | `extract_figure_data.py` | `data/` |
| | `timing.py` | inference time of one pore geometry on a single CPU core → `data/timing_measured.csv` |

### Monod pipeline

| folder | script | output |
|---|---|---|
| `1_preprocess/` | `geometry_features.py`, `velocity_input.py` | wg, GDF, velocity input |
| | `targets.py` | LB concentration of every sample and of the unseen time slices |
| `2_train/` | `train_monod.py` | `monod_variant_seed<s>.pt` (`SEED=<s>`) |
| `3_extract/` | `evaluate.py` | test RMSE of the concentration variant and the original PRT-DeepONet at the 21 training time slices (25200 samples) and the 5 unseen ones (6000) |
| | `extract_figure_data.py` | `data/` (field panels recomputed with the seed-42 weights) |
| | `timing.py` | inference time of one pore geometry on a single CPU core → `data/timing_measured.csv` |

## 5. Training, evaluation, weights and seeds

Proposed models: AdamW (lr 1e-3), batch 25, mixed precision, early stopping on the validation loss (patience 15; the
weights with the lowest validation loss are kept), at most 1000 epochs (Monod: 500). The pressure U-Net: full-image
MSE, batch 16, at most 300 epochs, patience 20. The test geometries are never used during training or model
selection.

Evaluation metrics:

* **Velocity**: NRMSE over the pore cells, RMSE / (max − min) of the LB field of the sample.
* **Concentration**: RMSE over the pore cells.

Training seeds: 42, 2, 3, 4, 5 for every model, proposed and baseline (released weights). The input-configuration
ablation of the velocity (Table 2) uses 10 seeds (42, 2–10), including its C0 row (vanilla DeepONet); its validation
metrics are shipped as data (`velocity/data/ablation_input_config.csv`, `baseline_val_metrics.csv`).

| folder | file | seeds |
|---|---|---|
| `velocity/weights/` | `pressure_unet.pt` | 42 |
| | `velocity_variant_seed<s>.pt` (λ = 10) | 42, 2, 3, 4, 5 |
| | `ablation/lambda/velocity_variant_lam<λ>_seed<s>.pt` (λ = 0, 1, 3, 5, 30, 100, 300, 500) | 42, 2, 3, 4, 5 |
| | `baseline/vanilla_deeponet_seed<s>.pt` | 42, 2, 3, 4, 5 |
| `irreversible_adsorption/weights/` | `adsorption_variant_seed<s>.pt`, `baseline/adsorption_original_seed<s>.pt` | 42, 2, 3, 4, 5 |
| `monod/weights/` | `monod_variant_seed<s>.pt`, `baseline/monod_original_seed<s>.pt` | 42, 2, 3, 4, 5 |

## 6. Figures and tables

Tables 2–4, 6, S1 and S2 are computed on the validation set (velocity: 900 samples; adsorption: 2700 samples),
together with Figure S1; all other figures use the test set. Outputs go to
`<physics>/4_figures/output/`.

Tables 5 and 7 are built from the measured times in `data/timing.csv`. `3_extract/timing.py` measures the inference
time of the released models on the machine it runs on, so its results depend on the hardware and its load.

### Velocity (`velocity/4_figures/`, data in `velocity/data/`)

| script | content | set | data |
|---|---|---|---|
| `table2_input_config.py` | Table 2: input configurations C0–C7 (10 seeds) | validation | `baseline_val_metrics.csv` (C0), `ablation_input_config.csv` |
| `table3_lambda_sweep.py` | Table 3: λ = 0–500 (5 seeds) | validation | `val_metrics.csv` |
| `table4_physics_diagnostics.py` | Table 4: physical consistency, LB and λ = 0, 10, 100 (5 seeds) | validation | `physics_diagnostics_val.csv` |
| `table5_timing.py` | Table 5: wall-clock time, LB vs velocity variant, and training time | – | `timing.csv` |
| `fig04_variant_vs_vanilla.py` | Figure 4: velocity variant vs vanilla DeepONet | test | `fields_test.npz`, `test_metrics.csv`, `baseline_test_metrics.csv` |
| `fig05_nrmse_vs_re.py` | Figure 5: NRMSE vs Re | test | `test_metrics.csv`, `baseline_test_metrics.csv` |
| `fig06_unseen_re_errors.py` | Figure 6: unseen-Re errors | test | `test_metrics.csv`, `unseen_re_metrics.csv` |
| `fig07_unseen_re_showcase.py` | Figure 7: unseen-Re fields | test | `fields_unseen_re.npz` |
| `fig08_permeability.py` | Figure 8: permeability, LB vs predicted (5 seeds) | test | `permeability.csv` |

### Irreversible adsorption (`irreversible_adsorption/4_figures/`, data in `irreversible_adsorption/data/`)

| script | content | set | data |
|---|---|---|---|
| `table_input_config.py` | Table 6: input configurations C0–C10 (5 seeds) | validation | `ablation_input_config.csv` |
| `table7_timing.py` | Table 7: wall-clock time, LB vs concentration variant, and training time (adsorption and Monod) | – | `timing.csv`, `../monod/data/timing.csv` |
| `table_lambda.py` | Table S1: λ of the velocity variant supplying the velocity (5 seeds) | validation | `ablation_lambda.csv` |
| `table_velocity_input.py` | Table S2: velocity input (matched / mismatched Re, predicted / LB, vector / magnitude; 5 seeds) | validation | `ablation_velocity_input.csv` |
| `fig_worst_and_histogram.py` | Figure 9: worst sample of the original model + RMSE distributions | test | `fields_worst.npz`, `test_metrics.csv`, `baseline_test_metrics.csv` |
| `fig_peda_heatmap.py` | Figure 10: mean RMSE over the 9 (Pe, Da) conditions (5 seeds) | test | `test_metrics.csv`, `baseline_test_metrics.csv` |
| `fig_unseen_peda.py` | Figure 12: unseen (Pe, Da) conditions | test | `fields_unseen.npz`, `test_metrics.csv`, `unseen_metrics.csv`, `baseline_unseen_metrics.csv` |
| `fig_velocity_physics_vs_concentration.py` | Figure S1: velocity physics metrics vs concentration RMSE | validation | `../velocity/data/physics_diagnostics_val.csv`, `val_metrics.csv` |

### Monod (`monod/4_figures/`, data in `monod/data/`)

| script | content | set | data |
|---|---|---|---|
| `fig_histogram.py` | Figure 11a: RMSE distributions, original vs variant | test | `test_metrics.csv`, `baseline_test_metrics.csv` |
| `fig_worst_domain_time.py` | Figure 11b: worst sample of the original model (Pe 10, Da_M 7.4), t = 1, 20, 40, 60, 80 | test | `fields_worst.npz` |
| `fig_unseen_t.py` | Figure 13: unseen time slices t = 6, 22, 38, 54, 70 | test | `fields_unseen_t.npz`, `test_metrics.csv`, `unseen_t_metrics.csv`, `baseline_unseen_t_metrics.csv` |

## 7. License

GPL-3.0 (see `LICENSE`).
