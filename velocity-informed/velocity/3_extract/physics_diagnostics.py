#!/usr/bin/env python3
"""Physical-consistency diagnostics of the LB velocity and of the velocity variant at lambda = 0, 10 (adopted)
and 100, per sample and training seed, on the validation samples (Table 4).

    flux_cov    std_j Q(j) / mean_j Q(j)  over all 148 cross-sections j, Q(j) = sum_i ux[i,j] over the pore cells of
                the whole cross-section
    flux_ratio  max_j Q(j) / min_j Q(j)
    io_err      |Q(inlet) - Q(outlet)| / mean_j Q(j)   (first and last column)
    wallnorm    near-wall normal velocity: at every pore cell next to the solid boundary, u_n = ux nx + uy ny with n the
                normalized sum of the unit vectors from its solid neighbours towards the cell; mean |u_n| over these
                cells / U, U = mean LB pore speed of the sample. The half-way bounce-back wall lies half a lattice
                spacing beyond these cells, so the LB and predicted values are taken at the same cells, without
                extrapolation.
    div         mean |dux/dx + duy/dy| over all pore cells / sp (central differences, zero velocity in the solid, as in
                the divergence penalty of the training loss); sp = mean pore speed of the field

Inputs : $PRT_WORK/velocity/eval/velocity_variant[_lam0|_lam100]_seed<S>_val.npz (evaluate.py with SPLIT=val), LB labels
Output : velocity/data/physics_diagnostics_val.csv
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import velocity_mask, VelocityInputs

SEEDS = [42, 2, 3, 4, 5]
SERIES = [("lam0", "velocity_variant_lam0_seed{}"), ("lam10", "velocity_variant_seed{}"), ("lam100", "velocity_variant_lam100_seed{}")]
KEYS = ["flux_cov", "flux_ratio", "io_err", "wallnorm", "div"]


def wall_normals(P, solid):
    S = solid.astype(float)
    nx_ = np.zeros(P.shape); ny_ = np.zeros(P.shape); nsol = np.zeros(P.shape)
    ny_[1:, :] += S[:-1, :]; ny_[:-1, :] -= S[1:, :]; nx_[:, 1:] += S[:, :-1]; nx_[:, :-1] -= S[:, 1:]
    nsol[1:, :] += S[:-1, :]; nsol[:-1, :] += S[1:, :]; nsol[:, 1:] += S[:, :-1]; nsol[:, :-1] += S[:, 1:]
    nn = np.hypot(nx_, ny_); cells = P & (nsol > 0) & (nn > 0)
    return cells, np.where(cells, nx_ / np.maximum(nn, 1e-12), 0), np.where(cells, ny_ / np.maximum(nn, 1e-12), 0)


def diagnostics(UX, UY, P, wall, U):
    sp = np.sqrt(UX[P] ** 2 + UY[P] ** 2).mean() + 1e-30
    Q = (UX * P).sum(axis=0)
    Q = Q * np.sign(np.median(Q)); meanQ = Q.mean() if Q.mean() != 0 else 1e-30
    flux_cov = float(Q.std() / abs(meanQ))
    flux_ratio = float(Q.max() / max(Q.min(), 0.02 * abs(meanQ)))
    io_err = float(abs(Q[0] - Q[-1]) / abs(meanQ))
    cells, nx_, ny_ = wall
    wallnorm = float(np.abs(UX[cells] * nx_[cells] + UY[cells] * ny_[cells]).mean() / U) if cells.any() else np.nan
    ux = UX * P; uy = UY * P
    uxp = np.concatenate([ux[:, :1], ux, ux[:, -1:]], 1); uyp = np.concatenate([np.zeros_like(uy[:1]), uy, np.zeros_like(uy[:1])], 0)
    dv = (uxp[:, 2:] - uxp[:, :-2]) * 0.5 + (uyp[2:] - uyp[:-2]) * 0.5
    div = float(np.abs(dv[P]).mean() / sp)
    return [flux_cov, flux_ratio, io_err, wallnorm, div]


VI = VelocityInputs(pressure_unet_weights=None)
samples = common.load_split()["velocity_samples"]["val"]
UXG, UYG, _ = VI.targets(samples)
P = np.stack([VI.mask[VI.k(i)] == 1 for i in samples])
W = [wall_normals(P[i], velocity_mask(idx) == 0) for i, idx in enumerate(samples)]
U = [np.hypot(UXG[i][P[i]], UYG[i][P[i]]).mean() for i in range(len(samples))]
EV = os.path.join(common.work_dir("velocity"), "eval")
rows = []
for i, idx in enumerate(samples):
    rows.append(["", "LB", idx, f"{VI.re[idx]:.6g}"] + [f"{x:.6g}" for x in diagnostics(UXG[i], UYG[i], P[i], W[i], U[i])] + [""])
for seed in SEEDS:
    for tag, pat in SERIES:
        z = np.load(os.path.join(EV, pat.format(seed) + "_val.npz"))
        assert np.array_equal(z["idx"], samples)
        uxp, uyp, nm = z["ux_pred"], z["uy_pred"], z["nrmse_mag"]
        for i, idx in enumerate(samples):
            d = diagnostics(uxp[i], uyp[i], P[i], W[i], U[i])
            rows.append([seed, tag, idx, f"{VI.re[idx]:.6g}"] + [f"{x:.6g}" for x in d] + [f"{nm[i]:.6g}"])
    print(f"seed {seed} done", flush=True)
out = os.path.join(common.data_dir("velocity"), "physics_diagnostics_val.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "series", "idx", "re"] + KEYS + ["nrmse_mag"]); w.writerows(rows)
print("[save]", out)
