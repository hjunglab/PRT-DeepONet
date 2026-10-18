#!/usr/bin/env python3
"""Velocity-field physics metrics vs downstream adsorption error, per validation geometry (300 points).
x: metric of the adopted velocity prediction (lambda = 10, seed 42) at the highest Re of the geometry, i.e. the
   velocity actually supplied to the concentration variant (../velocity/data/physics_diagnostics_val.csv);
y: pore RMSE of the concentration variant averaged over the 9 (Da, Pe) conditions of the geometry.
Panel title: R^2 (squared Pearson r), mean +/- std over the 5 concentration training seeds; scatter = seed 42."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import plt, paths, save, read_csv

DATA, OUT = paths("irreversible_adsorption")
P = read_csv(os.path.join(common.ROOT, "velocity", "data", "physics_diagnostics_val.csv"))
M = read_csv(f"{DATA}/val_metrics.csv"); SEEDS = [42, 2, 3, 4, 5]
sel = (P["series"] == "lam10") & (P["seed"] == 42) & (((P["idx"] - 1) // common.N_BASE) == 2)
pb = ((P["idx"][sel] - 1) % common.N_BASE + 1).astype(int)
bases = sorted(set(pb))
Y = {s: np.array([M["rmse"][(M["model"] == "variant") & (M["seed"] == s) & (M["base"] == b)].mean() for b in bases]) for s in SEEDS}
MET = [("flux_cov", "Sectional flux variation (std/mean)"), ("flux_ratio", "Sectional flux ratio (max/min)"),
       ("io_err", "Inlet-outlet flux error"), ("wallnorm", "Near-wall normal velocity"),
       ("div", "Velocity divergence"), ("nrmse_mag", "Velocity NRMSE")]
fig, axs = plt.subplots(2, 3, figsize=(17, 9)); axs = axs.ravel()
for ax, (k, lab) in zip(axs, MET):
    xv = dict(zip(pb, P[k][sel])); x = np.array([xv[b] for b in bases])
    r2 = []
    for s in SEEDS:
        m = np.isfinite(x) & np.isfinite(Y[s]); r2.append(np.corrcoef(x[m], Y[s][m])[0, 1] ** 2)
    r2 = np.array(r2); r2s = r2.std(ddof=1); y = Y[42]; m = np.isfinite(x) & np.isfinite(y); xp, yp = x[m], y[m]
    hi = float(np.percentile(xp, 98.0))
    ax.scatter(np.clip(xp, None, hi), yp, s=20, alpha=0.55, color="#5a9bd4", edgecolors="none")
    b1, b0 = np.polyfit(xp, yp, 1); xs = np.linspace(xp.min(), hi, 50); ax.plot(xs, b0 + b1 * xs, "--", color="#d62728", lw=1.8)
    ax.set_title(f"{lab}\n$R^2$={r2.mean():.3f}±{r2s:.3f}", fontsize=12)
    ax.set_xlabel("Velocity field metric"); ax.set_ylabel("Concentration RMSE"); ax.grid(alpha=0.3)
    print(f"{k:12s} R2 = {r2.mean():.3f} ± {r2s:.3f}")
plt.tight_layout()
save(fig, OUT, "fig_velocity_physics_vs_concentration")
