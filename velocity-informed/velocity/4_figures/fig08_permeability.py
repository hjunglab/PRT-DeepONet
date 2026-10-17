#!/usr/bin/env python3
"""Supplementary figure. Darcy permeability k = nu q L / dP from the predicted vs the LB velocity, test set,
5 training seeds (log-log, coloured by Re, 1:1 line, R^2 per seed)."""
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_csv

DATA, OUT = paths("velocity")
P = read_csv(f"{DATA}/permeability.csv"); SEEDS = [42, 2, 3, 4, 5]
K = {s: (P["k_lb"][P["seed"] == s], P["k_pred"][P["seed"] == s]) for s in SEEDS}
re = P["re"][P["seed"] == SEEDS[0]]
allk = np.concatenate([np.r_[a, b] for a, b in K.values()]); lo = allk[allk > 0].min() * 0.9; hi = allk.max() * 1.1
norm = LogNorm(vmin=re.min(), vmax=re.max())
r2 = lambda a, b: 1 - np.sum((b - a) ** 2) / np.sum((a - a.mean()) ** 2)
fig, axs = plt.subplots(2, 3, figsize=(15, 9)); axs = axs.ravel()
for k, s in enumerate(SEEDS):
    a = axs[k]; klb, kp = K[s]
    a.plot([lo, hi], [lo, hi], "k--", lw=1.1, zorder=1)
    sc = a.scatter(klb, kp, c=re, cmap="viridis", norm=norm, s=14, alpha=0.85, edgecolors="none", zorder=2)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal", "box")
    a.grid(True, which="both", ls=":", alpha=0.35); a.set_title(f"seed {s}", fontsize=12)
    a.text(0.05, 0.95, f"$R^2$={r2(klb, kp):.3f}", transform=a.transAxes, va="top", ha="left", fontsize=10,
           bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95))
    a.set_xlabel("$k_{LB}$"); a.set_ylabel("$k_{pred}$")
    print(f"seed {s}: R2 = {r2(klb, kp):.4f}")
axs[5].axis("off")
fig.colorbar(sc, ax=axs[:5].tolist(), fraction=0.02, pad=0.02).set_label("Reynolds $Re$ (log)")
save(fig, OUT, "fig08_permeability")
