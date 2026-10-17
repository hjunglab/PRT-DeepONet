#!/usr/bin/env python3
"""Monod kinetics at unseen time slices t = 6, 22, 38, 54, 70.
(a) RMSE distributions (densities): variant at the seen slices (25200), variant and original at the unseen slices
(6000 each); (b) LB and error maps of both models for the (geometry, Pe, Da_M) with the largest mean original-model
error over the unseen slices. Seed 42."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plt, paths, save, read_csv, read_metrics
from _fieldrows import draw

DATA, OUT = paths("monod"); SEED = 42
M = read_csv(f"{DATA}/test_metrics.csv"); U = read_metrics(DATA, "unseen_t_metrics")
seen = M["rmse"][(M["model"] == "variant") & (M["seed"] == SEED)]
uv = U["rmse"][(U["model"] == "variant") & (U["seed"] == SEED)]
uo = U["rmse"][(U["model"] == "original") & (U["seed"] == SEED)]
F = np.load(f"{DATA}/fields_unseen_t.npz"); nt = len(F["t"])
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(4, nt + 1, height_ratios=[1.35, 1, 1, 1], width_ratios=[1] * nt + [0.04], hspace=0.55, wspace=0.08)
ah = fig.add_subplot(gs[0, :nt])
ah.hist(seen, bins=45, color="#4C72B0", alpha=0.55, density=True, label="Seen time points : Concentration variant")
ah.hist(uv, bins=45, color="#C44E52", alpha=0.55, density=True, label="Unseen time points : Concentration variant")
ah.hist(uo, bins=45, color="#55A868", alpha=0.50, density=True, label="Unseen time points : Original PRT-DeepONet")
for a, c in ((seen, "#2f4b7c"), (uv, "#a4242a"), (uo, "#2a6a3a")):
    ah.axvline(a.mean(), color=c, ls="--", lw=1.6, label=f"mean = {a.mean():.4f}")
ah.set_xlabel("RMSE"); ah.set_ylabel("density"); ah.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
ah.set_title("(a)", loc="left", fontsize=13, fontweight="bold")
draw(fig, gs, 1, F, 0.6)
fig.axes[1].text(-0.18, 1.25, "(b)", transform=fig.axes[1].transAxes, fontsize=13, fontweight="bold")
save(fig, OUT, "fig_unseen_t")
print(f"seen variant {seen.mean():.4f} | unseen variant {uv.mean():.4f} | unseen original {uo.mean():.4f}")
print(f"base {int(F['base'])} (Pe {float(F['Pe']):g}, Da {float(F['Da']):g}) original {np.round(F['rmse_original'], 3)} variant {np.round(F['rmse_variant'], 3)}")
