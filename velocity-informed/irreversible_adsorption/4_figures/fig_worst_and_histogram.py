#!/usr/bin/env python3
"""Irreversible adsorption. (a) Test sample with the largest error of the original model (mask, LB concentration,
error maps of both models); (b) per-sample pore-RMSE distributions of the original model and the concentration
variant (2700 test samples, seed 42)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_metrics

DATA, OUT = paths("irreversible_adsorption")
F = np.load(f"{DATA}/fields_worst.npz"); seed = int(F["seed"])
M = read_metrics(DATA, "test_metrics")
rv = M["rmse"][(M["model"] == "variant") & (M["seed"] == seed)]
ro = M["rmse"][(M["model"] == "original") & (M["seed"] == seed)]


def figure(k, name):
    m = F["mask"][k] == 1; truth = np.where(m, F["gt"][k], 0)
    eo = np.ma.masked_where(~m, F["pred_original"][k] - F["gt"][k]); ev = np.ma.masked_where(~m, F["pred_variant"][k] - F["gt"][k])
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], hspace=0.5, wspace=0.3)
    am = fig.add_subplot(gs[0, 0]); at = fig.add_subplot(gs[0, 1])
    am.imshow(F["mask"][k], cmap="gray", vmin=0, vmax=1, origin="lower"); am.set_title("Mask", fontsize=12, fontweight="bold")
    it = at.imshow(truth, cmap="viridis", vmin=0, vmax=1, origin="lower"); at.set_title("True", fontsize=12, fontweight="bold")
    fig.colorbar(it, ax=at, fraction=0.046, pad=0.03).set_ticks([0, 0.5, 1.0])
    a0 = fig.add_subplot(gs[1, 0]); a1 = fig.add_subplot(gs[1, 1])
    a0.imshow(eo, cmap="seismic", vmin=-0.4, vmax=0.4, origin="lower")
    a0.set_title(f"Original PRT-DeepONet\nRMSE = {F['rmse_original'][k]:.4f}", fontsize=11)
    ie = a1.imshow(ev, cmap="seismic", vmin=-0.4, vmax=0.4, origin="lower")
    a1.set_title(f"Concentration variant\nRMSE = {F['rmse_variant'][k]:.4f}", fontsize=11)
    fig.colorbar(ie, ax=a1, fraction=0.046, pad=0.03).set_ticks([-0.4, 0, 0.4])
    for a in (am, at, a0, a1):
        a.set_xticks([]); a.set_yticks([])
    a0.text(0.0, 1.42, "Error map (pred-true)", transform=a0.transAxes, fontsize=11, fontweight="bold")
    am.text(0.0, 1.25, "(a)", transform=am.transAxes, fontsize=13, fontweight="bold")
    ah = fig.add_subplot(gs[:, 2])
    ah.hist(ro, bins=60, color="#6f8fc7", alpha=0.55, density=True, label="Original PRT-DeepONet")
    ah.hist(rv, bins=60, color="#cf6a6a", alpha=0.55, density=True, label="Concentration variant")
    ah.axvline(ro.mean(), color="#2f4b7c", ls="--", lw=1.8, label=f"mean = {ro.mean():.4f}")
    ah.axvline(rv.mean(), color="#a4242a", ls="--", lw=1.8, label=f"mean = {rv.mean():.4f}")
    ah.set_xlabel("RMSE"); ah.set_ylabel("density"); ah.legend(frameon=False)
    ah.set_title("(b)", fontsize=13, fontweight="bold", loc="left")
    save(fig, OUT, name)
    print(f"{name}: idx {F['idx'][k]} (Da {F['Da'][k]}, Pe {F['Pe'][k]}) original {F['rmse_original'][k]:.4f} variant {F['rmse_variant'][k]:.4f}")


names = list(F["names"])
figure(names.index("original_worst"), "fig_worst_and_histogram")
print(f"seed {seed}: mean RMSE original {ro.mean():.4f}  variant {rv.mean():.4f}  ({100 * (ro.mean() - rv.mean()) / ro.mean():.1f}% lower)")
