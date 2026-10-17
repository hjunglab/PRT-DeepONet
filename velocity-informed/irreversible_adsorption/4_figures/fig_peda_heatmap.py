#!/usr/bin/env python3
"""Irreversible adsorption. Mean pore RMSE of each of the nine (Da, Pe) training conditions for (a) the original
model and (b) the concentration variant; cell = 5-seed mean +/- std (population std) of the per-seed condition mean."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_metrics

DATA, OUT = paths("irreversible_adsorption")
M = read_metrics(DATA, "test_metrics"); SEEDS = [42, 2, 3, 4, 5]; DAS = [74, 296, 740]; PES = [1, 5, 10]


def cond(model):
    C = np.array([[[M["rmse"][(M["model"] == model) & (M["seed"] == s) & (M["Da"] == d) & (M["Pe"] == p)].mean()
                    for p in PES] for d in DAS] for s in SEEDS])
    return C.mean(0), C.std(0), C


fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for k, (title, model) in enumerate((("(a) Original PRT-DeepONet", "original"), ("(b) Concentration variant", "variant"))):
    MU, SD, C = cond(model)
    im = ax[k].imshow(MU, cmap="magma_r", vmin=0.01, vmax=0.06)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels([f"Pe{p}" for p in PES])
    ax[k].set_yticks(range(3)); ax[k].set_yticklabels([f"Da{d}" for d in DAS])
    for i in range(3):
        for j in range(3):
            ax[k].text(j, i, f"{MU[i, j]:.4f}\n±{SD[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8.5)
    ax[k].set_title(title, fontsize=12, fontweight="bold")
    o = C.mean(axis=(1, 2))
    print(f"{model}: mean over conditions {o.mean():.4f} ± {o.std():.4f}")
plt.tight_layout(); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
save(fig, OUT, "fig_peda_heatmap")
