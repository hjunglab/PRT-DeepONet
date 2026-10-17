#!/usr/bin/env python3
"""Irreversible adsorption at unseen (Pe, Da) = (3, 185), (3, 518), (7.5, 185), (7.5, 518).
(a) RMSE distributions (densities): variant at the seen conditions (2700), variant and original at the unseen
conditions (1200 each); (b) LB, prediction and error at the 80th-percentile case of each unseen condition. Seed 3."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_csv, read_metrics

DATA, OUT = paths("irreversible_adsorption")
F = np.load(f"{DATA}/fields_unseen.npz"); seed = int(F["seed"])
M = read_csv(f"{DATA}/test_metrics.csv"); U = read_metrics(DATA, "unseen_metrics")
seen = M["rmse"][(M["model"] == "variant") & (M["seed"] == seed)]
uv = U["rmse"][(U["model"] == "variant") & (U["seed"] == seed)]
uo = U["rmse"][(U["model"] == "original") & (U["seed"] == seed)]
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(4, 5, height_ratios=[1.35, 1, 1, 1], width_ratios=[1, 1, 1, 1, 0.04], hspace=0.33, wspace=0.10)
ah = fig.add_subplot(gs[0, :4])
ah.hist(seen, bins=45, color="#4C72B0", alpha=0.55, density=True, label="Seen Pe, Da : Concentration variant")
ah.hist(uv, bins=45, color="#C44E52", alpha=0.55, density=True, label="Unseen Pe, Da : Concentration variant")
ah.hist(uo, bins=45, color="#55A868", alpha=0.50, density=True, label="Unseen Pe, Da : Original PRT-DeepONet")
for a, c in ((seen, "#2f4b7c"), (uv, "#a4242a"), (uo, "#2a6a3a")):
    ah.axvline(a.mean(), color=c, ls="--", lw=1.6, label=f"mean = {a.mean():.4f}")
ah.set_xlabel("RMSE"); ah.set_ylabel("density"); ah.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
ah.set_title("(a)", loc="left", fontsize=13, fontweight="bold")
for c in range(4):
    m = F["mask"][c] == 1; gt, pr = F["gt"][c], F["pred"][c]
    at = fig.add_subplot(gs[1, c]); it = at.imshow(np.where(m, gt, 0), cmap="viridis", vmin=0, vmax=1, origin="lower")
    at.text(0.96, 0.93, f"Pe={F['Pe'][c]:g}, Da={F['Da'][c]:g}\nRMSE = {F['rmse'][c]:.4f}", transform=at.transAxes,
            ha="right", va="top", color="w", fontsize=9)
    ap = fig.add_subplot(gs[2, c]); ap.imshow(np.where(m, pr, 0), cmap="viridis", vmin=0, vmax=1, origin="lower")
    ae = fig.add_subplot(gs[3, c]); ie = ae.imshow(np.ma.masked_where(~m, pr - gt), cmap="seismic", vmin=-0.4, vmax=0.4, origin="lower")
    for a in (at, ap, ae):
        a.set_xticks([]); a.set_yticks([])
    if c == 0:
        at.set_ylabel("True", fontsize=12, fontweight="bold"); ap.set_ylabel("Prediction", fontsize=12, fontweight="bold")
        ae.set_ylabel("Error map\n(pred-true)", fontsize=11, fontweight="bold")
        at.text(-0.18, 1.15, "(b)", transform=at.transAxes, fontsize=13, fontweight="bold")
fig.colorbar(it, cax=fig.add_subplot(gs[1, 4])).set_ticks([0, 0.5, 1.0])
fig.colorbar(ie, cax=fig.add_subplot(gs[3, 4])).set_ticks([-0.4, 0, 0.4])
save(fig, OUT, "fig_unseen_peda")
print(f"seen variant {seen.mean():.4f} | unseen variant {uv.mean():.4f} | unseen original {uo.mean():.4f}")
for cond in sorted(set(U["cond"])):
    s = U["cond"] == cond
    print(f"  {cond}: variant {U['rmse'][s & (U['model'] == 'variant') & (U['seed'] == seed)].mean():.4f}  "
          f"original {U['rmse'][s & (U['model'] == 'original') & (U['seed'] == seed)].mean():.4f}")
