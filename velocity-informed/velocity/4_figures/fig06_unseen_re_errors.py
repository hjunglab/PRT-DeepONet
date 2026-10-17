#!/usr/bin/env python3
"""Figure 6. Seen Re (900 test samples) vs unseen Re (600 cases): (a) NRMSE distributions (densities),
(b) component NRMSE vs Re at the unseen conditions. Seed 42."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_csv

DATA, OUT = paths("velocity")
seed = int(np.load(f"{DATA}/fields_test.npz")["seed"])
M = read_csv(f"{DATA}/test_metrics.csv"); U = read_csv(f"{DATA}/unseen_re_metrics.csv")
seen = M["nrmse_mag"][(M["model"] == "variant") & (M["lambda"] == 10) & (M["seed"] == seed)]
s = U["seed"] == seed; unseen = U["nrmse_mag"][s]; re = U["re"][s]; nux, nuy = U["nrmse_ux"][s], U["nrmse_uy"][s]
C_UX, C_UY = "#4FB393", "#E8924A"
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
a = axs[0]
a.hist(seen, bins=45, color="#4C72B0", alpha=0.55, density=True, label="Seen Re")
a.hist(unseen, bins=45, color="#C44E52", alpha=0.60, density=True, label="Unseen Re")
a.axvline(seen.mean(), color="#2f4b7c", ls="--", lw=1.8, label=f"mean = {seen.mean():.4f}")
a.axvline(unseen.mean(), color="#a4242a", ls="--", lw=1.8, label=f"mean = {unseen.mean():.4f}")
a.set_xlabel("NRMSE"); a.set_ylabel("density"); a.set_title("(a)", fontsize=13); a.legend(loc="upper right", frameon=False, fontsize=10)
a = axs[1]
a.scatter(re, nux, s=12, color=C_UX, edgecolors="none", rasterized=True, label=r"$u_x$")
a.scatter(re, nuy, s=12, color=C_UY, edgecolors="none", rasterized=True, label=r"$u_y$")
a.axhline(nux.mean(), color=C_UX, ls="--", lw=1.6, label=f"mean = {nux.mean():.4f}")
a.axhline(nuy.mean(), color=C_UY, ls="--", lw=1.6, label=f"mean = {nuy.mean():.4f}")
a.set_xlabel("Reynolds number (Re)"); a.set_ylabel("NRMSE"); a.set_title("(b)", fontsize=13); a.legend(loc="upper right", frameon=False, fontsize=10)
plt.tight_layout()
save(fig, OUT, "fig06_unseen_re_errors")
print(f"seen {seen.mean():.4f} (N={len(seen)})  unseen {unseen.mean():.4f} (N={len(unseen)})  unseen u_x {nux.mean():.4f} u_y {nuy.mean():.4f}  "
      f"unseen Re range {re.min():.3f}-{re.max():.3f}")
