#!/usr/bin/env python3
"""Figure 5. Component NRMSE of the predicted velocity vs Re for (a) the vanilla DeepONet and
(b) the velocity variant (lambda = 10), test set, seed 42. Dashed lines = means."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_metrics

DATA, OUT = paths("velocity")
M = read_metrics(DATA, "test_metrics")
seed = int(np.load(f"{DATA}/fields_test.npz")["seed"])
var = (M["model"] == "variant") & (M["lambda"] == 10) & (M["seed"] == seed)
van = (M["model"] == "vanilla") & (M["seed"] == seed)
assert np.array_equal(M["idx"][var], M["idx"][van])
re = M["re"][var]
ymax = max(M["nrmse_ux"][var].max(), M["nrmse_uy"][var].max(), M["nrmse_ux"][van].max(), M["nrmse_uy"][van].max()) * 1.03
C_UX, C_UY = "#4FB393", "#E8924A"
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
for a, sel, tag in ((axs[0], van, "a"), (axs[1], var, "b")):
    nux, nuy = M["nrmse_ux"][sel], M["nrmse_uy"][sel]
    a.scatter(re, nux, s=12, color=C_UX, edgecolors="none", rasterized=True, label=r"$u_x$")
    a.scatter(re, nuy, s=12, color=C_UY, edgecolors="none", rasterized=True, label=r"$u_y$")
    a.axhline(nux.mean(), color=C_UX, ls="--", lw=1.6, label=f"mean = {nux.mean():.4f}")
    a.axhline(nuy.mean(), color=C_UY, ls="--", lw=1.6, label=f"mean = {nuy.mean():.4f}")
    a.set_xlabel("Reynolds number (Re)"); a.set_ylabel("NRMSE"); a.set_xlim(-0.03, 2.0); a.set_ylim(0, ymax)
    a.set_title(f"({tag})", fontsize=13); a.legend(loc="upper left", frameon=False, fontsize=10)
    print(f"({tag}) u_x mean {nux.mean():.4f}  u_y mean {nuy.mean():.4f}")
plt.tight_layout()
save(fig, OUT, "fig05_nrmse_vs_re")
