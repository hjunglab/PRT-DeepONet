#!/usr/bin/env python3
"""Figure 4. Velocity variant vs vanilla DeepONet at the 80th-percentile NRMSE case of the vanilla model:
mask, LB velocity, both predictions (vectors coloured by speed / max speed), and u_x, u_y error maps."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_metrics, NY, NX

DATA, OUT = paths("velocity")
F = np.load(f"{DATA}/fields_test.npz")
r = int(F["pick_rows"][list(F["pick_names"]).index("vanilla_p80")])
idx = int(F["idx"][r]); seed = int(F["seed"])
mask = F["mask"][r] == 1; roi = mask
uxg, uyg = F["ux_gt"][r], F["uy_gt"][r]; uxc, uyc = F["ux_variant"][r], F["uy_variant"][r]; uxv, uyv = F["ux_vanilla"][r], F["uy_vanilla"][r]
M = read_metrics(DATA, "test_metrics")


def comp(model, key):
    s = (M["model"] == model) & (M["seed"] == seed) & (M["idx"] == idx) & ((M["lambda"] == 10) | (M["model"] == "vanilla"))
    return float(M[key][s][0])


def nice(x):
    e = math.floor(math.log10(x)); b = 10 ** e; m = x / b
    for c in [1, 2, 4, 6, 10]:
        if m <= c + 1e-9:
            return c * b
    return 10 * b


allmag = np.concatenate([np.hypot(a, b)[roi] for a, b in [(uxg, uyg), (uxc, uyc), (uxv, uyv)]])
RNORM = float(allmag.max()); Q_SCALE = (1.0 / 4.6) * (nice(float(np.percentile(allmag, 80))) / RNORM); ELIM = 0.7


def quiver(a, ux, uy):
    a.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.30)
    st = 4; Y, X = np.mgrid[0:NY:st, 0:NX:st]
    QX = ux[::st, ::st] / RNORM; QY = uy[::st, ::st] / RNORM; P = roi[::st, ::st]
    q = a.quiver(X[P], Y[P], QX[P], QY[P], np.hypot(QX, QY)[P], cmap="viridis", clim=(0, 1), scale=Q_SCALE,
                 scale_units="xy", angles="xy", pivot="mid", width=0.005)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    return q


def err(a, p, g, title):
    im = a.imshow(np.ma.masked_where(~roi, (p - g) / RNORM), origin="lower", cmap="seismic", vmin=-ELIM, vmax=ELIM)
    a.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.12)
    a.set_title(title, fontsize=10.5); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    return im


fig = plt.figure(figsize=(13.5, 11.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.8], hspace=0.32, wspace=0.18)
am = fig.add_subplot(gs[0, 0:2]); at = fig.add_subplot(gs[0, 2:4])
am.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1); am.set_xticks([]); am.set_yticks([]); am.set_aspect("equal")
am.set_title("Mask", fontsize=13, fontweight="bold")
q = quiver(at, uxg, uyg); at.set_title("True", fontsize=13, fontweight="bold")
fig.colorbar(q, ax=at, fraction=0.025, pad=0.02).set_ticks([0, 0.5, 1.0])
ap = fig.add_subplot(gs[1, 0:2]); av = fig.add_subplot(gs[1, 2:4])
quiver(ap, uxc, uyc); ap.set_title("PRT-DeepONet", fontsize=13, fontweight="bold")
quiver(av, uxv, uyv); av.set_title("Vanilla DeepONet", fontsize=13, fontweight="bold")
ap.set_ylabel("Prediction", fontsize=13, fontweight="bold")
ae = [fig.add_subplot(gs[2, c]) for c in range(4)]
im = err(ae[0], uxc, uxg, f"$u_x$ NRMSE = {comp('variant', 'nrmse_ux'):.4f}")
err(ae[1], uyc, uyg, f"$u_y$ NRMSE = {comp('variant', 'nrmse_uy'):.4f}")
err(ae[2], uxv, uxg, f"$u_x$ NRMSE = {comp('vanilla', 'nrmse_ux'):.4f}")
err(ae[3], uyv, uyg, f"$u_y$ NRMSE = {comp('vanilla', 'nrmse_uy'):.4f}")
ae[0].set_ylabel("Error map\n(pred-true)", fontsize=12, fontweight="bold")
for a, lab in ((ae[0], "PRT-DeepONet"), (ae[2], "Vanilla DeepONet")):
    a.text(1.0, 1.42, lab, transform=a.transAxes, ha="center", va="bottom", fontsize=12, fontweight="bold",
           bbox=dict(boxstyle="round", fc="white", ec="0.6"))
fig.colorbar(im, ax=ae, fraction=0.012, pad=0.02).set_ticks([-0.7, 0, 0.7])
save(fig, OUT, "fig04_variant_vs_vanilla")
print(f"sample idx {idx} (seed {seed}): variant u_x {comp('variant', 'nrmse_ux'):.4f} u_y {comp('variant', 'nrmse_uy'):.4f} | "
      f"vanilla u_x {comp('vanilla', 'nrmse_ux'):.4f} u_y {comp('vanilla', 'nrmse_uy'):.4f}")
