#!/usr/bin/env python3
"""Figure 7. Unseen-Re cases at the (a) 70th, (b) 80th and (c) 90th percentiles of the NRMSE distribution:
mask, LB and predicted velocity (vectors coloured by speed / max LB speed), u_x and u_y error maps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, NY, NX

DATA, OUT = paths("velocity")
F = np.load(f"{DATA}/fields_unseen_re.npz")
EM, ST, ARROW = 0.3, 4, 12.0


def quiver(a, ux, uy, pore, mask, vm):
    a.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.30)
    Y, X = np.mgrid[0:NY:ST, 0:NX:ST]
    QX = ux[::ST, ::ST] / vm; QY = uy[::ST, ::ST] / vm; P = pore[::ST, ::ST]
    q = a.quiver(X[P], Y[P], QX[P], QY[P], np.hypot(QX, QY)[P], cmap="viridis", clim=(0, 1), scale=1.0 / ARROW,
                 scale_units="xy", angles="xy", pivot="mid", width=0.006)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    return q


def err(a, e, pore, title):
    im = a.imshow(np.ma.masked_where(~pore, e), origin="lower", cmap="seismic", vmin=-EM, vmax=EM)
    a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    return im


fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 6, hspace=0.30, wspace=0.12)
qa, ea = [], []
for c in range(3):
    mask = F["mask"][c] == 1; pore = mask
    uxg, uyg, uxp, uyp = F["ux_gt"][c], F["uy_gt"][c], F["ux_pred"][c], F["uy_pred"][c]
    vm = float(np.hypot(uxg, uyg)[pore].max()) + 1e-30; cc = 2 * c
    am = fig.add_subplot(gs[0, cc:cc + 2]); am.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1); am.set_xticks([]); am.set_yticks([])
    am.set_title(f"({'abc'[c]})\n{F['percentile'][c]}th percentile\nRe = {F['re'][c]:.3f}", fontsize=12, fontweight="bold")
    at = fig.add_subplot(gs[1, cc:cc + 2]); q = quiver(at, uxg, uyg, pore, mask, vm)
    ap = fig.add_subplot(gs[2, cc:cc + 2]); quiver(ap, uxp, uyp, pore, mask, vm); qa += [at, ap]
    ax_ = fig.add_subplot(gs[3, cc]); im = err(ax_, (uxp - uxg) / vm, pore, f"$u_x$ NRMSE = {F['nrmse_ux'][c]:.4f}")
    ay_ = fig.add_subplot(gs[3, cc + 1]); err(ay_, (uyp - uyg) / vm, pore, f"$u_y$ NRMSE = {F['nrmse_uy'][c]:.4f}"); ea += [ax_, ay_]
    if c == 0:
        am.set_ylabel("Mask", fontsize=12, fontweight="bold"); at.set_ylabel("True", fontsize=12, fontweight="bold")
        ap.set_ylabel("Prediction", fontsize=12, fontweight="bold"); ax_.set_ylabel("Error map\n(pred-true)", fontsize=11, fontweight="bold")
    print(f"P{F['percentile'][c]}: {F['cond'][c]} base {F['base'][c]} Re {F['re'][c]:.3f} u_x {F['nrmse_ux'][c]:.4f} u_y {F['nrmse_uy'][c]:.4f}")
fig.colorbar(q, ax=qa, fraction=0.010, pad=0.01).set_ticks([0, 0.5, 1.0])
fig.colorbar(im, ax=ea, fraction=0.010, pad=0.01).set_ticks([-EM, 0, EM])
save(fig, OUT, "fig07_unseen_re_showcase")
