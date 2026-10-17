"""LB row and the two error-map rows (original, variant) of a Monod time sequence, drawn into gs rows r0..r0+2."""
import numpy as np


def draw(fig, gs, r0, F, emax):
    m = F["mask"] == 1; nt = len(F["t"]); it = ie = None
    for c in range(nt):
        a = fig.add_subplot(gs[r0, c]); it = a.imshow(np.where(m, F["gt"][c], 0), cmap="viridis", vmin=0, vmax=1, origin="lower")
        a.set_title(f"t={int(F['t'][c])}", fontsize=12, fontweight="bold"); a.set_xticks([]); a.set_yticks([])
        if c == 0:
            a.set_ylabel("True", fontsize=12, fontweight="bold")
    for r, (k, lab) in enumerate((("original", "Original PRT-DeepONet"), ("variant", "Concentration variant")), start=1):
        for c in range(nt):
            a = fig.add_subplot(gs[r0 + r, c])
            ie = a.imshow(np.ma.masked_where(~m, F[f"pred_{k}"][c] - F["gt"][c]), cmap="seismic", vmin=-emax, vmax=emax, origin="lower")
            a.contour(m.astype(float), levels=[0.5], colors="0.55", linewidths=0.4, origin="lower")
            a.set_title(f"RMSE = {F[f'rmse_{k}'][c]:.3f}", fontsize=11, fontweight="bold"); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel("Error map\n(pred-true)", fontsize=11, fontweight="bold")
                a.text(-0.02, 1.32, lab, transform=a.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold",
                       bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    fig.colorbar(it, cax=fig.add_subplot(gs[r0, nt])).set_ticks([0, 0.5, 1.0])
    fig.colorbar(ie, cax=fig.add_subplot(gs[r0 + 1:r0 + 3, nt])).set_ticks([-emax, 0, emax])
