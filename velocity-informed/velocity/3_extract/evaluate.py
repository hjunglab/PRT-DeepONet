#!/usr/bin/env python3
"""Score the velocity variant on the 900 test samples (300 test geometries x 3 Re) and keep the predictions.
SPLIT=val scores the 900 validation samples instead (used for the model-selection tables; output <name>_val.npz).

The vanilla DeepONet (comparison baseline, core.py; released weights, seeds 42 2 3 4 5, test split) is
scored the same way. For every weight file: per-sample NRMSE of |u|, ux and uy over the pore cells,
    NRMSE = sqrt(mean((pred - LB)^2)) / (max(LB) - min(LB)),
and the normalized divergence  mean over all pore cells of ((dux/dx + duy/dy) / SD_UX)^2, with the central differences,
zero solid velocity and edge columns of the divergence penalty (core.divergence_penalty).

Weights: velocity/weights/ (released) or WEIGHTS=<dir> (e.g. $PRT_WORK/velocity/weights after retraining);
the vanilla DeepONet always uses the released weights velocity/weights/baseline/vanilla_deeponet_seed<s>.pt.
    python evaluate.py                       all velocity variant weights (+ the vanilla DeepONet on the test split)
    python evaluate.py velocity_variant_seed42 vanilla_deeponet_seed42 ...
Output: $PRT_WORK/velocity/eval/<name>.npz  (idx, re, nrmse_mag, nrmse_ux, nrmse_uy, div, ux_pred, uy_pred)
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import load_weights
from velocity.core import VelocityVariant, VelocityInputs, predict, nrmse, vanilla_inputs, VanillaDeepONet

WDIR = os.environ.get("WEIGHTS", common.weights_dir("velocity"))
OUT = os.path.join(common.work_dir("velocity"), "eval"); os.makedirs(OUT, exist_ok=True)
SPLIT = os.environ.get("SPLIT", "test")
names = sys.argv[1:] or (sorted(os.path.basename(p)[:-3] for p in
                                glob.glob(f"{WDIR}/velocity_*.pt") + glob.glob(f"{WDIR}/ablation/*/velocity_*.pt"))
                         + ([f"vanilla_deeponet_seed{s}" for s in common.SEEDS] if SPLIT == "test" else []))
dev = common.device()
VI = VelocityInputs(dev=dev)
test = common.load_split()["velocity_samples"][SPLIT]
SUF = "" if SPLIT == "test" else "_" + SPLIT
inp = {"variant": VI.build(test), "vanilla": vanilla_inputs(VI, test)}
UX, UY, _ = VI.targets(test)
MAG = np.sqrt(UX * UX + UY * UY)
roi = inp["variant"]["roi"].numpy() > 0.5


def divergence(ux, uy, pore, sd):
    ux = ux * pore; uy = uy * pore
    uxp = np.concatenate([ux[:, :1], ux, ux[:, -1:]], 1)
    uyp = np.concatenate([np.zeros_like(uy[:1]), uy, np.zeros_like(uy[:1])], 0)
    dn = ((uxp[:, 2:] - uxp[:, :-2]) * 0.5 + (uyp[2:] - uyp[:-2]) * 0.5) / sd
    return float((dn[pore] ** 2).mean())


def find(name):
    for p in [f"{WDIR}/{name}.pt", common.baseline_weights("velocity", name)] \
            + glob.glob(f"{WDIR}/ablation/*/{name}.pt"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


for name in names:
    kind = "vanilla" if name.startswith("vanilla_deeponet") else "variant"
    model = load_weights(VanillaDeepONet() if kind == "vanilla" else VelocityVariant(), find(name), dev)
    uxp, uyp = VI.to_physical(predict(model, inp[kind], dev))
    N = len(test); nm, nux, nuy, dv = (np.empty(N) for _ in range(4))
    for i in range(N):
        mk = roi[i]
        nm[i] = nrmse(MAG[i], np.sqrt(uxp[i] ** 2 + uyp[i] ** 2), mk)
        nux[i] = nrmse(UX[i], uxp[i], mk); nuy[i] = nrmse(UY[i], uyp[i], mk)
        dv[i] = divergence(uxp[i], uyp[i], mk, VI.stats["SD_UX"])
    np.savez_compressed(os.path.join(OUT, name + SUF + ".npz"), idx=np.array(test), re=VI.re[np.array(test)],
                        nrmse_mag=nm, nrmse_ux=nux, nrmse_uy=nuy, div=dv,
                        ux_pred=uxp.astype(np.float32), uy_pred=uyp.astype(np.float32))
    print(f"{name + SUF:44s} NRMSE |u| mean {nm.mean():.4f}  P50 {np.median(nm):.4f}  P90 {np.percentile(nm, 90):.4f}  "
          f"worst {nm.max():.4f} | ux {nux.mean():.4f} uy {nuy.mean():.4f}", flush=True)
