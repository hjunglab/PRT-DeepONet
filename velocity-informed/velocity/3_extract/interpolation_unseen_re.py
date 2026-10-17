#!/usr/bin/env python3
"""Velocity variant at two Reynolds conditions that were not used in training (ReA between Re groups 0 and 1,
ReB between groups 1 and 2), on the 300 test geometries (600 cases).

The inputs are built as in training from the geometry of each case; Re is computed from the LB
velocity of the case and normalized with the training Re_min/Re_max.
Output: $PRT_WORK/velocity/eval/unseen_re_velocity_variant_seed<S>.npz  for every seed in SEEDS (42 2 3 4 5)
        (cond, base, re, nrmse_mag, nrmse_ux, nrmse_uy, ux_pred, uy_pred, ux_gt, uy_gt, mask)
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import coords, load_weights
from velocity.core import pore_mask, block_top_bottom, flow_neck_radius_map, mis_radius_map, wall_distance, VelocityVariant, load_preprocessed, pressure_unet, nrmse, unseen_velocity, UNSEEN_RE

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42 2 3 4 5").split()]
WDIR = os.environ.get("WEIGHTS", common.weights_dir("velocity"))
OUT = os.path.join(common.work_dir("velocity"), "eval"); os.makedirs(OUT, exist_ok=True)
dev = common.device()
G, F = load_preprocessed()
re_min, re_max = float(F["re_min"]), float(F["re_max"])
S = {k: float(F[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY")}
Y, X = coords()

cases = [(c, b) for c in UNSEEN_RE for b in common.load_split()["test_bases"]]
M, UX, UY, RE = [], [], [], []
for c, b in cases:
    raw, ux, uy = unseen_velocity(c, b)
    m = block_top_bottom(pore_mask(raw)); ux = ux.copy(); uy = uy.copy()
    ux[[0, -1], :] = 0; uy[[0, -1], :] = 0
    M.append(m); UX.append(ux); UY.append(uy)
    RE.append(float(np.sqrt(ux * ux + uy * uy)[m == 1].mean()) * 148.0 / 0.25)
M = np.stack(M); UX = np.stack(UX); UY = np.stack(UY); RE = np.array(RE)
b1 = np.stack([M.astype(np.float32),
               (np.stack([flow_neck_radius_map(m) for m in M]) - float(G["uprm_mu"])) / float(G["uprm_sd"]),
               (np.stack([mis_radius_map(m) for m in M]) - float(G["mis_mu"])) / float(G["mis_sd"])], 1).astype(np.float32)
dw2 = np.stack([wall_distance(m) ** 2 for m in M])
dw2n = np.where(M == 1, (dw2 - float(G["dw2_min"])) / (float(G["dw2_max"]) - float(G["dw2_min"])), 0).astype(np.float32)
pgrad = pressure_unet(M, dev=dev)
N, L = len(cases), common.NY * common.NX
tr = np.stack([np.broadcast_to(Y.ravel(), (N, L)), np.broadcast_to(X.ravel(), (N, L)),
               pgrad[:, 0].reshape(N, L), pgrad[:, 1].reshape(N, L), dw2n.reshape(N, L)], -1).astype(np.float32)
b2 = ((RE - re_min) / (re_max - re_min)).astype(np.float32)[:, None]

for seed in SEEDS:
    model = load_weights(VelocityVariant(), os.path.join(WDIR, f"velocity_variant_seed{seed}.pt"), dev)
    outs = []
    with torch.no_grad():
        for s in range(0, N, 50):
            outs.append(model(torch.from_numpy(b1[s:s + 50]).to(dev), torch.from_numpy(b2[s:s + 50]).to(dev),
                              torch.from_numpy(tr[s:s + 50]).to(dev)).cpu().numpy())
    p = np.concatenate(outs)
    uxp = p[..., 0] * S["SD_UX"] + S["MU_UX"]; uyp = p[..., 1] * S["SD_UY"] + S["MU_UY"]
    uxp[M != 1] = 0; uyp[M != 1] = 0
    nm = np.array([nrmse(np.hypot(UX[i], UY[i]), np.hypot(uxp[i], uyp[i]), M[i] == 1) for i in range(N)])
    nux = np.array([nrmse(UX[i], uxp[i], M[i] == 1) for i in range(N)])
    nuy = np.array([nrmse(UY[i], uyp[i], M[i] == 1) for i in range(N)])
    np.savez_compressed(os.path.join(OUT, f"unseen_re_velocity_variant_seed{seed}.npz"),
                        cond=np.array([c for c, _ in cases]), base=np.array([b for _, b in cases]), re=RE,
                        nrmse_mag=nm, nrmse_ux=nux, nrmse_uy=nuy, ux_pred=uxp.astype(np.float32),
                        uy_pred=uyp.astype(np.float32), ux_gt=UX, uy_gt=UY, mask=M.astype(np.int8))
    print(f"seed {seed}: unseen Re |u| NRMSE mean {nm.mean():.4f}  P90 {np.percentile(nm, 90):.4f}  worst {nm.max():.4f}", flush=True)
