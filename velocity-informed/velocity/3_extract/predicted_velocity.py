#!/usr/bin/env python3
"""Predicted velocity fields of all 9000 velocity samples (3000 geometries x 3 Re), used as an input by the
concentration variants (irreversible_adsorption/, monod/).

The adopted velocity variant (lambda = 10, training seed VSEED) is run on every sample; the
prediction is returned to physical units and set to 0 outside the pore cells. The concentration models use,
for each geometry, the prediction at the highest of its three Re conditions.

Output: $PRT_WORK/velocity/predicted_velocity.npz
        uxuy (9000, 2, 64, 148) float32 (channels ux, uy), row = idx - 1;  group_mean_re (3,) mean normalized Re per Re group
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import load_weights
from velocity.core import VelocityVariant, VelocityInputs, predict

VSEED = int(os.environ.get("VSEED", "42"))
WDIR = os.environ.get("WEIGHTS", common.weights_dir("velocity"))
dev = common.device()
VI = VelocityInputs(dev=dev)
model = load_weights(VelocityVariant(), os.path.join(WDIR, f"velocity_variant_seed{VSEED}.pt"), dev)
N = 3 * common.N_BASE
uxuy = np.empty((N, 2, common.NY, common.NX), np.float32)
for s in range(0, N, 300):
    idx = list(range(s + 1, min(s + 300, N) + 1))
    inp = VI.build(idx)
    ux, uy = VI.to_physical(predict(model, inp, dev))
    pore = inp["roi"].numpy() > 0.5
    uxuy[s:s + len(idx), 0] = np.where(pore, ux, 0.0); uxuy[s:s + len(idx), 1] = np.where(pore, uy, 0.0)
gm = np.array([VI.re_norm[1 + g * common.N_BASE: 1 + (g + 1) * common.N_BASE].mean() for g in range(3)], np.float32)
out = os.path.join(common.work_dir("velocity"), "predicted_velocity.npz")
np.savez(out, uxuy=uxuy, group_mean_re=gm, vseed=VSEED)
print(f"[save] {out}  group mean normalized Re {gm.tolist()}")
