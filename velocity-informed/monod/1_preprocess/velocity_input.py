#!/usr/bin/env python3
"""Velocity input of the Monod variant: for every base geometry, the velocity predicted by the adopted velocity
variant at the highest of its three Re conditions (velocity/3_extract/predicted_velocity.py), used for both Pe
values. Standardization: mean / std of ux and of uy over the pore cells of the training (base, group) pairs.

Output: $PRT_WORK/monod/velocity_input.npz   (uxuy (3000, 2, 64, 148), MU_UX, SD_UX, MU_UY, SD_UY)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from monod.core import load_samples

P = np.load(os.path.join(common.work_dir("velocity"), "predicted_velocity.npz"))
high = int(np.argmax(P["group_mean_re"]))
bases = np.arange(1, common.N_BASE + 1)
uxuy = P["uxuy"][high * common.N_BASE + bases - 1]
M = np.load(os.path.join(common.work_dir("monod"), "geometry_features.npz"))["mask"]
pairs, seen = [], set()
for b, g, _ in load_samples("train"):
    if (b, g) not in seen:
        seen.add((b, g)); pairs.append(int(b))
tux = np.concatenate([uxuy[b - 1, 0][M[b - 1] == 1] for b in pairs])
tuy = np.concatenate([uxuy[b - 1, 1][M[b - 1] == 1] for b in pairs])
stats = dict(MU_UX=float(tux.mean()), SD_UX=float(tux.std()), MU_UY=float(tuy.mean()), SD_UY=float(tuy.std()))
out = os.path.join(common.work_dir("monod"), "velocity_input.npz")
np.savez(out, uxuy=uxuy, re_group=high, **stats)
print(stats); print("[save]", out)
