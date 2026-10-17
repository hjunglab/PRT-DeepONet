#!/usr/bin/env python3
"""Velocity input of the adsorption variant: for every base geometry, the velocity predicted by the adopted
velocity variant at the highest of its three Re conditions (velocity/3_extract/predicted_velocity.py), used for
every (Da, Pe) condition. Standardization: mean / std of ux and of uy over the pore cells of the training samples.

Output: $PRT_WORK/irreversible_adsorption/velocity_input.npz   (uxuy (3000, 2, 64, 148), MU_UX, SD_UX, MU_UY, SD_UY)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common

P = np.load(os.path.join(common.work_dir("velocity"), "predicted_velocity.npz"))
high = int(np.argmax(P["group_mean_re"]))
bases = np.arange(1, common.N_BASE + 1)
uxuy = P["uxuy"][high * common.N_BASE + bases - 1]
G = np.load(os.path.join(common.work_dir("irreversible_adsorption"), "geometry_features.npz"))
tr = common.load_split()["train_bases"]
tux = np.concatenate([uxuy[b - 1, 0][G["mask"][b - 1] == 1] for b in tr for _ in (1, 5, 10)])
tuy = np.concatenate([uxuy[b - 1, 1][G["mask"][b - 1] == 1] for b in tr for _ in (1, 5, 10)])
stats = dict(MU_UX=float(tux.mean()), SD_UX=float(tux.std()), MU_UY=float(tuy.mean()), SD_UY=float(tuy.std()))
out = os.path.join(common.work_dir("irreversible_adsorption"), "velocity_input.npz")
np.savez(out, uxuy=uxuy, re_group=high, **stats)
print(stats); print("[save]", out)
