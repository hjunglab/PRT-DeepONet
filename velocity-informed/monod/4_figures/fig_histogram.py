#!/usr/bin/env python3
"""Monod kinetics: per-sample pore-RMSE distributions of the original model and the concentration variant
(25200 test samples = 300 geometries x 4 (Pe, Da_M) x 21 time slices, seed 42)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import plt, paths, save, read_metrics

DATA, OUT = paths("monod"); SEED = 42
M = read_metrics(DATA, "test_metrics")
ro = M["rmse"][(M["model"] == "original") & (M["seed"] == SEED)]
rv = M["rmse"][(M["model"] == "variant") & (M["seed"] == SEED)]
fig = plt.figure(figsize=(9, 4.8))
plt.hist(ro, bins=60, color="#6f8fc7", alpha=0.55, density=True, label="Original PRT-DeepONet")
plt.hist(rv, bins=60, color="#cf6a6a", alpha=0.55, density=True, label="Concentration variant")
plt.axvline(ro.mean(), color="#2f6db5", ls="--", lw=2, label=f"mean = {ro.mean():.4f}")
plt.axvline(rv.mean(), color="#d62728", ls="--", lw=2, label=f"mean = {rv.mean():.4f}")
plt.xlabel("RMSE"); plt.ylabel("density")
plt.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
save(fig, OUT, "fig_histogram")
print(f"seed {SEED}: mean RMSE original {ro.mean():.4f}  variant {rv.mean():.4f}  ({100 * (ro.mean() - rv.mean()) / ro.mean():.1f}% lower)")
