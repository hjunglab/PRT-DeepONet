#!/usr/bin/env python3
"""Monod kinetics: LB concentration and error maps of both models at t = 1, 20, 40, 60, 80 for the test sample
with the largest original-model error (Pe = 10, Da_M = 7.4; seed 42)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import plt, paths, save
from _fieldrows import draw

DATA, OUT = paths("monod")
F = np.load(f"{DATA}/fields_worst.npz"); nt = len(F["t"])
fig = plt.figure(figsize=(3.4 * nt, 7.6))
gs = fig.add_gridspec(3, nt + 1, width_ratios=[1] * nt + [0.04], hspace=0.55, wspace=0.08)
draw(fig, gs, 0, F, 0.4)
save(fig, OUT, "fig_worst_domain_time")
print(f"base {int(F['base'])} (Pe {float(F['Pe']):g}, Da {float(F['Da']):g}) original {np.round(F['rmse_original'], 3)} variant {np.round(F['rmse_variant'], 3)}")
