#!/usr/bin/env python3
"""Table 4. Physical consistency of the predicted velocity: LB reference and the velocity variant at lambda = 0,
10 (adopted) and 100. Each predicted value = 5-seed mean +/- std of the per-seed median (P50) over the 900
validation samples; LB is seed-independent. Last row: mean |u| NRMSE (5-seed mean +/- std)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("velocity")
P = read_csv(f"{DATA}/physics_diagnostics_val.csv"); SEEDS = [42, 2, 3, 4, 5]
SERIES = [("lam0", "lambda = 0"), ("lam10", "lambda = 10"), ("lam100", "lambda = 100")]
LABEL = {"flux_cov": "Sectional flux variation (std/mean)", "flux_ratio": "Sectional flux ratio (max/min)",
         "io_err": "Inlet-outlet flux error", "wallnorm": "Near-wall normal velocity",
         "div": "Velocity divergence"}
fin = lambda a: a[np.isfinite(a)]
rows = [["Metric", "LB"] + [lab for _, lab in SERIES]]
for k, lab in LABEL.items():
    lb = f"{np.median(fin(P[k][P['series'] == 'LB'])):.3g}"
    row = [lab, lb]
    for tag, _ in SERIES:
        m, s = mean_std([np.median(fin(P[k][(P["series"] == tag) & (P["seed"] == sd)])) for sd in SEEDS])
        row.append(f"{m:.3g}±{s:.2g}")
    rows.append(row)
row = ["Velocity NRMSE", "-"]
for tag, _ in SERIES:
    m, s = mean_std([P["nrmse_mag"][(P["series"] == tag) & (P["seed"] == sd)].mean() for sd in SEEDS])
    row.append(f"{m:.3g}±{s:.2g}")
rows.append(row)
table_png(rows, "Physics-based diagnostics of the predicted velocity (validation set, P50, 5-seed mean±std)", OUT,
          "table4_physics_diagnostics", highlight_col=3)
for r in rows:
    print(" | ".join(r))
