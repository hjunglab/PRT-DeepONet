#!/usr/bin/env python3
"""Table 3. Effect of the divergence-penalty weight lambda on the velocity variant, on the validation samples (900):
NRMSE of u_x and u_y and the normalized divergence at the 50th / 90th percentile and the worst sample,
5-seed mean +/- std (data/val_metrics.csv)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("velocity")
M = read_csv(f"{DATA}/val_metrics.csv"); SEEDS = [42, 2, 3, 4, 5]
var = M["model"] == "variant"
STAT = {"50th": lambda a: np.percentile(a, 50), "90th": lambda a: np.percentile(a, 90), "worst": np.max}
KEYS = [("nrmse_ux", "u_x", "{:.4f}±{:.4f}"), ("nrmse_uy", "u_y", "{:.4f}±{:.4f}"),
        ("div", "divergence", "{:.2e}±{:.1e}")]
head = [["lambda"] + sum([[lab, "", ""] for _, lab, _ in KEYS], []), [""] + list(STAT) * len(KEYS)]
rows = []
for lam in sorted(set(M["lambda"][var])):
    row = [f"{lam:g}"]
    for key, _, fmt in KEYS:
        for f in STAT.values():
            row.append(fmt.format(*mean_std([f(M[key][var & (M["lambda"] == lam) & (M["seed"] == sd)]) for sd in SEEDS])))
    rows.append(row)
table_png(head + rows, "Velocity variant vs divergence-penalty weight (validation set, 5-seed mean±std)", OUT,
          "table3_lambda_sweep", header_rows=2, figw=18, fontsize=8)
for r in rows:
    print(" | ".join(r))
