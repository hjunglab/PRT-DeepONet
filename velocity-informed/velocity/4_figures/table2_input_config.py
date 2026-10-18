#!/usr/bin/env python3
"""Table 2. Velocity input configuration C0-C7 on the validation samples (900): NRMSE of |u| at the 25th / 50th /
75th / 90th percentile and the worst sample, 10-seed mean +/- std. C0 = vanilla DeepONet
(data/baseline_val_metrics.csv); C1-C7 = configurations trained without the divergence penalty (data/input_configuration.csv;
C5 is the adopted input set)."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("velocity")
A = read_csv(f"{DATA}/input_configuration.csv")
V = read_csv(f"{DATA}/baseline_val_metrics.csv"); O = V["model"] == "vanilla"
C0 = {k: [f(V["nrmse_mag"][O & (V["seed"] == s)]) for s in sorted(set(V["seed"][O]))]
      for k, f in [("p25", lambda a: np.percentile(a, 25)), ("p50", np.median), ("p75", lambda a: np.percentile(a, 75)),
                   ("p90", lambda a: np.percentile(a, 90)), ("worst", np.max)]}
SPEC = [("C0", "wg", "x"), ("C1", "wg", "x, dP/dx, dP/dy"), ("C2", "wg", "x, UPRM^2, dP/dx, dP/dy"),
        ("C3", "wg, UPRM", "x, UPRM^2, dP/dx, dP/dy"), ("C4", "wg, UPRM", "x, d_w^2, dP/dx, dP/dy"),
        ("C5", "wg, UPRM, MIS", "x, d_w^2, dP/dx, dP/dy"), ("C6", "wg, UPRM, dw", "x, d_w^2, dP/dx, dP/dy"),
        ("C7", "wg, UPRM, MIS, dw", "x, d_w^2, dP/dx, dP/dy")]
K = [("25th", "p25"), ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("worst", "worst")]
rows = [["ID", "BC(w)", "T(p)"] + [h for h, _ in K]]
for cid, bc, tp in SPEC:
    s = A["config"] == cid
    rows.append([cid, bc, tp] + ["{:.4f}±{:.4f}".format(*mean_std(C0[k] if cid == "C0" else A[k][s])) for _, k in K])
table_png(rows, f"Velocity input configurations (validation |u| NRMSE, {len(C0['p50'])}-seed mean±std)",
          OUT, "table2_input_config", figw=17)
for r in rows:
    print(" | ".join(r))
