#!/usr/bin/env python3
"""Effect of the divergence-penalty weight lambda of the velocity variant on the concentration variant (irreversible
adsorption): validation pore RMSE at the 25th / 50th / 75th / 90th percentile and the worst sample, 5-seed mean +/- std
(data/ablation_lambda.csv)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("irreversible_adsorption")
A = read_csv(f"{DATA}/ablation_lambda.csv")
K = [("25th", "p25"), ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("worst", "worst")]
rows = [["lambda"] + [h for h, _ in K]]
for lam in np.unique(A["lambda"]):
    s = A["lambda"] == lam
    rows.append([f"{lam:g}"] + ["{:.4f}±{:.4f}".format(*mean_std(A[k][s])) for _, k in K])
table_png(rows, "Concentration RMSE vs lambda of the velocity variant (validation, 5-seed mean±std)", OUT, "table_lambda", figw=12)
for r in rows:
    print(" | ".join(r))
