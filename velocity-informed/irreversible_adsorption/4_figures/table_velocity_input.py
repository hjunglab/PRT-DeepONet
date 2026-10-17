#!/usr/bin/env python3
"""Velocity input of the concentration variant (irreversible adsorption): validation pore RMSE at the 25th / 50th / 75th / 90th
percentile and the worst sample, 5-seed mean +/- std (data/ablation_velocity_input.csv).
matched     : velocity at the highest Re of the geometry for every Pe (adopted)
mismatched  : velocity at the Re group paired with each Pe (Pe 1, 5, 10 -> low, middle, high Re)
LB          : LB velocity instead of the predicted velocity;  magnitude : |u| (1 channel) instead of (u_x, u_y)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("irreversible_adsorption")
A = read_csv(f"{DATA}/ablation_velocity_input.csv")
SPEC = [("matched_predicted", "Matched predicted velocity"), ("mismatched_predicted", "Mismatched predicted velocity"),
        ("matched_lb", "Matched LB velocity"), ("predicted_magnitude", "Predicted velocity magnitude"),
        ("lb_magnitude", "LB velocity magnitude")]
K = [("25th", "p25"), ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("worst", "worst")]
rows = [["Velocity input configuration"] + [h for h, _ in K]]
for key, lab in SPEC:
    s = A["velocity_input"] == key
    rows.append([lab] + ["{:.4f}±{:.4f}".format(*mean_std(A[k][s])) for _, k in K])
table_png(rows, "Velocity input of the concentration variant (validation pore RMSE, 5-seed mean±std)", OUT, "table_velocity_input", figw=15)
for r in rows:
    print(" | ".join(r))
