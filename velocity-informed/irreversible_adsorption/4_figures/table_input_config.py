#!/usr/bin/env python3
"""Concentration input configuration C0-C10 (irreversible adsorption): validation pore RMSE (2700 samples) at the 25th / 50th / 75th /
90th percentile and the worst sample, 5-seed mean +/- std (data/input_configuration.csv). C0 = original PRT-DeepONet;
C8 = predicted velocity (u_x, u_y; velocity variant, lambda = 10, highest Re of the geometry) in the CNN branch
(adopted); C9 = in the trunk; C10 = C8 with the predicted velocity vectors randomly permuted
over the pore cells of each geometry (u_x^sh, u_y^sh)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, mean_std, table_png

DATA, OUT = paths("irreversible_adsorption")
A = read_csv(f"{DATA}/input_configuration.csv")
G = "dP/dx, dP/dy"
SPEC = [("C0", "wg", "x, GDF"), ("C1", "wg", f"x, GDF, {G}"), ("C2", "wg", f"x, GDF, UPRM^2, {G}"),
        ("C3", "wg, UPRM", f"x, GDF, UPRM^2, {G}"), ("C4", "wg, UPRM", f"x, GDF, d_w^2, {G}"),
        ("C5", "wg, UPRM, MIS", f"x, GDF, d_w^2, {G}"), ("C6", "wg, UPRM, dw", f"x, GDF, d_w^2, {G}"),
        ("C7", "wg, UPRM, MIS, dw", f"x, GDF, d_w^2, {G}"), ("C8", "wg, u_x, u_y", "x, GDF"), ("C9", "wg", "x, GDF, u_x, u_y"),
        ("C10", "wg, u_x^sh, u_y^sh", "x, GDF")]
K = [("25th", "p25"), ("50th", "p50"), ("75th", "p75"), ("90th", "p90"), ("worst", "worst")]
rows = [["ID", "BC(w)", "T(p)"] + [h for h, _ in K]]
for cid, bc, tp in SPEC:
    s = A["config"] == cid
    rows.append([cid, bc, tp] + ["{:.4f}±{:.4f}".format(*mean_std(A[k][s])) for _, k in K])
table_png(rows, "Concentration input configurations (validation pore RMSE, 5-seed mean±std)", OUT, "table_input_config", figw=18)
for r in rows:
    print(" | ".join(r))
