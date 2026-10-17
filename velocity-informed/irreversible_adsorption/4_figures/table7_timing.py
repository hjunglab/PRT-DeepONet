#!/usr/bin/env python3
"""Table 7. Wall-clock time (s) to obtain the concentration field of one pore geometry on a single CPU core at the most
expensive LB condition (Pe = 10; Da = 74 for irreversible adsorption, Da_M = 7.4 for Monod kinetics; PRT time: full
pipeline from the pore geometry including the velocity prediction, 3_extract/timing.py; LB time: LB simulation from the
initial state to steady state), and the GPU training time of the seed with the median best epoch
(data/timing.csv and ../monod/data/timing.csv)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import paths, read_csv, table_png

DATA, OUT = paths("irreversible_adsorption")
rows = [["Reaction regime", "LB time (s)", "PRT time (s)", "LB/PRT", "PRT training time (s)"]]
for path in (f"{DATA}/timing.csv", os.path.join(common.ROOT, "monod", "data", "timing.csv")):
    T = read_csv(path)
    rows += [[c, f"{a:.2f}", f"{b:.2f}", f"{a / b:.0f}", f"{t:,.0f}"]
             for c, a, b, t in zip(T["case"], T["lb_time_s"], T["prt_time_s"], T["train_time_s"])]
table_png(rows, "Concentration: wall-clock time per pore geometry (1 CPU core)", OUT, "table7_timing", figw=12)
for r in rows:
    print(" | ".join(r))
