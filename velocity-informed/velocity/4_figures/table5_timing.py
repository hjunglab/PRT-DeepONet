#!/usr/bin/env python3
"""Table 5. Wall-clock time (s) to obtain the steady velocity field of one 148 x 64 pore geometry, averaged over the Re
conditions, on a single CPU core (PRT time: full pipeline from the pore geometry, 3_extract/timing.py; LB time: LB
simulation to steady state), and the GPU training time of the seed with the median best epoch (data/timing.csv)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import paths, read_csv, table_png

DATA, OUT = paths("velocity")
T = read_csv(f"{DATA}/timing.csv")
rows = [["LB time (s)", "PRT time (s)", "LB/PRT", "PRT training time (s)"]] + \
       [[f"{a:.2f}", f"{b:.2f}", f"{a / b:.2f}", f"{c:.2f}"] for a, b, c in zip(T["lb_time_s"], T["prt_time_s"], T["train_time_s"])]
table_png(rows, "Velocity: wall-clock time per pore geometry (1 CPU core)", OUT, "table5_timing", figw=9)
for r in rows:
    print(" | ".join(r))
