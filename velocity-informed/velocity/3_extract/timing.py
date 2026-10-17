#!/usr/bin/env python3
"""Wall-clock time of the velocity variant for one pore geometry at one flow condition on a single CPU core, covering
the full pipeline from the pore geometry (UPRM, MIS, d_w^2, pressure U-Net, velocity variant). Median of REPS runs
after one warm-up, for every Re condition of the first N_GEOM test geometries; the reported time is their mean.
The pipeline is first checked against the predicted velocity of 3_extract/predicted_velocity.py.
Timings depend on the hardware and its load.

Output: velocity/data/timing_measured.csv   (the values of Table 5 are in velocity/data/timing.csv)
"""
import os, sys, csv
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import GeometryPipeline, median_time, velocity_mask

torch.set_num_threads(1)
N_GEOM, REPS, VSEED = int(os.environ.get("N_GEOM", "3")), int(os.environ.get("REPS", "5")), int(os.environ.get("VSEED", "42"))
P = GeometryPipeline(VSEED)
bases = common.load_split()["test_bases"][:N_GEOM]
ref = np.load(os.path.join(common.work_dir("velocity"), "predicted_velocity.npz"))["uxuy"]
chk = []
for b in bases:
    for g in range(3):
        idx = g * common.N_BASE + b
        u = np.stack(P(velocity_mask(b), float(P.re_norm[idx])))
        chk.append(float(np.abs(u - ref[idx - 1]).max() / np.abs(ref[idx - 1]).max()))
print(f"check against predicted_velocity.npz: max |diff| / max |u| = {max(chk):.2e}")
rows = []
for b in bases:
    raw = velocity_mask(b)
    for g in range(3):
        idx = g * common.N_BASE + b
        rows.append([int(b), g, median_time(lambda: P(raw, float(P.re_norm[idx])), REPS)])
        print(f"base {b} Re group {g}: {rows[-1][-1]:.3f} s", flush=True)
with open(os.path.join(common.data_dir("velocity"), "timing_measured.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["base", "re_group", "prt_time_s"]); w.writerows(rows)
print(f"mean {np.mean([r[-1] for r in rows]):.3f} s")
