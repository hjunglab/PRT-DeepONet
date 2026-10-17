#!/usr/bin/env python3
"""Wall-clock time of the concentration variant for one pore geometry on a single CPU core, covering the full pipeline
from the pore geometry (velocity variant at the highest Re of the geometry, GDF, concentration variant), at the most
expensive LB condition (Pe = 10, Da = 74). Median of REPS runs after one warm-up for each of the first N_GEOM test
geometries; the reported time is their mean. The pipeline is first checked against the evaluation inputs.
Timings depend on the hardware and its load.

Output: irreversible_adsorption/data/timing_measured.csv   (the values of Table 7 are in irreversible_adsorption/data/timing.csv)
"""
import os, sys, csv
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import inlet_distance_norm, load_weights
from velocity.core import GeometryPipeline, median_time, velocity_mask
from irreversible_adsorption.core import ConcentrationVariant, AdsorptionInputs, predict, adsorption_mask, mm, DA_RANGE, PE_RANGE

torch.set_num_threads(1)
N_GEOM, REPS, VSEED = int(os.environ.get("N_GEOM", "3")), int(os.environ.get("REPS", "5")), int(os.environ.get("VSEED", "42"))
dev = torch.device("cpu")
V = np.load(os.path.join(common.work_dir("irreversible_adsorption"), "velocity_input.npz"))
S = {k: float(V[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY")}; high = int(V["re_group"])
vel = GeometryPipeline(VSEED)
wdir = os.environ.get("WEIGHTS", common.weights_dir("irreversible_adsorption"))
net = load_weights(ConcentrationVariant(), os.path.join(wdir, f"adsorption_variant_seed{VSEED}.pt"), dev)
b2 = torch.tensor([[float(mm(74.0, DA_RANGE)), float(mm(10.0, PE_RANGE))]], dtype=torch.float32)


@torch.no_grad()
def run(raw_vel, raw_conc, base):
    ux, uy = vel(raw_vel, float(vel.re_norm[high * common.N_BASE + base]))
    m = (raw_conc > 0.5).astype(np.int32); gdf = inlet_distance_norm(m)
    b1 = np.stack([m.astype(np.float32), (ux - S["MU_UX"]) / S["SD_UX"], (uy - S["MU_UY"]) / S["SD_UY"]])[None].astype(np.float32)
    tr = np.stack([vel.Y, vel.X, gdf.ravel()], -1)[None, None].astype(np.float32)
    return net(torch.from_numpy(b1), b2, torch.from_numpy(tr)).numpy().reshape(common.NY, common.NX)


bases = common.load_split()["test_bases"][:N_GEOM]
inp = AdsorptionInputs().build_geometry([bases[0]], [74.0], [10.0])
ref = predict(net, inp, dev)[0]
c = run(velocity_mask(bases[0]), adsorption_mask(bases[0]), bases[0])
print(f"check against the evaluation inputs: max |C diff| over the pore cells = {np.abs(c - ref)[inp['pore'][0]].max():.2e}")
rows = []
for b in bases:
    rv, rm = velocity_mask(b), adsorption_mask(b)
    rows.append([int(b), median_time(lambda: run(rv, rm, b), REPS)])
    print(f"base {b}: {rows[-1][-1]:.3f} s", flush=True)
with open(os.path.join(common.data_dir("irreversible_adsorption"), "timing_measured.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["base", "prt_time_s"]); w.writerows(rows)
print(f"mean {np.mean([r[-1] for r in rows]):.3f} s")
