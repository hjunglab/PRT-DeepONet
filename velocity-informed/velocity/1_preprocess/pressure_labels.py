#!/usr/bin/env python3
"""Training labels of the pressure U-Net: harmonic pressure-gradient components (px, py) of every base geometry
(core.pressure_gradient), and the per-component normalization PX_MAX, PY_MAX = max |px|, max |py| over the pore cells of
the training geometries (train_bases).

Output: $PRT_WORK/velocity/pressure_labels.npz   (bases, px, py: (3000, 64, 148), PX_MAX, PY_MAX)
One SOR solve per geometry (NPROC parallel processes, default 16).
"""
import os, sys, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import pore_mask, block_top_bottom, pressure_gradient, velocity_mask


def solve(b):
    m = block_top_bottom(pore_mask(velocity_mask(b)))
    return (b, m.astype(np.uint8)) + pressure_gradient(m)


if __name__ == "__main__":
    bases = list(range(1, common.N_BASE + 1))
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        res = pool.map(solve, bases, chunksize=10)
    M = {b: m for b, m, _, _ in res}; PX = {b: px for b, _, px, _ in res}; PY = {b: py for b, _, _, py in res}
    tr = common.load_split()["train_bases"]
    PX_MAX = float(max(np.abs(PX[b][M[b] == 1]).max() for b in tr))
    PY_MAX = float(max(np.abs(PY[b][M[b] == 1]).max() for b in tr))
    out = os.path.join(common.work_dir("velocity"), "pressure_labels.npz")
    np.savez_compressed(out, bases=np.array(bases), px=np.stack([PX[b] for b in bases]),
                        py=np.stack([PY[b] for b in bases]), PX_MAX=PX_MAX, PY_MAX=PY_MAX)
    print(json.dumps(dict(PX_MAX=PX_MAX, PY_MAX=PY_MAX)))
    print("[save]", out)
