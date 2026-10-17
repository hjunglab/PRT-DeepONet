#!/usr/bin/env python3
"""Geometry features of the velocity model for the 3000 base geometries, and their normalization.

    wg    pore mask, top/bottom rows blocked                         (branch CNN channel 0)
    UPRM  flow-neck radius map  (core.flow_neck_radius_map)  (branch CNN channel 1, z-score)
    MIS   maximal-inscribed-sphere radius map (core.mis_radius_map) (branch CNN channel 2, z-score)
    d_w^2 squared wall distance, min-max over the pore cells of the training geometries (trunk input)

Normalization: UPRM mean/std over the pore cells of the training samples; MIS mean/std over the pore cells
of the training geometries (train_bases); d_w^2 min/max over the pore cells of the training geometries.
Output: $PRT_WORK/velocity/geometry_features.npz
"""
import os, sys, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import pore_mask, block_top_bottom, flow_neck_radius_map, mis_radius_map, wall_distance, velocity_mask


def features(b):
    m = block_top_bottom(pore_mask(velocity_mask(b)))
    d = wall_distance(m)
    return b, m.astype(np.uint8), flow_neck_radius_map(m, 4), mis_radius_map(m), (d * d).astype(np.float32)


if __name__ == "__main__":
    bases = list(range(1, common.N_BASE + 1))
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        res = pool.map(features, bases, chunksize=20)
    M = {b: m for b, m, _, _, _ in res}; UPRM = {b: g for b, _, g, _, _ in res}
    MIS = {b: r for b, _, _, r, _ in res}; DW2 = {b: d for b, _, _, _, d in res}

    split = common.load_split()
    tr_samples = [common.base_of(i) for i in split["velocity_samples"]["train"]]
    g = np.concatenate([UPRM[b][M[b] == 1] for b in tr_samples])
    r = np.concatenate([MIS[b][MIS[b] > 0] for b in split["train_bases"]])
    e = np.concatenate([DW2[b][M[b] == 1] for b in split["train_bases"]])
    stats = dict(uprm_mu=float(g.mean()), uprm_sd=float(g.std() + 1e-8),
                 mis_mu=float(r.mean()), mis_sd=float(r.std()),
                 dw2_min=float(e.min()), dw2_max=float(e.max()))

    out = os.path.join(common.work_dir("velocity"), "geometry_features.npz")
    np.savez_compressed(out, bases=np.array(bases),
                        mask=np.stack([M[b] for b in bases]), uprm=np.stack([UPRM[b] for b in bases]),
                        mis=np.stack([MIS[b] for b in bases]), dw2=np.stack([DW2[b] for b in bases]), **stats)
    print(json.dumps(stats, indent=1))
    print("[save]", out)
