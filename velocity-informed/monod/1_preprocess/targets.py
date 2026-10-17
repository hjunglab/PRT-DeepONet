#!/usr/bin/env python3
"""Monod training targets: LB concentration A[t] * mask for every sample of splits/monod_samples.npz
(order = training order), plus the unseen time slices t in {6, 22, 38, 54, 70} of every test (base, group).

Output: $PRT_WORK/monod/targets_{train,val,test}.npy  (N, 64, 148) float32
        $PRT_WORK/monod/targets_test_unseen_t.npz       pairs (1200, 2) base, group;  A (1200, 5, 64, 148)
"""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from monod.core import load_samples, T_UNSEEN, monod_series

W = common.work_dir("monod")
MASK = np.load(os.path.join(W, "geometry_features.npz"))["mask"]


def frames(args):
    base, g, ts = args
    A = monod_series(base + g * common.N_BASE)
    m = MASK[base - 1].reshape(-1).astype(np.float32)
    return np.stack([(A[t] * m).reshape(common.NY, common.NX) for t in ts]).astype(np.float32)


if __name__ == "__main__":
    for split in ("val", "test", "train"):
        S = load_samples(split)
        need = {}
        for b, g, t in S:
            need.setdefault((int(b), int(g)), set()).add(int(t))
        pairs = sorted(need); need = {p: sorted(v) for p, v in need.items()}
        with Pool(int(os.environ.get("NPROC", "16"))) as pool:
            res = pool.map(frames, [(b, g, need[(b, g)]) for b, g in pairs], chunksize=20)
        F = {p: dict(zip(need[p], r)) for p, r in zip(pairs, res)}
        out = np.lib.format.open_memmap(os.path.join(W, f"targets_{split}.npy"), mode="w+", dtype=np.float32,
                                        shape=(len(S), common.NY, common.NX))
        for i, (b, g, t) in enumerate(S):
            out[i] = F[(int(b), int(g))][int(t)]
        out.flush(); del out, F
        print(f"[save] targets_{split}.npy  N={len(S)}", flush=True)
        if split == "test":
            with Pool(int(os.environ.get("NPROC", "16"))) as pool:
                U = pool.map(frames, [(b, g, T_UNSEEN) for b, g in pairs], chunksize=20)
            np.savez(os.path.join(W, "targets_test_unseen_t.npz"), pairs=np.array(pairs), t=np.array(T_UNSEEN), A=np.stack(U))
            print("[save] targets_test_unseen_t.npz", flush=True)
