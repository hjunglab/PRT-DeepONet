#!/usr/bin/env python3
"""Score the Monod models on the test geometries.

    seen time slices   : 300 test geometries x 4 (Pe, Da_M) x 21 t in {1, 4, ..., 80}  = 25200 samples
    unseen time slices : 300 test geometries x 4 (Pe, Da_M) x 5 t in {6, 22, 38, 54, 70} = 6000 samples
Metric: pore RMSE of the concentration.

Weights: monod/weights/ (released) or WEIGHTS=<dir>: monod_variant_seed{42,2,3,4,5}; the original PRT-DeepONet
(comparison baseline) always uses the released weights weights/baseline/monod_original_seed<s>.pt (s = 42, 2, 3, 4, 5).
Output : $PRT_WORK/monod/eval/<name>.npz  (samples, rmse; u_pairs, u_t, u_rmse (1200, 5))
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import load_weights
from monod.core import ConcentrationVariant, OriginalPRTDeepONet, MonodInputs, load_samples, targets, predict, T_UNSEEN

WDIR = os.environ.get("WEIGHTS", common.weights_dir("monod"))
OUT = os.path.join(common.work_dir("monod"), "eval"); os.makedirs(OUT, exist_ok=True)
names = sys.argv[1:] or (sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{WDIR}/monod_variant_*.pt"))
                         + [f"monod_original_seed{s}" for s in common.SEEDS])
dev = common.device()
MI = MonodInputs()
S = load_samples("test"); Y = np.asarray(targets("test"))
U = np.load(os.path.join(common.work_dir("monod"), "targets_test_unseen_t.npz"))
SU = np.array([(b, g, t) for b, g in U["pairs"] for t in T_UNSEEN])
YU = U["A"].reshape(-1, common.NY, common.NX)
seen = {v: MI.build(S, v) for v in (True, False)}
unseen = {v: MI.build(SU, v) for v in (True, False)}


def rmse(pred, gt, pore):
    return np.array([np.sqrt(np.mean((pred[i][pore[i]] - gt[i][pore[i]]) ** 2)) for i in range(len(pred))])


for name in names:
    var = "variant" in name
    path = os.path.join(WDIR, name + ".pt") if var else common.baseline_weights("monod", name)
    model = load_weights((ConcentrationVariant(trunk_in=4) if var else OriginalPRTDeepONet(trunk_in=4)), path, dev)
    r = rmse(predict(model, seen[var], dev), Y, seen[var]["pore"])
    ru = rmse(predict(model, unseen[var], dev), YU, unseen[var]["pore"])
    np.savez_compressed(os.path.join(OUT, name + ".npz"), samples=S, rmse=r.astype(np.float32),
                        u_pairs=U["pairs"], u_t=np.array(T_UNSEEN), u_rmse=ru.reshape(len(U["pairs"]), len(T_UNSEEN)).astype(np.float32))
    print(f"{name:24s} seen RMSE mean {r.mean():.5f} worst {r.max():.4f} | unseen t mean {ru.mean():.5f}", flush=True)
