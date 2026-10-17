#!/usr/bin/env python3
"""Score the adsorption models on the test geometries and keep the predictions.

    seen conditions   : 300 test geometries x 9 (Da, Pe) training conditions = 2700 samples
    unseen conditions : 300 test geometries x 4 (Pe, Da) in {3, 7.5} x {185, 518}  = 1200 samples
Metric: pore RMSE of the concentration, sqrt(mean over pore cells of (pred - LB)^2).
SPLIT=val scores the concentration variant on the 300 validation geometries x 9 training conditions instead
(output <name>_val.npz: idx, base, Da, Pe, rmse).

Weights: irreversible_adsorption/weights/ (released) or WEIGHTS=<dir>: adsorption_variant_seed{42,2,3,4,5}; the original
PRT-DeepONet (comparison baseline) always uses the released weights weights/baseline/adsorption_original_seed<s>.pt (s = 42, 2, 3, 4, 5).
Output : $PRT_WORK/irreversible_adsorption/eval/<name>.npz   (seen: idx, base, Da, Pe, rmse, pred; unseen: u_cond, u_base, u_rmse, u_pred)
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import load_weights, inlet_distance_norm
from irreversible_adsorption.core import ConcentrationVariant, OriginalPRTDeepONet, AdsorptionInputs, samples, predict, pore_rmse, adsorption_condition, unseen_adsorption, UNSEEN_ADSORPTION

WDIR = os.environ.get("WEIGHTS", common.weights_dir("irreversible_adsorption"))
OUT = os.path.join(common.work_dir("irreversible_adsorption"), "eval"); os.makedirs(OUT, exist_ok=True)
SPLIT = os.environ.get("SPLIT", "test")
names = sys.argv[1:] or (sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{WDIR}/adsorption_variant_*.pt"))
                         + ([f"adsorption_original_seed{s}" for s in common.SEEDS] if SPLIT == "test" else []))
dev = common.device()
SI = AdsorptionInputs()
test_b = common.load_split()[SPLIT + "_bases"]
idx = samples(test_b)
gt = SI.targets(idx).numpy()[..., 0]
DaPe = np.array([adsorption_condition(i) for i in idx])
seen = {v: SI.build(idx, v) for v in (True, False)}

if SPLIT == "val":
    for name in names:
        model = load_weights(ConcentrationVariant(trunk_in=3), os.path.join(WDIR, name + ".pt"), dev)
        r = pore_rmse(predict(model, seen[True], dev), gt, seen[True]["pore"])
        np.savez_compressed(os.path.join(OUT, name + "_val.npz"), idx=np.array(idx),
                            base=np.array([common.base_of(i) for i in idx]), Da=DaPe[:, 0], Pe=DaPe[:, 1], rmse=r)
        print(f"{name:28s} validation RMSE mean {r.mean():.4f} P50 {np.median(r):.4f} worst {r.max():.4f}", flush=True)
    sys.exit(0)

ub, ucond, uDa, uPe, um, uA = [], [], [], [], [], []
for cond, Pe, Da in UNSEEN_ADSORPTION:
    for b in test_b:
        m, A = unseen_adsorption(cond, b)
        ub.append(b); ucond.append(cond); uDa.append(Da); uPe.append(Pe); um.append((m > 0.5).astype(int)); uA.append(A)
um = np.stack(um); uA = np.stack(uA)
unseen = {}
for v in (True, False):
    d = SI.build_geometry(ub, uDa, uPe, v)
    d["b1"][:, 0] = torch.from_numpy(um.astype(np.float32))
    gdf = np.stack([inlet_distance_norm(m) for m in um]).reshape(len(ub), -1)
    d["tr"][:, 0, :, 2] = torch.from_numpy(gdf); d["pore"] = um == 1
    unseen[v] = d

for name in names:
    var = "variant" in name
    path = os.path.join(WDIR, name + ".pt") if var else common.baseline_weights("irreversible_adsorption", name)
    model = load_weights((ConcentrationVariant(trunk_in=3) if var else OriginalPRTDeepONet(trunk_in=3)), path, dev)
    p = predict(model, seen[var], dev); r = pore_rmse(p, gt, seen[var]["pore"])
    pu = predict(model, unseen[var], dev); ru = pore_rmse(pu, uA, unseen[var]["pore"])
    np.savez_compressed(os.path.join(OUT, name + ".npz"), idx=np.array(idx), base=np.array([common.base_of(i) for i in idx]),
                        Da=DaPe[:, 0], Pe=DaPe[:, 1], rmse=r, pred=p.astype(np.float32),
                        u_cond=np.array(ucond), u_base=np.array(ub), u_Da=np.array(uDa), u_Pe=np.array(uPe),
                        u_rmse=ru, u_pred=pu.astype(np.float32))
    print(f"{name:28s} seen RMSE mean {r.mean():.4f} P90 {np.percentile(r, 90):.4f} worst {r.max():.4f} | "
          f"unseen mean {ru.mean():.4f}", flush=True)
