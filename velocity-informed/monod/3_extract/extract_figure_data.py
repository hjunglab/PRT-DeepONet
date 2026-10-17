#!/usr/bin/env python3
"""Extract the data behind the Monod figures into monod/data/.

Reads $PRT_WORK/monod/eval/ (evaluate.py), the LB labels and the released weights; writes
    test_metrics.csv       per test sample (base, group, t), concentration variant: pore RMSE (seen time slices, 5 seeds)
    unseen_t_metrics.csv   per test (base, group) and unseen time slice, concentration variant: pore RMSE (5 seeds)
    baseline_{test,unseen_t}_metrics.csv   the same for the original PRT-DeepONet (comparison baseline, 5 seeds)
    fields_worst.npz       domain of the largest original-model error: LB and both predictions at t = 1, 20, 40, 60, 80
    fields_unseen_t.npz    (base, group) with the largest mean original-model error over the unseen time slices:
                           LB and both predictions at t = 6, 22, 38, 54, 70
Figure fields use seed 42 (the two predictions of a panel are recomputed with the released weights).
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from common import load_weights
from monod.core import ConcentrationVariant, OriginalPRTDeepONet, MonodInputs, predict, T_UNSEEN, MONOD_GROUPS, monod_series

EV = os.path.join(common.work_dir("monod"), "eval"); D = common.data_dir("monod")
WDIR = os.environ.get("WEIGHTS", common.weights_dir("monod"))
SEEDS = [42, 2, 3, 4, 5]; FIG_SEED = 42; T_WORST = [1, 20, 40, 60, 80]
Z = {(m, s): dict(np.load(os.path.join(EV, f"monod_{m}_seed{s}.npz"))) for m in ("variant", "original") for s in SEEDS}


def cond(g):
    _, Pe, Da = MONOD_GROUPS[g]
    return f"{Pe:g}", f"{Da:g}"


for model, pre in (("variant", ""), ("original", "baseline_")):
    with open(os.path.join(D, f"{pre}test_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "base", "Pe", "Da", "t", "rmse"])
        for s in SEEDS:
            z = Z[(model, s)]
            for (b, g, t), r in zip(z["samples"], z["rmse"]):
                w.writerow([model, s, b, *cond(g), t, f"{r:.6g}"])
    with open(os.path.join(D, f"{pre}unseen_t_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "base", "Pe", "Da", "t", "rmse"])
        for s in SEEDS:
            z = Z[(model, s)]
            for (b, g), rr in zip(z["u_pairs"], z["u_rmse"]):
                for t, r in zip(z["u_t"], rr):
                    w.writerow([model, s, b, *cond(g), t, f"{r:.6g}"])
print("[save] test_metrics.csv, unseen_t_metrics.csv, baseline_test_metrics.csv, baseline_unseen_t_metrics.csv")

dev = common.device(); MI = MonodInputs()
models = {"variant": load_weights(ConcentrationVariant(trunk_in=4), os.path.join(WDIR, f"monod_variant_seed{FIG_SEED}.pt"), dev),
          "original": load_weights(OriginalPRTDeepONet(trunk_in=4), common.baseline_weights("monod", f"monod_original_seed{FIG_SEED}"), dev)}


def fields(base, g, ts, name):
    S = np.array([(base, g, t) for t in ts], np.int64)
    m = MI.mask[MI.pos[base]]; pore = m == 1
    A = monod_series(base + g * common.N_BASE)
    gt = np.stack([(A[t] * m.reshape(-1)).reshape(common.NY, common.NX) for t in ts]).astype(np.float32)
    out = dict(seed=FIG_SEED, base=base, Pe=MONOD_GROUPS[g][1], Da=MONOD_GROUPS[g][2], t=np.array(ts),
               mask=m.astype(np.int8), gt=gt)
    for k, model in models.items():
        p = predict(model, MI.build(S, k == "variant"), dev)
        out[f"pred_{k}"] = p.astype(np.float32)
        out[f"rmse_{k}"] = np.array([np.sqrt(np.mean((p[j][pore] - gt[j][pore]) ** 2)) for j in range(len(ts))], np.float32)
    np.savez_compressed(os.path.join(D, name + ".npz"), **out)
    print(f"[save] {name}.npz  base {base} Pe {out['Pe']:g} Da {out['Da']:g}  original {np.round(out['rmse_original'], 3)}"
          f"  variant {np.round(out['rmse_variant'], 3)}")


zo = Z[("original", FIG_SEED)]
b, g, _ = zo["samples"][int(np.argmax(zo["rmse"]))]
fields(int(b), int(g), T_WORST, "fields_worst")
b, g = zo["u_pairs"][int(np.argmax(zo["u_rmse"].mean(1)))]
fields(int(b), int(g), T_UNSEEN, "fields_unseen_t")
