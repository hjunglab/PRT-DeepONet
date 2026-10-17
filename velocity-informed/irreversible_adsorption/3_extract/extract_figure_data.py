#!/usr/bin/env python3
"""Extract the data behind the adsorption figures and tables into irreversible_adsorption/data/.

Reads $PRT_WORK/irreversible_adsorption/eval/ (evaluate.py) and the LB labels; writes
    test_metrics.csv     per test sample, concentration variant: base, Da, Pe, pore RMSE (seen conditions, 5 seeds)
    unseen_metrics.csv   per case at the unseen (Pe, Da) conditions: pore RMSE (5 seeds)
    baseline_{test,unseen}_metrics.csv   the same for the original PRT-DeepONet (comparison baseline, 5 seeds)
    val_metrics.csv      per validation sample, concentration variant (evaluate.py with SPLIT=val, 5 seeds)
    fields_worst.npz     mask, LB and predicted concentration of the worst samples of the original and of the variant (seed 42)
    fields_unseen.npz    mask, LB and variant prediction at the 80th-percentile case of each unseen condition (seed 3)
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from irreversible_adsorption.core import unseen_adsorption, UNSEEN_ADSORPTION, adsorption_concentration, adsorption_mask

EV = os.path.join(common.work_dir("irreversible_adsorption"), "eval"); D = common.data_dir("irreversible_adsorption")
SEEDS = [42, 2, 3, 4, 5]; FIG_SEED = 42; UNSEEN_FIG_SEED = 3
Z = {(m, s): dict(np.load(os.path.join(EV, f"adsorption_{m}_seed{s}.npz"))) for m in ("variant", "original") for s in SEEDS}

for model, pre in (("variant", ""), ("original", "baseline_")):
    with open(os.path.join(D, f"{pre}test_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "idx", "base", "Da", "Pe", "rmse"])
        for s in SEEDS:
            z = Z[(model, s)]
            for i in range(len(z["idx"])):
                w.writerow([model, s, z["idx"][i], z["base"][i], z["Da"][i], z["Pe"][i], f"{z['rmse'][i]:.6g}"])
    with open(os.path.join(D, f"{pre}unseen_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "cond", "base", "Pe", "Da", "rmse"])
        for s in SEEDS:
            z = Z[(model, s)]
            for i in range(len(z["u_base"])):
                w.writerow([model, s, z["u_cond"][i], z["u_base"][i], f"{z['u_Pe'][i]:g}", f"{z['u_Da'][i]:g}", f"{z['u_rmse'][i]:.6g}"])
print("[save] test_metrics.csv, unseen_metrics.csv, baseline_test_metrics.csv, baseline_unseen_metrics.csv")
if all(os.path.exists(os.path.join(EV, f"adsorption_variant_seed{s}_val.npz")) for s in SEEDS):
    with open(os.path.join(D, "val_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "seed", "idx", "base", "Da", "Pe", "rmse"])
        for s in SEEDS:
            z = np.load(os.path.join(EV, f"adsorption_variant_seed{s}_val.npz"))
            for i in range(len(z["idx"])):
                w.writerow(["variant", s, z["idx"][i], z["base"][i], z["Da"][i], z["Pe"][i], f"{z['rmse'][i]:.6g}"])
    print("[save] val_metrics.csv")

zv, zo = Z[("variant", FIG_SEED)], Z[("original", FIG_SEED)]
sel = {"original_worst": int(np.argmax(zo["rmse"]))}
rows = list(sel.values()); idx = zv["idx"][rows]
gt = np.stack([adsorption_concentration(i) for i in idx])
mask = np.stack([(adsorption_mask(common.base_of(i)) > 0.5).astype(np.int8) for i in idx])
np.savez_compressed(os.path.join(D, "fields_worst.npz"), seed=FIG_SEED, names=np.array(list(sel)), idx=idx,
                    Da=zv["Da"][rows], Pe=zv["Pe"][rows], mask=mask, gt=gt,
                    pred_original=zo["pred"][rows], pred_variant=zv["pred"][rows],
                    rmse_original=zo["rmse"][rows], rmse_variant=zv["rmse"][rows])

zv = Z[("variant", UNSEEN_FIG_SEED)]
pick = []
for cond, _, _ in UNSEEN_ADSORPTION:
    k = np.where(zv["u_cond"] == cond)[0]; o = k[np.argsort(zv["u_rmse"][k], kind="stable")]
    pick.append(int(o[int(len(o) * 0.80)]))
mk, gt = zip(*[unseen_adsorption(zv["u_cond"][i], zv["u_base"][i]) for i in pick])
np.savez_compressed(os.path.join(D, "fields_unseen.npz"), seed=UNSEEN_FIG_SEED, cond=zv["u_cond"][pick], base=zv["u_base"][pick],
                    Pe=zv["u_Pe"][pick], Da=zv["u_Da"][pick], rmse=zv["u_rmse"][pick],
                    mask=(np.stack(mk) > 0.5).astype(np.int8), gt=np.stack(gt), pred=zv["u_pred"][pick])
print("[save] fields_worst.npz, fields_unseen.npz")
