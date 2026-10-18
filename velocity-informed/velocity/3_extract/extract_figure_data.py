#!/usr/bin/env python3
"""Extract the data behind the velocity figures and tables into velocity/data/ (small CSV / npz files).

Reads $PRT_WORK/velocity/eval/ (evaluate.py, interpolation_unseen_re.py) and the LB labels; writes
    test_metrics.csv       per test sample, velocity variant (every lambda and seed): Re, NRMSE of |u|, ux, uy, normalized divergence
    val_metrics.csv        the same on the validation samples (evaluate.py with SPLIT=val; model-selection tables)
    fields_test.npz        velocity fields of the showcase samples (seed 42; variant and vanilla DeepONet)
    permeability.csv       Darcy permeability from the LB and the predicted velocity (5 seeds)
    unseen_re_metrics.csv  per case at the unseen Re conditions: Re, NRMSE (5 seeds)
    fields_unseen_re.npz   velocity fields of the 70th/80th/90th-percentile unseen-Re cases (seed 42)
    baseline_test_metrics.csv  the test_metrics.csv columns for the vanilla DeepONet (comparison baseline, 5 seeds)
baseline_val_metrics.csv (vanilla DeepONet = input configuration C0 of Table 2, 10 seeds like C1-C7) is shipped as it is,
like input_configuration.csv, and not rewritten here.
Seed 42 is also the seed whose mean test NRMSE of |u| is closest to the 5-seed mean.
"""
import os, sys, csv, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import VelocityInputs

EV = os.path.join(common.work_dir("velocity"), "eval"); D = common.data_dir("velocity")
SEEDS = [42, 2, 3, 4, 5]
VI = VelocityInputs(pressure_unet_weights=None)
test = np.array(common.load_split()["velocity_samples"]["test"])


def model_tag(name):
    s = int(name.split("seed")[-1])
    if name.startswith("vanilla_deeponet"):
        return "vanilla", 0, s
    lam = float(name.split("_lam")[1].split("_")[0]) if "_lam" in name else 10.0
    return "variant", lam, s


for split, pattern, out in (("test", "velocity_*.npz", "test_metrics"), ("val", "velocity_*.npz", "val_metrics"),
                            ("test", "vanilla_deeponet_*.npz", "baseline_test_metrics")):
    ref = np.array(common.load_split()["velocity_samples"][split]); suf = "" if split == "test" else "_val"
    rows = []
    for f in sorted(glob.glob(os.path.join(EV, pattern))):
        name = os.path.basename(f)[:-4]
        if name.endswith("_val") != (split == "val"):
            continue
        m, lam, s = model_tag(name[:len(name) - len(suf)]); z = np.load(f)
        assert np.array_equal(z["idx"], ref)
        zz = {k: z[k] for k in ("re", "nrmse_mag", "nrmse_ux", "nrmse_uy", "div")}
        for i, idx in enumerate(ref):
            rows.append([m, f"{lam:g}", s, idx, common.base_of(idx), common.group_of(idx), f"{zz['re'][i]:.6g}",
                         f"{zz['nrmse_mag'][i]:.6g}", f"{zz['nrmse_ux'][i]:.6g}", f"{zz['nrmse_uy'][i]:.6g}", f"{zz['div'][i]:.6g}"])
    if not rows:
        continue
    with open(os.path.join(D, f"{out}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "lambda", "seed", "idx", "base", "re_group", "re", "nrmse_mag", "nrmse_ux", "nrmse_uy", "div"])
        w.writerows(rows)
    print(f"[save] {out}.csv", len(rows), "rows")

rep = common.FIG_SEED

Zv = dict(np.load(os.path.join(EV, f"velocity_variant_seed{rep}.npz"))); Zo = dict(np.load(os.path.join(EV, f"vanilla_deeponet_seed{rep}.npz")))
ov = np.argsort(Zv["nrmse_mag"]); oo = np.argsort(Zo["nrmse_mag"]); n = len(ov)
pick = {"variant_best": ov[0], "variant_p50": ov[int(n * 0.5)], "variant_p80": ov[int(n * 0.8)], "variant_worst": ov[-1],
        "vanilla_p80": oo[int(n * 0.8)]}
sel = sorted(set(int(i) for i in pick.values()))
UXG, UYG, _ = VI.targets(list(test[sel]))
np.savez_compressed(os.path.join(D, "fields_test.npz"), seed=rep, idx=test[sel], re=VI.re[test[sel]],
                    pick_names=np.array(list(pick)), pick_rows=np.array([sel.index(int(pick[k])) for k in pick]),
                    mask=np.stack([VI.mask[VI.k(i)] for i in test[sel]]).astype(np.int8), ux_gt=UXG, uy_gt=UYG,
                    ux_variant=Zv["ux_pred"][sel], uy_variant=Zv["uy_pred"][sel],
                    ux_vanilla=Zo["ux_pred"][sel], uy_vanilla=Zo["uy_pred"][sel])
print("[save] fields_test.npz", dict((k, int(test[v])) for k, v in pick.items()))

NU, L, DP = 0.25, 148.0, {0: 0.0025, 1: 0.005, 2: 0.01}
P = np.stack([VI.mask[VI.k(i)] == 1 for i in test]); ncell = common.NY * common.NX
UXa, _, _ = VI.targets(list(test))
dp = np.array([DP[common.group_of(i)] for i in test])
k_lb = NU * (UXa * P).sum((1, 2)) / ncell * L / dp
with open(os.path.join(D, "permeability.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["seed", "idx", "re", "k_lb", "k_pred"])
    for s in SEEDS:
        uxp = np.load(os.path.join(EV, f"velocity_variant_seed{s}.npz"))["ux_pred"]
        k_pred = NU * (uxp * P).sum((1, 2)) / ncell * L / dp
        for i, idx in enumerate(test):
            w.writerow([s, idx, f"{VI.re[idx]:.6g}", f"{k_lb[i]:.6g}", f"{k_pred[i]:.6g}"])
print("[save] permeability.csv")

with open(os.path.join(D, "unseen_re_metrics.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["seed", "cond", "base", "re", "nrmse_mag", "nrmse_ux", "nrmse_uy"])
    for s in SEEDS:
        z = dict(np.load(os.path.join(EV, f"unseen_re_velocity_variant_seed{s}.npz")))
        for i in range(len(z["base"])):
            w.writerow([s, z["cond"][i], z["base"][i], f"{z['re'][i]:.6g}", f"{z['nrmse_mag'][i]:.6g}",
                        f"{z['nrmse_ux'][i]:.6g}", f"{z['nrmse_uy'][i]:.6g}"])
z = dict(np.load(os.path.join(EV, f"unseen_re_velocity_variant_seed{rep}.npz")))
o = np.argsort(z["nrmse_mag"], kind="stable"); n = len(o)
sel = [int(o[min(n - 1, int(n * p))]) for p in (0.70, 0.80, 0.90)]
np.savez_compressed(os.path.join(D, "fields_unseen_re.npz"), seed=rep, percentile=np.array([70, 80, 90]),
                    cond=z["cond"][sel], base=z["base"][sel], re=z["re"][sel], nrmse_mag=z["nrmse_mag"][sel],
                    nrmse_ux=z["nrmse_ux"][sel], nrmse_uy=z["nrmse_uy"][sel], mask=z["mask"][sel],
                    ux_gt=z["ux_gt"][sel], uy_gt=z["uy_gt"][sel], ux_pred=z["ux_pred"][sel], uy_pred=z["uy_pred"][sel])
print("[save] unseen_re_metrics.csv, fields_unseen_re.npz")
