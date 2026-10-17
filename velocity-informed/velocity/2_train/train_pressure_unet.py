#!/usr/bin/env python3
"""Train the pressure U-Net: pore mask (1 ch) -> normalized harmonic pressure gradient (py/PY_MAX, px/PX_MAX)
(2 ch; px = dP/dx streamwise, py = dP/dy cross-stream), solid cells = 0 in the labels.

    Labels      : $PRT_WORK/velocity/pressure_labels.npz (1_preprocess/pressure_labels.py); the released
                  velocity/weights/pressure_unet.pt is the pressure U-Net used by every velocity model
    Geometries  : the 2700 non-test geometries (training + validation), split at random into a part for fitting
                  and a part for early stopping. The 300 test geometries are only scored at the end.
    Optimization: full-image MSE, AdamW lr 1e-3, batch 16, mixed precision, at most 300 epochs, patience 20.

The pressure U-Net is a geometry-only map, trained once; it is frozen inside the velocity model.
Output: $PRT_WORK/velocity/weights/pressure_unet.pt (+ .json with the test NRMSE of px, py and |grad P|)
"""
import os, sys, copy, time, json
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import PressureUNet, nrmse

SEED = 42; BATCH, LR, EPOCHS, PATIENCE, HOLDOUT = 16, 1e-3, 300, 20, 0.1
if os.environ.get("SMOKE") == "1":
    EPOCHS = 2
dev = common.device(); torch.backends.cudnn.benchmark = True
np.random.seed(SEED); torch.manual_seed(SEED)
if dev.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

Z = np.load(os.path.join(common.work_dir("velocity"), "pressure_labels.npz"))
pos = {int(b): k for k, b in enumerate(Z["bases"])}
PX_MAX, PY_MAX = float(Z["PX_MAX"]), float(Z["PY_MAX"])
M = np.load(os.path.join(common.work_dir("velocity"), "geometry_features.npz"))["mask"]
split = common.load_split()


def build(bases):
    k = [pos[b] for b in bases]
    m = (M[k] == 1)
    y = np.stack([np.where(m, Z["py"][k] / PY_MAX, 0), np.where(m, Z["px"][k] / PX_MAX, 0)], 1).astype(np.float32)
    return torch.from_numpy(m[:, None].astype(np.float32)), torch.from_numpy(y)


fit = sorted(split["train_bases"] + split["val_bases"])
perm = np.random.RandomState(SEED).permutation(len(fit)); n_es = int(len(fit) * HOLDOUT)
Xtr, Ytr = build([fit[i] for i in perm[n_es:]]); Xva, Yva = build([fit[i] for i in perm[:n_es]])
Xte, Yte = build(split["test_bases"])
tl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
vl = DataLoader(TensorDataset(Xva, Yva), batch_size=BATCH)
print(f"fit {len(Xtr)}  early stopping {len(Xva)}  test {len(Xte)}", flush=True)

model = PressureUNet(1, 2).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
best = copy.deepcopy(model.state_dict()); bv = 1e9; ni = 0; t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in tl:
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            loss = ((model(xb.to(dev)) - yb.to(dev)) ** 2).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); tot = 0.0; nb = 0
    with torch.no_grad():
        for xb, yb in vl:
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                tot += float(((model(xb.to(dev)) - yb.to(dev)) ** 2).mean()); nb += 1
    va = tot / nb
    if va < bv - 1e-7:
        bv = va; best = copy.deepcopy(model.state_dict()); ni = 0
    else:
        ni += 1
    if ep % 5 == 0 or ep == 1:
        print(f"  ep{ep:3d} early-stopping MSE {va:.3e}  best {bv:.3e}  {time.time() - t0:.0f}s", flush=True)
    if ni >= PATIENCE:
        break
model.load_state_dict(best)
outdir = os.path.join(common.work_dir("velocity"), "weights"); os.makedirs(outdir, exist_ok=True)
torch.save(best, os.path.join(outdir, "pressure_unet.pt"))

model.eval(); pred = []
with torch.no_grad():
    for (xb,) in DataLoader(TensorDataset(Xte), batch_size=BATCH):
        pred.append(model(xb.to(dev)).float().cpu().numpy())
pred = np.concatenate(pred)
r = {"px": [], "py": [], "mag": []}
for i in range(len(Xte)):
    mk = Xte[i, 0].numpy() > 0.5
    py_lb, px_lb = Yte[i, 0].numpy() * PY_MAX, Yte[i, 1].numpy() * PX_MAX
    py_pr, px_pr = pred[i, 0] * PY_MAX, pred[i, 1] * PX_MAX
    r["px"].append(nrmse(px_lb, px_pr, mk)); r["py"].append(nrmse(py_lb, py_pr, mk))
    r["mag"].append(nrmse(np.hypot(px_lb, py_lb), np.hypot(px_pr, py_pr), mk))
summary = {k: float(np.mean(v)) for k, v in r.items()}
json.dump(dict(test_nrmse_mean=summary, best_early_stopping_mse=bv), open(os.path.join(outdir, "pressure_unet.json"), "w"), indent=1)
print("test NRMSE (mean):", summary)
