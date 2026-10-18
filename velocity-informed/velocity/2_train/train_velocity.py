#!/usr/bin/env python3
"""Train the velocity variant of PRT-DeepONet (outputs ux, uy: streamwise and cross-stream velocity).

    branch CNN [wg, UPRM, MIS], branch MLP [Re], trunk [y, x, dP/dy, dP/dx, d_w^2],
    loss function = pore Huber + LAMBDA * divergence penalty (LAMBDA default 10 = the adopted model); the penalty is
    the mean squared divergence over the same pore cells as the Huber loss, with zero velocity in the solid
    (core.divergence_penalty)

    AdamW (lr 1e-3), batch 25, mixed precision, at most 1000 epochs; the weights with the lowest validation loss are
    kept (early stopping, patience 15). The test set is not used here.

Inputs : velocity/lb_data/flow_conditions.npz, $PRT_WORK/velocity/geometry_features.npz (1_preprocess/), the
         pressure U-Net (velocity/weights/pressure_unet.pt, or PRESSURE_UNET=<path>), LB velocity labels.
Output : $PRT_WORK/velocity/weights/velocity_variant_seed<S>.pt (lambda = 10) or
         $PRT_WORK/velocity/weights/configuration/lambda/velocity_variant_lam<L>_seed<S>.pt  (+ .json training log)
Run    : python train_velocity.py        |   LAMBDA=0 SEED=2 python train_velocity.py
         SMOKE=1 for a 2-epoch check on a subset.
"""
import os, sys, copy, time, json
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from velocity.core import VelocityVariant, ROIHuberLoss, divergence_penalty, VelocityInputs, PRESSURE_UNET_WEIGHTS

SEED = int(os.environ.get("SEED", "42"))
LAM = float(os.environ.get("LAMBDA", "10"))
SMOKE = os.environ.get("SMOKE") == "1"
BATCH, LR, EPOCHS, PATIENCE = 25, 1e-3, (2 if SMOKE else 1000), 15
dev = common.device(); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
outdir = os.path.join(common.work_dir("velocity"), "weights")
if LAM == 10:
    name = f"velocity_variant_seed{SEED}"
else:
    name = f"velocity_variant_lam{LAM:g}_seed{SEED}"; outdir = os.path.join(outdir, "configuration", "lambda")
os.makedirs(outdir, exist_ok=True)
print(f"[{name}] device={dev}", flush=True)

split = common.load_split()
tr_idx = split["velocity_samples"]["train"]; va_idx = split["velocity_samples"]["val"]
if SMOKE:
    tr_idx, va_idx = tr_idx[:60], va_idx[:30]
VI = VelocityInputs(pressure_unet_weights=os.environ.get("PRESSURE_UNET", PRESSURE_UNET_WEIGHTS), dev=dev)
tr = VI.build(tr_idx); va = VI.build(va_idx)
_, _, tr_y = VI.targets(tr_idx); _, _, va_y = VI.targets(va_idx)
RV = VI.stats["SD_UY"] / VI.stats["SD_UX"]
CX = VI.stats["MU_UX"] / VI.stats["SD_UX"]; CY = VI.stats["MU_UY"] / VI.stats["SD_UY"]

np.random.seed(SEED); torch.manual_seed(SEED)
if dev.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
model = VelocityVariant().to(dev)
crit = ROIHuberLoss(1.0); opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
TR = [x.to(dev) for x in (tr["b1"], tr["b2"], tr["tr"], tr_y, tr["roi"])]
VA = [x.to(dev) for x in (va["b1"], va["b2"], va["tr"], va_y, va["roi"])]
tl = DataLoader(TensorDataset(torch.arange(len(tr_y))), batch_size=BATCH, shuffle=True)
vl = [torch.arange(i, min(i + 50, len(va_y)), device=dev) for i in range(0, len(va_y), 50)]
best = copy.deepcopy(model.state_dict()); bv = 1e9; bep = 0; ni = 0; t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for (ix,) in tl:
        b1, b2, t, y, rm = (x[ix.to(dev)] for x in TR)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            pred = model(b1, b2, t)
            loss = crit(pred, y, rm)
            if LAM > 0:
                loss = loss + LAM * divergence_penalty(pred.float(), rm, CX, CY, RV)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for ix in vl:
            b1, b2, t, y, rm = (x[ix] for x in VA)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                tot += float(crit(model(b1, b2, t), y, rm).item()); n += 1
    vloss = tot / n
    if vloss < bv - 1e-6:
        bv = vloss; bep = ep; ni = 0; best = copy.deepcopy(model.state_dict())
    else:
        ni += 1
        if ni >= PATIENCE:
            break
    if ep % 10 == 0 or ep == 1:
        print(f"  ep{ep:4d} val Huber {vloss:.5f}  best {bv:.5f} @ {bep}  {time.time() - t0:.0f}s", flush=True)

path = os.path.join(outdir, name + ".pt")
torch.save(best, path)
json.dump(dict(model="velocity_variant", lam=LAM, seed=SEED, best_epoch=bep, best_val_huber=bv, epochs=ep, minutes=(time.time() - t0) / 60),
          open(os.path.join(outdir, name + ".json"), "w"), indent=1)
print(f"[save] {path}  (best epoch {bep}, validation loss {bv:.5f})", flush=True)
