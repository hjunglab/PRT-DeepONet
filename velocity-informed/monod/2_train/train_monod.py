#!/usr/bin/env python3
"""Train the concentration variant for Monod kinetics (transient concentration C(x, y, t)).

    branch CNN [wg, ux, uy] (predicted velocity), branch MLP [Pe, Da_M], trunk [y, x, t, GDF]

    Huber loss, AdamW (lr 1e-3), batch 25, mixed precision, at most 500 epochs; the weights
    with the lowest validation Huber loss are kept (patience 15). The test set is
    not used here.

Inputs : $PRT_WORK/monod/{geometry_features,velocity_input}.npz, targets_{train,val}.npy (1_preprocess/)
Output : $PRT_WORK/monod/weights/monod_variant_seed<S>.pt (+ .json)
"""
import os, sys, copy, time, json
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import common
from monod.core import ConcentrationVariant, MonodInputs, load_samples, targets

SEED = int(os.environ.get("SEED", "42"))
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS, PAT = (3, 2) if SMOKE else (1000, 15)
dev = common.device(); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
MI = MonodInputs()
Str, Sva = load_samples("train"), load_samples("val")
ntr, nva = (800, 300) if SMOKE else (len(Str), len(Sva))
T = MI.build(Str[:ntr]); V = MI.build(Sva[:nva])
ty = torch.from_numpy(np.asarray(targets("train")[:ntr]))[..., None]
vy = torch.from_numpy(np.asarray(targets("val")[:nva]))[..., None]

np.random.seed(SEED); torch.manual_seed(SEED)
if dev.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
model = ConcentrationVariant(trunk_in=4).to(dev)
tl = DataLoader(TensorDataset(T["b1"], T["b2"], T["tr"], ty), batch_size=25, shuffle=True)
vl = DataLoader(TensorDataset(V["b1"], V["b2"], V["tr"], vy), batch_size=25, shuffle=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3); crit = nn.HuberLoss(delta=1.0)
scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
best = None; bestloss = 1e9; bestep = 0; noimp = 0; t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for bb in tl:
        b1, b2, tr, y = [x.to(dev) for x in bb]; opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
            loss = crit(model(b1, b2, tr), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for bb in vl:
            b1, b2, tr, y = [x.to(dev) for x in bb]
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                tot += float(crit(model(b1, b2, tr), y).item()); n += 1
    vloss = tot / n
    if vloss < bestloss - 1e-5:
        bestloss = vloss; bestep = ep; noimp = 0; best = copy.deepcopy(model.state_dict())
    else:
        noimp += 1
        if noimp >= PAT:
            break
    if ep % 5 == 0 or ep <= 2:
        print(f"  ep{ep:4d} val Huber {vloss:.6f}  best {bestloss:.6f} @ {bestep}  {time.time() - t0:.0f}s", flush=True)

outdir = os.path.join(common.work_dir("monod"), "weights"); os.makedirs(outdir, exist_ok=True)
name = f"monod_variant_seed{SEED}"
torch.save(best, os.path.join(outdir, name + ".pt"))
json.dump(dict(model="variant", seed=SEED, best_epoch=bestep, best_val_huber=bestloss, minutes=(time.time() - t0) / 60),
          open(os.path.join(outdir, name + ".json"), "w"), indent=1)
print(f"[save] {name}.pt (best epoch {bestep})")
