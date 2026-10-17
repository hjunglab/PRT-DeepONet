"""Monod pipeline: everything specific to the transient concentration with Monod kinetics.

A sample is (base, group, t): group = (Pe, Da_M) corner (MONOD_GROUPS), t = time index of the LB series
(t in {1, 4, 8, ..., 80} for training; Fo = t / 1000).
    branch CNN  [wg] (original) or [wg, ux, uy] (variant, predicted velocity at the highest Re, standardized)
    branch MLP  [(Pe - 1) / 9, (Da_M - 7.4) / 66.6]
    trunk       [y, x, t / 80, GDF]
Networks: concentration variant and original PRT-DeepONet (comparison baseline); parameter names match the released
weight files.
"""
import os
import numpy as np
import torch
import torch.nn as nn

import common
from common import NY, NX, read_npz, coords


MONOD_GROUPS = (("Pe1_Da7.4", 1, 7.4), ("Pe1_Da74", 1, 74.0), ("Pe10_Da7.4", 10, 7.4), ("Pe10_Da74", 10, 74.0))


def monod_mask(base, root=None):
    return read_npz(os.path.join(root or common.LB, "monod", "mask", f"m{base}.npz"), "m").reshape(NY, NX)


def monod_series(idx, root=None):
    return read_npz(os.path.join(root or common.LB, "monod", "temporal", f"A{idx}.npz"), "A")


class CustomCNN(nn.Module):

    def __init__(s, in_channels=1, out_dim=128, num_blocks=6):
        super().__init__(); ch = [in_channels, 16, 32, 64, 128, 256, 512][:num_blocks + 1]; L = []
        for i in range(num_blocks):
            L += [nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1), nn.SiLU(), nn.AvgPool2d(2)]
        s.features = nn.Sequential(*L); h, w = NY, NX
        for _ in range(num_blocks):
            h //= 2; w //= 2
        s.fc = nn.Linear(ch[num_blocks] * h * w, out_dim)

    def forward(s, x):
        x = s.features(x)
        return s.fc(x.view(x.size(0), -1))


class CondMLP(nn.Module):

    def __init__(s, in_dim=1, out_dim=128, hidden=128, layers=3):
        super().__init__(); m = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            m += [nn.Linear(hidden, hidden), nn.SiLU()]
        m += [nn.Linear(hidden, out_dim)]; s.net = nn.Sequential(*m)

    def forward(s, x):
        return s.net(x)


def make_trunk(in_dim, out_dim=128, layers=8, width=128):
    m = [nn.Linear(in_dim, width), nn.SiLU()]
    for _ in range(layers - 2):
        m += [nn.Linear(width, width), nn.SiLU()]
    m += [nn.Linear(width, out_dim)]
    return nn.Sequential(*m)


class ConcentrationVariant(nn.Module):

    def __init__(s, b1_ch=3, trunk_in=3, out_dim=128, b2_in=2, cnn_blocks=5):
        super().__init__(); s.branch1_net = CustomCNN(b1_ch, out_dim, num_blocks=cnn_blocks)
        s.branch2_net = CondMLP(b2_in, out_dim, 128, 3); s.trunk_net = make_trunk(trunk_in, out_dim, 8, 128)
        s.bias = nn.Parameter(torch.zeros(1))

    def forward(s, b1, b2, tr):
        N = b1.shape[0]; D = tr.shape[-1]
        to = s.trunk_net(tr.reshape(-1, D)).view(N, -1, s.trunk_net[-1].out_features).unsqueeze(1)
        b1o = s.branch1_net(b1).unsqueeze(1).unsqueeze(2); b2o = s.branch2_net(b2).unsqueeze(1).unsqueeze(2)
        return ((b1o * b2o * to).sum(-1) + s.bias).view(N, NY, NX, 1)


def OriginalPRTDeepONet(trunk_in=3):
    return ConcentrationVariant(b1_ch=1, trunk_in=trunk_in)


T_TRAIN = [1] + list(range(4, 81, 4))


T_UNSEEN = [6, 22, 38, 54, 70]


def load_samples(split):
    return np.load(os.path.join(common.SPLITS, "monod_samples.npz"))[split].astype(np.int64)


class MonodInputs:
    def __init__(self, with_velocity=True):
        w = common.work_dir("monod")
        G = np.load(os.path.join(w, "geometry_features.npz"))
        self.pos = {int(b): k for k, b in enumerate(G["bases"])}
        self.mask = G["mask"].astype(np.int32); self.gdf = G["gdf"]
        self.ux_z = self.uy_z = None
        if with_velocity:
            V = np.load(os.path.join(w, "velocity_input.npz"))
            self.stats = {k: float(V[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY")}
            self.ux_z = ((V["uxuy"][:, 0] - self.stats["MU_UX"]) / self.stats["SD_UX"]).astype(np.float32)
            self.uy_z = ((V["uxuy"][:, 1] - self.stats["MU_UY"]) / self.stats["SD_UY"]).astype(np.float32)
        Y, X = coords(); self.Y = Y.ravel().astype(np.float32); self.X = X.ravel().astype(np.float32)

    def build(self, S, variant=True):
        N = len(S); L = common.NY * common.NX
        ks = np.array([self.pos[int(b)] for b in S[:, 0]])
        b1 = np.empty((N, 3 if variant else 1, common.NY, common.NX), np.float32)
        b1[:, 0] = self.mask[ks]
        if variant:
            b1[:, 1] = self.ux_z[ks]; b1[:, 2] = self.uy_z[ks]
        Pe = np.array([MONOD_GROUPS[g][1] for g in S[:, 1]], np.float32)
        Da = np.array([MONOD_GROUPS[g][2] for g in S[:, 1]], np.float32)
        b2 = np.stack([(Pe - 1) / 9.0, (Da - 7.4) / 66.6], 1).astype(np.float32)
        tr = np.empty((N, L, 4), np.float32)
        tr[:, :, 0] = self.Y; tr[:, :, 1] = self.X
        tr[:, :, 2] = (S[:, 2] / 80.0).astype(np.float32)[:, None]
        tr[:, :, 3] = self.gdf[ks].reshape(N, L)
        return dict(b1=torch.from_numpy(b1), b2=torch.from_numpy(b2), tr=torch.from_numpy(tr), pore=self.mask[ks] == 1)


def targets(split):
    return np.load(os.path.join(common.work_dir("monod"), f"targets_{split}.npy"), mmap_mode="r")


@torch.no_grad()
def predict(model, inp, dev, bs=200, lo=0, hi=None):
    model.eval(); hi = inp["b1"].shape[0] if hi is None else hi; out = []
    for s in range(lo, hi, bs):
        e = min(s + bs, hi)
        out.append(model(inp["b1"][s:e].to(dev), inp["b2"][s:e].to(dev), inp["tr"][s:e].to(dev)).float().cpu().numpy()[..., 0])
    return np.concatenate(out)
