"""Irreversible-adsorption pipeline: everything specific to the steady concentration with irreversible adsorption.

    LB readers     samples idx = base + 3000 * (3 * Da_index + Pe_index) and the four unseen (Pe, Da) conditions
    networks       concentration variant and original PRT-DeepONet (comparison baseline); parameter names match the
                   released weight files
    model inputs   assembled from the outputs of irreversible_adsorption/1_preprocess/ and the LB labels
"""
import os
import numpy as np
import torch
import torch.nn as nn

import common
from common import NY, NX, read_npz, coords


ADSORPTION_DA = (74, 296, 740)


ADSORPTION_PE = (1, 5, 10)


def adsorption_condition(idx):
    idx = int(idx)
    Da = 74 if idx <= 9000 else (296 if idx <= 18000 else 740)
    r = (idx - 1) % 9000
    Pe = 1 if r < 3000 else (5 if r < 6000 else 10)
    return Da, Pe


def adsorption_mask(idx, root=None):
    return read_npz(os.path.join(root or common.LB, "irreversible_adsorption", "mask", f"m{idx}.npz"), "m").reshape(NY, NX)


def adsorption_concentration(idx, root=None):
    A = read_npz(os.path.join(root or common.LB, "irreversible_adsorption", "concentration", f"A{idx}.npz"), "A")
    return (A.reshape(NY, NX) if A.ndim == 1 else A[-1].reshape(NY, NX)).astype(np.float32)


UNSEEN_ADSORPTION = (("Pe3_Da185", 3.0, 185.0), ("Pe3_Da518", 3.0, 518.0),
                   ("Pe7.5_Da185", 7.5, 185.0), ("Pe7.5_Da518", 7.5, 518.0))


def unseen_adsorption(cond, base, root=None):
    z = np.load(os.path.join(root or common.LB, "irreversible_adsorption_unseen", cond, f"b{base}.npz"))
    return z["mask"].reshape(NY, NX), z["A"].reshape(NY, NX).astype(np.float32)


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


DA_RANGE, PE_RANGE = (74.0, 740.0), (1.0, 10.0)


def mm(v, r):
    return (v - r[0]) / (r[1] - r[0])


class AdsorptionInputs:
    def __init__(self, with_velocity=True):
        w = common.work_dir("irreversible_adsorption")
        G = np.load(os.path.join(w, "geometry_features.npz"))
        self.pos = {int(b): k for k, b in enumerate(G["bases"])}
        self.mask = G["mask"].astype(np.int32); self.gdf = G["gdf"]
        self.ux_z = self.uy_z = None
        if with_velocity:
            V = np.load(os.path.join(w, "velocity_input.npz"))
            self.stats = {k: float(V[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY")}
            self.ux_z = ((V["uxuy"][:, 0] - self.stats["MU_UX"]) / self.stats["SD_UX"]).astype(np.float32)
            self.uy_z = ((V["uxuy"][:, 1] - self.stats["MU_UY"]) / self.stats["SD_UY"]).astype(np.float32)
        Y, X = coords(); self.Y = Y.ravel(); self.X = X.ravel()

    def build_geometry(self, bases, Da, Pe, variant=True):
        N = len(bases); L = common.NY * common.NX; ks = [self.pos[int(b)] for b in bases]
        m = self.mask[ks]
        b1 = np.stack([m.astype(np.float32), self.ux_z[ks], self.uy_z[ks]], 1) if variant else m[:, None].astype(np.float32)
        b2 = np.stack([mm(np.asarray(Da, float), DA_RANGE), mm(np.asarray(Pe, float), PE_RANGE)], 1).astype(np.float32)
        tr = np.stack([np.broadcast_to(self.Y, (N, L)), np.broadcast_to(self.X, (N, L)), self.gdf[ks].reshape(N, L)], -1)
        return dict(b1=torch.from_numpy(b1), b2=torch.from_numpy(b2), tr=torch.from_numpy(tr[:, None].astype(np.float32)),
                    pore=m == 1)

    def build(self, idx_list, variant=True):
        DaPe = [adsorption_condition(i) for i in idx_list]
        return self.build_geometry([common.base_of(i) for i in idx_list], [d for d, _ in DaPe], [p for _, p in DaPe], variant)

    @staticmethod
    def targets(idx_list):
        return torch.from_numpy(np.stack([adsorption_concentration(i) for i in idx_list])[..., None])


def samples(bases):
    return [k * common.N_BASE + b for b in bases for k in range(9)]


@torch.no_grad()
def predict(model, inp, dev, bs=100):
    model.eval(); out = []
    for s in range(0, inp["b1"].shape[0], bs):
        out.append(model(inp["b1"][s:s + bs].to(dev), inp["b2"][s:s + bs].to(dev), inp["tr"][s:s + bs].to(dev)).float().cpu().numpy()[..., 0])
    return np.concatenate(out)


def pore_rmse(pred, gt, pore):
    return np.array([float(np.sqrt(np.mean((pred[i][pore[i]] - gt[i][pore[i]]) ** 2))) for i in range(len(pred))])
