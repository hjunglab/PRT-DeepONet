"""Velocity pipeline: everything specific to the steady velocity field (ux, uy).

    LB readers        velocity samples idx = base + 3000 * Re_group (groups ordered by Reynolds number) and the two
                      unseen Reynolds conditions (interpolation set)
    geometry          pore mask, top/bottom walls, wall distance, UPRM (flow-neck radius), MIS (maximal inscribed sphere)
    pressure labels   harmonic pressure field of the pore space and its gradient (px, py)
    networks          pressure U-Net, velocity variant, vanilla DeepONet (comparison baseline), Huber loss,
                      divergence penalty; parameter names match the released weight files
    model inputs      assembled from velocity/lb_data/flow_conditions.npz, the outputs of velocity/1_preprocess/ and
                      the pressure U-Net, so that training, evaluation and extraction use the same inputs

Conventions: arrays are (NY, NX) = (64, 148), rows = y (cross-stream), columns = x (streamwise, the flow direction);
column 0 is the inlet, column NX-1 the outlet. A raw LB mask holds 0 = solid, 1 = pore, 2 = pore cell next to a grain;
1 and 2 are both pore.
"""
import os
import heapq
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from common import NY, NX, read_npz, coords, load_weights


def velocity_mask(idx, root=None):
    return read_npz(os.path.join(root or common.LB, "velocity", "mask", f"m{idx}.npz"), "mask").reshape(NY, NX)


def velocity_uxuy(idx, root=None):
    vel = read_npz(os.path.join(root or common.LB, "velocity", "velocity", f"v{idx}.npz"), "vel").ravel()
    h = vel.size // 2
    return vel[:h].reshape(NY, NX).astype(np.float32), vel[h:].reshape(NY, NX).astype(np.float32)


UNSEEN_RE = ("ReA", "ReB")


def unseen_velocity(cond, base, root=None):
    z = np.load(os.path.join(root or common.LB, "velocity_unseen_re", cond, f"b{base}.npz"))
    return z["mask"].reshape(NY, NX), z["ux"].reshape(NY, NX).astype(np.float32), z["uy"].reshape(NY, NX).astype(np.float32)


def pore_mask(raw):
    m = np.asarray(raw).reshape(NY, NX)
    return (np.where(m == 2, 1, m) > 0).astype(np.int32)


def block_top_bottom(m):
    m = m.copy()
    m[0, :] = 0
    m[-1, :] = 0
    return m


def wall_distance(m):
    return distance_transform_edt(m == 1).astype(np.float32)


def flow_neck_radius_map(mask, connectivity=4):
    H, W = mask.shape
    void = (mask.astype(np.uint8) > 0)
    edt = ndi.distance_transform_edt(void).astype(np.float32)
    seeds = [(y, 0) for y in range(H) if void[y, 0]]
    if not seeds:
        return np.zeros_like(edt)
    acc = np.zeros_like(edt)
    visited = np.zeros_like(void, bool)
    heap = []
    for y, x in seeds:
        v = float(edt[y, x]); acc[y, x] = v
        heapq.heappush(heap, (-v, y, x))
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else \
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while heap:
        nv, y, x = heapq.heappop(heap)
        cur = -nv
        if visited[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < H and 0 <= xx < W) or not void[yy, xx]:
                continue
            cand = min(cur, float(edt[yy, xx]))
            if cand > acc[yy, xx]:
                acc[yy, xx] = cand
                heapq.heappush(heap, (-cand, yy, xx))
    acc *= void.astype(np.float32)
    return acc.astype(np.float32)


BUF = 10


def local_thickness_map(m, r_step=0.5):
    pore = (m > 0)
    edt = distance_transform_edt(pore).astype(np.float32)
    rmax = float(edt.max())
    if rmax <= 0:
        return np.zeros_like(edt)
    res = np.zeros_like(edt)
    for r in np.arange(rmax, 0, -r_step):
        ce = pore & (edt >= r - 1e-6)
        if not ce.any():
            continue
        rr = int(np.ceil(r)); yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]; dk = (xx * xx + yy * yy) <= r * r
        painted = binary_dilation(ce, structure=dk) & pore
        res[painted & (res == 0)] = r
    res[~pore] = 0.0
    return res.astype(np.float32)


def mis_radius_map(m, buf=BUF, r_step=0.5):
    H, W = m.shape
    pore = (m > 0)
    lt = local_thickness_map(m[:, buf:W - buf], r_step)
    full = np.zeros((H, W), np.float32); full[:, buf:W - buf] = lt
    left = np.zeros((H, W), bool); left[:, :buf] = True
    right = np.zeros((H, W), bool); right[:, W - buf:] = True
    seed = np.zeros((H, W), np.float32); seed[:, buf:W - buf] = lt
    for r in np.unique(seed[seed > 0])[::-1]:
        ce = seed >= r - 1e-6
        rr = int(np.ceil(r)); yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]; dk = (xx * xx + yy * yy) <= r * r
        pt = binary_dilation(ce, structure=dk)
        for reg in (left, right):
            up = pt & reg & (full < r); full[up] = r
    bp = pore & (left | right) & (full <= 0)
    vmin = float(lt[lt > 0].min()) if (lt > 0).any() else 0.0
    full[bp] = vmin
    full[m <= 0] = 0.0
    return full.astype(np.float32)


P_IN, P_OUT, OMEGA, ITERS, TOL = 0.1, 0.0, 1.85, 20000, 1e-6


def harmonic_pressure(m, w):
    ny, nx = m.shape
    pore = (m == 1)
    inl = pore & (np.arange(nx) == 0)[None, :]
    out = pore & (np.arange(nx) == nx - 1)[None, :]
    unk = pore & ~inl & ~out
    cL = np.zeros((ny, nx), np.float32); cR = np.zeros((ny, nx), np.float32)
    cD = np.zeros((ny, nx), np.float32); cU = np.zeros((ny, nx), np.float32)
    cL[:, 1:] = np.where(pore[:, 1:] & pore[:, :-1], w[:, 1:], 0.0)
    cR[:, :-1] = np.where(pore[:, :-1] & pore[:, 1:], w[:, :-1], 0.0)
    cD[1:, :] = np.where(pore[1:, :] & pore[:-1, :], w[1:, :], 0.0)
    cU[:-1, :] = np.where(pore[:-1, :] & pore[1:, :], w[:-1, :], 0.0)
    n = cL + cR + cD + cU
    unk &= (n > 0)
    x = np.arange(nx, dtype=np.float32)[None, :]
    ramp = P_OUT + (P_IN - P_OUT) * (1.0 - x / max(1, nx - 1))
    p = np.where(inl, P_IN, np.where(out, P_OUT, ramp)).astype(np.float32)
    Y, X = np.indices((ny, nx)); red = ((X + Y) & 1) == 0; black = ~red
    den = np.where(n > 0, n, 1.0)
    for _ in range(ITERS):
        md = 0.0
        for col in (red, black):
            mk = unk & col
            pL = np.zeros_like(p); pL[:, 1:] = p[:, :-1]
            pR = np.zeros_like(p); pR[:, :-1] = p[:, 1:]
            pD = np.zeros_like(p); pD[1:, :] = p[:-1, :]
            pU = np.zeros_like(p); pU[:-1, :] = p[1:, :]
            pn = (cL * pL + cR * pR + cD * pD + cU * pU) / den
            dl = OMEGA * (pn - p)
            if np.any(mk):
                bf = p[mk].copy(); p[mk] = p[mk] + dl[mk]
                np.clip(p, 0.0, 1.0, out=p)
                md = max(md, float(np.max(np.abs(p[mk] - bf))))
        if md < TOL:
            break
    p[inl] = P_IN; p[out] = P_OUT
    return p


def pressure_gradient(m):
    d = wall_distance(m)
    p = harmonic_pressure(m, (d * d).astype(np.float32))
    pore = (m == 1)
    px = np.zeros_like(p); cx = np.zeros_like(p)
    mx = pore[:, :-1] & pore[:, 1:]; dx = (p[:, 1:] - p[:, :-1]).astype(np.float32)
    px[:, :-1][mx] += dx[mx]; cx[:, :-1][mx] += 1; px[:, 1:][mx] += dx[mx]; cx[:, 1:][mx] += 1
    px /= np.maximum(cx, 1)
    py = np.zeros_like(p); cy = np.zeros_like(p)
    my = pore[:-1, :] & pore[1:, :]; dy = (p[1:, :] - p[:-1, :]).astype(np.float32)
    py[:-1, :][my] += dy[my]; cy[:-1, :][my] += 1; py[1:, :][my] += dy[my]; cy[1:, :][my] += 1
    py /= np.maximum(cy, 1)
    px[~pore] = 0.0; py[~pore] = 0.0
    return px.astype(np.float32), py.astype(np.float32)


class ConvBlock(nn.Module):
    def __init__(s, cin, cout, p=0.1):
        super().__init__(); s.conv = nn.Conv2d(cin, cout, 3, padding=1); s.drop = nn.Dropout2d(p); s.bn = nn.BatchNorm2d(cout)

    def forward(s, x):
        return s.bn(s.drop(F.relu(s.conv(x))))


class Container(nn.Module):
    def __init__(s, cin, cout, n=4, p=0.1):
        super().__init__(); s.net = nn.Sequential(*([ConvBlock(cin, cout, p)] + [ConvBlock(cout, cout, p) for _ in range(n - 1)]))

    def forward(s, x):
        return s.net(x)


class PressureUNet(nn.Module):

    def __init__(s, in_ch=1, out_ch=2, base=64, depth=6, n=4, p=0.1):
        super().__init__(); s.depth = depth; s.enc = nn.ModuleList(); c = in_ch
        for _ in range(depth):
            s.enc.append(Container(c, base, n, p)); c = base
        s.pool = nn.MaxPool2d(2); s.bottleneck = Container(base, base, n, p)
        s.up = nn.ModuleList([nn.ConvTranspose2d(base, base, 2, stride=2) for _ in range(depth)])
        s.dec = nn.ModuleList([Container(base * 2, base, n, p) for _ in range(depth)])
        s.head_container = Container(base, base, n, p); s.head = nn.Conv2d(base, out_ch, 1)

    def forward(s, x):
        skips = []
        for d in range(s.depth):
            x = s.enc[d](x); skips.append(x); x = s.pool(x)
        x = s.bottleneck(x)
        for d in range(s.depth):
            skip = skips[s.depth - 1 - d]; x = s.up[d](x)
            dh = skip.shape[-2] - x.shape[-2]; dw = skip.shape[-1] - x.shape[-1]
            if dh or dw:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], 1); x = s.dec[d](x)
        return s.head(s.head_container(x))


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


class VelocityVariant(nn.Module):

    def __init__(s, b1_ch=3, trunk_in=5, out_dim=128, n_out=2, cnn_blocks=5):
        super().__init__(); s.branch1 = CustomCNN(b1_ch, out_dim, num_blocks=cnn_blocks); s.branch2 = CondMLP(1, out_dim, 128, 3)
        s.trunk = make_trunk(trunk_in, out_dim, 8, 128); s.bias = nn.Parameter(torch.zeros(n_out))
        s.n_out = n_out; s.half = out_dim // n_out

    def forward(s, b1, b2, tr):
        N, L, D = tr.shape
        t = s.trunk(tr.reshape(-1, D)).view(N, L, -1)
        prod = (s.branch1(b1).unsqueeze(1) * s.branch2(b2).unsqueeze(1) * t).view(N, L, s.n_out, s.half).sum(-1)
        return (prod + s.bias).view(N, NY, NX, s.n_out)


class VanillaDeepONet(VelocityVariant):

    def __init__(s):
        super().__init__(b1_ch=1, trunk_in=2, out_dim=256, n_out=2, cnn_blocks=6)


class ROIHuberLoss(nn.Module):

    def __init__(s, delta=1.0):
        super().__init__(); s.delta = delta

    def forward(s, pred, target, roi):
        mask = (roi > 0.5).unsqueeze(-1); diff = (pred - target) * mask; absd = torch.abs(diff)
        quad = 0.5 * absd.pow(2); lin = s.delta * (absd - 0.5 * s.delta)
        loss = torch.where(absd <= s.delta, quad, lin)
        return loss.sum() / (mask.sum().clamp(min=1.0) * pred.shape[-1])


def divergence_penalty(pred, pore, cx, cy, rv):
    p = (pore > 0.5).to(pred.dtype)
    ux = (pred[..., 0] + cx) * p; uy = rv * (pred[..., 1] + cy) * p
    uxp = torch.cat([ux[:, :, :1], ux, ux[:, :, -1:]], 2)
    uyp = torch.cat([torch.zeros_like(uy[:, :1]), uy, torch.zeros_like(uy[:, :1])], 1)
    div = (uxp[:, :, 2:] - uxp[:, :, :-2]) * 0.5 + (uyp[:, 2:] - uyp[:, :-2]) * 0.5
    return (div.pow(2) * p).sum() / p.sum().clamp(min=1.0)


PRESSURE_UNET_WEIGHTS = os.path.join(os.environ.get("WEIGHTS", common.weights_dir("velocity")), "pressure_unet.pt")


FLOW_CONDITIONS = os.path.join(common.lb_data_dir("velocity"), "flow_conditions.npz")


def load_preprocessed():
    G = np.load(os.path.join(common.work_dir("velocity"), "geometry_features.npz"))
    F = np.load(FLOW_CONDITIONS)
    return G, F


@torch.no_grad()
def pressure_unet(masks, weights=PRESSURE_UNET_WEIGHTS, dev=None):
    dev = dev or common.device()
    net = load_weights(PressureUNet(1, 2), weights, dev)
    out = np.empty((len(masks), 2, common.NY, common.NX), np.float32)
    for i, m in enumerate(masks):
        x = torch.from_numpy((m == 1).astype(np.float32)[None, None]).to(dev)
        o = net(x).float().cpu().numpy()[0]
        o[:, m != 1] = 0.0
        out[i] = o
    return out


class VelocityInputs:

    def __init__(self, pressure_unet_weights=PRESSURE_UNET_WEIGHTS, dev=None):
        G, F = load_preprocessed()
        self.bases = G["bases"]; pos = {int(b): k for k, b in enumerate(self.bases)}
        self.pos = pos
        self.mask = G["mask"].astype(np.int32)
        self.uprm_z = ((G["uprm"] - float(G["uprm_mu"])) / float(G["uprm_sd"])).astype(np.float32)
        self.mis_z = ((G["mis"] - float(G["mis_mu"])) / float(G["mis_sd"])).astype(np.float32)
        dw2 = G["dw2"]; lo, hi = float(G["dw2_min"]), float(G["dw2_max"])
        self.dw2n = np.where(dw2 > 0, (dw2 - lo) / (hi - lo), 0.0).astype(np.float32)
        self.re = F["re"]; self.re_norm = F["re_norm"]
        self.stats = {k: float(F[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY", "re_min", "re_max")}
        self.pgrad = pressure_unet(self.mask, pressure_unet_weights, dev) if pressure_unet_weights else None
        Y, X = coords(); self.Y = Y.ravel(); self.X = X.ravel()

    def k(self, idx):
        return self.pos[common.base_of(idx)]

    def build(self, idx_list):
        N = len(idx_list); L = common.NY * common.NX
        ks = [self.k(i) for i in idx_list]
        m = self.mask[ks]
        b1 = np.stack([m.astype(np.float32), self.uprm_z[ks], self.mis_z[ks]], 1)
        tr = np.empty((N, L, 5), np.float32)
        tr[:, :, 0] = self.Y; tr[:, :, 1] = self.X
        tr[:, :, 2] = self.pgrad[ks, 0].reshape(N, L); tr[:, :, 3] = self.pgrad[ks, 1].reshape(N, L)
        tr[:, :, 4] = self.dw2n[ks].reshape(N, L)
        b2 = self.re_norm[np.asarray(idx_list)].astype(np.float32)[:, None]
        pore = (m == 1)
        inter = np.zeros_like(pore)
        inter[:, 1:-1, 1:-1] = pore[:, 1:-1, 1:-1] & pore[:, 1:-1, 2:] & pore[:, 1:-1, :-2] & pore[:, 2:, 1:-1] & pore[:, :-2, 1:-1]
        return dict(b1=torch.from_numpy(b1), b2=torch.from_numpy(b2), tr=torch.from_numpy(tr),
                    roi=torch.from_numpy(pore.astype(np.float32)), interior=torch.from_numpy(inter))

    def targets(self, idx_list):
        UX = np.empty((len(idx_list), common.NY, common.NX), np.float32); UY = np.empty_like(UX)
        for i, idx in enumerate(idx_list):
            UX[i], UY[i] = velocity_uxuy(idx)
        s = self.stats
        y = np.stack([(UX - s["MU_UX"]) / s["SD_UX"], (UY - s["MU_UY"]) / s["SD_UY"]], -1).astype(np.float32)
        return UX, UY, torch.from_numpy(y)

    def to_physical(self, pred):
        s = self.stats
        return pred[..., 0] * s["SD_UX"] + s["MU_UX"], pred[..., 1] * s["SD_UY"] + s["MU_UY"]


def vanilla_inputs(VI, idx_list):
    N = len(idx_list); L = common.NY * common.NX
    m = VI.mask[[VI.k(i) for i in idx_list]]
    tr = np.empty((N, L, 2), np.float32); tr[:, :, 0] = VI.Y; tr[:, :, 1] = VI.X
    b2 = VI.re_norm[np.asarray(idx_list)].astype(np.float32)[:, None]
    return dict(b1=torch.from_numpy(m[:, None].astype(np.float32)), b2=torch.from_numpy(b2), tr=torch.from_numpy(tr))


@torch.no_grad()
def predict(model, inp, dev, bs=50):
    model.eval(); outs = []
    for s in range(0, inp["b1"].shape[0], bs):
        outs.append(model(inp["b1"][s:s + bs].to(dev), inp["b2"][s:s + bs].to(dev), inp["tr"][s:s + bs].to(dev)).float().cpu())
    return torch.cat(outs, 0).numpy()


def nrmse(t, p, mk):
    tt, pp = t[mk], p[mk]
    return float(np.sqrt(np.mean((pp - tt) ** 2)) / (tt.max() - tt.min() + 1e-12))


class GeometryPipeline:
    def __init__(self, seed=42, dev=None):
        self.dev = dev or torch.device("cpu")
        G, Fc = load_preprocessed()
        self.g = {k: float(G[k]) for k in ("uprm_mu", "uprm_sd", "mis_mu", "mis_sd", "dw2_min", "dw2_max")}
        self.s = {k: float(Fc[k]) for k in ("MU_UX", "SD_UX", "MU_UY", "SD_UY")}
        self.re_norm = Fc["re_norm"]
        wdir = os.environ.get("WEIGHTS", common.weights_dir("velocity"))
        self.unet = load_weights(PressureUNet(1, 2), PRESSURE_UNET_WEIGHTS, self.dev)
        self.net = load_weights(VelocityVariant(), os.path.join(wdir, f"velocity_variant_seed{seed}.pt"), self.dev)
        Y, X = coords(); self.Y = Y.ravel(); self.X = X.ravel()

    @torch.no_grad()
    def __call__(self, raw, re_norm):
        g = self.g; m = block_top_bottom(pore_mask(raw)); p = m == 1
        uprm = (flow_neck_radius_map(m) - g["uprm_mu"]) / g["uprm_sd"]
        mis = (mis_radius_map(m) - g["mis_mu"]) / g["mis_sd"]
        d = wall_distance(m); dw2 = (d * d).astype(np.float32)
        dw2n = np.where(dw2 > 0, (dw2 - g["dw2_min"]) / (g["dw2_max"] - g["dw2_min"]), 0.0)
        pg = self.unet(torch.from_numpy(p.astype(np.float32)[None, None]).to(self.dev)).float().cpu().numpy()[0]
        pg[:, ~p] = 0.0
        b1 = np.stack([m.astype(np.float32), uprm, mis])[None].astype(np.float32)
        tr = np.stack([self.Y, self.X, pg[0].ravel(), pg[1].ravel(), dw2n.ravel()], -1)[None].astype(np.float32)
        o = self.net(torch.from_numpy(b1).to(self.dev), torch.tensor([[re_norm]], dtype=torch.float32, device=self.dev),
                     torch.from_numpy(tr).to(self.dev)).float().cpu().numpy()[0]
        ux = o[..., 0] * self.s["SD_UX"] + self.s["MU_UX"]; uy = o[..., 1] * self.s["SD_UY"] + self.s["MU_UY"]
        return np.where(p, ux, 0.0), np.where(p, uy, 0.0)


def median_time(fn, reps=5):
    import time
    fn(); ts = []
    for _ in range(reps):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    return float(np.median(ts))
