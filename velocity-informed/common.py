"""Code shared by the three pipelines (velocity, irreversible adsorption, Monod): paths, grid, split, LB file reading,
query coordinates, the inlet geodesic distance, weight loading, baseline seeds and the figure helpers. Everything that
belongs to one physics is in <physics>/core.py.

Every location can be overridden with an environment variable:
    PRT_LB    LB labels                                                             default: <repo>/lb_labels
    PRT_WORK  intermediate files written by 1_preprocess/, 2_train/ and 3_extract/   default: <repo>/work
    PRT_DEVICE  torch device                                                          default: cuda if available
"""
import os, csv, json
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
LB = os.environ.get("PRT_LB", os.path.join(ROOT, "lb_labels"))
WORK = os.environ.get("PRT_WORK", os.path.join(ROOT, "work"))
SPLITS = os.path.join(ROOT, "splits")

NY, NX = 64, 148
N_BASE = 3000
BLOCK_TOP_BOTTOM = True


def work_dir(physics):
    d = os.path.join(WORK, physics)
    os.makedirs(d, exist_ok=True)
    return d


def weights_dir(physics):
    return os.path.join(ROOT, physics, "weights")


def data_dir(physics):
    d = os.path.join(ROOT, physics, "data")
    os.makedirs(d, exist_ok=True)
    return d


def lb_data_dir(physics):
    return os.path.join(ROOT, physics, "lb_data")


def device():
    import torch
    d = os.environ.get("PRT_DEVICE")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_of(idx):
    return (int(idx) - 1) % N_BASE + 1


def group_of(idx):
    return (int(idx) - 1) // N_BASE


def load_split():
    return json.load(open(os.path.join(SPLITS, "split.json")))


def porosity(raw):
    m = np.asarray(raw)
    return (np.sum((m == 1) | (m == 2)) - 64 * 20) / (64 * 128)


def read_npz(path, key):
    with np.load(path) as z:
        return np.asarray(z[key])


def coords():
    ys = np.arange(NY, dtype=np.float32) / (NY - 1)
    xs = np.arange(NX, dtype=np.float32) / (NX - 1)
    return np.meshgrid(ys, xs, indexing="ij")


def _inlet_bfs(m):
    H, W = m.shape
    dist = np.zeros_like(m, dtype=np.float32)
    starts = [(i, 0) for i in range(H) if m[i, 0] == 1]
    for (y, x) in starts:
        dist[y, x] = 0.5
    q = deque(starts)
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            a, b = y + dy, x + dx
            if 0 <= a < H and 0 <= b < W and m[a, b] == 1:
                nd = dist[y, x] + 1.0
                if dist[a, b] == 0 or dist[a, b] > nd:
                    dist[a, b] = nd; q.append((a, b))
    return dist


def inlet_distance_norm(m):
    d = _inlet_bfs(m)
    mx = np.nanmax(d[m == 1]); d[m == 0] = mx + 1; d = -d
    v = np.isfinite(d); lo, hi = d[v].min(), d[v].max()
    out = np.zeros_like(d)
    if hi > lo:
        out[v] = (d[v] - lo) / (hi - lo)
    out[m == 0] = 0
    return out.astype(np.float32)


SEEDS = [42, 2, 3, 4, 5]
FIG_SEED = 42


def load_weights(model, path, dev):
    import torch
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
    return model.to(dev).eval()


def baseline_weights(physics, name):
    return os.path.join(weights_dir(physics), "baseline", f"{name}.pt")


if "Noto Sans CJK JP" in {f.name for f in font_manager.fontManager.ttflist}:
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


def paths(physics):
    data = os.environ.get("DATA", os.path.join(ROOT, physics, "data"))
    out = os.environ.get("OUT", os.path.join(ROOT, physics, "4_figures", "output"))
    os.makedirs(out, exist_ok=True)
    return data, out


def save(fig, out_dir, name, dpi=300):
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    cols = {}
    for j, h in enumerate(head):
        col = [r[j] for r in body]
        try:
            cols[h] = np.array([float(x) if x != "" else np.nan for x in col])
        except ValueError:
            cols[h] = np.array(col)
    return cols


def read_metrics(data_dir, name):
    a = read_csv(os.path.join(data_dir, f"{name}.csv")); b = read_csv(os.path.join(data_dir, f"baseline_{name}.csv"))
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def mean_std(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def table_png(rows, title, out_dir, name, header_rows=1, figw=13, fontsize=10, highlight_col=None):
    fig, ax = plt.subplots(figsize=(figw, 0.5 * len(rows) + 1.2)); ax.axis("off")
    t = ax.table(cellText=rows, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.5)
    t.auto_set_column_width(list(range(max(len(r) for r in rows))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r < header_rows:
            cell.set_facecolor("#333"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif highlight_col is not None and c == highlight_col:
            cell.set_facecolor("#fff2cc")
        else:
            cell.set_facecolor("#f5f5f5" if r % 2 else "#eef5ff")
        if c == 0 and r >= header_rows:
            cell.get_text().set_ha("left")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    save(fig, out_dir, name, dpi=170)
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
