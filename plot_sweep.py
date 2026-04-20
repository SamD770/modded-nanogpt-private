"""Plot sweep_results.csv as 4 heatmaps + paired-comparison stats for left vs right padding."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent
CSV_PATH = REPO / "sweep_results.csv"
OUT_PATH = REPO / "sweep_heatmaps.png"

WINDOWS = [4, 8, 16]
DIMS = [16, 32, 64]

# --- load ---
rows = []
with CSV_PATH.open() as f:
    for r in csv.DictReader(f):
        if r["status"] != "ok":
            continue
        rows.append({
            "w": int(r["byte_window_size"]),
            "d": int(r["byte_embedding_dim"]),
            "pad": r["byte_pad_side"],
            "val_loss": float(r["val_loss"]),
            "train_time_ms": int(r["train_time_ms"]),
        })

def grid(pad: str, key: str) -> np.ndarray:
    g = np.full((len(WINDOWS), len(DIMS)), np.nan)
    for r in rows:
        if r["pad"] != pad:
            continue
        i = WINDOWS.index(r["w"])
        j = DIMS.index(r["d"])
        g[i, j] = r[key]
    return g

# --- 4 heatmaps ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
panels = [
    ("left",  "val_loss",      "val_loss (left-pad / suffix-aligned)",  "viridis_r", "{:.4f}"),
    ("right", "val_loss",      "val_loss (right-pad / prefix-aligned)", "viridis_r", "{:.4f}"),
    ("left",  "train_time_ms", "train_time_ms (left-pad)",              "magma",     "{:.0f}"),
    ("right", "train_time_ms", "train_time_ms (right-pad)",             "magma",     "{:.0f}"),
]

# Shared color scales per metric so left vs right are visually comparable
loss_all = np.concatenate([grid("left", "val_loss").ravel(), grid("right", "val_loss").ravel()])
time_all = np.concatenate([grid("left", "train_time_ms").ravel(), grid("right", "train_time_ms").ravel()])
vmin_loss, vmax_loss = np.nanmin(loss_all), np.nanmax(loss_all)
vmin_time, vmax_time = np.nanmin(time_all), np.nanmax(time_all)

for ax, (pad, key, title, cmap, fmt) in zip(axes.flat, panels):
    g = grid(pad, key)
    vmin, vmax = (vmin_loss, vmax_loss) if key == "val_loss" else (vmin_time, vmax_time)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DIMS)), DIMS)
    ax.set_yticks(range(len(WINDOWS)), WINDOWS)
    ax.set_xlabel("byte_embedding_dim")
    ax.set_ylabel("byte_window_size")
    ax.set_title(title)
    for i in range(len(WINDOWS)):
        for j in range(len(DIMS)):
            ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center",
                    color="white" if (g[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Token-strided byte conv sweep (3 windows × 3 dims × 2 pad sides)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=130)
print(f"wrote {OUT_PATH}")

# --- paired stats: left vs right, paired by (window, dim) ---
pairs = []
for w in WINDOWS:
    for d in DIMS:
        l = next((r for r in rows if r["pad"] == "left" and r["w"] == w and r["d"] == d), None)
        r2 = next((r for r in rows if r["pad"] == "right" and r["w"] == w and r["d"] == d), None)
        if l and r2:
            pairs.append((w, d, l["val_loss"], r2["val_loss"], l["train_time_ms"], r2["train_time_ms"]))

print(f"\n=== Paired comparison (n={len(pairs)} grid cells) ===")
print(f"{'win':>4} {'dim':>4}   {'left_loss':>9} {'right_loss':>10}  {'Δ(L-R)':>8}    {'left_ms':>7} {'right_ms':>8}")
for w, d, ll, rl, lt, rt in pairs:
    print(f"{w:>4} {d:>4}   {ll:>9.4f} {rl:>10.4f}  {ll - rl:>+8.4f}    {lt:>7d} {rt:>8d}")

l_loss = np.array([p[2] for p in pairs])
r_loss = np.array([p[3] for p in pairs])
diff = l_loss - r_loss   # negative -> left is better

print(f"\nval_loss left  : mean={l_loss.mean():.4f}  std={l_loss.std(ddof=1):.4f}")
print(f"val_loss right : mean={r_loss.mean():.4f}  std={r_loss.std(ddof=1):.4f}")
print(f"paired Δ (L-R) : mean={diff.mean():+.4f}  std={diff.std(ddof=1):.4f}  (negative => left better)")

t_stat, p_two = stats.ttest_rel(l_loss, r_loss)
w_stat, w_p = stats.wilcoxon(l_loss, r_loss)
print(f"paired t-test   : t={t_stat:+.3f}  p(two-sided)={p_two:.3f}")
print(f"Wilcoxon signed : W={w_stat:.1f}  p(two-sided)={w_p:.3f}")
n_left_wins = int((diff < 0).sum())
print(f"left wins (Δ<0): {n_left_wins}/{len(diff)}")

l_time = np.array([p[4] for p in pairs])
r_time = np.array([p[5] for p in pairs])
print(f"\ntrain_time_ms left  : mean={l_time.mean():.0f}  std={l_time.std(ddof=1):.0f}")
print(f"train_time_ms right : mean={r_time.mean():.0f}  std={r_time.std(ddof=1):.0f}")
print(f"paired Δ time (L-R) : mean={(l_time - r_time).mean():+.1f}ms  (essentially identical, as expected)")

# Note: each cell is a single seed, so even a "significant" p-value here is weak evidence —
# inter-seed noise on this codebase is ~0.001-0.002 loss per the README.
