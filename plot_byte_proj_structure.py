"""Two more views of byte_proj convergence beyond block-pair similarity.

(1) Singular-value spectra of each window-position block, init vs trained.
    At init, R^w @ M with R orthogonal => identical sigma_i across all w; lines overlap.
    Training may push some blocks to lower effective rank.

(2) PCA of the effective per-byte directions d[w, b] = W_w @ E_b.
    Reveals whether learned directions cluster by window position or by byte category
    (ASCII letter / digit / whitespace / punct / extended / control).
"""
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
    "logs/medium_runs/token_strided_conv_w8_e32_right_ckpt_20260503_024858/state_step004150.pt")


class _IgnoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ParamConfig":
            return type("ParamConfig", (), {})
        return super().find_class(module, name)
class _PicklerProxy:
    Unpickler = _IgnoreUnpickler

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, pickle_module=_PicklerProxy)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

W_full = sd["byte_proj.weight"].float()
E_byte = sd["byte_embed.weight"].float()                     # (256, byte_embedding_dim)
model_dim, in_dim = W_full.shape
n_w, E = 8, 32
W_trained = W_full.reshape(model_dim, n_w, E).permute(1, 0, 2).contiguous()


def build_init_blocks(model_dim: int, E: int, n_w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    bound = 1.0 / math.sqrt(E)
    M = torch.empty(model_dim, E).uniform_(-bound, bound, generator=g)
    angle = math.pi / 12
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = torch.eye(model_dim)
    half = model_dim // 2
    for j in range(half, model_dim, 2):
        R[j,     j    ] =  cos_a; R[j,     j + 1] = -sin_a
        R[j + 1, j    ] =  sin_a; R[j + 1, j + 1] =  cos_a
    blocks = []
    Rn = torch.eye(model_dim)
    for _ in range(n_w):
        blocks.append(Rn @ M)
        Rn = R @ Rn
    return torch.stack(blocks, dim=0)


W_init = build_init_blocks(model_dim, E, n_w)


# -------------------------------- (1) SVD spectra --------------------------------
cmap = plt.get_cmap("viridis", n_w)

sigmas_init = torch.stack([torch.linalg.svdvals(W_init[w]) for w in range(n_w)])    # (n_w, E)
sigmas_trnd = torch.stack([torch.linalg.svdvals(W_trained[w]) for w in range(n_w)]) # (n_w, E)

# Effective rank (participation ratio): (sum sigma_i^2)^2 / sum sigma_i^4
def eff_rank(s):
    return (s.pow(2).sum() ** 2) / s.pow(4).sum()

print("effective rank per block:")
print(f"{'w':>3}  {'init':>8}  {'trained':>8}")
for w in range(n_w):
    print(f"{w:>3}  {eff_rank(sigmas_init[w]).item():>8.2f}  {eff_rank(sigmas_trnd[w]).item():>8.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for w in range(n_w):
    ax1.plot(range(E), sigmas_init[w].numpy(),  marker="o", markersize=3, color=cmap(w), label=f"w={w}")
    ax2.plot(range(E), sigmas_trnd[w].numpy(),  marker="o", markersize=3, color=cmap(w), label=f"w={w}")
ax1.set_title(f"Init (R^w·M; R orthogonal => sigma identical)")
ax2.set_title("Trained (right-pad, end of training)")
for ax in (ax1, ax2):
    ax.set_yscale("log")
    ax.set_xlabel("singular value index i")
    ax.grid(True, alpha=0.3)
ax1.set_ylabel("sigma_i")
ax1.legend(ncol=2, fontsize=8, loc="lower left")
fig.suptitle(f"Singular-value spectra of byte_proj blocks (each block: {model_dim}x{E})")
fig.tight_layout()
fig.savefig("byte_proj_svd.png", dpi=130, bbox_inches="tight")
print("wrote byte_proj_svd.png")
plt.close(fig)


# -------------------- (2) PCA of effective per-byte directions -------------------
# d[w, b, :] = W_trained[w] @ E_byte[b] -> (8, 256, 1024)
directions = torch.einsum("wde,be->wbd", W_trained, E_byte)
flat = directions.reshape(n_w * 256, model_dim)
flat_centered = flat - flat.mean(dim=0, keepdim=True)
U, S_pca, V = torch.linalg.svd(flat_centered, full_matrices=False)
pcs = flat_centered @ V[:3].T   # (2048, 3)

# Variance explained
var_explained = (S_pca ** 2) / (S_pca ** 2).sum()
print(f"\nvariance explained by top-3 PCs: {var_explained[:3].sum().item()*100:.1f}%")
print(f"variance explained by top-10 PCs: {var_explained[:10].sum().item()*100:.1f}%")


def byte_category(b: int) -> str:
    if 65 <= b <= 90 or 97 <= b <= 122:
        return "letter"
    if 48 <= b <= 57:
        return "digit"
    if b == 32:
        return "space"
    if b in (9, 10, 13):                       # \t \n \r
        return "whitespace"
    if 33 <= b <= 47 or 58 <= b <= 64 or 91 <= b <= 96 or 123 <= b <= 126:
        return "punct"
    if b >= 128:
        return "extended_utf8"
    return "control_other"


cats = ["letter", "digit", "space", "whitespace", "punct", "extended_utf8", "control_other"]
cat_color = {
    "letter":         "#1f77b4",  # blue
    "digit":          "#ff7f0e",  # orange
    "space":          "#2ca02c",  # green
    "whitespace":     "#17becf",  # cyan
    "punct":          "#d62728",  # red
    "extended_utf8":  "#9467bd",  # purple
    "control_other":  "#7f7f7f",  # gray
}

# Per-byte category for all 2048 points
cat_per_pt = []
for w in range(n_w):
    for b in range(256):
        cat_per_pt.append(byte_category(b))


fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

# Panel A: PC1 vs PC2 colored by window position
for w in range(n_w):
    pts = pcs[w * 256:(w + 1) * 256]
    axes[0].scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.45, color=cmap(w), label=f"w={w}")
axes[0].set_title("PC1 vs PC2 — colored by window position")
axes[0].set_xlabel(f"PC1  ({100*var_explained[0]:.1f}%)")
axes[0].set_ylabel(f"PC2  ({100*var_explained[1]:.1f}%)")
axes[0].legend(ncol=2, fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel B: PC1 vs PC2 colored by byte category
for cat in cats:
    idxs = [k for k, c in enumerate(cat_per_pt) if c == cat]
    if not idxs:
        continue
    pts = pcs[idxs]
    axes[1].scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55, color=cat_color[cat], label=cat)
axes[1].set_title("PC1 vs PC2 — colored by byte category")
axes[1].set_xlabel(f"PC1  ({100*var_explained[0]:.1f}%)")
axes[1].set_ylabel(f"PC2  ({100*var_explained[1]:.1f}%)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Panel C: cumulative variance explained
cum = (var_explained.cumsum(0)).numpy()
axes[2].plot(range(1, len(cum) + 1), cum * 100, marker="o", markersize=3)
axes[2].axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95%")
axes[2].axhline(99, color="gray", linestyle=":",  linewidth=0.8, label="99%")
axes[2].set_xlabel("# PCs")
axes[2].set_ylabel("cumulative variance explained (%)")
axes[2].set_title("PCA variance accumulation")
axes[2].set_xscale("log")
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

fig.suptitle(f"PCA of {n_w * 256} effective per-byte directions  d[w,b] = W_w · E_b   (1024-dim, then projected to top-3 PCs)")
fig.tight_layout()
fig.savefig("byte_proj_directions_pca.png", dpi=130, bbox_inches="tight")
print("wrote byte_proj_directions_pca.png")
