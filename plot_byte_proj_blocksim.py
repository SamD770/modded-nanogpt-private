"""Compare byte_proj block-pair cosine similarity: analytical init expectation vs
a fresh random init draw vs the trained matrix.

byte_proj weight has shape (model_dim, W*E) where W=byte_window_size, E=byte_embedding_dim.
We split it into W blocks of shape (model_dim, E), flatten each, and compute the
WxW cosine-similarity matrix between block pairs.

At init, block w = R^w @ M where R is identity on the first half of model_dim and
2D paired rotations by pi/12 on the second half. This implies
    cos(W_0, W_w) = 0.5 * (1 + cos(pi*|w|/12))
to leading order (in expectation over M, when ||first half||^2 ~ ||second half||^2).
"""
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
    "logs/medium_runs/token_strided_conv_w8_e32_right_ckpt_20260503_024858/state_step004150.pt")
OUT = Path("byte_proj_blocksim.png")


# Optimizer state stores `ParamConfig` from train_gpt_medium.py — stub it out so we can
# unpickle the checkpoint from a standalone script.
class _IgnoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ParamConfig":
            return type("ParamConfig", (), {})
        return super().find_class(module, name)
class _PicklerProxy:
    Unpickler = _IgnoreUnpickler

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, pickle_module=_PicklerProxy)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

W_full = sd["byte_proj.weight"].float()                       # (model_dim, W*E)
model_dim, in_dim = W_full.shape
E = 32                                                        # byte_embedding_dim (held fixed)
n_w = in_dim // E                                             # byte_window_size (auto-detected)
assert in_dim == n_w * E, f"unexpected in_dim {in_dim}, not a multiple of E={E}"
print(f"detected n_w={n_w}, E={E} (byte_proj.weight shape={tuple(W_full.shape)})")
OUT = Path(f"byte_proj_blocksim_w{n_w}.png")

# Reshape into (n_w, model_dim, E) — block w occupies columns [w*E : (w+1)*E]
W_trained = W_full.reshape(model_dim, n_w, E).permute(1, 0, 2).contiguous()


def build_init_blocks(model_dim: int, E: int, n_w: int, seed: int = 0) -> torch.Tensor:
    """Reproduce the rotation init exactly: block w = R^w @ M."""
    g = torch.Generator().manual_seed(seed)
    # nn.init.kaiming_uniform_(M, a=sqrt(5)) with default fan_in mode and leaky_relu:
    #   gain = sqrt(2 / (1 + 5)) = sqrt(1/3); std = gain / sqrt(fan_in=E); bound = sqrt(3) * std = 1/sqrt(E)
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
    return torch.stack(blocks, dim=0)  # (n_w, model_dim, E)


def block_cos(W3: torch.Tensor) -> torch.Tensor:
    flat = W3.reshape(W3.shape[0], -1)
    flat = flat / flat.norm(dim=1, keepdim=True)
    return flat @ flat.T


W_init = build_init_blocks(model_dim, E, n_w)
S_init = block_cos(W_init)
S_trained = block_cos(W_trained)

# Analytical expectation (in expectation over M; converges quickly with model_dim*E~32k entries).
S_expected = torch.zeros(n_w, n_w)
for i in range(n_w):
    for j in range(n_w):
        S_expected[i, j] = 0.5 * (1 + math.cos(math.pi * abs(i - j) / 12))


_annotate = n_w <= 12  # too cramped for larger windows

def heatmap(ax, S, title):
    im = ax.imshow(S.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("window pos w'")
    ax.set_ylabel("window pos w")
    ax.set_xticks(range(n_w))
    ax.set_yticks(range(n_w))
    if _annotate:
        for i in range(n_w):
            for j in range(n_w):
                ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if S[i,j] < 0.5 else "black")
    return im


_w = max(15.5, 1.5 * n_w + 4)
fig, axes = plt.subplots(1, 3, figsize=(_w, _w / 3.4))
heatmap(axes[0], S_expected, "Analytical init expectation\n0.5*(1 + cos(pi*|w-w'|/12))")
heatmap(axes[1], S_init,     f"Actual init (random M, seed=0)\n{model_dim}x{E} per block")
im = heatmap(axes[2], S_trained,  f"Trained ({CKPT.parent.name})")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="cosine similarity")

fig.suptitle("byte_proj block-pair cosine similarity (flattened)", fontsize=12)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")

# Numeric summary: how much did the structure shift?
delta = (S_trained - S_init).abs()
print(f"\n||S_trained - S_init||_F = {delta.norm().item():.3f}")
print(f"max |S_trained - S_init|   = {delta.max().item():.3f}")
print(f"\ndiagonal-1 (adjacent w):  init={S_init.diagonal(1).mean():.3f}  trained={S_trained.diagonal(1).mean():.3f}")
print(f"diagonal-3 (3 apart):     init={S_init.diagonal(3).mean():.3f}  trained={S_trained.diagonal(3).mean():.3f}")
print(f"diagonal-7 (7 apart):     init={S_init.diagonal(7).mean():.3f}  trained={S_trained.diagonal(7).mean():.3f}")
