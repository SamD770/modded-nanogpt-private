"""PCA of the byte embedding table (256, D) restricted to lowercase / uppercase / digits."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
import torch
import matplotlib.pyplot as plt


# Stub so torch.load can unpickle the optimizer state from train_gpt.py's __main__.
# We only read sd["model"]; the class just needs to accept arbitrary attrs.
class ParamConfig:
    pass

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "logs/e90deb53-9d08-4013-bf0f-76af07743f5a/state_step001480.pt")

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
W = next(v for k, v in sd.items() if k.endswith("byte_embed.weight")).float()
assert W.shape[0] == 256
print(f"byte_embed shape: {tuple(W.shape)}")

groups = {
    "lower": (range(ord("a"), ord("z") + 1), "tab:blue"),
    "upper": (range(ord("A"), ord("Z") + 1), "tab:orange"),
    "digit": (range(ord("0"), ord("9") + 1), "tab:green"),
}

# Fit PCA on all 256 byte embeddings, then project the highlighted subset.
mean = W.mean(dim=0, keepdim=True)
Wc = W - mean
U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
var = (S ** 2) / (W.shape[0] - 1)
evr = (var / var.sum()).tolist()
print(f"explained variance ratio (top 5, all 256): {['%.3f' % v for v in evr[:5]]}")

idx = torch.tensor([i for rng, _ in groups.values() for i in rng])
coords = (W[idx] - mean) @ Vh[:2].T  # (62, 2)

fig, ax = plt.subplots(figsize=(8, 7))
offset = 0
for name, (rng, color) in groups.items():
    n = len(list(rng))
    pts = coords[offset:offset + n]
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=60, alpha=0.7, label=name, edgecolors="k", linewidths=0.5)
    for j, b in enumerate(rng):
        ax.annotate(chr(b), (pts[j, 0].item(), pts[j, 1].item()),
                    fontsize=9, ha="center", va="center")
    offset += n

ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
ax.set_title(f"byte_embed PCA (lower/upper/digit) — {CKPT.parent.name}")
ax.legend()
ax.grid(alpha=0.3)
out = f"byte_embed_pca_{CKPT.parent.name}.png"
plt.tight_layout()
plt.savefig(out, dpi=140)
print(f"wrote {out}")
