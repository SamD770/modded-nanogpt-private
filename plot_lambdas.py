"""Parse 'lambdas ...' lines from a train log and plot byte_lambda + bigram_lambdas over steps."""
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/modded-nanogpt-private/logs/9aa70110-ca41-49bd-8810-c5df3622f965.txt")
OUT = Path("/root/modded-nanogpt-private/lambda_trajectory.png")

pat = re.compile(r"lambdas step:(\d+) byte_lambda:(\[[^\]]*\]) bigram_lambdas:(\[[^\]]*\])")
steps, bl, bg = [], [], []
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if not m:
        continue
    steps.append(int(m[1]))
    bl.append(ast.literal_eval(m[2])[0])
    bg.append(ast.literal_eval(m[3]))

print(f"parsed {len(steps)} samples from {LOG.name}")
print(f"steps: {steps}")

num_layers = len(bg[0])
bg_per_layer = list(zip(*bg))  # (num_layers, n_samples)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(steps, bl, marker="o", color="crimson", linewidth=2, label="byte_lambda")
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.set_xlabel("step")
ax1.set_ylabel("value")
ax1.set_title("byte_lambda over training")
ax1.grid(True, alpha=0.3)
ax1.legend()

cmap = plt.get_cmap("viridis", num_layers)
for i in range(num_layers):
    ax2.plot(steps, bg_per_layer[i], marker="o", color=cmap(i), linewidth=1.5, label=f"layer {i}")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.set_xlabel("step")
ax2.set_ylabel("value")
ax2.set_title(f"bigram_lambdas[i] over training ({num_layers} layers)")
ax2.grid(True, alpha=0.3)
ax2.legend(ncol=2, fontsize=8)

fig.suptitle("Injection-coefficient trajectories — W=8, D=32, pad=left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")

# summary table
print("\nfinal values (step {}):".format(steps[-1]))
print(f"  byte_lambda        = {bl[-1]:+.4f}  (init {bl[0]:+.4f})")
for i in range(num_layers):
    print(f"  bigram_lambdas[{i:2d}] = {bg_per_layer[i][-1]:+.4f}  (init {bg_per_layer[i][0]:+.4f})")
