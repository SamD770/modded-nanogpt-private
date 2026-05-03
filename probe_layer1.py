"""Analyze layer-1 component norms/cosines captured by train_gpt_medium.py PROBE_NORMS=1.

Usage:
    python probe_layer1.py <probe_pt_file>
"""
import sys
from pathlib import Path

import torch

PROBE = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/medium_runs/probe_layer1.pt")
data = torch.load(PROBE, map_location="cpu", weights_only=False)
records = data["records"]
print(f"loaded {len(records)} forward-pass records from {PROBE}")
print(f"checkpoint: {data.get('ckpt_path', '?')}")

# Each record: {"norms": {name: (1, T)}, "cosines": {name: (1, T)}, "x_norm": (1, T)}
names = list(records[0]["norms"].keys())

# Concatenate per-token stats across all records
norms = {n: torch.cat([r["norms"][n].flatten().float() for r in records]) for n in names}
cosines = {n: torch.cat([r["cosines"][n].flatten().float() for r in records]) for n in names}
x_norm = torch.cat([r["x_norm"].flatten().float() for r in records])
n_tokens = x_norm.numel()
print(f"aggregate over {n_tokens:,} tokens\n")

# Share = mean_t ||c_i[t]|| / mean_t Σ_j ||c_j[t]||
sum_norms = sum(norms[n] for n in names)  # per-token sum of magnitudes
shares = {n: (norms[n] / sum_norms.clamp_min(1e-12)).mean().item() for n in names}

print(f"{'component':<14}{'mean_norm':>12}{'norm_share':>12}{'cos(c_i,x)':>14}{'|cos|':>10}")
print("-" * 62)
for n in names:
    mn = norms[n].mean().item()
    sh = shares[n]
    co = cosines[n].mean().item()
    aco = cosines[n].abs().mean().item()
    print(f"{n:<14}{mn:>12.3f}{100*sh:>11.2f}%{co:>14.3f}{aco:>10.3f}")
print("-" * 62)
print(f"{'x (sum)':<14}{x_norm.mean().item():>12.3f}{100.0:>11.2f}%")
print()

# Variance breakdown (||c||^2 share, more honest when components partially cancel)
sq_norms = {n: norms[n].pow(2) for n in names}
sum_sq = sum(sq_norms[n] for n in names)
energy_share = {n: (sq_norms[n] / sum_sq.clamp_min(1e-12)).mean().item() for n in names}
print("energy share (||c_i||^2 / Σ ||c_j||^2):")
for n in names:
    print(f"  {n:<14}{100*energy_share[n]:>6.2f}%")
print()

# Sanity: ||x||^2 vs Σ ||c_i||^2 (ratio < 1 means components cancel)
ratio = (x_norm.pow(2) / sum_sq.clamp_min(1e-12)).mean().item()
print(f"mean(||x||^2 / Σ||c_i||^2) = {ratio:.3f}  "
      f"(<1 → components partially cancel; =1 → orthogonal; >1 → constructive)")
