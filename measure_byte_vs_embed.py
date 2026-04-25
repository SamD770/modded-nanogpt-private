"""Measure norm of byte-channel vs token-embedding contribution in the residual stream.

Loads a trained checkpoint and compares, per layer:
  * ||x0_lambdas[i] * x0||   — token-embedding contribution
  * ||byte_lambdas[i] * x_byte|| — byte-conv contribution
where x0 is the RMS-normalized token embedding (||x0|| = sqrt(model_dim) per token)
and x_byte = byte_proj(byte_embed(byte_table[tokens]).flatten(1)).
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class ParamConfig:  # stub so torch.load can unpickle the optimizer section (we don't use it)
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2:
            _, slots = state
            if slots:
                for k, v in slots.items():
                    object.__setattr__(self, k, v)
    def __reduce__(self):
        return (self.__class__, (), self.__dict__)

# Inlined from train_gpt.py (importing it triggers distributed-init side effects).
def _build_token_byte_table(vocab_size, window, pad_side="left"):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    table = torch.zeros(vocab_size, window, dtype=torch.uint8)
    for tok in range(enc.n_vocab):
        raw = enc.decode_single_token_bytes(tok)
        if not raw:
            continue
        if pad_side == "left":
            b = raw[-window:]
            table[tok, window - len(b):] = torch.frombuffer(bytearray(b), dtype=torch.uint8)
        else:
            b = raw[:window]
            table[tok, :len(b)] = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    return table


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520 and header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

DEFAULT_CKPT = "logs/54378d35-1b01-43ae-91da-df608aa30390/state_step001480.pt"
DEFAULT_SHARD = "data/fineweb10B/fineweb_val_000000.bin"
N_SAMPLE = 65536

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
print(f"Loading checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()}

embed_w      = sd["embed.weight"].float()        # (vocab, model_dim)
byte_embed_w = sd["byte_embed.weight"].float()   # (256, E)
byte_proj_w  = sd["byte_proj.weight"].float()    # (model_dim, W*E)
byte_lambdas = sd["byte_lambdas"].float()        # (num_layers,)
x0_lambdas   = sd["x0_lambdas"].float()          # (num_layers,)
bigram_lmbd  = sd["bigram_lambdas"].float()      # (num_layers,)

vocab_size, model_dim = embed_w.shape
num_bytes, emb_dim = byte_embed_w.shape
out_dim, inp_dim = byte_proj_w.shape
window = inp_dim // emb_dim
num_layers = byte_lambdas.numel()

byte_pad_side = os.environ.get("BYTE_PAD_SIDE", "left")
print(f"vocab_size={vocab_size} model_dim={model_dim} byte_window={window} "
      f"byte_embed_dim={emb_dim} num_layers={num_layers} pad_side={byte_pad_side}")

shard_path = Path(DEFAULT_SHARD)
if shard_path.exists():
    toks = _load_data_shard(shard_path)[:N_SAMPLE].to(torch.int64)
    print(f"Sampled {toks.numel()} tokens from {shard_path}")
else:
    toks = torch.arange(vocab_size, dtype=torch.int64)
    print(f"No val shard at {shard_path}; falling back to full vocab ({vocab_size} tokens)")

byte_table = _build_token_byte_table(vocab_size, window, byte_pad_side).to(torch.int64)

byte_ids = byte_table[toks]                                     # (T, W)
bemb = F.embedding(byte_ids, byte_embed_w).flatten(1)           # (T, W*E)
x_byte = F.linear(bemb, byte_proj_w)                            # (T, model_dim)

x_byte_norms = x_byte.norm(dim=-1)                              # (T,)
E_byte = x_byte_norms.mean().item()
x0_norm = math.sqrt(model_dim)

print()
print(f"mean ||x_byte||   = {E_byte:.4f}   (std {x_byte_norms.std().item():.4f})")
print(f"     ||x0||       = sqrt(model_dim) = {x0_norm:.4f}")
print(f"raw ratio ||x_byte|| / ||x0|| = {E_byte / x0_norm:.4f}  (lambdas not applied)")

print()
print(f"{'layer':>5} | {'x0_lambda':>10} | {'byte_lambda':>11} | "
      f"{'||x0*lam||':>11} | {'||byte*lam||':>12} | {'byte/embed':>10}")
print("-" * 78)
for i in range(num_layers):
    e = abs(x0_lambdas[i].item()) * x0_norm
    b = abs(byte_lambdas[i].item()) * E_byte
    ratio = b / e if e > 0 else float("inf")
    print(f"{i:>5d} | {x0_lambdas[i].item():>10.4f} | {byte_lambdas[i].item():>11.4f} | "
          f"{e:>11.4f} | {b:>12.4f} | {ratio:>10.4f}")

print()
print("Layer-0 seed (pre-attention): x = x0 + bigram_lambdas[0]*x0_bigram + byte_lambdas[0]*x_byte")
print(f"  ||x0||                   = {x0_norm:.4f}")
print(f"  ||byte_lambdas[0]*x_byte|| = {abs(byte_lambdas[0].item()) * E_byte:.4f}  "
      f"(byte_lambdas[0]={byte_lambdas[0].item():.4f})")
print(f"  bigram_lambdas[0]        = {bigram_lmbd[0].item():.4f}  "
      f"(bigram norm not measured — would require bigram input tensor)")
