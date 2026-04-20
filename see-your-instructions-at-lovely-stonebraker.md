# Plan: Token-strided byte convolutions

## Context

Current input embedding in `train_gpt.py:1298-1318` is `x = embed(input_seq) + bigram_lambdas[0] * bigram_embed(bigram_input_seq)`. These views throw away per-token UTF-8 byte structure — e.g. "The " vs "the " share no signal at init, and features like "ends in whitespace" or "ends in -ing" are invisible to layer 0.

The user wants to add a cheap byte-level signal by:
1. Taking each token id's UTF-8 bytes (suffix-aligned in a fixed `byte_window_size=8` window),
2. Embedding each byte (256 × `byte_embedding_dim=32`),
3. Flattening and linearly projecting the `W * byte_embedding_dim` vector to `model_dim`,
4. Adding the result to the residual stream at layer 0 alongside the existing bigram injection.

Goal for this first pass: lower val loss at the current preprogrammed step count. Ablations (`byte_window_size`, `byte_embedding_dim`, pad side) come after the feature lands.

## User-confirmed design decisions

- **Injection site:** layer 0 only (same site as `x0_bigram`). No per-layer `byte_lambdas` vector — a single scalar `byte_lambda` parameter.
- **Pad direction:** left-pad / suffix-aligned (last byte of each token sits at `W-1`).
- **Table source:** GPU buffer built once at model init from tiktoken. No disk artifact, no change to `distributed_data_generator`, no CPU per-step work.

## Critical files

- `/root/modded-nanogpt-private/train_gpt.py` — single file where all changes land:
  - GPT `__init__` (~line 1229 onward): add byte embedding, projection, scalar lambda, and the precomputed byte table buffer.
  - GPT `forward` (~line 1298): compute `x_byte` and add it to the residual beside the bigram injection.
  - `Hyperparameters` dataclass (~line 1558): expose `byte_window_size` and `byte_embedding_dim`.
  - `args.bigram_vocab_size` site (~line 1576): add matching args entries.
  - `param_table` / `work_order` (~lines 1693-1719): register new parameter labels with sensible bank assignments.

## Implementation plan

### 1. Hyperparameters

Add to the `Hyperparameters` dataclass (the module-level `args` object built around line 1558):
- `byte_window_size: int = 8`
- `byte_embedding_dim: int = 32`

### 2. Precomputed byte table (model init)

Add a helper near the other data helpers:
```python
def _build_token_byte_table(vocab_size: int, window: int) -> torch.Tensor:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    table = torch.zeros(vocab_size, window, dtype=torch.uint8)
    for tok in range(enc.n_vocab):                   # 50257
        b = enc.decode_single_token_bytes(tok)[-window:]   # keep tail bytes
        if not b:
            continue
        table[tok, window - len(b):] = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    # ids in [enc.n_vocab, vocab_size) are unused padding; leave zero.
    return table
```

In `GPT.__init__` (near where `self.embed` / `self.bigram_embed` are created, line ~1229):
```python
self.byte_window_size = args.byte_window_size
self.byte_embed = nn.Embedding(256, args.byte_embedding_dim)
nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)

self.byte_proj = nn.Linear(
    args.byte_window_size * args.byte_embedding_dim, model_dim, bias=False,
)
nn.init.zeros_(self.byte_proj.weight)   # zero-init so training starts identical to baseline

self.register_buffer(
    "byte_table",
    _build_token_byte_table(self.vocab_size, args.byte_window_size).to(torch.int64),
    persistent=False,
)

self.byte_lambda = nn.Parameter(torch.tensor(0.05))
```

Notes:
- Store the table as `int64` on the GPU so the embedding lookup needs no per-step cast. Memory cost: `50304 * 8 * 8 = ~3MB`. Trivial.
- Zero-init `byte_proj` means the added signal is identically zero at step 0 — so this change is strictly additive and cannot regress the baseline at init (same trick used for `bigram_embed`).
- Suffix alignment is implemented by the `table[tok, window - len(b):]` slice.

### 3. Forward path

In `GPT.forward` around line 1315-1316:
```python
# existing bigram block
x0_bigram = self.bigram_embed(bigram_input_seq)[None]
# new byte block
byte_seq = self.byte_table[input_seq]                      # (T, W) int64
x_byte_emb = self.byte_embed(byte_seq)                     # (T, W, D_byte)
x_byte = self.byte_proj(x_byte_emb.flatten(1))[None]       # (1, T, model_dim)

# existing layer-0 injection, extended
x = x + x0_bigram * bigram_lambdas[0] + x_byte * self.byte_lambda
```

Keep `x0_inject` / per-layer bigram mixing untouched — byte signal is only added pre-layer-0.

### 4. Optimizer / parameter banks

Auto-labeling at line 1259 (`param.label = name.replace('.weight', '')`) will give the new params these labels:
- `byte_embed` — the nn.Embedding weight
- `byte_proj` — the Linear weight
- `byte_lambda` — scalar Parameter

Add to `self.param_table` (line 1693):
```python
"byte_embed":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.75, 0.95], "lr_mul": 75., "wd_mul": 5.0},
"byte_proj":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0, "wd_mul": 0.0},
"byte_lambda":  {"optim": "adam", "comms": "replicated", "adam_betas": [0.9,  0.95], "lr_mul": 1.0, "wd_mul": 0.0},
```
Rationale:
- `byte_embed` mirrors `bigram_embed`'s adam betas / lr_mul / wd_mul, but uses `replicated` (not `sharded_sparse`) — table is only 256×32=8k params, no need for sharded-sparse infra.
- `byte_proj` is a small dense matmul (256×768 ≈ 200k params), Adam replicated. Muon is overkill; Adam matches other small projections.
- `byte_lambda` is a scalar, mirrors `bigram_lambdas`.

Add to `self.work_order` (line 1714) in the "small, fast" prefix group:
```python
"scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank",
"post_lambdas", "x0_lambdas", "bigram_lambdas", "byte_lambda", "resid_lambdas",
"value_embeds", "bigram_embed", "byte_embed", "byte_proj",
"lm_head", "embed",
"qk_bank", "vo_bank", "mlp_bank",
```

### 5. No data pipeline changes

`distributed_data_generator` / `get_bigram_hash` untouched. The byte signal is derived purely from `input_seq` on GPU inside the model — this satisfies rule #1 ("Not modify the train or validation data pipelines").

## Speed considerations (flagged per user request)

- **GPU gather** `byte_table[input_seq]`: one fused kernel, contiguous rows; effectively free at this scale.
- **Byte embed + flatten + linear:** for `model_dim=768`, per-token work is `W*D_byte + W*D_byte*model_dim ≈ 256 + 200k ≈ 200k FLOPs`, << attention.
- **No CPU work.** The table is built once in init; lookup happens on GPU.
- **torch.compile:** all ops (gather, embedding, linear, flatten) are compile-safe.
- **Init overhead:** `_build_token_byte_table` runs tiktoken 50257 times at startup. Likely <1s; if noticeable, cache it to `/tmp/gpt2_token_bytes_W{W}.pt` keyed by `W`. Not required for first iteration.

## Verification

1. **Shape smoke test** (fast): launch `python -c` that instantiates the `GPT` with `byte_window_size=8, byte_embedding_dim=32`, feeds a tiny `input_seq`, and asserts `x_byte.shape == (1, T, model_dim)` and a forward pass runs end-to-end. Confirms buffer registration, device placement, and dtype.
2. **Param-bank wiring check:** after model init, iterate `named_parameters()` and confirm `byte_embed`, `byte_proj`, `byte_lambda` appear in the optimizer groups with the expected lr_mul / wd_mul.
3. **Baseline preservation at step 0:** because `byte_proj.weight` is zero-init, the loss at step 0 should match a no-byte baseline to within floating-point noise. Run ~5 steps on both branches and diff the loss curves.
4. **Short training run:** run `./run.sh` with a reduced step count (e.g., `num_iterations` cut to ~200) as a sanity check that training is stable and wallclock per step hasn't regressed.
5. **Full run at the preprogrammed step count** via `./run.sh`; compare final val loss to the latest baseline log under `records/track_1_short/`.

## Follow-ups (out of scope for this PR)

- Sweep `byte_window_size ∈ {4, 8, 16}` and `byte_embedding_dim ∈ {16, 32, 64}`.
- Ablate pad side (right-pad variant).
- If val loss improves, consider per-layer `byte_lambdas` like `bigram_lambdas`.
- If startup cost from tiktoken shows up, cache the table to disk.
