# sfcompute setup notes — modded-nanogpt

Notes from getting `train_gpt.py` running via the Dockerfile on an sfcompute 8×H100 instance (driver 565.57.01 / CUDA 12.7, no pre-installed CUDA toolkit, no Python 3.12).

## TL;DR

Two upstream issues block a fresh Docker run. Neither is documented in the README.

1. **Dockerfile is stale** — pinned to CUDA 12.6, but recent training code needs a CUDA 12.8+ intrinsic.
2. **The `varunneal/flash-attention-3` kernel has a broken non-relative import** that fails once `torch.compile` traces the backward under fake tensors.

With both patched, training runs on 8×H100 via:

```bash
sudo docker build -t modded-nanogpt .
sudo docker run -it --rm --gpus all \
    -v $(pwd):/modded-nanogpt \
    modded-nanogpt python data/cached_fineweb10B.py 8
sudo docker run -it --rm --gpus all --ipc=host \
    -v $(pwd):/modded-nanogpt \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    modded-nanogpt sh run.sh
```

(`--ipc=host` for NCCL shared memory; HF cache mount so the flash-attn patch persists.)

## Issue 1: CUDA 12.6 base image vs. `__tanhf` intrinsic

`triton_kernels.py:708` uses `__tanhf` (a fast-math PTX intrinsic). The declaration lives in NVRTC's built-in headers starting with **CUDA 12.8**.

PR [#251](https://github.com/KellerJordan/modded-nanogpt/pull/251) (*Fuse CE fwd and bwd*, merged 2026-04-04) added that kernel. The Dockerfile was last touched in commit `a5402c4 Update Dockerfile to CUDA 12.6` — nobody bumped it when #251 landed. Official records run on bare-metal PrimeIntellect boxes, so this regression went unnoticed.

**Failure mode:**
```
RuntimeError: Kernel compilation failed:
ce_fwd_bwd_kernel.cu(43): error: identifier "__tanhf" is undefined
```

**Fix (single line edit in `Dockerfile`):**
```diff
-RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
+RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

Base image (`nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`) does *not* need to change — `torch.cuda._compile_kernel` uses the NVRTC from the pip-installed `nvidia-cuda-nvrtc-cu12` wheel, not the system `/usr/local/cuda`. The cu128 torch wheel pulls in `nvidia-cuda-nvrtc-cu12==12.8.93`, which knows `__tanhf`.

Resulting stack: `torch-2.12.0.dev20260408+cu128`, `triton-3.7.0+git282c8251`, `nvidia-cuda-nvrtc-cu12-12.8.93`, Python 3.12.7.

Earlier Docker layers (Python build, requirements.txt) are cached, so the rebuild after the one-line change is ~2–3 min (vs ~20 min cold).

## Issue 2: flash-attn kernel's non-relative import

After the CUDA fix, training fails ~1 min into `torch.compile` warmup:

```
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
...  got ModuleNotFoundError("No module named 'flash_attn_config'")
```

Traced to `flash_attn_interface.py:25` inside `varunneal/flash-attention-3` (the FA3 kernel loaded at runtime via `kernels.get_kernel`):

```python
def round_up_headdim(head_size: int) -> int:
    from flash_attn_config import CONFIG   # <-- non-relative import
```

The kernel's package directory is only on `sys.path` transiently during initial load. When dynamo retraces the backward under fake tensors, the import fires in a context where that dir is no longer on `sys.path`.

**Fix** (in the cached kernel — the real fix is upstream in the `varunneal/flash-attention-3` HF repo):

```bash
sed -i 's/from flash_attn_config import CONFIG/from .flash_attn_config import CONFIG/' \
    /root/.cache/huggingface/hub/models--varunneal--flash-attention-3/snapshots/*/build/torch212-cxx11-cu128-x86_64-linux/flash_attention_3/flash_attn_interface.py
```

Since the fix is in `/root/.cache/huggingface`, the training container must mount that cache (`-v /root/.cache/huggingface:/root/.cache/huggingface`) — otherwise each `docker run --rm` re-downloads the unpatched kernel.

## Undocumented requirements in README

- No mention of Python version, CUDA version, driver version, or which PyTorch channel (`cu126` vs `cu128`) is needed.
- README advertises Docker as "recommended for precise timing" but doesn't note the Dockerfile can drift behind the training code.
- `--ipc=host` is not mentioned; needed for multi-GPU NCCL shared-memory allocation (torch will crash with `/dev/shm too small` warnings otherwise).
- HF cache mount is not mentioned; without it every run downloads ~800 MB of dynamic kernels.

## Suggested upstream PRs

1. Bump `Dockerfile` to `cu128` index-url (one line).
2. Fix `varunneal/flash-attention-3` to use `from .flash_attn_config import CONFIG`.
3. Add a small section to the README noting `--ipc=host` and HF cache mount for Docker.

## Image published

- `samd01/modded-nanogpt:cu128` — the rebuilt image (torch cu128 nightly). Does **not** bundle the HF kernel patch; follow the `sed` command above after first run.
