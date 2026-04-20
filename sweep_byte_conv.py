"""
Sweep driver for token-strided byte convolution hyperparameters.

Runs ./run.sh once per (byte_window_size, byte_embedding_dim, byte_pad_side) combo,
overriding each knob via env var (read by train_gpt.py's Hyperparameters). Captures
stdout/stderr into per-run files under sweep_logs/ and appends final val_loss +
wall-clock fields to sweep_results.csv.

Resumable: rows already present in sweep_results.csv are skipped.

Usage:
    python sweep_byte_conv.py                      # full 3x3x2 = 18 run sweep
    python sweep_byte_conv.py --windows 4,8        # subset
    python sweep_byte_conv.py --dry-run            # print what would run
"""
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "sweep_logs"
CSV_PATH = REPO / "sweep_results.csv"
RESULT_LINE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<val_loss>[\d.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[\d.]+)ms"
)
PEAK_MEM_LINE = re.compile(r"peak memory allocated:\s*(?P<alloc>\d+)\s*MiB\s*reserved:\s*(?P<reserved>\d+)\s*MiB")

FIELDS = [
    "byte_window_size", "byte_embedding_dim", "byte_pad_side",
    "val_loss", "train_time_ms", "step_avg_ms",
    "peak_alloc_mib", "peak_reserved_mib",
    "wallclock_s", "status", "log_path",
]


def load_done() -> set[tuple[int, int, str]]:
    if not CSV_PATH.exists():
        return set()
    done = set()
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            done.add((int(row["byte_window_size"]), int(row["byte_embedding_dim"]), row["byte_pad_side"]))
    return done


def append_csv(row: dict):
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def parse_log(text: str) -> dict:
    """Return {val_loss, train_time_ms, step_avg_ms, peak_alloc_mib, peak_reserved_mib}
    from the final matching lines, or {} on miss."""
    result = {}
    # Scan from the end; the last val_loss line is the final eval.
    for line in reversed(text.splitlines()):
        m = RESULT_LINE.search(line)
        if m:
            result["val_loss"] = float(m["val_loss"])
            result["train_time_ms"] = int(m["train_time_ms"])
            result["step_avg_ms"] = float(m["step_avg_ms"])
            break
    for line in reversed(text.splitlines()):
        m = PEAK_MEM_LINE.search(line)
        if m:
            result["peak_alloc_mib"] = int(m["alloc"])
            result["peak_reserved_mib"] = int(m["reserved"])
            break
    return result


def run_one(window: int, dim: int, pad: str) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    tag = f"w{window}_d{dim}_{pad}"
    log_path = LOG_DIR / f"{tag}_{int(time.time())}.log"

    env = os.environ.copy()
    env["BYTE_WINDOW_SIZE"] = str(window)
    env["BYTE_EMBEDDING_DIM"] = str(dim)
    env["BYTE_PAD_SIDE"] = pad

    print(f"\n=== RUN {tag} -> {log_path.name} ===", flush=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(
            ["bash", "run.sh"],
            cwd=REPO,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    wall = time.time() - t0

    text = log_path.read_text(errors="replace")
    parsed = parse_log(text)
    status = "ok" if proc.returncode == 0 and "val_loss" in parsed else f"fail_rc{proc.returncode}"

    row = {
        "byte_window_size": window,
        "byte_embedding_dim": dim,
        "byte_pad_side": pad,
        "val_loss": parsed.get("val_loss", ""),
        "train_time_ms": parsed.get("train_time_ms", ""),
        "step_avg_ms": parsed.get("step_avg_ms", ""),
        "peak_alloc_mib": parsed.get("peak_alloc_mib", ""),
        "peak_reserved_mib": parsed.get("peak_reserved_mib", ""),
        "wallclock_s": round(wall, 1),
        "status": status,
        "log_path": str(log_path.relative_to(REPO)),
    }
    append_csv(row)
    print(f"  -> status={status} val_loss={row['val_loss']} wall={wall:.0f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="4,8,16")
    ap.add_argument("--dims", default="16,32,64")
    ap.add_argument("--pads", default="left,right")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run combos already in CSV")
    a = ap.parse_args()

    windows = [int(x) for x in a.windows.split(",")]
    dims = [int(x) for x in a.dims.split(",")]
    pads = [x.strip() for x in a.pads.split(",")]
    combos = list(itertools.product(windows, dims, pads))

    done = set() if a.force else load_done()
    pending = [c for c in combos if c not in done]

    print(f"Total combos: {len(combos)}  already done: {len(combos) - len(pending)}  to run: {len(pending)}")
    for w, d, p in pending:
        print(f"  pending: window={w} dim={d} pad={p}")

    if a.dry_run:
        return

    for w, d, p in pending:
        try:
            run_one(w, d, p)
        except KeyboardInterrupt:
            print("Interrupted by user; CSV has results so far. Re-run to resume.", file=sys.stderr)
            sys.exit(130)


if __name__ == "__main__":
    main()
