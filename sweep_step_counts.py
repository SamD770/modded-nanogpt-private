"""
Sweep driver over total training step count, holding byte convolution knobs fixed
at byte_window_size=8, byte_embedding_dim=32.

total_steps = NUM_SCHEDULED_ITERATIONS + NUM_EXTENSION_ITERATIONS.
This sweep varies NUM_SCHEDULED_ITERATIONS and keeps NUM_EXTENSION_ITERATIONS at 40,
so total_steps = scheduled + 40. All defaults stay below 1280.

Usage:
    python sweep_step_counts.py
    python sweep_step_counts.py --totals 400,800,1200
    python sweep_step_counts.py --dry-run
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "sweep_logs"
CSV_PATH = REPO / "sweep_step_counts.csv"
RESULT_LINE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<val_loss>[\d.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[\d.]+)ms"
)
PEAK_MEM_LINE = re.compile(r"peak memory allocated:\s*(?P<alloc>\d+)\s*MiB\s*reserved:\s*(?P<reserved>\d+)\s*MiB")

BYTE_WINDOW_SIZE = 8
BYTE_EMBEDDING_DIM = 32
EXTENSION_ITERS = 40

FIELDS = [
    "total_steps", "num_scheduled_iterations", "num_extension_iterations",
    "byte_window_size", "byte_embedding_dim",
    "val_loss", "train_time_ms", "step_avg_ms",
    "peak_alloc_mib", "peak_reserved_mib",
    "wallclock_s", "status", "log_path",
]


def load_done() -> set[int]:
    if not CSV_PATH.exists():
        return set()
    done = set()
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            done.add(int(row["total_steps"]))
    return done


def append_csv(row: dict):
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def parse_log(text: str) -> dict:
    result = {}
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


def run_one(total_steps: int) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    scheduled = total_steps - EXTENSION_ITERS
    assert scheduled > 0, f"total_steps {total_steps} must exceed EXTENSION_ITERS {EXTENSION_ITERS}"

    tag = f"steps{total_steps}_w{BYTE_WINDOW_SIZE}_d{BYTE_EMBEDDING_DIM}"
    log_path = LOG_DIR / f"{tag}_{int(time.time())}.log"

    env = os.environ.copy()
    env["BYTE_WINDOW_SIZE"] = str(BYTE_WINDOW_SIZE)
    env["BYTE_EMBEDDING_DIM"] = str(BYTE_EMBEDDING_DIM)
    env["NUM_SCHEDULED_ITERATIONS"] = str(scheduled)
    env["NUM_EXTENSION_ITERATIONS"] = str(EXTENSION_ITERS)

    print(f"\n=== RUN {tag} (scheduled={scheduled} + ext={EXTENSION_ITERS}) -> {log_path.name} ===", flush=True)
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
        "total_steps": total_steps,
        "num_scheduled_iterations": scheduled,
        "num_extension_iterations": EXTENSION_ITERS,
        "byte_window_size": BYTE_WINDOW_SIZE,
        "byte_embedding_dim": BYTE_EMBEDDING_DIM,
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
    ap.add_argument("--totals", default="1280,1270,1260,1250,1240,1230,1220,1210,1200",
                    help="comma-separated total_steps values, swept in given order")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run totals already in CSV")
    a = ap.parse_args()

    totals = [int(x) for x in a.totals.split(",")]
    for t in totals:
        if t <= EXTENSION_ITERS:
            sys.exit(f"total_steps {t} must exceed extension iters {EXTENSION_ITERS}")

    done = set() if a.force else load_done()
    pending = [t for t in totals if t not in done]

    print(f"Totals: {totals}  already done: {len(totals) - len(pending)}  to run: {len(pending)}")
    for t in pending:
        print(f"  pending: total_steps={t}  (scheduled={t - EXTENSION_ITERS} + ext={EXTENSION_ITERS})")

    if a.dry_run:
        return

    for t in pending:
        try:
            run_one(t)
        except KeyboardInterrupt:
            print("Interrupted; CSV has results so far. Re-run to resume.", file=sys.stderr)
            sys.exit(130)


if __name__ == "__main__":
    main()
