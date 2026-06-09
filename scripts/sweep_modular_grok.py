#!/usr/bin/env python3
"""Modular-grokking CPU sweep for H001 go/no-go gate.

Sweeps schedule × train_frac × base_wd × seed on the (a+b) mod 97 task.
Key question: does any catapult schedule (cyclic_lr, cyclic_wd, onecycle)
grok *sooner* or *more robustly* than baseline_cosine?

The cyclic_weight_decay multiplier is co-tuned to base_wd so peak WD stays
in the 1–10 range (not the catastrophic wd_max_mult=20 from the first probe).

Tier 0: CPU only, no cost, no approval needed.
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

SCHEDULES = ["baseline_cosine", "onecycle_high_lr", "cyclic_lr", "cyclic_weight_decay"]
TRAIN_FRACS = [0.3, 0.4, 0.5]
BASE_WDS = [0.3, 1.0, 3.0]
SEEDS = [0, 1, 2]
MAX_STEPS = 10000
EVAL_EVERY = 200
LOG_EVERY = 2000

# Co-tune cyclic-WD multiplier so peak WD ≈ 3× base (not 20×).
WD_MAX_MULT_BY_BASE = {0.3: 5.0, 1.0: 3.0, 3.0: 2.0}


def build_cmd(schedule: str, train_frac: float, base_wd: float, seed: int) -> list[str]:
    cmd = [
        "uv", "run", "ga", "train",
        "+exp=arithmetic_modular_grok",
        f"schedule={schedule}",
        f"seed={seed}",
        f"data.train_frac={train_frac}",
        f"train.weight_decay={base_wd}",
        f"train.max_steps={MAX_STEPS}",
        f"train.eval_every={EVAL_EVERY}",
        f"train.log_every={LOG_EVERY}",
    ]
    if schedule == "cyclic_weight_decay":
        mult = WD_MAX_MULT_BY_BASE[base_wd]
        cmd.append(f"schedule.wd_max_mult={mult}")
        cmd.append("schedule.wd_min_mult=0.1")
    return cmd


def main():
    grid = list(itertools.product(SCHEDULES, TRAIN_FRACS, BASE_WDS, SEEDS))
    total = len(grid)
    print(f"Modular-grokking sweep: {total} runs ({len(SCHEDULES)} schedules × "
          f"{len(TRAIN_FRACS)} train_fracs × {len(BASE_WDS)} base_wds × {len(SEEDS)} seeds)")
    print(f"Steps per run: {MAX_STEPS}, eval every {EVAL_EVERY}")
    print()

    results = []
    failed = []
    for i, (sched, frac, wd, seed) in enumerate(grid, 1):
        label = f"[{i}/{total}] {sched} frac={frac} wd={wd} seed={seed}"
        print(f"▸ {label}", flush=True)
        cmd = build_cmd(sched, frac, wd, seed)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(f"  FAILED (rc={proc.returncode}): {proc.stderr[-200:]}")
                failed.append({"label": label, "error": proc.stderr[-500:]})
            else:
                print(f"  done")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>300s)")
            failed.append({"label": label, "error": "timeout"})

    print(f"\nSweep complete: {total - len(failed)}/{total} succeeded, {len(failed)} failed.")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  - {f['label']}: {f['error'][:100]}")


if __name__ == "__main__":
    main()
