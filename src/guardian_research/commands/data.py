"""`ga data` — generate/inspect synthetic datasets (no private data, ever)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.logging import console
from ..data.arithmetic import ArithmeticConfig, build_splits

NAME = "data"
HELP = "Generate synthetic data, e.g. ga data arithmetic --digits-train 1-3 --digits-hard 4-5 --out data/processed/arith_v0"


def _digits(spec: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        if "-" in spec:
            a, b = spec.split("-")
            return int(a), int(b)
        return int(spec), int(spec)
    except Exception:
        return default


def run(argv: list[str]) -> int:
    if not argv or argv[0] != "arithmetic":
        console.print("usage: [bold]ga data arithmetic [--digits-train 1-3] [--digits-hard 4-5] "
                      "[--op +] [--carry-heavy 0.5] [--out DIR][/bold]")
        return 2
    ap = argparse.ArgumentParser(prog="ga data arithmetic", add_help=True)
    ap.add_argument("--digits-train", default="1-3")
    ap.add_argument("--digits-hard", default="4-5")
    ap.add_argument("--op", default="+")
    ap.add_argument("--carry-heavy", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/arith_v0")
    args = ap.parse_args(argv[1:])

    tr_min, tr_max = _digits(args.digits_train, (1, 3))
    hd_min, hd_max = _digits(args.digits_hard, (4, 5))
    cfg = ArithmeticConfig(
        op=args.op,
        train_min_digits=tr_min,
        train_max_digits=tr_max,
        hard_min_digits=hd_min,
        hard_max_digits=hd_max,
        n_train=args.n_train,
        carry_heavy_frac=args.carry_heavy,
        seed=args.seed,
    )
    splits = build_splits(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Manifest + a small human-readable sample of each split.
    manifest = {"config": cfg.__dict__, "meta": splits["meta"], "data_class": "synthetic"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for split_name in ("train", "easy_eval", "hard_eval"):
        ds = splits[split_name]
        sample = [f"{p}{a}" for p, a in (ds.text(i) for i in range(min(20, len(ds))))]
        (out / f"{split_name}.sample.txt").write_text("\n".join(sample))

    console.print(f"[green]✓ generated synthetic arithmetic[/green] → {out}")
    console.print(f"  {splits['meta']}")
    return 0
