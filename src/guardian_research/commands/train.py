"""`ga train` — run one experiment locally from a composed Hydra config."""

from __future__ import annotations

from ..common.hydra_utils import compose_config, split_overrides, to_container
from ..common.logging import console
from ..launchers.local import run_local

NAME = "train"
HELP = "Train one experiment locally (Hydra overrides), e.g. +exp=arithmetic_catapult seed=0"


def run(argv: list[str]) -> int:
    flags, overrides = split_overrides(argv)
    if flags:
        console.print(f"[yellow]ignoring unrecognized flags for train:[/yellow] {flags}")
    cfg = to_container(compose_config(overrides))
    if not cfg.get("runner"):
        console.print("[red]No runner set.[/red] Pass an experiment, e.g. "
                      "[cyan]ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0[/cyan]")
        return 2
    out = run_local(cfg)
    console.print(f"[green]✓ run complete[/green] → {out}")
    return 0
