"""`ga grok` — run the dynamic-grokking ("pondering") toy locally.

Thin wrapper, identical in shape to ``commands/train.py``: compose a Hydra
config (so the user passes ``+exp=dynamic_grokking`` plus any overrides) and
hand it to ``run_local``. The interesting logic lives in the experiment runner
``experiments/dynamic_grokking/run.py``.
"""

from __future__ import annotations

from ..common.hydra_utils import compose_config, split_overrides, to_container
from ..common.logging import console
from ..launchers.local import run_local

NAME = "grok"
HELP = "Run the dynamic-grokking dynamic-eval toy locally, e.g. +exp=dynamic_grokking seed=0"


def run(argv: list[str]) -> int:
    flags, overrides = split_overrides(argv)
    if flags:
        console.print(f"[yellow]ignoring unrecognized flags for grok:[/yellow] {flags}")
    cfg = to_container(compose_config(overrides))
    if not cfg.get("runner"):
        console.print(
            "[red]No runner set.[/red] Pass the experiment, e.g. "
            "[cyan]ga grok +exp=dynamic_grokking seed=0[/cyan]"
        )
        return 2
    out = run_local(cfg)
    console.print(f"[green]✓ run complete[/green] → {out}")
    return 0
