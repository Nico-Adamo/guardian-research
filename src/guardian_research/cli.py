"""``ga`` — the Guardian research CLI.

A thin dispatcher: it discovers command modules in ``guardian_research.commands``
(each exposes ``NAME``, ``HELP``, and ``run(argv) -> int``) and routes to them.
Auto-discovery means new commands (dynamic grokking, persona, ...) register
themselves just by adding a module — no edits here.

Commands mix ordinary ``--flags`` with Hydra overrides (``+exp=...``, ``seed=0``);
the per-command modules handle the split via ``common.hydra_utils``.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

from . import commands as commands_pkg
from .common.logging import console, get_logger


def _load_dotenv() -> None:
    """Load .env from the repo root into os.environ (won't overwrite existing)."""
    root = Path(__file__).resolve().parents[2]
    env_file = root / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)

log = get_logger("ga")


def discover_commands() -> dict[str, ModuleType]:
    registry: dict[str, ModuleType] = {}
    for mod in pkgutil.iter_modules(commands_pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"{commands_pkg.__name__}.{mod.name}")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load command module %s: %s", mod.name, exc)
            continue
        name = getattr(m, "NAME", None)
        if name and hasattr(m, "run"):
            registry[name] = m
    return registry


def _print_help(registry: dict[str, ModuleType]) -> None:
    console.print("[bold]ga[/bold] — Guardian research experiment factory\n")
    console.print("Usage: [bold]ga <command> [args...][/bold]\n")
    console.print("Commands:")
    for name in sorted(registry):
        help_text = getattr(registry[name], "HELP", "")
        console.print(f"  [cyan]{name:<18}[/cyan] {help_text}")
    console.print("\nExamples:")
    console.print("  ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0")
    console.print("  ga analyze --experiment arithmetic_catapult --write reports/runs/arith.md")
    console.print("  ga launch --dry-run --provider skypilot +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0")
    console.print("  ga propose --experiment arithmetic_catapult --budget-usd 25 --write reports/proposals/next.yaml")
    console.print("  ga validate-proposal reports/proposals/next.yaml")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    registry = discover_commands()

    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(registry)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd not in registry:
        console.print(f"[red]unknown command:[/red] {cmd}")
        _print_help(registry)
        return 2

    try:
        return int(registry[cmd].run(rest) or 0)
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
