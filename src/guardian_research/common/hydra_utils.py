"""Hydra composition helpers + argv splitting.

We deliberately do NOT use the ``@hydra.main`` decorator (it hijacks argv and a
single config). Instead the ``ga`` CLI dispatches to subcommands and uses
Hydra's *compose API* so that commands can mix ordinary ``--flags`` with Hydra
overrides like ``+exp=arithmetic_catapult model=tiny_transformer seed=0``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from .paths import conf_dir


def is_override_token(tok: str) -> bool:
    """True if ``tok`` looks like a Hydra override rather than an argparse flag.

    Examples that ARE overrides: ``seed=0``, ``+exp=foo``, ``++a.b=1``, ``~x``,
    ``model=tiny_transformer``. Examples that are NOT: ``--dry-run``,
    ``--provider``, ``skypilot`` (a flag *value*).
    """
    if tok.startswith("-"):
        return False
    return ("=" in tok) or (tok[:1] in "+~")


def split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    """Partition argv into (argparse_args, hydra_overrides), order-preserving."""
    flags: list[str] = []
    overrides: list[str] = []
    for tok in argv:
        (overrides if is_override_token(tok) else flags).append(tok)
    return flags, overrides


def compose_config(overrides: list[str], config_name: str = "config") -> DictConfig:
    """Compose a config from ``conf/`` with the given Hydra overrides."""
    # Imported lazily so that non-Hydra commands stay fast and import-light.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a config to a plain, JSON-serialisable dict."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def try_compose(overrides: list[str], config_name: str = "config") -> tuple[bool, str]:
    """Return (ok, message) — used to validate that a proposal's config composes."""
    try:
        compose_config(overrides, config_name=config_name)
        return True, "composed OK"
    except Exception as exc:  # noqa: BLE001 — we want the message
        return False, f"{type(exc).__name__}: {exc}"
