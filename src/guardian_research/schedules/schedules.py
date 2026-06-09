"""Learning-rate and weight-decay schedules.

These are the experimental knob of the catapult thesis. All schedules expose the
same interface: given ``(step, total_steps)`` return ``(lr, wd)``. The training
loop sets the optimizer's lr and (decoupled) weight_decay every step, so a single
loop can run any recipe.

Implemented:

* ``baseline_cosine``      — warmup + cosine decay; constant WD. The control.
* ``onecycle_high_lr``     — one up-then-down cycle to a *high* peak LR.
* ``cyclic_lr``            — N cosine LR cycles ("explore / exploit" repeatedly).
* ``cyclic_weight_decay``  — constant-ish LR, WD cycles low->high ("sleep" step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ScheduleConfig:
    name: str = "baseline_cosine"
    base_lr: float = 1e-3
    base_wd: float = 0.01
    warmup_frac: float = 0.05
    min_lr_mult: float = 0.1  # floor as a fraction of base_lr
    max_lr_mult: float = 1.0  # peak as a multiple of base_lr (onecycle/cyclic)
    n_cycles: int = 1
    wd_max_mult: float = 1.0  # cyclic WD peak as a multiple of base_wd
    wd_min_mult: float = 0.0  # cyclic WD floor

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScheduleConfig:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _warmup_factor(step: int, total: int, warmup_frac: float) -> float | None:
    warmup_steps = max(1, int(total * warmup_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return None


def lr_wd_at(step: int, total_steps: int, cfg: ScheduleConfig) -> tuple[float, float]:
    total = max(1, total_steps)
    step = min(step, total)
    name = cfg.name

    if name == "baseline_cosine":
        wu = _warmup_factor(step, total, cfg.warmup_frac)
        if wu is not None:
            return cfg.base_lr * wu, cfg.base_wd
        warmup_steps = max(1, int(total * cfg.warmup_frac))
        progress = (step - warmup_steps) / max(1, total - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        lr = cfg.base_lr * (cfg.min_lr_mult + (1 - cfg.min_lr_mult) * cos)
        return lr, cfg.base_wd

    if name == "onecycle_high_lr":
        # Triangular-ish one cycle: ramp up to peak in the first ~30%, anneal down.
        peak = cfg.base_lr * cfg.max_lr_mult
        up = 0.3
        if step / total < up:
            frac = (step / total) / up
            lr = cfg.base_lr * cfg.min_lr_mult + frac * (peak - cfg.base_lr * cfg.min_lr_mult)
        else:
            frac = (step / total - up) / (1 - up)
            cos = 0.5 * (1 + math.cos(math.pi * frac))
            lr = cfg.base_lr * cfg.min_lr_mult + cos * (peak - cfg.base_lr * cfg.min_lr_mult)
        return lr, cfg.base_wd

    if name == "cyclic_lr":
        peak = cfg.base_lr * cfg.max_lr_mult
        floor = cfg.base_lr * cfg.min_lr_mult
        cycle_len = total / max(1, cfg.n_cycles)
        phase = (step % cycle_len) / cycle_len  # 0..1 within a cycle
        # Cosine that starts high (explore) and decays to floor (exploit).
        cos = 0.5 * (1 + math.cos(math.pi * phase))
        lr = floor + cos * (peak - floor)
        return lr, cfg.base_wd

    if name in ("cyclic_weight_decay", "cyclic_wd_tuned"):
        wu = _warmup_factor(step, total, cfg.warmup_frac)
        lr = cfg.base_lr * wu if wu is not None else cfg.base_lr
        cycle_len = total / max(1, cfg.n_cycles)
        phase = (step % cycle_len) / cycle_len
        wd_lo = cfg.base_wd * cfg.wd_min_mult
        wd_hi = cfg.base_wd * cfg.wd_max_mult
        # Ramp WD up across each cycle, then reset (a periodic "sleep").
        wd = wd_lo + phase * (wd_hi - wd_lo)
        return lr, wd

    raise ValueError(f"unknown schedule '{name}'")


def get_schedule(cfg: ScheduleConfig):
    """Return a closure ``f(step, total_steps) -> (lr, wd)``."""

    def f(step: int, total_steps: int) -> tuple[float, float]:
        return lr_wd_at(step, total_steps, cfg)

    return f
