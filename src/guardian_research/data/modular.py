"""Modular-arithmetic dataset — the canonical grokking testbed.

This is the setting from the original grokking paper (small algorithmic task,
heavy regularization, train on a *fraction* of all pairs): the model memorizes
the training pairs quickly (train_acc -> 1) while held-out accuracy stays at
chance, then *later* — if weight decay / schedule push it out of the
memorization basin — it "groks" the algorithm and val_acc jumps to 1.

Unlike base-10 addition (which the tiny model just learns + generalizes, leaving
no memorize-but-don't-generalize gap to study cheaply), modular arithmetic
reliably exhibits the memorize→generalize transition on a CPU, so the H001
question "do high-LR / cyclic-WD schedules change the transition?" is actually
*observable* here. Each number 0..p-1 is a single token (standard setup).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MOD_OPS = {
    "+": lambda a, b, p: (a + b) % p,
    "-": lambda a, b, p: (a - b) % p,
    "*": lambda a, b, p: (a * b) % p,
}


@dataclass
class ModularConfig:
    p: int = 97  # prime modulus
    op: str = "+"
    train_frac: float = 0.4
    seed: int = 0


def build_modular_splits(cfg: ModularConfig) -> dict[str, object]:
    """Build train/val tensors for ``(a op b) mod p``.

    Token ids: numbers 0..p-1 are themselves; op = p; eq = p+1.
    Each example is the 4-token sequence ``[a, op, b, eq]`` and the target is the
    answer token (a number in 0..p-1), predicted at the final (eq) position.
    """
    p = cfg.p
    op_id, eq_id = p, p + 1
    vocab_size = p + 2
    fn = MOD_OPS[cfg.op]

    pairs = [(a, b) for a in range(p) for b in range(p)]
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_train = int(round(len(pairs) * cfg.train_frac))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    def to_tensors(idxs):
        X = torch.empty((len(idxs), 4), dtype=torch.long)
        y = torch.empty((len(idxs),), dtype=torch.long)
        for row, i in enumerate(idxs):
            a, b = pairs[i]
            X[row] = torch.tensor([a, op_id, b, eq_id])
            y[row] = fn(a, b, p)
        return X, y

    Xtr, ytr = to_tensors(train_idx)
    Xval, yval = to_tensors(val_idx)
    return {
        "vocab_size": vocab_size,
        "seq_len": 4,
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xval,
        "y_val": yval,
        "meta": {
            "p": p,
            "op": cfg.op,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "chance_acc": 1.0 / p,
            "vocab_size": vocab_size,
        },
    }
