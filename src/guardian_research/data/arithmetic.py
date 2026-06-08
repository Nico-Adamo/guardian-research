"""Synthetic arithmetic dataset with easy / hard splits.

The catapult thesis predicts that a high-LR / heavy-regularization recipe will
eventually *generalize* an arithmetic algorithm and beat a standard recipe on
**hard** held-out data even if it looks worse on average for a long time
("the curves cross"). To see that, the hard split must stress exactly what
memorization cannot fake:

* **out-of-distribution digit lengths** — operands longer than anything seen in
  training (the model must have learned carry propagation as an *algorithm*); and
* **carry-chain-heavy examples** — operand pairs engineered so that almost every
  digit position triggers a carry, producing long carry chains.

Everything is character-level so a tiny decoder-only transformer can learn it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed character vocabulary. Index 0/1/2 are reserved control tokens.
SPECIALS = ["<pad>", "<bos>", "<eos>"]
SYMBOLS = list("0123456789+-*=")


class CharTokenizer:
    def __init__(self) -> None:
        self.itos = SPECIALS + SYMBOLS
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            c = self.itos[i]
            if c in ("<pad>", "<bos>"):
                continue
            if c == "<eos>":
                break
            out.append(c)
        return "".join(out)


OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


@dataclass
class ArithmeticConfig:
    op: str = "+"
    train_min_digits: int = 1
    train_max_digits: int = 3
    hard_min_digits: int = 4  # strictly OOD: longer than train_max_digits
    hard_max_digits: int = 5
    n_train: int = 4000
    n_easy_eval: int = 256
    n_hard_eval: int = 256
    # The hard split is two distinct kinds of "hard" (per the catapult essay):
    #   * carry-chain-heavy examples at IN-DISTRIBUTION lengths (tests whether
    #     the model truly carries, not whether it memorized); and
    #   * out-of-distribution (longer) lengths (tests length extrapolation).
    hard_ood_frac: float = 0.5  # share of hard split that is OOD-length
    carry_heavy_frac: float = 0.5  # within the OOD portion, share that is carry-heavy
    reverse_answer: bool = True  # least-significant digit first eases learning
    seed: int = 0
    # Derived at build time; not user-set.
    max_len: int = field(default=0)


def _rand_int(n_digits: int, rng: random.Random) -> int:
    if n_digits <= 1:
        return rng.randint(0, 9)
    lo = 10 ** (n_digits - 1)
    hi = 10**n_digits - 1
    return rng.randint(lo, hi)


def _carry_heavy_pair(n_digits: int, rng: random.Random) -> tuple[int, int]:
    """Construct (a, b) so most digit-positions sum to >= 9 (forcing carries)."""
    da, db = [], []
    for pos in range(n_digits):
        leading = pos == n_digits - 1
        # Pick digit_a, then digit_b biased so a+b >= 9 (a carry at this position).
        a = rng.randint(1 if leading else 0, 9)
        need = max(0, 9 - a)
        b = rng.randint(min(need, 9), 9)
        da.append(a)
        db.append(b)
    a = int("".join(str(d) for d in reversed(da)))
    b = int("".join(str(d) for d in reversed(db)))
    return a, b


def make_example(a: int, b: int, op: str, reverse_answer: bool) -> tuple[str, str]:
    """Return (prompt, answer_text). prompt ends with '='; answer is the result."""
    result = OPS[op](a, b)
    answer = str(result)
    if reverse_answer:
        # Reverse only the magnitude digits, keep a leading '-' in front.
        sign = "-" if answer.startswith("-") else ""
        digits = answer[len(sign):]
        answer = sign + digits[::-1]
    prompt = f"{a}{op}{b}="
    return prompt, answer


def _gen_examples(
    cfg: ArithmeticConfig,
    n: int,
    min_d: int,
    max_d: int,
    rng: random.Random,
    carry_heavy_frac: float = 0.0,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < n and attempts < n * 50:
        attempts += 1
        nd = rng.randint(min_d, max_d)
        if carry_heavy_frac and rng.random() < carry_heavy_frac:
            a, b = _carry_heavy_pair(nd, rng)
        else:
            a, b = _rand_int(nd, rng), _rand_int(rng.randint(min_d, max_d), rng)
        if cfg.op == "-" and b > a:
            a, b = b, a  # keep subtraction non-negative for the toy
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def build_splits(cfg: ArithmeticConfig) -> dict[str, object]:
    """Build train / easy_eval / hard_eval splits + a tokenizer.

    Returns a dict with: tokenizer, max_len, and the three ``ArithmeticDataset``
    splits. ``hard_eval`` operands are disjoint from ``train`` by construction
    (different digit lengths and/or carry-heavy), the core requirement for
    measuring real generalization rather than memorization.
    """
    rng = random.Random(cfg.seed)
    tok = CharTokenizer()

    train_pairs = _gen_examples(cfg, cfg.n_train, cfg.train_min_digits, cfg.train_max_digits, rng)
    train_set = set(train_pairs)

    easy_pairs = [
        p
        for p in _gen_examples(cfg, cfg.n_easy_eval * 2, cfg.train_min_digits, cfg.train_max_digits, rng)
        if p not in train_set
    ][: cfg.n_easy_eval]

    # Hard split: OOD-length examples + carry-heavy examples at training lengths.
    n_ood = int(round(cfg.n_hard_eval * cfg.hard_ood_frac))
    n_carry = cfg.n_hard_eval - n_ood
    hard_ood_pairs = _gen_examples(
        cfg, n_ood, cfg.hard_min_digits, cfg.hard_max_digits, rng, carry_heavy_frac=cfg.carry_heavy_frac
    )
    hard_carry_pairs = [
        p
        for p in _gen_examples(
            cfg, n_carry * 2, cfg.train_min_digits, cfg.train_max_digits, rng, carry_heavy_frac=1.0
        )
        if p not in train_set
    ][:n_carry]
    hard_pairs = hard_ood_pairs + hard_carry_pairs

    def _len(pair: tuple[int, int]) -> int:
        prompt, answer = make_example(pair[0], pair[1], cfg.op, cfg.reverse_answer)
        return len(prompt) + len(answer) + 2  # bos + eos

    all_pairs = train_pairs + easy_pairs + hard_pairs
    max_len = max(_len(p) for p in all_pairs)
    cfg.max_len = max_len

    return {
        "tokenizer": tok,
        "max_len": max_len,
        "train": ArithmeticDataset(train_pairs, cfg, tok),
        "easy_eval": ArithmeticDataset(easy_pairs, cfg, tok),
        "hard_eval": ArithmeticDataset(hard_pairs, cfg, tok),
        "hard_ood_eval": ArithmeticDataset(hard_ood_pairs, cfg, tok) if hard_ood_pairs else None,
        "hard_carry_eval": ArithmeticDataset(hard_carry_pairs, cfg, tok) if hard_carry_pairs else None,
        "meta": {
            "n_train": len(train_pairs),
            "n_easy_eval": len(easy_pairs),
            "n_hard_eval": len(hard_pairs),
            "n_hard_ood": len(hard_ood_pairs),
            "n_hard_carry": len(hard_carry_pairs),
            "max_len": max_len,
            "vocab_size": tok.vocab_size,
        },
    }


class ArithmeticDataset(Dataset):
    """Char-level autoregressive arithmetic.

    Each item yields ``(input_ids, target_ids, loss_mask)`` where the loss mask
    is 1 only over the *answer* region (everything after ``=``), so the model is
    scored on producing the answer, not on copying the prompt.
    """

    def __init__(self, pairs: list[tuple[int, int]], cfg: ArithmeticConfig, tok: CharTokenizer):
        self.pairs = pairs
        self.cfg = cfg
        self.tok = tok
        self.max_len = cfg.max_len if cfg.max_len else self._compute_max_len()

    def _compute_max_len(self) -> int:
        return max(
            len(make_example(a, b, self.cfg.op, self.cfg.reverse_answer)[0])
            + len(make_example(a, b, self.cfg.op, self.cfg.reverse_answer)[1])
            + 2
            for a, b in self.pairs
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def text(self, idx: int) -> tuple[str, str]:
        a, b = self.pairs[idx]
        return make_example(a, b, self.cfg.op, self.cfg.reverse_answer)

    def __getitem__(self, idx: int):
        a, b = self.pairs[idx]
        prompt, answer = make_example(a, b, self.cfg.op, self.cfg.reverse_answer)
        seq = [self.tok.bos_id] + self.tok.encode(prompt + answer) + [self.tok.eos_id]
        seq = seq[: self.max_len]
        # Answer region = positions after '=' up to and including <eos>.
        prompt_len = 1 + len(prompt)  # bos + prompt chars (the '=' is the last prompt char)
        input_ids = seq[:-1]
        target_ids = seq[1:]
        loss_mask = [0] * len(target_ids)
        for i in range(len(target_ids)):
            # target position i predicts seq[i+1]; it's in the answer region if
            # the *target* token index (i+1) is past the prompt.
            if (i + 1) >= prompt_len:
                loss_mask[i] = 1
        # Pad to max_len - 1.
        pad_to = self.max_len - 1
        pad_n = pad_to - len(input_ids)
        if pad_n > 0:
            input_ids = input_ids + [self.tok.pad_id] * pad_n
            target_ids = target_ids + [self.tok.pad_id] * pad_n
            loss_mask = loss_mask + [0] * pad_n
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )

    def prompt_and_answer(self, idx: int) -> tuple[str, str]:
        return self.text(idx)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
