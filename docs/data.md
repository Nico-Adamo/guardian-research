# Datasets

All datasets in this repo are **synthetic**. See `data/README.md` for the on-disk
layout and the (optional) DVC versioning convention. This document covers *how to
generate* datasets and the policies that keep the project safe and reproducible.

## Synthetic-only policy

> No private data is ever generated, committed, or uploaded here.

- Training and evaluation data are produced by **seeded code**, not by copying
  real corpora.
- A disposable cloud worker regenerates whatever it needs from the same seeded
  code after checking out an exact git SHA. It never receives a local data file.
- Private persona data (if it ever exists outside this repo) is gated behind
  `GUARDIAN_ALLOW_PRIVATE_PERSONA_DATA` (default `false`) plus an approved
  encryption plan — see `docs/security.md`. The default code paths never touch it,
  and synthetic personas live under `data/persona_synthetic/`.

## Generating the arithmetic dataset

The arithmetic "catapult / grokking" vertical uses a generated character-level
addition task. Generate it with the `ga data` command
(`src/guardian_research/commands/data.py`, backed by
`src/guardian_research/data/arithmetic.py`):

```bash
uv run ga data arithmetic \
  --digits-train 1-3 \
  --digits-hard 4-5 \
  --op + \
  --carry-heavy 0.5 \
  --n-train 2000 \
  --seed 0 \
  --out data/processed/arith_v0
```

Flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `--digits-train` | digit range for the easy/train distribution (`a-b` or `n`) | `1-3` |
| `--digits-hard` | digit range for the out-of-distribution HARD eval | `4-5` |
| `--op` | binary operator | `+` |
| `--carry-heavy` | fraction of HARD examples that are carry-heavy | `0.5` |
| `--n-train` | number of training examples | `2000` |
| `--seed` | RNG seed (reproducibility) | `0` |
| `--out` | output directory under `data/processed/` | `data/processed/arith_v0` |

This writes into `--out`:

- `manifest.json` — the exact `ArithmeticConfig`, split `meta`, and
  `data_class: "synthetic"`. This makes the dataset self-describing.
- `train.sample.txt`, `easy_eval.sample.txt`, `hard_eval.sample.txt` — small
  human-readable samples of each split for inspection.

The full splits themselves are built deterministically in memory by
`build_splits(cfg)` (`CharTokenizer`, `ArithmeticConfig`), so training does not
require any persisted dataset file — the manifest + seed are enough to recreate
it. Note `data/processed/` is gitignored (see `data/README.md`); the committed,
inspectable trace is the small sample files only if you choose to track them via
DVC, plus the manifest's seeded config.

> No GPU, no network, and no model download is needed to generate data — it is
> pure CPU code, consistent with the repo's offline-by-default test/toy posture.

## Reproducibility: seed + git SHA

Two coordinates fully pin any dataset (and any run that consumes it):

1. **Seed** — every generator and training run flows through
   `seed_everything(seed)` (`src/guardian_research/common/seeding.py`), and the
   seed is recorded in both the dataset `manifest.json` and each run's
   `results.json`.
2. **Git SHA** — each run's `results.json` (via `RunWriter`) records the git SHA
   and whether the working tree was dirty. Cloud launches check out an **exact
   SHA**, and proposals (`ga propose`) carry `git_sha` / `requires_exact_sha`.

So "reproduce dataset X" reduces to: *check out the SHA, run the same `ga data`
command with the same seed.* You get byte-identical synthetic data without
shipping the data itself. This is also why DVC is optional rather than required
(see `data/README.md`): the seeded code is the canonical reproduction path.

## Synthetic personas

The persona personalization vertical
(`src/guardian_research/experiments/persona/`, runner key `persona`) is
**synthetic-only by default**: any persona corpora it uses are generated, live
under `data/persona_synthetic/`, and are safe to inspect and commit. Real persona
data is never a default code path and is gated as described above.
