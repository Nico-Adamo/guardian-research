# `data/` — synthetic-only, DVC-compatible layout

This directory holds **datasets and dataset caches**. The hard rule for this repo:

> **No private data is ever committed to git or uploaded to a cloud worker.**
> Everything under `data/` that ships is **synthetic**. Real persona data, if it
> ever exists, lives only on an operator's machine under a path that is
> gitignored, and is gated behind an explicit opt-in (see `docs/security.md`).

The control plane (your laptop) is the source of truth. Cloud GPU workers are
disposable: they `git clone` an exact SHA, regenerate any synthetic data they
need from seeded code, run, and die. They never receive a local data file.

## Layout

```
data/
  raw/                # immutable inputs (downloaded/generated upstream sources)
  processed/          # tokenized / split-ready datasets produced by `ga data`
  splits/             # explicit train / eval split manifests
  persona_synthetic/  # synthetic persona corpora ONLY (safe to inspect, never real)
```

| Path | What goes here | git | DVC |
|------|----------------|-----|-----|
| `data/raw/` | immutable source inputs | ignored | trackable |
| `data/processed/` | generated datasets + `manifest.json` + `*.sample.txt` | ignored | trackable |
| `data/splits/` | split manifests | ignored | trackable |
| `data/persona_synthetic/` | synthetic personas only | **not ignored** (small, safe) | optional |
| `data/persona_private/` | (never created here) real persona data | **always ignored** | never |

The `manifest.json` written by `ga data arithmetic` records `data_class:
"synthetic"`, the exact `ArithmeticConfig`, and the split `meta` — so a dataset
is self-describing and reproducible from its seed.

## What git ignores vs. what DVC tracks

`.gitignore` deliberately ignores the large, re-creatable caches so the repo
stays small and never accidentally carries data:

```gitignore
/data/raw/
/data/processed/
/data/splits/
...
/data/persona_private/      # private persona data: never, under any circumstances
*.principal.private.md
!**/.gitkeep
```

Note that `data/persona_synthetic/` is **intentionally not ignored**: synthetic
personas are safe to commit and inspect, and committing a small synthetic corpus
makes a result reproducible without any external store.

These ignored directories are **DVC-tracked, not git-tracked**. DVC (Data Version
Control) stores small pointer files (`*.dvc`) in git while the actual bytes live
in a DVC cache / remote. This gives you content-addressed, versioned data without
bloating the git history.

## Using DVC — without making it a hard dependency

DVC is **optional**. None of the code paths, tests, or toys import or require it;
everything here works with DVC absent. Datasets are always regenerable from
seeded code (`ga data ...`), so DVC is a *convenience for caching/sharing large
artifacts*, not a correctness requirement.

If you do want content-addressed data versioning:

```bash
# one-time, only if you want DVC:
uv pip install dvc            # or: pipx install dvc   (NOT in default deps)
dvc init

# track a generated dataset (writes data/processed/arith_v0.dvc into git):
dvc add data/processed/arith_v0
git add data/processed/arith_v0.dvc data/.gitignore

# optional: push the bytes to a remote you control (never a public bucket of private data)
dvc remote add -d store s3://your-bucket/guardian-dvc
dvc push
```

The committed `*.dvc` pointer is tiny and contains only a content hash + path —
no data. A teammate runs `dvc pull` (or just re-runs `ga data ...`) to materialize
the bytes.

`.gitignore` already keeps DVC's own caches out of git:

```gitignore
/.dvc/cache/
/.dvc/tmp/
```

## The non-negotiable rule

- **Synthetic only** in this repo and on any worker.
- **Private persona data is never committed and never uploaded.** It is enabled
  only by the `GUARDIAN_ALLOW_PRIVATE_PERSONA_DATA` gate (default `false`) and an
  approved encryption plan — see `docs/security.md` and `docs/data.md`.
- Generating a dataset is reproducible from a **seed + git SHA**; that is the
  authoritative way to recreate any `data/` contents, with or without DVC.
