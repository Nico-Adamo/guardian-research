# guardian-research

Experiment factory for investigating **catapult / grokking training dynamics**,
**dynamic grokking** (test-time "pondering"), and **Guardian-Angel-style
personalization**. See `planning/guardian/` for the source ideas and
`planning/hypotheses/` for the falsifiable claims this repo is built to test.

> This README is a placeholder; the full version is written by the docs workstream.

Design in one line: **local machine = control plane, cloud GPUs = disposable
stateless workers, the repo = source of truth.**

```bash
uv sync          # create the environment
just test        # or: make test
just smoke       # or: make smoke  — a CPU-only toy training run
```
