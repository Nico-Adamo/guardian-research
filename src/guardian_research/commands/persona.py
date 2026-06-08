"""`ga persona` — Guardian-Angel personalization on a SYNTHETIC persona corpus.

Subcommands:
  ga persona prepare [--out DIR] [--n-personas 3] [--docs 12] [--seed 0]
      Generate the fictional persona corpus (docs, summaries, Q&A, PRINCIPAL.md).
  ga persona eval [--out DIR] [--prepare]
      Run authorship + preference + reconstruction evals across all variants and
      print the comparison table.
  ga persona questions [--out DIR] [--top-k 3]
      Print the top active-learning questions per persona (transparent heuristic).
  ga persona run [Hydra overrides...]
      Run the full comparison runner via the local launcher (writes a RunResult).
      e.g. `ga persona run +exp=persona_dynamic_eval seed=0`

No private data, no model downloads — everything is synthetic and CPU-fast.
"""

from __future__ import annotations

import argparse

from ..common.hydra_utils import compose_config, split_overrides, to_container
from ..common.logging import console
from ..experiments.persona import active_questions as aq
from ..experiments.persona.prepare_corpus import CorpusConfig, load_corpus, prepare_corpus
from ..experiments.persona.train_persona import _variant_systems, evaluate_variants
from ..launchers.local import run_local

NAME = "persona"
HELP = ("Guardian-Angel synthetic-persona pipeline: "
        "ga persona prepare --out data/persona_synthetic/v0 | eval | questions | run +exp=persona_dynamic_eval")


def _cmd_prepare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga persona prepare", add_help=True)
    ap.add_argument("--out", default="data/persona_synthetic/v0")
    ap.add_argument("--n-personas", type=int, default=3)
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--holdout-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = CorpusConfig(out_dir=args.out, n_personas=args.n_personas,
                       docs_per_persona=args.docs, holdout_frac=args.holdout_frac, seed=args.seed)
    manifest = prepare_corpus(cfg)
    console.print(f"[green]✓ generated synthetic persona corpus[/green] → {args.out}")
    for k, c in manifest["counts"].items():
        console.print(f"  [cyan]{k}[/cyan]: {c['train']} train / {c['holdout']} holdout docs, {c['qa']} Q&A")
    console.print("  [dim]data_class=synthetic, fictional=True — safe to commit[/dim]")
    return 0


def _cmd_eval(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga persona eval", add_help=True)
    ap.add_argument("--out", default="data/persona_synthetic/v0")
    ap.add_argument("--prepare", action="store_true", help="(re)generate the corpus first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.prepare:
        prepare_corpus(CorpusConfig(out_dir=args.out, seed=args.seed))
    corpus = load_corpus(args.out)
    results = evaluate_variants(corpus["personas"])

    console.print("[bold]Persona variant comparison[/bold] (REAL: base, rag-only | STUB: lora, dynamic-eval)")
    header = f"  {'variant':<14}{'pref_acc':>10}{'author_acc':>12}{'recon_cos':>11}{'stub':>6}"
    console.print(header)
    for name in ("base", "rag-only", "lora", "dynamic-eval"):
        m = results.get(name, {})
        console.print(
            f"  {name:<14}{m.get('preference_accuracy', 0):>10.3f}"
            f"{m.get('authorship_accuracy', 0):>12.3f}"
            f"{m.get('reconstruction_cosine_mean', 0):>11.3f}"
            f"{int(m.get('is_stub', 0)):>6}"
        )
    base_p = results.get("base", {}).get("preference_accuracy", 0.0)
    rag_p = results.get("rag-only", {}).get("preference_accuracy", 0.0)
    console.print(f"  [dim]preference lift (rag-only over base): {rag_p - base_p:+.3f}[/dim]")
    return 0


def _cmd_questions(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga persona questions", add_help=True)
    ap.add_argument("--out", default="data/persona_synthetic/v0")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)
    corpus = load_corpus(args.out)
    personas = corpus["personas"]
    systems = _variant_systems(personas)
    active = aq.propose_active_questions(
        personas, systems["rag-only"]["pref"], systems["base"]["pref"], top_k=args.top_k
    )
    console.print("[bold]Proposed active-learning questions[/bold] (transparent heuristic)")
    for key, qs in active.items():
        console.print(f"[cyan]{key}[/cyan]:")
        for q in qs:
            console.print(f"  [{q.score:.2f}] {q.text}")
            console.print(f"      [dim]{q.rationale}[/dim]")
    return 0


def _cmd_run(argv: list[str]) -> int:
    _flags, overrides = split_overrides(argv)
    if not any(o.startswith("+exp=") or o.startswith("experiment=") for o in overrides):
        overrides = ["+exp=persona_dynamic_eval", *overrides]
    cfg = to_container(compose_config(overrides))
    if cfg.get("runner") != "persona":
        console.print("[red]config does not select the persona runner.[/red] "
                      "Use [cyan]+exp=persona_dynamic_eval[/cyan].")
        return 2
    out = run_local(cfg)
    console.print(f"[green]✓ persona comparison complete[/green] → {out}")
    return 0


_SUB = {
    "prepare": _cmd_prepare,
    "eval": _cmd_eval,
    "questions": _cmd_questions,
    "run": _cmd_run,
}


def run(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        console.print(HELP)
        console.print("\nSubcommands: " + ", ".join(_SUB))
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    fn = _SUB.get(sub)
    if fn is None:
        console.print(f"[red]unknown persona subcommand:[/red] {sub}")
        console.print("Subcommands: " + ", ".join(_SUB))
        return 2
    return fn(rest)
