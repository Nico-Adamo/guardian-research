"""Persona runner: prepare/load the synthetic corpus, run evals, compare variants.

This is the ``persona`` runner registered in ``launchers/local.py``. It ties the
Guardian-Angel (GA) personalization pipeline together at toy scale and COMPARES
four systems on the same synthetic personas:

    base          frozen / generic "assistant" — ignores who the principal is.
    rag-only      retrieves from the principal's own docs + PRINCIPAL.md.
    lora          (HONEST STUB) parameter-efficient finetuning of an LM.
    dynamic-eval  online adaptation over the principal's append-only log.

Which are REAL vs. STUBBED at this CPU/dependency-light scale:
  * base       — REAL. Non-personalized majority predictor + generic boilerplate
                 reconstructor. This is the floor the GA must beat.
  * rag-only   — REAL. Reads the maintained PRINCIPAL.md for preferences and
                 retrieves the nearest persona doc for reconstruction. Runs on the
                 synthetic data with no model download.
  * lora       — HONEST STUB. A real LoRA needs `transformers`+`peft` and a base
                 LM (gated behind the `llm` extra; no downloads in default paths).
                 We *simulate* its expected behavior by reusing retrieval (so the
                 harness compares something), and record `lora_is_stub=1.0`.
  * dynamic-eval — HONEST STUB (lightweight real-ish behavior). True dynamic
                 evaluation finetunes the LM's weights on the fly over the
                 append-only log. We approximate the *information* it would gain by
                 letting the predictor additionally consult the persona's Q&A log
                 (the elicited answers), but we do NOT update any neural weights.
                 Recorded as `dynamic_eval_is_stub=1.0`.

GA principles in play: enhancement-not-replacement (every personalized variant is
measured against the generic `base` floor), mental sovereignty (preferences come
from the principal's own PRINCIPAL.md / log), self-actualization (active questions
target what the principal is still becoming). See planning/guardian/.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...common.artifacts import RunWriter, new_run_id
from ...common.logging import get_logger
from ...common.seeding import seed_everything
from ...tracking.mlflow_client import start_run
from . import active_questions as aq
from . import eval_persona as ev
from .prepare_corpus import CorpusConfig, load_corpus, prepare_corpus, strip_annotations

log = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")


# --------------------------------------------------------------------------- #
# Reconstructors (per system) — return a text reconstruction of a held-out doc  #
# --------------------------------------------------------------------------- #
def _generic_reconstructor(personas: dict) -> Callable[[str, str], str]:
    """base: a generic, persona-blind reconstruction (the "assistant" default)."""
    boiler = ("Here is a helpful, balanced response about the topic. "
              "It is clear and neutral and tries to be useful to anyone.")

    def recon(persona_key: str, title: str) -> str:
        return f"{boiler} The topic appears to be: {title}."

    return recon


def _build_retrieval_index(personas: dict) -> dict[str, list[tuple[str, str]]]:
    """Index each persona's TRAIN docs (annotations stripped) for retrieval."""
    idx: dict[str, list[tuple[str, str]]] = {}
    for k, pdata in personas.items():
        idx[k] = [(_extract_title(text), strip_annotations(text))
                  for _, text in pdata["train_docs"]]
    return idx


def _extract_title(text: str) -> str:
    m = re.search(r"<!--\s*title:\s*(.*?)\s*-->", text)
    return m.group(1) if m else ""


def _rag_reconstructor(personas: dict) -> Callable[[str, str], str]:
    """rag-only: retrieve the principal's nearest own doc + their PRINCIPAL.md.

    Nearest by bag-of-words cosine on the title (we are not allowed to see the
    held-out body). Prepends the principal's style fingerprint so the
    reconstruction is conditioned on the maintained PRINCIPAL.md.
    """
    index = _build_retrieval_index(personas)

    def recon(persona_key: str, title: str) -> str:
        best, best_sim = "", -1.0
        for cand_title, body in index[persona_key]:
            sim = ev.cosine_overlap(title, cand_title) if cand_title else 0.0
            if sim > best_sim:
                best, best_sim = body, sim
        principal = personas[persona_key]["principal_md"]
        return f"{principal}\n\n{best}"

    return recon


def _dynamic_eval_reconstructor(personas: dict) -> Callable[[str, str], str]:
    """dynamic-eval (STUB): RAG retrieval + the persona's elicited Q&A answers.

    Approximates the *information* gained by online finetuning over the append-only
    log by additionally injecting the principal's own answers. No weights change.
    """
    base = _rag_reconstructor(personas)

    def recon(persona_key: str, title: str) -> str:
        answers = " ".join(item["answer"] for item in personas[persona_key]["qa"])
        return f"{base(persona_key, title)}\n\n{answers}"

    return recon


# --------------------------------------------------------------------------- #
# Preference predictors (per system)                                            #
# --------------------------------------------------------------------------- #
def _qa_aware_predictor(personas: dict) -> Callable[[str, str, str, str], str]:
    """dynamic-eval (STUB) predictor: PRINCIPAL.md + the append-only Q&A log.

    Reads the maintained principal text *and* the elicited answers, modeling a GA
    that has trained on the log. Falls back to the structured preference table.
    """
    lookup = ev.principal_lookup_predictor(personas)

    def predict(persona_key: str, topic: str, a: str, b: str) -> str:
        qa_text = " ".join(item["answer"] for item in personas[persona_key]["qa"]).lower()
        ia, ib = qa_text.find(a.lower()), qa_text.find(b.lower())
        if ia != -1 and (ib == -1 or ia <= ib):
            return a
        if ib != -1:
            return b
        return lookup(persona_key, topic, a, b)

    return predict


# --------------------------------------------------------------------------- #
# Variant registry                                                              #
# --------------------------------------------------------------------------- #
def _variant_systems(personas: dict) -> dict[str, dict[str, Any]]:
    """Map variant name -> {pref_predictor, reconstructor, flags}.

    Keep this the single source of truth for what each variant *is*.
    """
    lookup = ev.principal_lookup_predictor(personas)     # personalized (RAG/LoRA)
    majority = ev.majority_predictor(personas)           # generic (base)
    qa_aware = _qa_aware_predictor(personas)             # dynamic-eval stub

    return {
        "base": {
            "pref": majority,
            "recon": _generic_reconstructor(personas),
            "flags": {"is_personalized": 0.0, "is_stub": 0.0},
            "real": True,
        },
        "rag-only": {
            "pref": lookup,
            "recon": _rag_reconstructor(personas),
            "flags": {"is_personalized": 1.0, "is_stub": 0.0},
            "real": True,
        },
        "lora": {
            # STUB: reuse retrieval-quality personalization; no real PEFT weights.
            "pref": lookup,
            "recon": _rag_reconstructor(personas),
            "flags": {"is_personalized": 1.0, "is_stub": 1.0},
            "real": False,
        },
        "dynamic-eval": {
            # STUB: consults the append-only Q&A log in addition to PRINCIPAL.md.
            "pref": qa_aware,
            "recon": _dynamic_eval_reconstructor(personas),
            "flags": {"is_personalized": 1.0, "is_stub": 1.0},
            "real": False,
        },
    }


def evaluate_variants(personas: dict) -> dict[str, dict[str, float]]:
    """Run the full eval suite for every variant; return per-variant metric dicts."""
    systems = _variant_systems(personas)
    results: dict[str, dict[str, float]] = {}
    for name, sys in systems.items():
        metrics = ev.run_all_evals(personas, sys["pref"], sys["recon"])
        metrics.update(sys["flags"])
        results[name] = metrics
    return results


# --------------------------------------------------------------------------- #
# Runner entry point                                                            #
# --------------------------------------------------------------------------- #
def run(cfg: dict[str, Any]) -> Path:
    """Runner contract: run(cfg) -> path to results.json.

    Config (all optional; toy defaults):
      cfg["seed"], cfg["experiment"]
      cfg["persona"]["out_dir"]      where the synthetic corpus lives / is written
      cfg["persona"]["prepare"]      bool: (re)generate the corpus (default True)
      cfg["persona"]["n_personas"], ["docs_per_persona"], ["holdout_frac"]
      cfg["persona"]["variants"]     list of variant names to compare
    """
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    experiment = str(cfg.get("experiment", "persona_dynamic_eval"))
    pcfg = dict(cfg.get("persona", {}) or {})

    out_dir = str(pcfg.get("out_dir", "data/persona_synthetic/v0"))
    do_prepare = bool(pcfg.get("prepare", True))
    variants = list(pcfg.get("variants", ["base", "rag-only", "lora", "dynamic-eval"]))

    # 1) Prepare or load the synthetic corpus.
    if do_prepare or not (Path(out_dir) / "manifest.json").exists():
        ccfg = CorpusConfig(
            out_dir=out_dir,
            n_personas=int(pcfg.get("n_personas", 3)),
            docs_per_persona=int(pcfg.get("docs_per_persona", 12)),
            holdout_frac=float(pcfg.get("holdout_frac", 0.34)),
            seed=seed,
        )
        prepare_corpus(ccfg)
    corpus = load_corpus(out_dir)
    personas = corpus["personas"]

    # 2) Evaluate all variants on the same personas.
    all_results = evaluate_variants(personas)

    # 3) Active-learning questions (transparent heuristic).
    systems = _variant_systems(personas)
    active = aq.propose_active_questions(
        personas,
        personalized_predictor=systems["rag-only"]["pref"],
        generic_predictor=systems["base"]["pref"],
        top_k=int(pcfg.get("active_top_k", 3)),
    )

    # 4) Write the RunResult.
    run_id = new_run_id(prefix="persona")
    writer = RunWriter(experiment=experiment, run_id=run_id, seed=seed)
    writer.set_config(cfg)
    writer.set_params(
        out_dir=out_dir,
        n_personas=len(personas),
        variants=",".join(variants),
        real_variants="base,rag-only",
        stub_variants="lora,dynamic-eval",
    )

    with start_run(experiment, run_id, tags={"seed": str(seed)}) as mlf:
        mlf.log_params({**writer.result.params})

        # Per-variant metrics (selected variants only, in order).
        final: dict[str, float] = {}
        for step, name in enumerate(variants):
            if name not in all_results:
                log.warning("unknown variant '%s' — skipping", name)
                continue
            m = all_results[name]
            tag = name.replace("-", "_")
            scoped = {f"{tag}__{k}": float(v) for k, v in m.items()}
            writer.log_metrics(step, **scoped)
            mlf.log_metrics(scoped, step=step)
            # Promote headline metrics to final_metrics for cross-run comparison.
            for headline in ("preference_accuracy", "authorship_accuracy",
                             "reconstruction_cosine_mean", "is_stub"):
                final[f"{tag}__{headline}"] = float(m.get(headline, 0.0))
            log.info(
                "variant=%-12s pref_acc=%.3f author_acc=%.3f recon_cos=%.3f stub=%d",
                name, m.get("preference_accuracy", 0.0), m.get("authorship_accuracy", 0.0),
                m.get("reconstruction_cosine_mean", 0.0), int(m.get("is_stub", 0.0)),
            )

        # Headline comparison: does personalization beat the generic floor?
        base_pref = all_results.get("base", {}).get("preference_accuracy", 0.0)
        rag_pref = all_results.get("rag-only", {}).get("preference_accuracy", 0.0)
        final["preference_lift_rag_over_base"] = float(rag_pref - base_pref)
        writer.set_final(**final)

        # Persist the per-variant table + active questions as artifacts.
        _write_comparison_artifact(writer, variants, all_results)
        _write_active_questions_artifact(writer, active)
        mlf.log_metrics(final, step=len(variants))

    out = writer.finish(status="completed")
    log.info("wrote %s", out)
    return out


# --------------------------------------------------------------------------- #
# Artifact writers                                                              #
# --------------------------------------------------------------------------- #
def _write_comparison_artifact(writer: RunWriter, variants: list[str],
                               results: dict[str, dict[str, float]]) -> None:
    cols = ["preference_accuracy", "authorship_accuracy",
            "reconstruction_cosine_mean", "is_personalized", "is_stub"]
    lines = ["# Persona variant comparison (synthetic corpus)\n",
             "REAL: base, rag-only.  HONEST STUB: lora, dynamic-eval.\n",
             "| variant | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name in variants:
        m = results.get(name, {})
        row = " | ".join(f"{m.get(c, 0.0):.3f}" for c in cols)
        lines.append(f"| {name} | {row} |")
    path = writer.add_artifact("variant_comparison.md")
    path.write_text("\n".join(lines) + "\n")


def _write_active_questions_artifact(writer: RunWriter, active: dict) -> None:
    lines = ["# Proposed active-learning questions (transparent heuristic)\n",
             "Ranked by coverage_gap (0.4) + disagreement (0.4) + expected_info (0.2).\n"]
    for key, qs in active.items():
        lines.append(f"## {key}")
        for q in qs:
            lines.append(f"- [{q.score:.2f}] {q.text}")
            lines.append(f"    - rationale: {q.rationale}")
        lines.append("")
    path = writer.add_artifact("active_questions.md")
    path.write_text("\n".join(lines) + "\n")
