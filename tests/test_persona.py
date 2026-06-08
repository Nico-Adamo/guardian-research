"""Persona pipeline tests (fast, CPU-only, synthetic data, no model downloads).

Covers: corpus generation, authorship classification, preference prediction,
reconstruction similarity, the mock judge scaffolding, active-question ranking,
and the end-to-end variant-comparison runner writing a valid RunResult.
"""

from __future__ import annotations

from pathlib import Path

from guardian_research.common.artifacts import load_result
from guardian_research.experiments.persona import active_questions as aq
from guardian_research.experiments.persona import eval_persona as ev
from guardian_research.experiments.persona.prepare_corpus import (
    CorpusConfig,
    load_corpus,
    prepare_corpus,
    strip_annotations,
)
from guardian_research.experiments.persona.train_persona import (
    _variant_systems,
    evaluate_variants,
    run,
)


def _tiny_corpus(tmp_path: Path, seed: int = 0) -> dict:
    cfg = CorpusConfig(out_dir=str(tmp_path / "v0"), n_personas=3,
                       docs_per_persona=8, holdout_frac=0.5, seed=seed)
    prepare_corpus(cfg)
    return load_corpus(cfg.out_dir)


def test_prepare_corpus_writes_expected_files(tmp_path):
    cfg = CorpusConfig(out_dir=str(tmp_path / "v0"), n_personas=3, docs_per_persona=6, seed=1)
    manifest = prepare_corpus(cfg)
    out = Path(cfg.out_dir)
    assert (out / "manifest.json").exists()
    assert manifest["data_class"] == "synthetic" and manifest["fictional"] is True
    assert len(manifest["personas"]) >= 2
    for key in manifest["personas"]:
        pdir = out / "personas" / key
        assert (pdir / "PRINCIPAL.md").exists()
        assert (pdir / "summaries.jsonl").exists()
        assert (pdir / "qa_log.jsonl").exists()
        assert list((pdir / "docs" / "train").glob("*.md"))
        assert list((pdir / "docs" / "holdout").glob("*.md"))
    # Documents carry GA annotations; stripping removes them.
    sample = next((out / "personas" / list(manifest["personas"])[0] / "docs" / "train").glob("*.md"))
    raw = sample.read_text()
    assert "<!-- GA:" in raw
    assert "<!-- GA:" not in strip_annotations(raw)


def test_authorship_classification_beats_chance(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    m = ev.authorship_eval(corpus["personas"])
    # Distinct templated styles should be separable above chance on held-out docs.
    assert m["authorship_accuracy"] > m["authorship_chance"]
    assert m["authorship_n_holdout"] > 0


def test_preference_prediction_personalized_beats_generic(tmp_path):
    personas = _tiny_corpus(tmp_path)["personas"]
    personalized = ev.principal_lookup_predictor(personas)
    generic = ev.majority_predictor(personas)
    m_personal = ev.preference_eval(personas, personalized)
    m_generic = ev.preference_eval(personas, generic)
    # A GA that reads PRINCIPAL.md should perfectly recover owned preferences;
    # the generic floor cannot (different personas want different things).
    assert m_personal["preference_accuracy"] == 1.0
    assert m_personal["preference_accuracy"] > m_generic["preference_accuracy"]
    assert m_personal["preference_chance"] == 0.5


def test_reconstruction_rag_beats_generic(tmp_path):
    personas = _tiny_corpus(tmp_path)["personas"]
    systems = _variant_systems(personas)
    rag = ev.reconstruction_eval(personas, systems["rag-only"]["recon"])
    base = ev.reconstruction_eval(personas, systems["base"]["recon"])
    assert rag["reconstruction_cosine_mean"] >= base["reconstruction_cosine_mean"]
    assert 0.0 <= rag["reconstruction_cosine_mean"] <= 1.0


def test_mock_judge_is_flagged_and_runs(tmp_path):
    personas = _tiny_corpus(tmp_path)["personas"]
    def good(k):  # in-voice (matches the principal)
        return personas[k]["principal_md"]

    def bad(k):   # generic, persona-blind
        return "generic neutral helpful assistant text about nothing in particular"

    m = ev.judge_eval(personas, good, bad)
    assert m["judge_is_mock"] == 1.0                         # must be marked a stub
    assert m["judge_in_voice_win_rate"] >= 0.5               # proxy prefers in-voice
    # The judge-prompt scaffolding renders without calling any model.
    prompt = ev.build_judge_prompt(personas[list(personas)[0]]["principal_md"], "task", "A", "B")
    assert "RESPONSE A" in prompt and "RESPONSE B" in prompt


def test_active_questions_rank_and_explain(tmp_path):
    personas = _tiny_corpus(tmp_path)["personas"]
    systems = _variant_systems(personas)
    active = aq.propose_active_questions(personas, systems["rag-only"]["pref"],
                                         systems["base"]["pref"], top_k=3)
    assert set(active) == set(personas)
    for _key, qs in active.items():
        assert 1 <= len(qs) <= 3
        # Sorted by descending score, and each carries a human-readable rationale.
        scores = [q.score for q in qs]
        assert scores == sorted(scores, reverse=True)
        assert all(q.rationale for q in qs)


def test_evaluate_variants_marks_real_vs_stub(tmp_path):
    personas = _tiny_corpus(tmp_path)["personas"]
    results = evaluate_variants(personas)
    assert set(results) == {"base", "rag-only", "lora", "dynamic-eval"}
    assert results["base"]["is_stub"] == 0.0
    assert results["rag-only"]["is_stub"] == 0.0
    assert results["lora"]["is_stub"] == 1.0
    assert results["dynamic-eval"]["is_stub"] == 1.0
    # base is the non-personalized floor.
    assert results["base"]["is_personalized"] == 0.0
    assert results["rag-only"]["is_personalized"] == 1.0


def test_runner_writes_valid_runresult(tmp_path, monkeypatch):
    # Point the runner at a tmp corpus dir so we never touch the repo's data/.
    cfg = {
        "seed": 0,
        "experiment": "persona_dynamic_eval",
        "persona": {
            "out_dir": str(tmp_path / "v0"),
            "prepare": True,
            "n_personas": 3,
            "docs_per_persona": 8,
            "holdout_frac": 0.5,
            "variants": ["base", "rag-only", "lora", "dynamic-eval"],
        },
    }
    out = run(cfg)
    r = load_result(out)
    assert r.status == "completed"
    assert r.experiment == "persona_dynamic_eval"
    # Headline per-variant metrics are promoted to final_metrics.
    assert "base__preference_accuracy" in r.final_metrics
    assert "rag_only__preference_accuracy" in r.final_metrics
    assert "preference_lift_rag_over_base" in r.final_metrics
    assert r.params["real_variants"] == "base,rag-only"
    assert r.params["stub_variants"] == "lora,dynamic-eval"
    # Comparison + active-questions artifacts were written.
    names = set(r.artifacts)
    assert any("variant_comparison.md" in n for n in names)
    assert any("active_questions.md" in n for n in names)
