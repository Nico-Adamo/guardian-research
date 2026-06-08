"""Active-learning question proposal for a Guardian Angel (transparent heuristic).

The GA design leans on *active learning*: rather than passively ingesting more
text, the agent asks the principal an optimally adaptive sequence of questions,
each of which is expected to update PRINCIPAL.md the most (or reduce held-out
loss / regret the most). This is the DAgger-style "ask, then never make that
mistake again" loop.

Here we implement a transparent, inspectable scoring rule over a bank of
candidate questions. We do NOT train a model to pick questions; we score them by
three legible signals and rank. The whole point is that a principal can read why
a question was asked.

Scoring (all in [0, 1], higher = ask sooner):

* **coverage_gap** — does PRINCIPAL.md currently say nothing about this question's
  topic? Unknown topics are worth more (information we don't have).
* **disagreement** — do our current systems (e.g. a personalized predictor vs. a
  generic majority predictor) disagree on the answer? Disagreement marks a
  decision boundary where one answer flips behavior, so the principal's reply is
  maximally informative.
* **expected_info** — a simple proxy for expected information gain: questions
  about binary preferences with no recorded answer score high; already-answered
  or low-entropy questions score low.

The composite score is a documented weighted sum. Nothing is learned; everything
is auditable. This honors *mental sovereignty*: the principal sees the rationale
and stays in control of what gets elicited.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class Candidate:
    """A candidate question to ask the principal."""

    topic: str
    text: str
    # If this question maps to a binary choice we already track, fill these so
    # the disagreement signal can be computed.
    option_a: str | None = None
    option_b: str | None = None


@dataclass
class ScoredQuestion:
    topic: str
    text: str
    score: float
    coverage_gap: float
    disagreement: float
    expected_info: float
    rationale: str


# Default weights for the composite score. Documented and tunable.
WEIGHTS = {"coverage_gap": 0.4, "disagreement": 0.4, "expected_info": 0.2}


def default_question_bank(personas: dict) -> dict[str, list[Candidate]]:
    """Build per-persona candidate questions from the corpus's preference topics.

    We generate one direct preference question per known topic plus a couple of
    open value/aspiration questions. In a real GA this bank would be far larger
    and partly model-generated; here it is template-based and synthetic.
    """
    bank: dict[str, list[Candidate]] = {}
    for k, pdata in personas.items():
        cands: list[Candidate] = []
        for pref in pdata["preferences"]:
            a, b = sorted([pref["choice"], pref["alt"]])
            cands.append(Candidate(
                topic=pref["topic"],
                text=f"For {pref['topic']}, do you prefer {a} or {b}?",
                option_a=a, option_b=b,
            ))
        cands.append(Candidate(topic="tiebreak",
                               text="When two options tie on results, what breaks the tie for you?"))
        cands.append(Candidate(topic="refusal",
                               text="What kind of output would you refuse to publish under your name?"))
        bank[k] = cands
    return bank


def _topic_covered(principal_md: str, topic: str) -> bool:
    """Heuristic: is ``topic`` already addressed in PRINCIPAL.md?"""
    return topic.lower() in principal_md.lower()


def score_questions(
    personas: dict,
    persona_key: str,
    candidates: list[Candidate],
    personalized_predictor: Callable[[str, str, str, str], str],
    generic_predictor: Callable[[str, str, str, str], str],
    weights: dict[str, float] | None = None,
) -> list[ScoredQuestion]:
    """Score and rank candidate questions for one persona (descending score).

    ``personalized_predictor`` and ``generic_predictor`` have the
    ``(persona_key, topic, a, b) -> choice`` signature (see eval_persona). Their
    disagreement on a candidate's options is the high-value signal.
    """
    w = {**WEIGHTS, **(weights or {})}
    principal_md = personas[persona_key]["principal_md"]
    scored: list[ScoredQuestion] = []

    for c in candidates:
        # coverage_gap: 1.0 if the topic is not yet in PRINCIPAL.md.
        coverage_gap = 0.0 if _topic_covered(principal_md, c.topic) else 1.0

        # disagreement: 1.0 if the two predictors disagree on this binary question.
        disagreement = 0.0
        if c.option_a is not None and c.option_b is not None:
            p = personalized_predictor(persona_key, c.topic, c.option_a, c.option_b)
            g = generic_predictor(persona_key, c.topic, c.option_a, c.option_b)
            disagreement = 1.0 if p != g else 0.0

        # expected_info: binary preference questions carry ~1 bit when unanswered;
        # open questions get a moderate prior; covered topics decay.
        if c.option_a is not None:
            expected_info = 1.0 if not _topic_covered(principal_md, c.topic) else 0.3
        else:
            expected_info = 0.6

        score = (w["coverage_gap"] * coverage_gap
                 + w["disagreement"] * disagreement
                 + w["expected_info"] * expected_info)
        rationale = (
            f"coverage_gap={coverage_gap:.2f} (topic {'absent from' if coverage_gap else 'present in'} PRINCIPAL.md), "
            f"disagreement={disagreement:.2f} (predictors {'differ' if disagreement else 'agree'}), "
            f"expected_info={expected_info:.2f}"
        )
        scored.append(ScoredQuestion(
            topic=c.topic, text=c.text, score=score,
            coverage_gap=coverage_gap, disagreement=disagreement,
            expected_info=expected_info, rationale=rationale,
        ))

    scored.sort(key=lambda s: (-s.score, s.topic, s.text))
    return scored


def propose_active_questions(
    personas: dict,
    personalized_predictor: Callable[[str, str, str, str], str],
    generic_predictor: Callable[[str, str, str, str], str],
    top_k: int = 3,
) -> dict[str, list[ScoredQuestion]]:
    """Top-k ranked questions per persona, expected to most update PRINCIPAL.md.

    Returns ``{persona_key: [ScoredQuestion, ...]}``. The orchestrator/CLI can
    surface these to the (synthetic) principal; in production the answers would be
    appended to the GA's log and trained on.
    """
    bank = default_question_bank(personas)
    out: dict[str, list[ScoredQuestion]] = {}
    for k in personas:
        ranked = score_questions(personas, k, bank[k], personalized_predictor, generic_predictor)
        out[k] = ranked[:top_k]
    return out
