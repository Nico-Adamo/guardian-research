"""Generate a *synthetic* Guardian-Angel persona corpus (no private data, ever).

This module fabricates 2-3 entirely fictional personas with rule/template-based
writing style and explicit preferences, then emits the kind of append-only log a
real Guardian Angel (GA) would accumulate about its principal:

* documents (short essays/notes) written in each persona's style,
* per-document summaries,
* Q&A logs (the principal answering elicitation questions),
* intra-file annotations like ``<!-- GA: important: preference -->`` that mark
  load-bearing spans, and
* a ``PRINCIPAL.md`` per persona summarizing values / preferences / style.

Why synthetic? The GA design doctrine is uncompromising: *no private/persona
data ever leaves the principal's machine, and none is committed to a repo*. To
build and test the tooling we therefore use fictional personas whose every token
is generated here from fixed templates and seeded RNG. The output is safe to
commit and trivially reproducible.

GA principles this corpus is meant to exercise (see planning/guardian/):
  * **Enhancement, not replacement** — the artifacts describe how to *amplify*
    a principal's own voice and choices, never to substitute a generic persona.
  * **Mental sovereignty** — preferences are owned by the principal; the corpus
    records them as first-class, queryable facts rather than inferring a docile
    "assistant" default.
  * **Self-actualization** — PRINCIPAL.md captures aspirations/ideals, not just a
    static snapshot, so the GA has "something meaningful to learn to emulate".
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Persona definitions (fictional; rule/template based)                          #
# --------------------------------------------------------------------------- #
@dataclass
class Persona:
    """A fictional principal: a bundle of style knobs and explicit preferences."""

    key: str
    display_name: str
    # Style knobs that drive deterministic, *distinguishable* prose.
    sentence_opener: str          # signature way to start a sentence
    connective: str               # signature connective / discourse marker
    closer: str                   # signature sign-off phrase
    avg_sentence_words: int       # short vs. long sentences
    exclaim: bool                 # uses exclamation marks
    lexicon: list[str] = field(default_factory=list)  # pet words sprinkled in
    # Explicit, owned preferences. Each is a binary choice the GA must predict.
    # ``choice`` is the option this persona prefers; ``alt`` is the rejected one.
    preferences: list[dict[str, str]] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    aspirations: list[str] = field(default_factory=list)


def default_personas() -> list[Persona]:
    """Three deliberately distinct fictional personas.

    The contrasts (terse vs. ornate vs. plain; tea vs. coffee; etc.) are chosen so
    that stylometry and preference prediction have real signal to recover at a tiny
    scale, without any real person being depicted.
    """
    return [
        Persona(
            key="ada",
            display_name="Ada (the terse engineer)",
            sentence_opener="Concretely,",
            connective="therefore",
            closer="— shipping it.",
            avg_sentence_words=9,
            exclaim=False,
            lexicon=["invariant", "throughput", "boring", "measurable", "tradeoff"],
            preferences=[
                {"topic": "drink", "choice": "black coffee", "alt": "herbal tea"},
                {"topic": "editor", "choice": "a terminal editor", "alt": "a heavy IDE"},
                {"topic": "prose", "choice": "short declarative sentences", "alt": "ornate flourishes"},
                {"topic": "meeting", "choice": "an async written update", "alt": "a live meeting"},
            ],
            values=["correctness over cleverness", "reproducibility", "owning one's tools"],
            aspirations=["build systems that outlive their author", "mentor without micromanaging"],
        ),
        Persona(
            key="bram",
            display_name="Bram (the ornate essayist)",
            sentence_opener="One might venture that",
            connective="and yet",
            closer="So it seems to me, at least.",
            avg_sentence_words=24,
            exclaim=False,
            lexicon=["labyrinthine", "reverie", "palimpsest", "twilight", "filigree"],
            preferences=[
                {"topic": "drink", "choice": "herbal tea", "alt": "black coffee"},
                {"topic": "editor", "choice": "longhand on paper", "alt": "a terminal editor"},
                {"topic": "prose", "choice": "ornate flourishes", "alt": "short declarative sentences"},
                {"topic": "meeting", "choice": "a long walking conversation", "alt": "an async written update"},
            ],
            values=["beauty in expression", "patience with ambiguity", "the long view of history"],
            aspirations=["write one essay worth rereading in a decade", "cultivate attention as a craft"],
        ),
        Persona(
            key="cleo",
            display_name="Cleo (the plain-spoken organizer)",
            sentence_opener="Here's the plan:",
            connective="so",
            closer="Let's get it done.",
            avg_sentence_words=14,
            exclaim=True,
            lexicon=["folks", "checklist", "deadline", "fair", "follow-up"],
            preferences=[
                {"topic": "drink", "choice": "iced tea", "alt": "black coffee"},
                {"topic": "editor", "choice": "a shared doc", "alt": "longhand on paper"},
                {"topic": "prose", "choice": "plain bullet points", "alt": "ornate flourishes"},
                {"topic": "meeting", "choice": "a short stand-up", "alt": "a long walking conversation"},
            ],
            values=["fairness", "following through", "keeping people informed"],
            aspirations=["run a team where nobody is surprised", "make hard calls kindly"],
        ),
    ]


# --------------------------------------------------------------------------- #
# Template-based text generation                                                #
# --------------------------------------------------------------------------- #
_TOPICS = [
    ("a morning routine", "starting the day"),
    ("a code review", "reviewing a teammate's change"),
    ("a decision about a deadline", "deciding what to cut"),
    ("a walk outside", "noticing the weather"),
    ("a book worth finishing", "choosing what to read"),
    ("a difficult conversation", "giving honest feedback"),
    ("a weekend plan", "balancing rest and work"),
    ("a tool that broke", "fixing or replacing it"),
]


def _sentence(p: Persona, rng: random.Random, core: str) -> str:
    """Build one sentence in persona ``p``'s style around a ``core`` clause."""
    word = rng.choice(p.lexicon)
    # Pad/trim toward the persona's target sentence length using filler clauses.
    fillers = [
        f"the {word} part matters",
        f"this is {word} but true",
        f"I keep coming back to the {word} of it",
        f"({word}, if I am honest)",
    ]
    body = core
    target = p.avg_sentence_words
    while len(body.split()) < target:
        body = f"{body}, {p.connective} {rng.choice(fillers)}"
    s = f"{p.sentence_opener} {body}"
    s = s[0].upper() + s[1:]
    return s + ("!" if p.exclaim and rng.random() < 0.5 else ".")


def _document(p: Persona, rng: random.Random) -> tuple[str, str, list[str]]:
    """Return (title, body_with_annotations, list_of_summary_bullets)."""
    topic, gerund = rng.choice(_TOPICS)
    pref = rng.choice(p.preferences)
    cores = [
        f"today I am thinking about {topic}",
        f"when {gerund}, I reach for {pref['choice']} rather than {pref['alt']}",
        f"the thing people miss about {topic} is the cost of getting it wrong",
        "I would rather do less and finish than do more and drift",
    ]
    rng.shuffle(cores)
    sentences = [_sentence(p, rng, c) for c in cores[:3]]
    # Embed an intra-file GA annotation on the preference-bearing sentence so the
    # tooling can learn to find and lift load-bearing spans into PRINCIPAL.md.
    lines = []
    for s in sentences:
        if pref["choice"] in s:
            lines.append(f"<!-- GA: important: preference ({pref['topic']}) -->")
        lines.append(s)
    lines.append("<!-- GA: style: signature-closer -->")
    lines.append(p.closer)
    title = f"{p.display_name.split(' ')[0]} on {topic}"
    summary = [
        f"Reflection on {topic} in {p.key}'s voice.",
        f"States a preference: {pref['choice']} over {pref['alt']} ({pref['topic']}).",
    ]
    return title, "\n".join(lines), summary


def _qa_log(p: Persona, rng: random.Random, n: int = 4) -> list[dict[str, str]]:
    """Elicitation Q&A: the GA asks, the (fictional) principal answers in-voice.

    This mirrors the GA active-learning loop: questions that pin down preferences
    and values, answered by the principal so the answer can be trained on forever
    rather than discarded with the session.
    """
    qs = [
        ("When two approaches tie on results, how do you break the tie?",
         f"I pick the one closer to my values: {', '.join(p.values[:2])}."),
        ("What do you reach for by default?",
         f"For most things, {p.preferences[0]['choice']} — not {p.preferences[0]['alt']}."),
        ("What are you trying to become better at?",
         f"{p.aspirations[0].capitalize()}."),
        ("How should an agent act when it is unsure?",
         "Ask me. Then never make that mistake again."),
        ("What writing do you refuse to publish under your name?",
         f"Anything that isn't {p.preferences[2]['choice']}."),
    ]
    rng.shuffle(qs)
    out = []
    for q, a in qs[:n]:
        out.append({"question": q, "answer": _answerify(p, a)})
    return out


def _answerify(p: Persona, a: str) -> str:
    """Lightly restyle a canned answer into the persona's voice."""
    return f"{a}" if not p.exclaim else (a if a.endswith(".") else a + "!")


def render_principal_md(p: Persona) -> str:
    """Render a PRINCIPAL.md summarizing values/preferences/style for persona ``p``.

    PRINCIPAL.md is the GA's living model of *who the principal is and is becoming*
    — the thing dynamic-evaluation finetuning and RAG both condition on.
    """
    prefs = "\n".join(
        f"- **{x['topic']}**: prefers _{x['choice']}_ over _{x['alt']}_." for x in p.preferences
    )
    vals = "\n".join(f"- {v}" for v in p.values)
    asps = "\n".join(f"- {a}" for a in p.aspirations)
    return f"""# PRINCIPAL.md — {p.display_name}

> SYNTHETIC / FICTIONAL persona. Generated by `prepare_corpus.py`. No real person
> is depicted. Safe to commit. This file is what a real Guardian Angel would
> maintain about its principal — here, a fixture for building and testing tooling.

## Who I am (and am becoming)
A fictional principal used to exercise the GA personalization pipeline. The GA's
job is to *amplify* this voice, never to replace it with a generic assistant.

## Style fingerprint
- Signature opener: "{p.sentence_opener}"
- Signature connective: "{p.connective}"
- Sign-off: "{p.closer}"
- Sentence length: ~{p.avg_sentence_words} words; exclamations: {"yes" if p.exclaim else "no"}
- Pet words: {", ".join(p.lexicon)}

## Preferences (owned by the principal — mental sovereignty)
{prefs}

## Values
{vals}

## Aspirations (self-actualization)
{asps}
"""


# --------------------------------------------------------------------------- #
# Corpus assembly                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class CorpusConfig:
    out_dir: str = "data/persona_synthetic/v0"
    n_personas: int = 3            # 2 or 3 (clamped to available templates)
    docs_per_persona: int = 12
    holdout_frac: float = 0.34     # fraction of docs reserved for held-out eval
    seed: int = 0


def prepare_corpus(cfg: CorpusConfig) -> dict[str, Any]:
    """Generate the full synthetic corpus on disk and return a manifest dict.

    Layout (all under ``cfg.out_dir``):
      personas/<key>/PRINCIPAL.md
      personas/<key>/docs/train/<i>.md      (with intra-file GA annotations)
      personas/<key>/docs/holdout/<i>.md
      personas/<key>/summaries.jsonl
      personas/<key>/qa_log.jsonl
      manifest.json                          (data_class=synthetic + counts)
    """
    rng = random.Random(cfg.seed)
    personas = default_personas()[: max(2, min(3, cfg.n_personas))]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "data_class": "synthetic",
        "fictional": True,
        "seed": cfg.seed,
        "personas": {},
        "counts": {},
    }

    for p in personas:
        pdir = out / "personas" / p.key
        (pdir / "docs" / "train").mkdir(parents=True, exist_ok=True)
        (pdir / "docs" / "holdout").mkdir(parents=True, exist_ok=True)

        (pdir / "PRINCIPAL.md").write_text(render_principal_md(p))

        summaries: list[dict[str, Any]] = []
        n = cfg.docs_per_persona
        n_holdout = max(1, int(round(n * cfg.holdout_frac)))
        for i in range(n):
            title, body, summary = _document(p, rng)
            split = "holdout" if i >= (n - n_holdout) else "train"
            doc_path = pdir / "docs" / split / f"{i:03d}.md"
            doc_path.write_text(f"<!-- title: {title} -->\n{body}\n")
            summaries.append({"doc": str(doc_path.relative_to(out)), "split": split,
                              "title": title, "summary": summary})

        with open(pdir / "summaries.jsonl", "w") as fp:
            for s in summaries:
                fp.write(json.dumps(s) + "\n")

        qa = _qa_log(p, rng)
        with open(pdir / "qa_log.jsonl", "w") as fp:
            for item in qa:
                fp.write(json.dumps(item) + "\n")

        manifest["personas"][p.key] = {
            "display_name": p.display_name,
            "principal_md": str((pdir / "PRINCIPAL.md").relative_to(out)),
            "preferences": p.preferences,
        }
        manifest["counts"][p.key] = {
            "docs": n,
            "holdout": n_holdout,
            "train": n - n_holdout,
            "qa": len(qa),
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_corpus(out_dir: str | Path) -> dict[str, Any]:
    """Load a prepared corpus into memory: per-persona docs/summaries/qa/principal.

    Returns ``{"manifest": ..., "personas": {key: {...}}}`` where each persona has
    ``train_docs``/``holdout_docs`` (lists of (path, text)), ``summaries``,
    ``qa``, ``principal_md`` text, and the structured ``preferences``.
    """
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    personas: dict[str, Any] = {}
    for key, meta in manifest["personas"].items():
        pdir = out / "personas" / key

        def _read_split(split: str, _pdir: Path = pdir) -> list[tuple[str, str]]:
            d = _pdir / "docs" / split
            return [(str(fp), fp.read_text()) for fp in sorted(d.glob("*.md"))]

        summaries = [json.loads(line) for line in (pdir / "summaries.jsonl").read_text().splitlines() if line]
        qa = [json.loads(line) for line in (pdir / "qa_log.jsonl").read_text().splitlines() if line]
        personas[key] = {
            "display_name": meta["display_name"],
            "preferences": meta["preferences"],
            "principal_md": (pdir / "PRINCIPAL.md").read_text(),
            "train_docs": _read_split("train"),
            "holdout_docs": _read_split("holdout"),
            "summaries": summaries,
            "qa": qa,
        }
    return {"manifest": manifest, "personas": personas}


def strip_annotations(text: str) -> str:
    """Remove ``<!-- GA: ... -->`` annotation lines and title comments from a doc."""
    keep = []
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("<!--") and ls.endswith("-->"):
            continue
        keep.append(line)
    return "\n".join(keep).strip()
