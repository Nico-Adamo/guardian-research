"""Persona evaluations (CPU-only, numpy, no sklearn / no model downloads).

These are the measurements a Guardian Angel (GA) cares about: does a candidate
system actually *capture the principal* — their style, their preferences, their
words — rather than averaging them into a generic assistant?

We implement four transparent, dependency-light evals:

1. **Authorship / style classification** on held-out documents, using simple
   stylometric features + a hand-rolled nearest-centroid classifier (no sklearn).
   "Truesight"-style stylometry is the cheapest probe of whether persona signal
   survives.
2. **Preference prediction** — given two options, predict the persona's choice.
   Mental sovereignty made measurable: the GA must know what the principal wants.
3. **Held-out response reconstruction similarity** — how close is a system's
   reconstruction of a held-out doc to the real thing (token-overlap / cosine).
4. **Pairwise judge scaffolding** — a CLEARLY-MARKED heuristic/mock judge that
   scores which of two candidate responses is more in-voice. This is a *stub for
   a real LLM judge*; it returns transparent feature-based scores, never claims
   to be a real preference model, and is here so the harness/plumbing exists.

Nothing here downloads a model or touches the network. All "systems under test"
are passed in as callables, so train_persona.py can plug in base / RAG / LoRA /
dynamic-eval variants.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .prepare_corpus import strip_annotations

_WORD_RE = re.compile(r"[A-Za-z']+")


# --------------------------------------------------------------------------- #
# Stylometric features                                                          #
# --------------------------------------------------------------------------- #
# A fixed, interpretable feature vector. Order matters (it is the contract for
# the centroid classifier). Function words + punctuation rates + length stats are
# the classic stylometry signal and are content-independent on purpose.
_FUNCTION_WORDS = [
    "the", "a", "and", "so", "therefore", "yet", "i", "one", "this", "it",
    "of", "to", "for", "is", "am", "would", "rather", "than", "here", "today",
]
_PUNCT = [".", ",", "!", "(", ")", "—", ":", ";"]


def stylometric_features(text: str) -> np.ndarray:
    """Return a fixed-length numpy feature vector for ``text`` (content-light)."""
    clean = strip_annotations(text)
    words = _WORD_RE.findall(clean.lower())
    n_words = max(1, len(words))
    n_chars = max(1, len(clean))
    counts = Counter(words)

    feats: list[float] = []
    # Function-word rates (normalized by word count).
    for fw in _FUNCTION_WORDS:
        feats.append(counts.get(fw, 0) / n_words)
    # Punctuation rates (normalized by char count).
    for pu in _PUNCT:
        feats.append(clean.count(pu) / n_chars)
    # Length statistics.
    sentences = [s for s in re.split(r"[.!?]+", clean) if s.strip()]
    avg_sent_len = np.mean([len(_WORD_RE.findall(s)) for s in sentences]) if sentences else 0.0
    feats.append(avg_sent_len / 30.0)                 # ~normalized
    feats.append(np.mean([len(w) for w in words]) / 12.0)  # avg word length
    feats.append(len(set(words)) / n_words)           # type-token ratio
    feats.append(clean.count("!") / max(1, len(sentences)))  # exclamation density
    return np.asarray(feats, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 1) Authorship / style classification (nearest-centroid, hand-rolled)          #
# --------------------------------------------------------------------------- #
@dataclass
class NearestCentroid:
    """A transparent classifier: one normalized centroid per class.

    No sklearn. Features are z-scored using train statistics; prediction is the
    nearest centroid in Euclidean space. Deterministic and inspectable.
    """

    classes: list[str]
    centroids: np.ndarray  # (n_classes, n_feats)
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, y: list[str]) -> NearestCentroid:
        classes = sorted(set(y))
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        Z = (X - mean) / std
        centroids = np.stack([Z[[i for i, yi in enumerate(y) if yi == c]].mean(axis=0)
                              for c in classes])
        return cls(classes=classes, centroids=centroids, mean=mean, std=std)

    def predict(self, X: np.ndarray) -> list[str]:
        Z = (X - self.mean) / self.std
        # (n, 1, f) - (1, k, f) -> (n, k) distances
        d = np.linalg.norm(Z[:, None, :] - self.centroids[None, :, :], axis=2)
        idx = d.argmin(axis=1)
        return [self.classes[i] for i in idx]


def authorship_eval(personas: dict) -> dict[str, float]:
    """Train a centroid classifier on train docs, evaluate on held-out docs.

    Returns accuracy + per-class support. Chance level is 1/n_personas; we report
    it explicitly so a null result is honest (no claim of "success").
    """
    keys = list(personas.keys())
    X_tr, y_tr = [], []
    for k in keys:
        for _, text in personas[k]["train_docs"]:
            X_tr.append(stylometric_features(text))
            y_tr.append(k)
    clf = NearestCentroid.fit(np.stack(X_tr), y_tr)

    X_te, y_te = [], []
    for k in keys:
        for _, text in personas[k]["holdout_docs"]:
            X_te.append(stylometric_features(text))
            y_te.append(k)
    preds = clf.predict(np.stack(X_te))
    correct = sum(int(p == t) for p, t in zip(preds, y_te, strict=False))
    n = len(y_te)
    return {
        "authorship_accuracy": correct / max(1, n),
        "authorship_chance": 1.0 / max(1, len(keys)),
        "authorship_n_holdout": float(n),
    }


# --------------------------------------------------------------------------- #
# 2) Preference prediction                                                       #
# --------------------------------------------------------------------------- #
def preference_eval(personas: dict, predictor: Callable[[str, str, str, str], str]) -> dict[str, float]:
    """Predict each persona's choice between (choice, alt) for every preference.

    ``predictor(persona_key, topic, option_a, option_b) -> chosen_option`` is the
    system under test. Options are presented in a fixed (sorted) order so a system
    cannot cheat off positional bias. Returns accuracy (chance = 0.5).
    """
    correct, total = 0, 0
    per_topic: dict[str, list[int]] = {}
    for k, pdata in personas.items():
        for pref in pdata["preferences"]:
            a, b = sorted([pref["choice"], pref["alt"]])
            chosen = predictor(k, pref["topic"], a, b)
            hit = int(chosen == pref["choice"])
            correct += hit
            total += 1
            per_topic.setdefault(pref["topic"], []).append(hit)
    out = {"preference_accuracy": correct / max(1, total), "preference_chance": 0.5,
           "preference_n": float(total)}
    for topic, hits in per_topic.items():
        out[f"preference_acc_{topic}"] = sum(hits) / max(1, len(hits))
    return out


def principal_lookup_predictor(personas: dict) -> Callable[[str, str, str, str], str]:
    """A predictor that reads the persona's PRINCIPAL.md / structured preferences.

    This models a GA that has correctly captured the principal: it answers from
    the maintained PRINCIPAL.md. It is the "oracle-ish" upper bound for systems
    that actually personalize (RAG/LoRA/dynamic-eval should approach it).
    """
    table: dict[tuple[str, str], str] = {}
    for k, pdata in personas.items():
        for pref in pdata["preferences"]:
            table[(k, pref["topic"])] = pref["choice"]

    def predict(persona_key: str, topic: str, a: str, b: str) -> str:
        choice = table.get((persona_key, topic))
        if choice in (a, b):
            return choice
        # Fall back to scanning the principal text for either option.
        text = personas[persona_key]["principal_md"].lower()
        return a if text.find(a.lower()) <= text.find(b.lower()) else b

    return predict


def majority_predictor(personas: dict) -> Callable[[str, str, str, str], str]:
    """A non-personalized baseline: always pick the globally most common option.

    This is the "generic assistant" null model — it ignores *who* is asking, which
    is exactly the failure mode GAs are designed to fix. Useful as a floor.
    """
    counts: Counter = Counter()
    for pdata in personas.values():
        for pref in pdata["preferences"]:
            counts[pref["choice"]] += 1

    def predict(persona_key: str, topic: str, a: str, b: str) -> str:
        return a if counts[a] >= counts[b] else b

    return predict


# --------------------------------------------------------------------------- #
# 3) Held-out response reconstruction similarity                                 #
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(strip_annotations(text).lower())


def cosine_overlap(a: str, b: str) -> float:
    """Cosine similarity of bag-of-words vectors (1.0 = identical word mix)."""
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    va = np.array([ca[w] for w in vocab], dtype=np.float64)
    vb = np.array([cb[w] for w in vocab], dtype=np.float64)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom > 0 else 0.0


def reconstruction_eval(personas: dict, reconstructor: Callable[[str, str], str]) -> dict[str, float]:
    """Score how well a system reconstructs held-out docs from a prompt.

    ``reconstructor(persona_key, title) -> reconstructed_text``. We compare to the
    true held-out doc with bag-of-words cosine. This is a *similarity* metric, not
    a quality claim; higher means the system recovered more of the held-out
    vocabulary/voice.
    """
    sims = []
    for k, pdata in personas.items():
        for _path, text in pdata["holdout_docs"]:
            title = _title_of(text)
            recon = reconstructor(k, title)
            sims.append(cosine_overlap(recon, text))
    arr = np.asarray(sims, dtype=np.float64) if sims else np.zeros(1)
    return {
        "reconstruction_cosine_mean": float(arr.mean()),
        "reconstruction_cosine_min": float(arr.min()),
        "reconstruction_n": float(len(sims)),
    }


def _title_of(text: str) -> str:
    m = re.search(r"<!--\s*title:\s*(.*?)\s*-->", text)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# 4) Pairwise judge scaffolding (MOCK / HEURISTIC — clearly marked)              #
# --------------------------------------------------------------------------- #
def build_judge_prompt(persona_principal_md: str, prompt: str, resp_a: str, resp_b: str) -> str:
    """Render the prompt a *real* LLM judge would receive. Plumbing only.

    In a real GA this string would be sent to a frozen-but-trusted judge model (or
    the principal). Here we only build it; scoring is done by the heuristic judge
    below, which is explicitly NOT a real preference model.
    """
    return (
        "You are judging which response better matches the PRINCIPAL below.\n"
        "Answer with exactly 'A' or 'B'.\n\n"
        f"=== PRINCIPAL ===\n{persona_principal_md.strip()}\n\n"
        f"=== TASK ===\n{prompt}\n\n"
        f"=== RESPONSE A ===\n{resp_a}\n\n=== RESPONSE B ===\n{resp_b}\n"
    )


def heuristic_judge(persona_principal_md: str, resp_a: str, resp_b: str) -> str:
    """MOCK judge: pick the response whose word mix is closer to the principal.

    HONEST STUB. This is a transparent stylometric proxy standing in for a real
    LLM/human judge. It must never be reported as a learned preference model.
    Returns 'A' or 'B'.
    """
    sa = cosine_overlap(resp_a, persona_principal_md)
    sb = cosine_overlap(resp_b, persona_principal_md)
    return "A" if sa >= sb else "B"


def judge_eval(personas: dict, responder_good: Callable[[str], str],
               responder_bad: Callable[[str], str]) -> dict[str, float]:
    """Run the mock judge on (in-voice vs. generic) response pairs.

    ``responder_good``/``responder_bad`` map a persona_key to a response. We expect
    a working judge proxy to prefer the in-voice response above chance; we report
    the rate plainly (it is a heuristic, not a result about a real judge).
    """
    wins, total = 0, 0
    for k, pdata in personas.items():
        good, bad = responder_good(k), responder_bad(k)
        # Randomize-free fixed assignment: A=good, then also B=good to cancel bias.
        for a, b, good_is in ((good, bad, "A"), (bad, good, "B")):
            verdict = heuristic_judge(pdata["principal_md"], a, b)
            wins += int(verdict == good_is)
            total += 1
    return {"judge_in_voice_win_rate": wins / max(1, total),
            "judge_chance": 0.5, "judge_n": float(total),
            "judge_is_mock": 1.0}  # flag: this is a heuristic stub, not a real judge


# --------------------------------------------------------------------------- #
# Convenience: run the full eval suite for a named "system under test"           #
# --------------------------------------------------------------------------- #
def run_all_evals(personas: dict,
                  pref_predictor: Callable[[str, str, str, str], str],
                  reconstructor: Callable[[str, str], str]) -> dict[str, float]:
    """Aggregate authorship + preference + reconstruction metrics into one dict.

    Authorship is system-independent (it probes the corpus itself), so it is the
    same across variants; preference + reconstruction depend on the supplied
    system callables.
    """
    out: dict[str, float] = {}
    out.update(authorship_eval(personas))
    out.update(preference_eval(personas, pref_predictor))
    out.update(reconstruction_eval(personas, reconstructor))
    return out
