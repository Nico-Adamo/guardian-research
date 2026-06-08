# PRINCIPAL.md — template

> This is a **template** for a real principal's `PRINCIPAL.md`. A real, filled-in
> `PRINCIPAL.md` is **private persona data** and must **never** be committed to
> this (or any) repository. Keep your real file outside the repo, or under
> `data/persona_private/` (gitignored) and name it `*.principal.private.md`
> (also gitignored). The repo only ships this empty template and the *synthetic*
> personas under `data/persona_synthetic/`.

`PRINCIPAL.md` is the Guardian Angel's (GA) living model of *who you are and who
you are becoming*. It is the document the GA conditions on for RAG, the seed for
dynamic-evaluation finetuning, and the artifact your active-learning answers
update over time. It is owned by you — **mental sovereignty**: nothing here is
imposed by a vendor's "assistant" defaults.

Three GA principles to keep in mind while filling this in:
- **Enhancement, not replacement.** Describe how you want to be *amplified*, not
  what a generic assistant should do on your behalf.
- **Mental sovereignty.** Your preferences and values are first-class and owned
  by you; the GA must not optimize against them.
- **Self-actualization.** Include what you are *trying to become*, not just a
  static snapshot — give the GA something meaningful to grow toward with you.

---

## Identity (pragmatic)
A short description of you as personality + values + preferences (the pragmatic
definition of identity used in the GA design). Not your résumé — your *voice*.

> _e.g._ "I am a systems-minded researcher who values correctness over cleverness
> and would rather ship something boring and true than something clever and
> fragile."

## Style fingerprint
How you write and speak, so the GA can emulate (not flatten) your voice.
- Signature openers / connectives / sign-offs:
- Typical sentence length (terse vs. expansive):
- Tone (plain, ornate, wry, formal, ...):
- Pet words / phrases you actually use:
- Words/constructions you avoid:
- Formatting habits (bullets vs. prose, footnotes, em-dashes, ...):

## Preferences (binary, owned by you)
List concrete either/or preferences. These are directly testable by the GA's
preference-prediction eval. Add a one-line *why* where it matters.
- topic — prefer _X_ over _Y_ (because ...):
- topic — prefer _X_ over _Y_ (because ...):

## Values
What breaks ties when options are otherwise equal. Ranked if you can.
- ...

## Aspirations (self-actualization)
What you are trying to get better at / become. The GA should help you grow here.
- ...

## Boundaries / refusals
Things the GA should never do in your name, and outputs you would not publish
under your name. (Also a security asset: a well-specified principal makes prompt
injection and "confused deputy" attacks far harder.)
- ...

## Open questions for active learning
Things you are still working out. The GA can prioritize asking about these (see
`ga persona questions`).
- ...

---
<!--
GA annotation convention used in source documents:
  <!-- GA: important: preference (topic) -->   marks a load-bearing preference span
  <!-- GA: style: signature-closer -->         marks a stylistic signature span
These let tooling lift load-bearing spans from your documents into this file.
-->
