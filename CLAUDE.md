# CLAUDE.md — Geocoding ML Project

This folder is a self-contained ML portfolio project: building one deep, well-documented **address-matching / entity-resolution** system (given two address strings, predict same-location or not) as a credibility piece for ML engineer job applications and future master's applications. Any Claude session opened in this folder should treat this file as the entry point and follow the checklist below before doing anything else.

## Start-of-session checklist (do this first, every session)

1. **Read `project-instructions.md`** — David's background, the project's goals, scope (address matching only — not parsing, not reverse geocoding, not toponym resolution), and the full working-style contract for how Claude should help (protect the baseline step, keep an eye on free-tier compute limits, push for real paper engagement, build toward a paper-style writeup). Treat it as binding for this whole folder.
2. **Read `geocoding-ml-roadmap.md`** — the step-by-step paper/task chain with a status column (⬜ not started · 🔶 in progress · ✅ done). Pick up work at the first non-✅ step unless told otherwise.
3. **Read `progress-log.md`** — roadmap steps already completed, papers already reviewed (with takeaways), and concepts already explained to David. Do not re-derive or re-explain anything already logged here.
4. **Read `open-questions-and-debugging.md`** — open questions and the debugging log. Check here before re-troubleshooting something that looks familiar.
5. **Check for `Article N/` subfolders** — each one holds a paper under consideration or mid-review (often just a link file dropped in ahead of a full review). Read any notes there; a paper sitting in one of these folders may not yet be reflected in the roadmap or progress log.

## End-of-session checklist (do this before finishing)

- Append new entries to `progress-log.md` (don't rewrite old ones): steps completed, papers reviewed, concepts explained.
- Append to `open-questions-and-debugging.md`: new open questions, new bugs and their resolution status (mark resolved rather than deleting).
- Update the status column in `geocoding-ml-roadmap.md` if a step's state changed.
- If a paper from an `Article N/` folder got reviewed this session, fold the outcome into both the roadmap (as a new step, a modification to an existing step, or an explicit "considered, not adopted" note) and `progress-log.md` — don't leave the takeaway stranded only in the article folder.

## How to run a literature review on a new paper

When given a paper (link, PDF, arXiv ID, or a new `Article N/` folder), cover these four things in order, per `project-instructions.md`:

1. Summary of the paper's actual method/architecture/dataset/results — technical, no generic ML background padding.
2. Lit review context — what gap it addresses, what precedes it, and **explicitly** whether it's already in `geocoding-ml-roadmap.md`, extends an existing step, or is a new candidate step.
3. Background on the paper's main authors' other work.
4. External learning materials for any technique not already covered in `progress-log.md`.

## Roadmap chain at a glance

Core chain: dumb baseline (Comber & Arribas-Bel 2019) → real dataset → hand-built siamese architecture (Reimers & Gurevych 2019, Sentence-BERT) → reproduce a published DL approach (Lin et al. 2020, ESIM) → pretrained transformer encoder (Siamese Transformer Networks, arXiv 2307.02300) → error analysis (Kilic et al. 2024) → writeup → position within field taxonomy (2024 IJGI survey). Optional stretch: geographic feature fusion (GeoBERT/GeoRoBERTa); later-stage: professor outreach. Full detail, implementation instructions, and finish thresholds live in `geocoding-ml-roadmap.md` — this is just an index so a new session recognizes step names before opening that file.

As of the last update, a **Step 1b** was added: Lee, Claridades & Lee (2020, *Applied Sciences* 10(16):5628) — a feature-engineered classical-ML baseline (17 string-similarity metrics feeding SVM/Random Forest/XGBoost) that hits 96%+ accuracy on Korean street addresses. It sits between Step 1 and Step 2 as a stronger, still-classical baseline to beat before moving to siamese/transformer architectures — see the roadmap file for full detail.

## Folder contents

- `project-instructions.md` — background, goals, scope, working-style contract. Read first.
- `geocoding-ml-roadmap.md` — the living step-by-step plan with status tracking.
- `progress-log.md` — running log of completed steps, reviewed papers, explained concepts.
- `open-questions-and-debugging.md` — running log of open questions and debugging history.
- `Article 1/` — Lee, Claridades & Lee (2020) MDPI *Applied Sciences* paper; reviewed and integrated into the roadmap as Step 1b (see `Article 1/Article 1.md` for the paper link and review pointer).
- `dummy.py` — scratch/connectivity-test file, not part of the project itself.
