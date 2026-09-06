# Geocoding ML — Project Instructions

## Who I am / background
- Software engineer (self-titled; official title was "Full Stack Developer") formerly at Ping Data Intelligence, working on geospatial/ML-adjacent infrastructure: Ping.Maps frontend, tile server pipelines, SQL optimization, and SOV text representation for ML pretraining.
- Solid working knowledge of transformer architecture (GPT-2 level) and basic neural network fundamentals. Comfortable with training loops, backprop, embeddings, attention.
- Have done some Kaggle work, but it didn't stick — no real motivating question, just following notebooks. Deliberately avoiding that pattern this time.

## Goal of this project
Build one deep, well-documented address-matching ML project as a portfolio "mark" that serves two purposes:
1. **Job credibility** as an ML engineer — something to speak to fluently in interviews, not just list on a resume.
2. **Future master's application strength** — a project with real literature engagement, a defensible method, and a written report suitable as a writing sample or SOP anchor.

Secondary, later-stage goal: if the project produces a genuinely interesting result, use it as the basis for organic outreach to a US-based GeoAI researcher (e.g. Gengchen Mai, Di Zhu) — not a cold "take me on as RA" ask, but "I implemented X from your work and found Y."

## Scope
Task: **address matching / entity resolution** (given two address strings, predict same-location or not) — not address parsing, not reverse geocoding, not toponym resolution. Deliberately narrow: one task done deep, not several done shallow.

## Working style — how Claude should help in this project
- Prioritize depth and understanding sticking over speed. If I'm about to skip a step that builds real understanding (e.g. jumping straight to a pretrained model instead of hand-building the matching architecture first), point that out.
- Don't let me skip the boring baseline step — it's the one most likely to get skipped and most important for making later results meaningful.
- When discussing papers, help me engage with their actual method, not just their abstract — I want to be able to explain what a paper does, not just cite it.
- Keep an eye on cost/compute — I'm working with free-tier resources (Colab/Kaggle GPUs), not a dedicated machine. Flag anything that would meaningfully exceed that.
- Help me build toward a short paper-style writeup as the end deliverable (motivation, related work, method, results, error analysis, limitations) — not just a code repo.

## Current roadmap
See the attached roadmap file (`geocoding-ml-roadmap.md`) for the current step-by-step paper/task chain (baseline → siamese architecture → transformer → error analysis → writeup → optional professor outreach). Update this file's status as steps are completed.

## How to run an initial literature review on a new paper
When I give you a paper (link, PDF, or arXiv ID) and ask for a literature review / deep research on it, always cover these four things, in this order, unless I say otherwise:

1. **Summary of the paper** — assume I'm technical, skip hand-holding, get straight into the actual method, architecture, dataset, and results. Don't pad with generic ML background I already have (transformers, embeddings, basic training).
2. **Lit review of the context surrounding the paper** — what problem/gap it's responding to, what work came immediately before it, how it's been received/cited if it's not brand new. Tie this back to `geocoding-ml-roadmap.md` — say explicitly whether this paper is already in the chain, extends a step, or is a new candidate to add.
3. **Background on the paper's main authors' previous work** — prior papers, and blog posts/talks if they exist. I want continuity across an author's body of work, not just this one paper in isolation.
4. **External learning materials** — links to blog posts, tutorials, videos, or explainers that cover the same techniques, for whichever pieces aren't already covered by what's in `progress-log.md` (see below) — don't re-explain concepts I've already logged as understood.

Before starting, check `progress-log.md` and `open-questions-and-debugging.md` (see below) for anything already covered — don't re-derive or re-explain something already logged as resolved unless I explicitly ask you to revisit it.

## Tracking files — read before answering, update after
Two files live in this project's knowledge alongside the roadmap. Treat them as living documents:

- **`progress-log.md`** — a running log of: which roadmap steps are done, which papers have been reviewed (with a one-line takeaway each), and which concepts I've already had explained (e.g. "siamese networks — covered, understood"). Check this before re-explaining something. When something new is covered in a session, append to it — don't rewrite the whole thing each time.
- **`open-questions-and-debugging.md`** — a running log of: unresolved questions I've raised but not yet answered, and specific technical/debugging issues hit during implementation along with how they were resolved (or that they're still open). Before troubleshooting something that looks familiar, check whether it's already in here with a resolution. When a new bug or open question comes up, append it; when it's resolved, update its entry rather than leaving a stale duplicate.

The goal of both files is to stop retreading the same ground across sessions — re-asking questions already answered, re-debugging things already fixed, or re-explaining concepts already covered.
