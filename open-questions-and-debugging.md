# Open Questions & Debugging Log — Geocoding ML Project

Update this file as work happens. Mark items resolved rather than deleting them, so there's a record.

## Open questions
(none yet)

## Resolved questions
(none yet)

## Debugging log
| Issue | Context (step/file/code) | Resolution | Status |
|---|---|---|---|
| Claude fully rebuilt the Step 1 baseline pipeline (generate_dataset.py/features.py/train.py) instead of guiding David through writing it himself, defeating the point of the project. | 2026-09-13, "start with a dumb baseline" request | Added an explicit "do not write implementation code for David" rule to the top of CLAUDE.md and to project-instructions.md's Working style section — Claude explains/reviews/debugs but does not generate finished implementation files, at any step. | Resolved |
