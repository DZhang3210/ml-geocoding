# Progress Log — Geocoding ML Project

Update this file as work happens. Append new entries; don't rewrite old ones unless correcting them.

## Roadmap steps completed
- **Step 1 (dumb baseline) — pipeline built, 🔶 in progress (2026-09-05).** Code lives in `step1-baseline/` (`generate_dataset.py`, `features.py`, `train.py`). Features: raw/number/street-name/street-type/zip similarity via jellyfish Jaro-Winkler, fed into sklearn LogisticRegression. Ran against a synthetic placeholder dataset (positives = reformatted addresses, hard negatives = number-offset / similar-street-name) split by address id 70/15/15. Test F1 = 0.9945, but flagged as inflated by easy synthetic negatives — real benchmark number pending Step 2's real dataset. See `step1-baseline/results.md` for full writeup and the honest caveat.

## Papers reviewed
| Paper | Date reviewed | One-line takeaway |
|---|---|---|
| Lee, Claridades & Lee (2020), "Improving a Street-Based Geocoding Algorithm Using Machine Learning Techniques" (Applied Sciences 10(16):5628) | 2026-09-05 | Feature-engineered classical baseline (17 string-similarity metrics -> XGBoost) hits 96%+ accuracy on Korean street addresses without any embeddings/neural net; added to roadmap as Step 1b, a stronger classical floor to beat before Steps 3-5. |

## Concepts already explained / understood
- (none logged yet)

## Background already established
- Software engineer, ex-Ping Data Intelligence (Ping.Maps, tile server pipelines, SQL optimization, SOV text representation for ML pretraining)
- Strong GPT-2-level transformer/NN background coming in
- Working on address matching as the chosen narrow task (not parsing, not reverse geocoding)
