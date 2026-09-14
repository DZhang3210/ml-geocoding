# Progress Log — Geocoding ML Project

Update this file as work happens. Append new entries; don't rewrite old ones unless correcting them.

## Roadmap steps completed
- **Step 1 (dumb baseline) — CORRECTION (2026-09-13): never actually saved, treat as not done.** The pipeline described below was reported as built on 2026-09-05, but `step1-baseline/` does not exist on disk and never appears in git history — it was lost (likely built in a session whose work wasn't committed). Resetting roadmap status to ⬜. Original (now-stale) description, kept for reference: code was reported to live in `step1-baseline/` (`generate_dataset.py`, `features.py`, `train.py`), features being raw/number/street-name/street-type/zip similarity via jellyfish Jaro-Winkler fed into sklearn LogisticRegression, run against a synthetic placeholder dataset split 70/15/15 by address id, reporting test F1 = 0.9945 (flagged even then as inflated by easy synthetic negatives). None of this should be cited or relied on.
- **Step 1b (stronger feature-engineered baseline) — 🔶 in progress (as of 2026-09-13).** Actively being worked on now, in parallel with Step 2 dataset setup.
- **Step 2 (real dataset) — 🔶 in progress (as of 2026-09-13), not previously logged.** Downloaded a real Stamford, CT address dataset (shapefile) into `Data/us-ct-city_of_stamford/`. Wrote `Data/Data_Helpers/convert_file_to_csv.py` (uses `pyshp`) to convert it to `output.csv` — 28,294 rows with columns StreetName, Address, ZipCode, X_COORD, Y_COORD. Started exploring it in `Data/Data_Helpers/stamford_analysis.ipynb` (currently just loads and previews the CSV; has uncommitted changes). Not yet done: address-component parsing, positive/hard-negative pair generation, or the location-based train/val/test split.

## Papers reviewed
| Paper | Date reviewed | One-line takeaway |
|---|---|---|
| Comber & Arribas-Bel (2019), "Machine learning innovations in address matching: A practical comparison of word2vec and CRFs" (Transactions in GIS 23(2):334-348) | 2026-09-13 | This is Step 1's actual paper -- CRF-based address parsing (Libpostal, pretrained on OSM) feeding Jaro-Winkler comparison vectors into logistic/RF/XGBoost (best: XGBoost, precision 0.955, recall 0.902), vs. swapping in word2vec cosine-similarity comparison vectors (best: XGBoost, precision 0.950, recall 0.870 -- competitive but doesn't win). Negative pairs generated synthetically via FEBRL character-level corruption of real matched pairs, not random pairing. Notably, their "baseline" is already parser+comparison-vector+classifier, not a single raw-string score -- worth knowing before implementing our own dumb baseline. |
| Lee, Claridades & Lee (2020), "Improving a Street-Based Geocoding Algorithm Using Machine Learning Techniques" (Applied Sciences 10(16):5628) | 2026-09-05 | Feature-engineered classical baseline (17 string-similarity metrics -> XGBoost) hits 96%+ accuracy on Korean street addresses without any embeddings/neural net; added to roadmap as Step 1b, a stronger classical floor to beat before Steps 3-5. |

## Concepts already explained / understood
- (none logged yet)

## Background already established
- Software engineer, ex-Ping Data Intelligence (Ping.Maps, tile server pipelines, SQL optimization, SOV text representation for ML pretraining)
- Strong GPT-2-level transformer/NN background coming in
- Working on address matching as the chosen narrow task (not parsing, not reverse geocoding)
