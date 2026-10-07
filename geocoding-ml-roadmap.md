# Geocoding ML Roadmap — Address Matching Project

**Task scope:** Address matching / entity resolution only (not parsing, not reverse geocoding, not toponym resolution).
**Sequence:** Depth-first on one task — each step builds on the last; only the last two steps branch outward.
**Time estimates assume full-time study: ~6-8 hours/day.**

---

## At a glance

| #   | Step                                  | Paper                                           | Est. time | Status |
| --- | ------------------------------------- | ----------------------------------------------- | --------- | ------ |
| 1   | Dumb baseline                         | Comber & Arribas-Bel (2019)                     | 1-2 days  | ✅     |
| 1b  | Stronger feature-engineered baseline  | Lee, Claridades & Lee (2020)                    | 1-2 days  | 🔶 (paused) |
| 2   | Real dataset                          | —                                               | 2-3 days  | ✅     |
| 2b  | Real cross-source pair dataset        | — (Overture Maps Places + bridge files)         | 3-5 days  | 🔶     |
| 2c  | Rule-based baseline                   | libpostal (normalization + dedupe)              | 1 day     | ⬜      |
| 3   | Hand-built matching architecture      | Reimers & Gurevych (2019)                       | 4-6 days  | ⬜      |
| 4   | Reproduce a published DL approach     | Lin et al. (2020)                               | 4-5 days  | ⬜      |
| 5   | Swap in pretrained transformer        | Siamese Transformer Networks (arXiv 2307.02300) | 3-4 days  | ⬜      |
| 6   | Real error analysis                   | Kilic et al. (2024)                             | 2-3 days  | ⬜      |
| 7   | Write it up                           | —                                               | 3-4 days  | ⬜      |
| 8   | Position within field taxonomy        | 2024 survey (IJGI 13(4):138)                    | 1 day     | ⬜      |
| 9   | *(stretch)* Geographic feature fusion | GeoBERT, GeoRoBERTa                             | 3-5 days  | ⬜      |
| 10  | *(later-stage)* Professor outreach    | —                                               | 1 day     | ⬜      |

**Core chain (Steps 1-8) total: roughly 20-28 working days**, i.e. about 4-6 weeks at full-time pace. This is a rough planning estimate, not a deadline — expect steps 3-5 in particular to run long the first time through, since that's where most of the actual new learning happens.

Update the status column as you go: ⬜ not started · 🔶 in progress · ✅ done

**Two-dataset evaluation (added 2026-10-05):** from Step 2b onward, report every model on both the Stamford synthetic set (Step 2) and the Overture real cross-source set (Step 2b). Train-on-synthetic → test-on-real is a planned writeup result.

---

## Week-by-week timeline (assuming a 5-day working week at 6-8 hrs/day)

| Week            | Steps covered                           | Notes                                                                                                                |
| --------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Week 1          | Step 1 (finish) + Step 2 + start Step 3 | Baseline and dataset can overlap — build the dataset first, then the baseline drops in quickly.                      |
| Week 2          | Finish Step 3                           | Budget the full week here if needed — this is the hardest new concept in the chain. Don't compress it to hit a date. |
| Week 3          | Step 4 + start Step 5                   | Reproducing Lin et al. should feel noticeably faster than Step 3 since the siamese scaffolding already exists.       |
| Week 4          | Finish Step 5 + Step 6                  | Transformer fine-tuning debugging can spill a day or two into error analysis time — that's fine.                     |
| Week 5          | Step 7 + Step 8                         | Writing typically takes longer than expected — don't schedule anything else this week if avoidable.                  |
| Week 6 (buffer) | Overflow / Step 9 if ahead of schedule  | Use as slack first; only start Step 9 if Weeks 1-5 actually finished on time.                                        |

This is a planning aid, not a commitment — if Week 2 runs into Week 3, that's normal and expected, not a sign of falling behind.

---

## Core chain (do in order)

### 1 — Build the dumb baseline first
- **Paper:** Comber & Arribas-Bel (2019), *"Machine learning innovations in address matching: A practical comparison of word2vec and CRFs"*
  - DOI / publisher: https://doi.org/10.1111/tgis.12522
  - Open-access copy: https://livrepository.liverpool.ac.uk/3038194/
- **Est. time:** 1-2 days.
- **Task:** Levenshtein / Jaro-Winkler string similarity + a simple classifier (logistic regression) on hand-crafted similarity features.
- **Implementation instructions:**
  1. Use `python-Levenshtein` or `jellyfish` for string distance functions — don't hand-roll the edit-distance algorithm, that's not where the learning value is.
  2. For each address pair, compute a small feature vector: raw string similarity, similarity of just the street-number token, similarity of just the street-name token, similarity of ZIP if present.
  3. Fit `sklearn.linear_model.LogisticRegression` on these features against your labeled match/no-match pairs.
  4. Report precision, recall, F1 — not just accuracy, since match/no-match is usually imbalanced.
- **Finish threshold:** You have a single number (F1 on a held-out test split) that every later model must be compared against, and you can explain in one sentence why accuracy alone would have been misleading for this data.
- **Status note (2026-09-13):** The pipeline previously reported here (`step1-baseline/`, F1 = 0.9945 on synthetic data) was never committed to git and is no longer on disk — it did not survive between sessions. Treating Step 1 as **not actually done**; resetting status to ⬜. Not currently being redone in isolation — work has moved to Step 1b and to Step 2 dataset setup in parallel (see their notes below), and Step 1's simple single-feature baseline can be rebuilt quickly once real pair data exists, as a first row in the same results table Step 1b feeds into.
- **Status note (2026-09-21): ✅ done.** Built in `Data/Data_Helpers_old/dumb_baseline.ipynb` — three Jaro-Winkler features (street name, house number, ZIP) into plain default `LogisticRegression`. Along the way, found and fixed a real bug in Step 2's dataset (train/val/test had inconsistent label ratios with each other; see Step 2's own 2026-09-21 status note) and recalibrated the positive:negative ratio to ~1:9 to match Comber & Arribas-Bel's own reported pair counts. Final test-set result: class 1 (match) F1 0.81 (precision 0.88, recall 0.75), class 0 (no-match) F1 0.98, accuracy 0.96 vs. a 90.0% trivial-baseline floor. Full detail in `progress-log.md`. This is the number every later step (1b, 3, 4, 5) must now be compared against.

### 1b — Build a stronger feature-engineered baseline (optional but recommended)
- **Paper:** Lee, Claridades & Lee (2020), *"Improving a Street-Based Geocoding Algorithm Using Machine Learning Techniques"*, Applied Sciences 10(16):5628.
  - DOI / publisher: https://www.mdpi.com/2076-3417/10/16/5628
- **Est. time:** 1-2 days.
- **Task:** Extend the Step 1 baseline from a single string-distance feature to a wider bank of similarity metrics feeding a gradient-boosted classifier, rather than jumping straight to a learned architecture. This is still classical ML (no embeddings, no neural net), so it belongs before Step 3, not after — it raises the bar Step 3 actually has to clear.
- **How it fits the chain:** New candidate step, not a replacement for Step 1. Comber & Arribas-Bel gives you the "one obvious feature, one linear classifier" floor; this paper shows that floor can be pushed much higher (96%+ on their Korean street-address data) purely with more string-similarity features and a stronger classical classifier, before any representation learning is involved. Worth reproducing so the later siamese/transformer steps (3-5) are compared against a genuinely strong classical baseline, not a token one.
- **Implementation instructions:**
  1. Compute a bank of string-similarity features per address-component pair rather than just one: edit-based (Levenshtein, Jaro, Jaro-Winkler sorted, Jaro-Winkler reversed, Hamming) and token-based (Jaccard, Cosine, Sorensen-Dice, Overlap, Tversky, Monge-Elkan) — `jellyfish` and `py_stringmatching`/`textdistance` cover most of these without hand-rolling.
  2. Train and compare three classifiers on the resulting feature matrix: `sklearn.svm.SVC`, `sklearn.ensemble.RandomForestClassifier`, and `xgboost.XGBClassifier` — the paper finds XGBoost best and most stable across noise levels.
  3. Do a feature-selection pass (their paper finds 9 of 17 metrics sufficient for >97% accuracy) — this is a good opportunity to practice feature-importance analysis (XGBoost's built-in importances, or a simple ablation) rather than keeping every feature by default.
  4. Use your own Step 2 dataset and its train/val/test split (don't build a separate Korean-address dataset) so the comparison to Step 1 and later steps stays apples-to-apples; note explicitly if the noise characteristics of your dataset differ from their 30/50/70%-correct synthetic setup.
  5. Report precision/recall/F1 the same way as Step 1, plus a short note on which similarity metrics mattered most for your data (may differ from theirs, since English/US addresses behave differently from Korean road-name addresses).
- **Status note (2026-09-13):** In progress — actively being worked on now, in parallel with Step 2 dataset setup.
- **Status note (2026-09-21):** Substantial progress. Full edit-based bank done (JW from Step 1, plus normalized Levenshtein and Hamming). Token-based bank done for 5 of 6 planned metrics (Jaccard, Sørensen-Dice, Cosine, Overlap, Tversky via `textdistance`, character-bigram-based; Monge-Elkan deliberately skipped — degenerates on single-word street names). Current result with `LogisticRegression` on all 22 features: F1=0.88 on matches (up from Step 1's 0.81). Found several real bugs along the way (all logged in `open-questions-and-debugging.md`) plus some genuine, non-bug mathematical redundancies among the metrics (Tversky-default≡Jaccard; Dice≡Overlap≡Cosine on fixed-length `ZipCode`; Levenshtein≡Hamming on `ZipCode`) worth accounting for in the feature-selection pass. **Still remaining:** the `SVC`/`RandomForestClassifier`/`XGBClassifier` comparison (only `LogisticRegression` tried so far), and the actual feature-selection/ablation pass (deliberately deferred, not done).
- **Status note (2026-10-03):** Paused the classifier comparison for a timeboxed side lab: reimplementing a decision tree (then RF, then simple gradient boosting) from scratch in `Algorithm_Reimplementation/decision_tree.ipynb` to understand the models before using them — boolean-column splits first. Not part of the Step 1b pipeline; reported Step 1b numbers will still come from sklearn/xgboost. Also decided: all model selection/threshold tuning on **val**, test only once at the end (prior 0.81/0.88 numbers were test-set, untuned). See `progress-log.md`.
- **Status note (2026-10-03, later):** Side-lab decision tree done — from-scratch Gini tree with continuous midpoint thresholds produces identical predictions to sklearn's `DecisionTreeClassifier` on `make_classification`, and 2/114 disagreements on `load_breast_cancer` (attributed to `score_cutoff=0.01` and correlated-feature near-ties; not yet isolated). Next in the side lab: random forest (bootstrap + per-split feature subsets), then simple gradient boosting; then back to the SVC/RF/XGBoost comparison on val. Step 1b stays 🔶.
- **Status note (2026-10-04):** Side-lab random forest implemented (bootstrap + per-split feature subsampling; within a couple of points of sklearn at B=3). Ablation deferred. Moving on to simple gradient boosting (regression tree with variance criterion first).
- **Status note (2026-10-04, later):** Side-lab gradient boosting (regression) done — matches sklearn's `GradientBoostingRegressor` (R² 0.958 vs 0.958). Remaining side-lab option: boosting for classification (log-loss); otherwise return to Step 1b's sklearn/xgboost comparison on val.
- **Status note (2026-10-04, end of side lab):** Classification boosting (log-loss, Newton leaves) done and matching sklearn (max |Δp| 2.3e-5). **Side lab complete** — decision tree, random forest, GB regression, GB classification all reimplemented and validated. Next: Step 1b housekeeping, then SVC/RF/XGBoost comparison on val, feature-selection pass, single final test evaluation.
- **Status note (2026-10-04, later):** Added XGBoost (regression + classification, exact greedy, λ/γ/min_child_weight) to the side lab — both match the `xgboost` library (Δ ≤ 7e-5 regression, 5.6e-8 classification). Side lab now fully closed; `xgboost` 2.1.4 installed in venv (needed `brew install libomp`). Step 1b resumes at housekeeping.
- **Status note (2026-10-05): paused — pivoting to Step 2b.** Housekeeping done (feature cache fixed, requirements.txt complete). Val comparison, default hyperparameters, no class weighting, 22 features — class-1 (match) results: LogReg P 0.93 / R 0.94 / F1 0.94; SVC (RBF, scaled) 0.93 / 0.95 / 0.94; RF 0.94 / 0.93 / 0.94; XGBoost 0.92 / 0.94 / 0.93. Differences are 0–2 of 87 positive pairs — noise. The models mostly miss the **same** val rows → the feature set, not the classifier, is the bottleneck. Inspection of the shared misses: **false negatives** = heavily corrupted street names (e.g. `NEWFIELD AVE` vs `NUTFIEL AVN`); **false positives** = same house number, similar street name, *different street type* — and the 22 features never see `street_extension` or `directional` (parsing moved them out of `StreetName`). Planned fix (not built): standardize suffix via the USPS table, then an exact-match 0/1 feature (same for directional). **Deferred, not abandoned:** Step 1 3-feature model on val, the street-type feature, feature selection, single final test evaluation — David's call (2026-10-05) to do these once on the Overture data rather than twice, since the same issue is expected to recur there.
- **Finish threshold:** A results-table row for this model sitting between Step 1's baseline and Step 3's hand-built siamese model, plus one paragraph on whether the added feature engineering + XGBoost meaningfully beat the Step 1 single-feature baseline on your data, and by how much.

### 2 — Assemble a real, small dataset
- **Est. time:** 2-3 days.
- **Task:** Pull address data from OpenAddresses (https://openaddresses.io) or Census TIGER (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html). Construct positive pairs (same location, different formatting) and hard negative pairs (similar-looking, different location).
- **Implementation instructions:**
  1. Download one city/region extract rather than a whole country — tens of thousands of addresses is plenty.
  2. Generate positive pairs by applying synthetic formatting noise to real addresses (abbreviate "Street" to "St", drop unit numbers, swap word order) so you know the ground truth.
  3. Generate hard negatives deliberately — addresses on the same street with different numbers, or same number on visually similar street names — not just random pairs, which are too easy and won't stress-test your model.
  4. Split into train/val/test by *location*, not by row, so the same address doesn't leak across splits.
- **Status note (2026-09-13):** In progress — downloaded a real address dataset for Stamford, CT (`Data/us-ct-city_of_stamford/`, shapefile) and converted it to CSV (`Data/Data_Helpers_old/convert_file_to_csv.py` → `output.csv`, ~28,294 rows: StreetName, Address, ZipCode, X/Y coords). Exploratory look at the data underway in `Data/Data_Helpers_old/stamford_analysis.ipynb`. Positive/hard-negative pair generation and the location-based train/val/test split have not started yet.
- **Status note (2026-09-14):** Still in progress — built out street-suffix/prefix frequency analysis in `stamford_analysis.ipynb` and identified/handled directional qualifiers (N/S/E/W) that were polluting naive street-type suffix extraction (see progress-log.md and open-questions-and-debugging.md for full detail). This is real progress on the "address-component parsing" sub-task, but still only covers the street-name side — house-number and ZIP component parsing, positive/hard-negative pair generation, and the location-based train/val/test split are all still not started.
- **Status note (2026-09-15):** Street-name-side address-component parsing now essentially complete — `Directional`, `is_extensional` (`EXT`), and `street_extension` (real street-type suffix) all cleanly extracted and stripped from a base `StreetName`, each verified against the full dataset rather than assumed. Added a stable `id` column. Fixed a real ZIP leading-zero data bug (`pd.read_csv` dtype inference, not the source data). House-number component confirmed to need no parsing (already clean integers). Positive-pair generation is essentially done: sourced official USPS Pub 28 abbreviation tables, built abbreviation-substitution corruption (`street_extension`/`directional`, 70% per-field trigger) and character-level replace/delete corruption (`StreetName`, 10% per-char, capped at half the string length via rejection sampling) — both wired into `_corrupt_row`/`corrupt_rows`, producing corrupted rows that reference their source row via `original_id`. Hard-negative generation (both strategies) is also built now: Case 1 (exact street identity → same street/different number) via `dropna=False` groupby; Case 2 (Jaro-Winkler fuzzy street-name similarity, threshold 0.85, chosen empirically) via a full pairwise similarity table over unique street names (`jellyfish`, added to `requirements.txt`). Still remaining: explode both hard-negative candidate sets into actual labeled `(id_a, id_b, label)` pair-rows (via shared house number for Case 2), build the final combined positive+negative pairs table, decide the train/eval hard-negative-proportion policy (flagged open), and the location-based train/val/test split itself. Corruption/blocking logic intentionally still lives in the notebook, not `corrupt_data.py` — promotion to a module remains deliberately deferred until this logic fully stabilizes.
- **Status note (2026-09-17):** Easy-negative and hard-negative datasets built, positive-dataset generator reworked to sample without replacement, all three schemas converged, and combined into one final table (`_generate_training_data()`: 14,818 positive + 14,818 easy-negative + 14,818 hard-negative = 44,450 rows). Location-based train/val/test split built on top of that: split the 28,294 unique address ids (~68/17/15), then filtered pair-rows requiring both `id_a` and `id_b` to fall in the same id-group, dropping boundary-crossing pairs — 23,900 train / 3,356 val / 2,866 test (30,122 of 44,450 rows retained; ~32% dropped, concentrated in negatives, accepted as-is for now). Also reversed the 2026-09-16 "randomly corrupt one side of negatives" decision in favor of deterministic corruption (side `b` always corrupted) — see progress-log.md/open-questions file for full reasoning. **Update (2026-09-17, later same day): persisted to disk, step complete.** Saved `train_dataset`/`val_dataset`/`test_dataset` as `Data/Data_Helpers_old/{train,val,test}.parquet` (Parquet chosen over CSV specifically to avoid a repeat of the earlier `ZipCode` leading-zero dtype-inference bug — verified via round-trip reload that dtypes, including `ZipCode` as `str`, survive correctly). Final counts: 23,886 train / 3,376 val / 2,907 test rows. Finish threshold met — Step 2 marked ✅ done.
- **Status note (2026-09-16):** Both hard-negative cases fully exploded into pair-rows, and the positive dataset is now actually built. Case 1: fixed an append/extend nesting bug and a first-N-vs-random-sample bias, capped at `NUM_NEGATIVE=4` random ids/group (down from an uncapped 932,661 pairs dominated by a handful of streets) → 7,334 pairs, confirmed no duplicates. Case 2: fixed a `groupby(["StreetName","Address"])` bug that forced self-pairs, replaced a catastrophic ~400M-pair full-combinations fallback loop with proper `Address`-only blocking, switched `jw_lookup` from name-keyed to id-keyed so it emits `(id_a,id_b)` pairs directly, added directional/extension/is_extensional bonuses on top of raw Jaro-Winkler (fixed a bug where `is_extensional` rewarded "both False" — true for 99.99% of rows — instead of the intended rare "both True" case), and ran a component-ablation analysis confirming `street_extension`-match is the only bonus doing real discriminating work → 15,435 pairs at combined score ≥0.85, confirmed not concentrated (top-10 house numbers = only 4.7% of pairs, vs. Case 1's ~35% top-5-streets). Case 1 + Case 2 combined and deduped into `train_df_lst` (~22,767 pairs). `_generate_positive_dataset()` now correctly joins `corrupted_df` to `new_df` via `original_id`/`id` (`pd.merge(..., suffixes=("_a","_b"))`) → 28,294 genuine clean-vs-corrupted pairs; an earlier version that independently sampled both sides was a real bug (produced non-corresponding pairs) and was replaced. Final table schema locked in: `a_id, a_StreetName, a_Address, a_street_extension, a_directional, a_ZipCode` + same `b_`-prefixed set + `label` (ids kept as metadata for later error-analysis traceability, excluded only at feature-selection time, not dropped; coordinates excluded, `ZipCode` included). Two new design decisions, not yet implemented: (1) negative pairs will also get one randomly-chosen side corrupted via the same `_corrupt_row` mechanism used for positives, so "is this row corrupted" can't become a spurious signal correlated with `label`; (2) a third negative source, "easy negatives" (random unrelated clean/corrupted pairs, `label=0`), will be added alongside Case 1/2's hard negatives — stubbed but not built. Still remaining: build the easy-negative generator, implement symmetric negative-side corruption, assemble the final combined table, resolve the hard-negative train/eval proportion policy (still open), and the location-based train/val/test split. Full debugging/design history in progress-log.md and open-questions-and-debugging.md.
- **Status note (2026-09-21): dataset rebuilt.** Found (via an unexpectedly-high Step 1 result) that train/val/test had inconsistent label ratios with each other (train 58% no-match, test 77% match) due to a quadratic negative-retention asymmetry in the split-then-filter method. Rebuilt by splitting the id space into train/val/test *first*, then generating each split's positive/easy-negative/hard-negative pairs independently within its own id pool. Also recalibrated the target positive:negative ratio to ~1:9 (from Comber & Arribas-Bel's own reported 110,742:934,150), padding hard-negative shortfall (which shrinks quadratically at smaller pool sizes) with easy negatives. New verified balance: train 9.00:1, val 9.07:1, test 9.03:1. Full detail in `progress-log.md`. Step 2 remains ✅, but treat the `train`/`val`/`test.parquet` files as v2 — anything evaluated against the old files (none currently) would need rerunning.
- **Finish threshold:** A single dataset file (or a small set of them) with clearly labeled positive/negative pairs and a fixed train/val/test split you will reuse for every later step without re-splitting.

### 2b — Real cross-source pair dataset (Overture Maps) *(added 2026-10-05)*
- **Why:** Step 2's positives come from our own synthetic corruption (USPS abbreviations, typos), so a model can partly learn to invert our corruption function rather than real-world variation. Overture's Places theme conflates records from several providers (Meta, Microsoft, Foursquare, …) under one GERS ID; bridge files map each GERS ID to the provider record IDs. Two provider records under the same GERS ID = a naturally occurring positive pair with real formatting differences.
- **Starts after:** originally Step 1b ✅; revised same day (2026-10-05) — started with Step 1b paused after the val model comparison. Its deferred items (street-type feature, feature selection, final test eval) are folded into this step's re-scoring.
- **Scope decision (2026-10-05): address-only.** Use only the address fields of Places records (street, house number, locality, postcode, …). Place names, categories, phones and websites are excluded as features, even though Overture's own matcher uses them — keeps the project inside the address-matching scope.
- **Sources:** Places guide https://docs.overturemaps.org/guides/places/ · Addresses guide https://docs.overturemaps.org/guides/addresses/ · bridge files https://docs.overturemaps.org/blog/2025/05/21/release-notes/ · GERS https://docs.overturemaps.org/gers/gers-tutorial/
- **Implementation instructions:**
  1. **Feasibility check first (timebox ~half a day).** Bridge files give provider *IDs*; matching needs the provider's *raw address text*. Confirm which providers publish their raw records openly (Foursquare Open Source Places does; others unverified). If only one provider is retrievable, positives may have to be "provider raw vs. Overture conflated record" rather than provider-vs-provider — decide and log which.
  2. Pick one region (a US metro, ideally overlapping Connecticut/Stamford for comparability). Pull via Overture's GeoParquet on S3/Azure with a bounding-box filter (DuckDB) — never download a whole theme; stay within free-tier disk/RAM.
  3. **Label-noise caveat:** positives come from Overture's ML matcher, not human judgement — training on them means imitating another model, errors included. Hand-label a few hundred pairs as a clean test set and report agreement between Overture's labels and yours.
  4. Hard negatives: nearby records with different GERS IDs (same street different number, same number similar street) — reuse Step 2's blocking logic. Keep the ~1:9 positive:negative ratio for comparability, and split by location/id pool *before* pair generation (the Step 2 v2 lesson).
  5. Keep Stamford as the synthetic benchmark — don't delete or replace it.
- **Finish threshold:** Feasibility outcome logged; a train/val/test parquet set of real cross-source address pairs (address fields only), a hand-labeled clean test subset with measured Overture-label agreement, and Step 1/1b models re-scored on it — the first "synthetic vs. real" comparison.

- **Feasibility research note (2026-10-05, Claude, from docs + public S3 listing — David still to verify hands-on):** Bridge files live at `s3://overturemaps-us-west-2/bridgefiles/<release>/provider=<p>/theme=<t>/type=<t>/` (latest seen: `2026-09-23.1`); columns `id` (GERS), `record_id` (source id), `dataset`, `provider`, `update_time`, …; one GERS id appears on multiple rows when conflated from multiple sources. Providers with `theme=places`: alltheplaces, brightquery, dac, foursquare, krick, meta, microsoft, pinmeto, renderseo. Raw records openly downloadable: **Foursquare OS Places** (Apache 2.0, Parquet on S3/Hugging Face, has `address`/`locality`/`region`/`postcode`) and **AllThePlaces** (CC0, weekly GeoJSON, chain-store locators). Meta (≈59M of ≈81M Overture places) and Microsoft: no bulk raw download found. Likely pair designs: (a) Foursquare raw vs AllThePlaces raw under a shared GERS id; (b) Foursquare raw vs Overture's published address where the place's `sources` attribute the address to another provider (e.g. Meta). Labelling trap: different businesses at the same street address are *different places* but the *same location* — in this project's task definition those are positives, not negatives.

- **Positives built (2026-10-06):** `Data/Overture/ct_positives_2026-09-23.1.parquet` — 37,680 Foursquare cross-source pairs (bridge `provider='foursquare' AND provider <> published_from`, joined to `ct_places` = side A and `fsq_ct` = side B). Composition: 26,286 (70%) identical address strings (flag `identical_address`); 1,778 different leading house number (likely label noise — flag `number_mismatch`, exclude from training positives by default, oversample into the hand-labeled test set); 1,177 number missing on one side (flag `number_missing`, keep); 544 NULL Foursquare address (drop); ≈7,900 genuine formatting variation (abbreviations, unit/suite formats, ranges, ZIP+4, junk tokens like `\N\N`, real typos e.g. `DANBURY EOAD`, locality variants e.g. `SPRAGUE`/`BALTIC`). AllThePlaces deferred (see open questions).
- **Match definition decided (2026-10-06): building-level.** Same street address → same location; unit/suite designators are ignored. Consequences: different businesses at the same street address are *matches*; `STE 1` vs `STE 2` is a match; when generating negatives from nearby places with different GERS IDs, **exclude pairs whose unit-stripped street addresses are identical** (they'd be contradictory labels). Unit-level matching noted as a possible extension for the writeup.

- **Status note (2026-10-06, end of day): 🔶 in progress.** Done: `ct_places` (193,270), `ct_bridge` (58,694), `fsq_ct` (99.94% bridge coverage), `ct_positives` (37,680, with `identical_address` flag), building-level match definition. In progress: unit-stripping regex for `same_street_address` — the `#<token>` piece works (`\s*#\s*[A-Z0-9]+$`); designator words (UNIT/STE/SUITE/APT/STORE…) not yet. **Next:** finish unit stripping (word-or-`#` alternation, must not strip the last word of a plain address) → whitespace/punctuation normalization → rebuild `ct_positives` with `same_street_address`, `number_mismatch`, `number_missing` flags (drop NULL `address_b`) → hard negatives with the same-street-address exclusion rule → location-based split (by town/county) → hand-labeled clean test subset.
- **Status note (2026-10-07): 🔶 in progress — address cleaning done.** Built a 5-step DuckDB cleaning chain (notebook section 6): ordinal floors → designator units (optionally two designators) → `#` units → punctuation→space → whitespace collapse/trim. Saved `Data/Overture/ct_positives_clean_2026-09-23.1.parquet` (37,680 rows, 18 columns: the 15 originals + `address_a_clean`, `address_b_clean`, `same_street_address`). Results: same street address 29,802 (79%) vs. 26,286 (70%) identical raw strings; 3,516 of the 10,850 non-identical non-NULL pairs (32%) differed only by units/floors/punctuation; 7,334 positives still differ after cleaning; sanity check identical→not-same = 0. Remaining gaps (each <0.3%) accepted as known gaps — see open questions. **Next:** `number_mismatch` / `number_missing` flags (and drop NULL `address_b`) → hard negatives with the same-street-address exclusion rule (use `address_*_clean`) → location-based split → hand-labeled clean test subset.

### 2c — Rule-based baseline: libpostal *(added 2026-10-05)*
- **Source:** https://github.com/openvenues/libpostal — considered as a *data source* and **not adopted** for that: its training data (OSM + OpenAddresses, >100 GB) is tagged-token data for *parsing*, which is out of scope and has no match/non-match labels. Adopted instead as a **non-ML baseline**.
- **Task:** Use libpostal's normalization (`expand_address`) and its duplicate-detection functions to predict match/no-match on both the Stamford and Overture test sets. This is the standard rule-based tool for the problem — every learned model should be reported against it.
- **Implementation instructions:** install the C library + Python bindings locally (the model files are a few GB — check disk); run on test sets only (nothing to train); evaluate with the same metrics/threshold policy as Step 1b.
- **Finish threshold:** a "libpostal" row in the results table on both datasets, plus a sentence on where it beats/loses to the Step 1b model.

### 3 — Hand-build the matching architecture
- **Paper:** Reimers & Gurevych (2019), *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*
  - arXiv: https://arxiv.org/abs/1908.10084
- **Est. time:** 4-6 days.
- **Task:** From scratch — two shared-weight encoders, a distance function, contrastive or binary cross-entropy loss, on embeddings you also build.
- **Implementation instructions:**
  1. Start with a simple encoder (e.g. a small char-level embedding + mean pooling, or a single-layer LSTM) before reaching for anything fancy — the point of this step is the siamese/pairwise structure, not encoder sophistication.
  2. Implement weight sharing explicitly (one encoder module, called twice) rather than two separate encoders — this is the actual siamese property and easy to get subtly wrong.
  3. Implement contrastive loss (or start with plain binary cross-entropy on a similarity score if contrastive loss trips you up — it's a legitimate simpler starting point) yourself rather than importing a loss function you haven't inspected.
  4. Validate on a tiny synthetic toy set (a handful of obviously-matching and obviously-non-matching pairs) before touching your real dataset — this isolates architecture bugs from data bugs.
- **Finish threshold:** A trained model, built without using a pre-existing siamese-network library, that beats the Step 1 baseline's F1 on your held-out test split — and you can explain, without notes, why the two encoders share weights.

### 4 — Reproduce a published deep learning approach
- **Paper:** Lin et al. (2020), *"A deep learning architecture for semantic address matching"*
  - DOI / publisher: https://doi.org/10.1080/13658816.2019.1681431
- **Est. time:** 4-5 days.
- **Task:** Reproduce their word2vec + ESIM approach on your dataset. Record precision/recall/F1 against the Step 1 baseline.
- **Implementation instructions:**
  1. Train word2vec on your address corpus using `gensim` — don't use a generic pretrained word2vec, since address vocabulary (abbreviations, numbers) is domain-specific.
  2. Implement ESIM's core idea (encode both sequences, cross-attend between them, aggregate) — you don't need to match their exact layer count, just the mechanism.
  3. Keep your Step 3 dataset splits identical here — the whole point is a fair comparison.
  4. Note where your reproduction diverges from their reported numbers, and have a hypothesis why (different dataset, different address conventions, smaller scale).
- **Finish threshold:** A results table with three rows (Step 1 baseline, your Step 3 model, this reproduction) on the same test split, plus one paragraph on how your numbers compare to what Lin et al. reported and why.

### 5 — Swap in a pretrained transformer encoder
- **Paper:** *"Improving Address Matching using Siamese Transformer Networks"*
  - arXiv: https://arxiv.org/abs/2307.02300
- **Est. time:** 3-4 days.
- **Task:** Replace the Step 3 hand-built encoder with a fine-tuned small pretrained transformer (e.g. DistilBERT).
- **Implementation instructions:**
  1. Reuse your Step 3 siamese scaffolding (pairing logic, loss, evaluation code) unchanged — only the encoder itself should change. This is what makes the comparison meaningful.
  2. Use Hugging Face `transformers` for DistilBERT, but write your own fine-tuning loop rather than a one-line `Trainer.fit()` call — you want to see the gradient updates happening, not abstract them away.
  3. Try freezing most of DistilBERT's layers and only fine-tuning the top 1-2 plus your similarity head first — full fine-tuning on a small dataset can overfit fast.
  4. Watch for the classic small-dataset-transformer failure mode: near-perfect train accuracy with poor validation performance. If you see this, it's a real, useful finding for your error analysis, not just a problem to fix.
- **Finish threshold:** A fourth row on your results table, and a one-paragraph comparison of this model's F1 and qualitative behavior against Step 3's hand-built version — specifically naming what the pretrained encoder bought you.

### 6 — Real error analysis
- **Paper:** Kilic et al. (2024), *"Unveiling the impact of machine learning algorithms on the quality of online geocoding services: a case study using COVID-19 data"*
  - DOI / publisher: https://doi.org/10.1007/s10109-023-00435-8
- **Est. time:** 2-3 days.
- **Task:** Categorize the best model's failures (abbreviations, unit numbers, directionals, typos) using their "quality impact" framing.
- **Implementation instructions:**
  1. Pull every false positive and false negative from your best model's test-set predictions into one table.
  2. Manually tag each with a failure category (abbreviation mismatch, missing unit number, directional confusion, typo, other).
  3. Count categories and note which ones are frequent vs. rare, and which ones seem "fixable" (a preprocessing step would catch it) vs. genuinely hard (requires real semantic understanding).
  4. Connect at least one category back to a specific real-world consequence (e.g. Kilic et al.'s framing of downstream data-quality impact) rather than treating error analysis as a pure accuracy exercise.
- **Finish threshold:** A short table of failure categories with counts and 2-3 concrete example pairs per category, plus one paragraph on which failure mode matters most in practice and why.

### 7 — Write it up like a short paper
- **Est. time:** 3-4 days.
- **Task:** Motivation → related work → method → results table → error analysis → limitations.
- **Implementation instructions:**
  1. Draft the results table and error-analysis section first — they're already done from Steps 4-6, so writing around solid numbers is easier than staring at a blank motivation section.
  2. Write the related-work section using the papers you've actually implemented or closely engaged with, in the order you engaged with them — this doubles as an honest record of your own learning path.
  3. Keep the limitations section genuinely honest (dataset size, single-locale scope, English-only) — a thoughtful limitations section reads as more credible than one that pretends the project has none.
  4. Do at least one full read-through revision pass after a day away from the draft.
- **Finish threshold:** A single document you'd be comfortable sending cold to a technical stranger (an interviewer or a professor) without walking them through it first.

### 8 — Position within the field's own taxonomy
- **Paper:** 2024 survey, *"A Novel Address-Matching Framework Based on Region Proposal"* (geodesic-grid prediction vs. text-semantic address matching), IJGI 13(4):138
  - DOI / publisher: https://doi.org/10.3390/ijgi13040138
- **Est. time:** 1 day.
- **Task:** Use their taxonomy to frame the writeup's related-work/positioning section.
- **Implementation instructions:**
  1. Re-read your Step 7 related-work section and explicitly state which taxonomy branch (grid-prediction vs. text-semantic matching) your whole project sits in.
  2. Add one or two sentences noting what the other branch does differently, so a reader unfamiliar with the field understands the scope choice was deliberate.
- **Finish threshold:** Your writeup's introduction or related-work section names this taxonomy explicitly and states, in your own words, where your project sits within it.

---

## Optional extensions (only after the core chain is solid)

### 9 — Geographic feature fusion *(stretch)*
- **Papers:**
  - GeoBERT (introduced in the "Experience: Enhancing Address Matching with Geocoding and Similarity Measure Selection" line of work) — https://www.researchgate.net/publication/327521706_Experience_Enhancing_Address_Matching_with_Geocoding_and_Similarity_Measure_Selection
  - GeoRoBERTa, Guermazi, Sellami & Boucelma (2023) — https://ceur-ws.org/Vol-3379/DARLI-AP_2023_1.pdf
- **Est. time:** 3-5 days if attempted.
- **Task:** Read; if time allows, add a small ablation (e.g. ZIP-code proximity as an auxiliary input) to the Step 5 model.
- **Implementation instructions:**
  1. Pick one geographic feature only (e.g. ZIP-code match, or lat/long distance if you have coordinates) — resist adding several at once, since the value here is a clean ablation, not a maximal feature set.
  2. Concatenate the feature into your Step 5 model's similarity head rather than redesigning the whole architecture.
- **Finish threshold:** A fifth results-table row showing whether the added feature measurably changed F1, plus a sentence on whether the change was worth the added complexity.

### 10 — Professor outreach *(later-stage)*
- **Targets:** Gengchen Mai (UT) — https://gengchenmai.github.io/ ; Di Zhu (University of Minnesota, GeoDI Lab) — https://geography.wisc.edu/geods/ (lab page found under UW-Madison GeoDS listing; check for his current UMN lab page directly if this has moved)
- **Est. time:** 1 day to draft and send once a specific result exists.
- **Task:** Only after a specific, non-generic result exists ("I implemented X from your paper and found Y"), send a direct, specific email.
- **Implementation instructions:**
  1. Reference a specific finding from your writeup, not the project in general.
  2. Ask one specific, answerable question rather than a broad "would you take me on."
- **Finish threshold:** A short, specific email drafted and sent — not a milestone to perfect endlessly.