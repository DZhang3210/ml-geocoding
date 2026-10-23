# Results — address matching

The project's running results table: every model gets a row, scored the same way on the **test** split, and steps 3–5 add rows here. Val numbers are in brackets (val was used to choose settings, so it is slightly optimistic). The writeup's results section is built from this file (Step 7).

**How to read it.** Every number is a **percentage of the pairs in that column's slice** (the *n* row gives how many pairs that is), never a count or a frequency. For a column of true matches it is the % the model *found* (recall); for a column of non-matches it is the % it correctly *rejected*. Higher is always better. Each cell is **test (val)**: the first number is the test split, the number in brackets is the same model on the val split. Example: 79.0 under "Non-identical matches — % found" means 79.0% of those 843 test pairs were predicted as matches. The slices are:
- **Identical matches**: true matches whose two address strings are identical.
- **Non-identical matches**: true matches whose strings differ. This is the slice the project is about.
- **Case1 non-matches**: same street, different house number.
- **Case2 non-matches**: same number, similar but different street (Jaro-Winkler > 0.8).
- **Easy non-matches**: random pairs.

Overall accuracy is left out because easy negatives dominate it. Data: shared pair schema (`helpers/pairs.py`). Overture CT release `2026-09-23.1`, test split by ZIP; Stamford is synthetic (side B is corrupted at the character level).

## Overture (real cross-source pairs) — test (val in brackets)

| Model | Step | Identical matches — **% found** | **Non-identical matches — % found** | Case1 non-matches — **% rejected** | Case2 non-matches — **% rejected** | Easy non-matches — **% rejected** |
|---|---|---|---|---|---|---|
| LogReg (3 Jaro-Winkler features) | 1 | 99.4 (99.5) | **79.2** (80.3) | 98.5 (98.3) | 6.7 (8.8) | 100.0 (100.0) |
| LogReg (23 similarity features) | 1b | 99.9 (100.0) | **79.0** (81.8) | 99.9 (99.8) | 6.7 (2.9) | 100.0 (100.0) |
| Random forest | 1b | 100.0 (100.0) | **83.0** (85.4) | 99.5 (99.5) | 6.7 (23.5) | 100.0 (100.0) |
| XGBoost | 1b | 100.0 (100.0) | **81.7** (84.9) | 99.7 (99.5) | 46.7 (20.6) | 99.9 (100.0) |
| libpostal, strict/strict/strict | 2c | 99.1 (99.0) | **52.3** (56.5) | 100.0 (100.0) | 80.0 (88.2) | 100.0 (100.0) |
| libpostal, loose/off/loose | 2c | 100.0 (100.0) | **71.9** (75.7) | 97.9 (98.3) | 73.3 (50.0) | 98.9 (99.4) |
| Siamese BiLSTM (characters only, mean pooling, seed 0) | 3 | 100.0 (100.0) | **83.6** (87.1) | 99.4 (99.4) | 33.3 (29.4) | 99.9 (99.9) |
| *test n (pairs in the slice)* | | *3,543* | *843* | *14,448* | *15* | *24,995* |

## Stamford (synthetic, character-level corruption) — test (val in brackets)

| Model | Step | Identical matches — **% found** | **Non-identical matches — % found** | Case1 non-matches — **% rejected** | Case2 non-matches — **% rejected** | Easy non-matches — **% rejected** |
|---|---|---|---|---|---|---|
| LogReg (3 Jaro-Winkler features) | 1 | 100.0 (100.0) | **83.1** (88.4) | 100.0 (100.0) | 94.2 (94.8) | 100.0 (100.0) |
| LogReg (23 similarity features) | 1b | 100.0 (100.0) | **88.1** (94.2) | 100.0 (100.0) | 94.2 (92.7) | 100.0 (100.0) |
| Random forest | 1b | 100.0 (100.0) | **88.1** (94.2) | 100.0 (100.0) | 97.1 (96.9) | 100.0 (100.0) |
| XGBoost | 1b | 100.0 (100.0) | **86.4** (93.0) | 100.0 (100.0) | 97.1 (95.8) | 100.0 (100.0) |
| libpostal, strict/strict/strict | 2c | 100.0 (100.0) | **39.0** (36.0) | 100.0 (100.0) | 98.6 (100.0) | 99.8 (100.0) |
| libpostal, loose/off/loose | 2c | 100.0 (100.0) | **39.0** (36.0) | 100.0 (98.8) | 92.8 (93.8) | 99.3 (100.0) |
| *test n (pairs in the slice)* | | *17* | *59* | *174* | *69* | *445* |

**Setup.**
- **Step 1:** the same default LogReg as Step 1b, on only the three Jaro-Winkler features (street, house number, ZIP), so Step 1 vs 1b differs only in the features. The old Step 1 number (F1 0.81, 2026-09-21) used the pre-schema Stamford columns and isn't comparable.
- **Step 1b:** default hyperparameters, no class weighting, trained on the train split. This is Step 1b's single test run (2026-10-15).
- **Step 3 (2026-10-23):** character-level siamese bi-encoder: one shared encoder (char embedding 32 → BiLSTM 1 × 128 → masked mean pooling) applied to both strings, head `[|u−v|; u⊙v]` → one logit, `BCEWithLogitsLoss`, Adam 1e-3, threshold 0.5. Input = the two address strings only: **no ZIP features and no hand-built similarity features**. Trained on Overture train only (Stamford was dropped from Step 3, so there is no Stamford row). Best epoch = highest val non-identical found with val case1 ≥ 98%. Mean vs max pooling × 3 seeds tied within seed noise on val; mean / seed 0 went to test (seed 0 was the default before any run, not picked on val). Single test run 2026-10-23, `siamese_model/14_training_analysis.ipynb` §4.
- **libpostal settings** are `road/zip/number`: strict/strict/strict is the rejection end, loose/off/loose the recall end. Both were chosen on val before test was run (`model_baseline/05_dumb_libpostal_baseline.ipynb`).
- **Where the numbers come from:** `model_baseline/06_compare_libpostal_and_dumb.ipynb`, section 4 builds both tables (nothing here is copied by hand).
- **Stamford re-run (2026-10-16):** 30 negatives whose clean source rows are the same address (28 train, 2 test) were dropped from the Stamford pairs, and the Stamford rows re-run; no setting changed. Effect: libpostal (no training) gained exactly those 2 easy rows, i.e. both were "errors" that were really correct; the 1b models' easy slice went 99.6 → 100.0. RF and XGB moved by one row (1.7 pts) on non-identical, from the 28 fewer training rows — noise.
- **Small slices:** Overture case2 has only 15 test rows (one row = 6.7 pts), mostly mirrored pairs, so its val↔test swings are noise. Stamford non-identical has 59 rows (one row = 1.7 pts).

## Step 3 — hand-built siamese model vs the baselines

**Characters alone match the best feature-engineered model.** On Overture test the siamese model finds 83.6% of non-identical matches, against 83.0% for RF, 81.7% for XGBoost and 79.0% for the 23-feature LogReg. The 0.6-pt gap to RF is 5 of 843 pairs, well inside the seed-to-seed spread of mean pooling on val, so the two are indistinguishable rather than ranked. It **beats the Step 1 baseline** (79.2%) by 4.4 pts, the Step 3 finish line, but by a margin close to seed-to-seed noise (mean pooling's three seeds span 3.8 pts of val found). Case1 (99.4%) and easy (99.9%) rejection are at the 1b models' level.

**It gets there without the ZIP.** Every baseline has ZIP features; the siamese model sees only the two strings. The cost of that falls on different slices for different reasons:
- **Case1 and case2:** no cost. These negatives were built by ZIP blocking, so both sides share a ZIP and it carries no information for any model.
- **Easy negatives:** almost no cost (99.9 vs 100.0%). They are the one slice where a different ZIP means "no match", but the street text already separates random pairs. The prediction that easy rejection would fall below 98% without the ZIP was wrong. LR leaned on the ZIP because its street feature was weak (Step 1 weight 0.6 vs ZIP 16).
- **Non-identical matches:** this is where the point bites. Both sides of a true match usually share a ZIP, a strong "yes" signal for the baselines on messy strings, and the siamese model reaches the same found rate from characters alone.
- Qualifier: some address strings contain town + ZIP text (e.g. a UConn Foursquare string), so the exact claim is "no ZIP features", not "no ZIP information"; the ZIP-text diagnostic (open questions) would size this.

**Val → test.** The siamese model drops 87.1 → 83.6 on non-identical, a drop of similar size to the 1b models' (LogReg 81.8 → 79.0, RF 85.4 → 83.0, XGBoost 84.9 → 81.7). The 1b models used default settings, so nothing in them was tuned on val, and their drop is the difference between the splits (split by ZIP: different towns, some with messier strings). The siamese model had choices made on val (best epoch, guardrail, pooling); since its drop is about the same, those choices did not inflate its val numbers much.

**Caveats.** (1) One seed: the test number carries that run's seed luck; on val, mean pooling's three seeds span 3.8 pts of found. (2) Case2 (33.3% = 5 of 15) can't support a claim: one pair moves it 6.7 pts and the slice has label noise (Step 2b). (3) Mean vs max pooling made no detectable difference on val at 3 seeds, itself a finding: on this data, how the character vectors are pooled matters less than the seed.

## Step 2c — libpostal vs the Step 1b models

On real data, libpostal recovers fewer true matches than the learned baselines (52–72% vs 79–83% of non-identical positives on Overture test). But it is far better at rejecting look-alike streets (73–80% of case2 vs 7–47%). Both results come from one difference.
- **The learned models score how *similar the letters* are.** A typo like `CONNECTICUT` / `CONNETICUT` still scores high and is accepted. But so is `70 MAIN ST` / `70 PINE ST`: same number, same ZIP, and look-alike letters.
- **libpostal asks whether two streets are the *same name* after normalisation.** It accepts abbreviations (`LN` / `LANE`), rejects `MAIN` vs `PINE`, and also rejects any typo.

On Stamford, the street names carry random character edits, so libpostal's name check fails on most true matches (39% recall vs ~88%). Meanwhile the learned models reject case2 well there (94–97%). Overture's training data has only 172 case2 pairs, against about 1,800 for Stamford, so imbalance in the training data is the leading explanation for the learned models' weakness on Overture case2. That has not been tested yet: the controlled test is to up-weight case2 in Overture train. Neither approach handles both a typo in a real match and a different street that looks alike, which is what the learned encoders in Steps 3–5 are meant to do.

## Step 1 → Step 1b — what 20 extra similarity features bought

**On real data, almost nothing.** On Overture test, the 3-feature LogReg finds 79.2% of non-identical matches against 79.0% for the 23-feature LogReg (val 80.3 vs 81.8). Case2 is equally bad (6.7% both). The extra features help only at the edges: case1 rejection 98.5 → 99.9% (about 200 of 14,448 rows) and identical matches 99.4 → 99.9%, most likely from `number_missing` telling the model when a house number failed to parse. The only real gain on non-identical matches comes from switching to trees (RF 83.0%), not from the features.

**On Stamford, more:** 83.1 → 88.1% of non-identical matches (val 88.4 → 94.2), i.e. 3 of 59 test rows. The extra edit-distance and n-gram views of the street help when the street has random character edits, which is exactly how Stamford's side B is built. Case2 is unchanged (94.2%).

**Why Step 1 fails case2 on Overture.** Its fitted weights on Overture are street 0.6, house number 8.7, ZIP 16.0 (Stamford: 17.8, 13.6, 12.0). On Overture the model has learned to almost ignore the street. That follows from the training mix: case1 negatives (same street, different number) are rejected by the number, easy negatives by the ZIP and number, and case2 (172 of 202,568 training pairs) is the only kind where the street decides. So "same number + same ZIP" is enough to predict a match — the case2 failure, now visible directly in the weights rather than inferred from the error rates.

**For Step 3:** the floor to beat is ~79–83% of Overture non-identical matches at ≥ 98.5% case1 rejection, and the hand-built features have stopped improving it. A learned encoder has to find street differences the training mix gives little reason to look at, so the case2 share in training matters as much as the architecture.

