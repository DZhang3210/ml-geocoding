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
| LogReg (23 similarity features) | 1b | 99.9 (100.0) | **79.0** (81.8) | 99.9 (99.8) | 6.7 (2.9) | 100.0 (100.0) |
| Random forest | 1b | 100.0 (100.0) | **83.0** (85.4) | 99.5 (99.5) | 6.7 (23.5) | 100.0 (100.0) |
| XGBoost | 1b | 100.0 (100.0) | **81.7** (84.9) | 99.7 (99.5) | 46.7 (20.6) | 99.9 (100.0) |
| libpostal, strict/strict/strict | 2c | 99.1 (99.0) | **52.3** (56.5) | 100.0 (100.0) | 80.0 (88.2) | 100.0 (100.0) |
| libpostal, loose/off/loose | 2c | 100.0 (100.0) | **71.9** (75.7) | 97.9 (98.3) | 73.3 (50.0) | 98.9 (99.4) |
| *test n (pairs in the slice)* | | *3,543* | *843* | *14,448* | *15* | *24,995* |

## Stamford (synthetic, character-level corruption) — test (val in brackets)

| Model | Step | Identical matches — **% found** | **Non-identical matches — % found** | Case1 non-matches — **% rejected** | Case2 non-matches — **% rejected** | Easy non-matches — **% rejected** |
|---|---|---|---|---|---|---|
| LogReg | 1b | 100.0 (100.0) | **88.1** (94.2) | 100.0 (100.0) | 92.8 (92.7) | 99.6 (100.0) |
| Random forest | 1b | 100.0 (100.0) | **86.4** (91.9) | 100.0 (100.0) | 97.1 (96.9) | 99.6 (100.0) |
| XGBoost | 1b | 100.0 (100.0) | **88.1** (94.2) | 100.0 (100.0) | 98.6 (95.8) | 99.6 (100.0) |
| libpostal, strict/strict/strict | 2c | 100.0 (100.0) | **39.0** (36.0) | 100.0 (100.0) | 98.6 (100.0) | 99.3 (100.0) |
| libpostal, loose/off/loose | 2c | 100.0 (100.0) | **39.0** (36.0) | 100.0 (98.8) | 92.8 (93.8) | 98.9 (100.0) |
| *test n (pairs in the slice)* | | *17* | *59* | *174* | *69* | *447* |

**Setup.**
- **Step 1b:** default hyperparameters, no class weighting, trained on the train split. This is Step 1b's single test run (2026-10-15).
- **libpostal settings** are `road/zip/number`: strict/strict/strict is the rejection end, loose/off/loose the recall end. Both were chosen on val before test was run (`model_baseline/dumb_libpostal_baseline.ipynb`).
- **Where the numbers come from:** `model_baseline/compare_libpostal_and_dumb.ipynb`.
- **Small slices:** Overture case2 has only 15 test rows (one row = 6.7 pts), mostly mirrored pairs, so its val↔test swings are noise. Stamford non-identical has 59 rows (one row = 1.7 pts).

## Step 2c — libpostal vs the Step 1b models

On real data, libpostal recovers fewer true matches than the learned baselines (52–72% vs 79–83% of non-identical positives on Overture test). But it is far better at rejecting look-alike streets (73–80% of case2 vs 7–47%). Both results come from one difference.
- **The learned models score how *similar the letters* are.** A typo like `CONNECTICUT` / `CONNETICUT` still scores high and is accepted. But so is `70 MAIN ST` / `70 PINE ST`: same number, same ZIP, and look-alike letters.
- **libpostal asks whether two streets are the *same name* after normalisation.** It accepts abbreviations (`LN` / `LANE`), rejects `MAIN` vs `PINE`, and also rejects any typo.

On Stamford, the street names carry random character edits, so libpostal's name check fails on most true matches (39% recall vs ~88%). Meanwhile the learned models reject case2 well there (93–99%). Overture's training data has only 172 case2 pairs, against about 1,800 for Stamford, so imbalance in the training data is the leading explanation for the learned models' weakness on Overture case2. That has not been tested yet: the controlled test is to up-weight case2 in Overture train. Neither approach handles both a typo in a real match and a different street that looks alike, which is what the learned encoders in Steps 3–5 are meant to do.
