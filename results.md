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

## Overture `case2s` (look-alike street swaps replace some easy negatives) — test (val in brackets)

Same positives, case1 and case2 as the Overture table above. The difference is that 1,053 test easy negatives (val 894, train 5,441) are replaced by **case2s** pairs. A case2s pair is a real address with its street swapped for a different real street in the same ZIP that looks alike (Jaro-Winkler > 0.8). Every learned model except the first siamese row is **retrained on the `case2s` train split**. libpostal has no training. Source: `model_baseline/07_backtest_case2s.ipynb` §6.

| Model | Step | Identical matches — **% found** | **Non-identical matches — % found** | Case1 non-matches — **% rejected** | Case2 non-matches — **% rejected** | **Case2s non-matches — % rejected** | Easy non-matches — **% rejected** |
|---|---|---|---|---|---|---|---|
| LogReg (3 Jaro-Winkler features) | 1 | 99.9 (100.0) | **59.2** (62.4) | 99.7 (99.5) | 80.0 (61.8) | **59.6** (63.4) | 100.0 (100.0) |
| LogReg (23 similarity features) | 1b | 99.9 (100.0) | **50.5** (57.1) | 99.1 (99.2) | 66.7 (47.1) | **74.4** (77.0) | 100.0 (100.0) |
| Random forest | 1b | 100.0 (100.0) | **70.1** (76.8) | 99.6 (99.5) | 53.3 (55.9) | **88.7** (91.3) | 99.9 (99.9) |
| XGBoost | 1b | 100.0 (100.0) | **69.9** (76.3) | 99.6 (99.5) | 53.3 (52.9) | **88.5** (90.2) | 100.0 (100.0) |
| libpostal, strict/strict/strict | 2c | 99.1 (99.0) | **52.3** (56.5) | 100.0 (100.0) | 80.0 (88.2) | **97.4** (98.2) | 100.0 (100.0) |
| libpostal, loose/off/loose | 2c | 100.0 (100.0) | **71.9** (75.7) | 97.9 (98.3) | 73.3 (50.0) | **75.1** (74.9) | 98.9 (99.4) |
| Siamese BiLSTM, trained on the **old** train (= Step 3 row) | 3 | 100.0 (100.0) | **83.6** (87.1) | 99.4 (99.4) | 33.3 (29.4) | **22.2** (24.8) | 99.9 (99.9) |
| Siamese BiLSTM, trained on `case2s` (mean pooling, seed 0) | 3 | 100.0 (100.0) | **80.1** (83.1) | 99.3 (99.3) | 53.3 (64.7) | **75.7** (74.5) | 100.0 (100.0) |
| ESIM, word2vec `min_count=2` (reproduction; 3 seeds, test min–max) | 4 | 95.5–100.0 (97.9–100.0) | **77.9–80.7** (80.9–82.7) | 96.1–98.4 (98.2–99.1) | 40.0–46.7 (55.9) | **73.9–75.5** (76.1–80.0) | 99.9 (99.9–100.0) |
| ESIM, word2vec `min_count=2` + random row per rare token (3 seeds, test min–max) | 4 | 99.9–100.0 (100.0) | **74.9–78.9** (77.8–81.8) | 98.6–98.9 (99.0–99.2) | 26.7–46.7 (55.9) | **72.9–75.9** (74.6–82.2) | 99.9–100.0 (100.0) |
| *test n (pairs in the slice)* | | *3,543* | *843* | *14,448* | *15* | *1,053* | *23,942* |

**Setup.**
- **`case2s` backtest (2026-10-29):** the Step 1 / 1b models use the same features and default settings as before, refitted on the `case2s` train split. libpostal uses the same two settings, re-run because val/test now hold different pairs. The rows on the positive, case1 and case2 slices are the same pairs as before, which is why the libpostal numbers there are unchanged. The `case2s`-trained siamese model uses the Step 3 protocol unchanged (same architecture, guardrail and best-epoch rule; `siamese_model/15_train_on_case2s.ipynb`), with one seed only. Case2s has known noise: some same-street spelling variants are labelled as non-matches (open questions).
- **Step 4 (2026-11-05 → 11-08):** word-level ESIM (Chen et al. 2017, as in Lin et al. 2020): frozen word2vec 200-d trained on all splits' address text (skip-gram, window 2, 50 epochs) → BiLSTM 200 → soft alignment → `[a; ã; a−ã; a⊙ã]` → projection → BiLSTM 200 → mean + max pooling → MLP 200 (tanh) → one logit, threshold 0.5; trained on the `case2s` train split, Step 3's guardrail and best-epoch rule. **Test was scored for every seed and setting before a choice was made, so each cell is the min–max over 3 seeds and no run is picked.** The setting (random row per rare token) was chosen on val only, by a rule fixed beforehand (`esim/03_esim_extension.ipynb` §5). Swap test (val, `rand_mc2`): swapping sides flips 2.3–4.3% of non-identical and 2.3–4.1% of `case2s` predictions; averaging both orders moves no slice by more than 0.6 pts, so the one-order numbers are reported. Row 1 maps every word seen once to a shared `UNK`; row 2 gives each such word its own fixed random (frozen) row. Source: `esim/02_esim.ipynb` §14, `esim/03_esim_extension.ipynb` §5.
- **Step 1:** the same default LogReg as Step 1b, on only the three Jaro-Winkler features (street, house number, ZIP), so Step 1 vs 1b differs only in the features. The old Step 1 number (F1 0.81, 2026-09-21) used the pre-schema Stamford columns and isn't comparable.
- **Step 1b:** default hyperparameters, no class weighting, trained on the train split. This is Step 1b's single test run (2026-10-15).
- **Step 3 (2026-10-23):** character-level siamese bi-encoder: one shared encoder (char embedding 32 → BiLSTM 1 × 128 → masked mean pooling) applied to both strings, head `[|u−v|; u⊙v]` → one logit, `BCEWithLogitsLoss`, Adam 1e-3, threshold 0.5. Input = the two address strings only: **no ZIP features and no hand-built similarity features**. Trained on Overture train only (Stamford was dropped from Step 3, so there is no Stamford row). Best epoch = highest val non-identical found with val case1 ≥ 98%. Mean vs max pooling × 3 seeds tied within seed noise on val; mean / seed 0 went to test (seed 0 was the default before any run, not picked on val). Single test run 2026-10-23, `siamese_model/14_training_analysis.ipynb` §4.
- **libpostal settings** are `road/zip/number`: strict/strict/strict is the rejection end, loose/off/loose the recall end. Both were chosen on val before test was run (`model_baseline/05_dumb_libpostal_baseline.ipynb`).
- **Where the numbers come from:** `model_baseline/06_compare_libpostal_and_dumb.ipynb`, section 4 builds both tables (nothing here is copied by hand).
- **Stamford re-run (2026-10-16):** 30 negatives whose clean source rows are the same address (28 train, 2 test) were dropped from the Stamford pairs, and the Stamford rows re-run; no setting changed. Effect: libpostal (no training) gained exactly those 2 easy rows, i.e. both were "errors" that were really correct; the 1b models' easy slice went 99.6 → 100.0. RF and XGB moved by one row (1.7 pts) on non-identical, from the 28 fewer training rows — noise.
- **Small slices:** Overture case2 has only 15 test rows (one row = 6.7 pts), mostly mirrored pairs, so its val↔test swings are noise. Stamford non-identical has 59 rows (one row = 1.7 pts).

## Step 4 — ESIM: the rare-number collision and its fix

**With the paper's set-up, rare house numbers are invisible.** word2vec with `min_count=2` drops every house number seen once, and ESIM maps all of them to one `UNK`. A same-street case1 pair like `2147 SUMMER ST` / `1867 SUMMER ST` then becomes the same input as an identical match, with the opposite label. Each seed settles on one answer for that input: seeds 0 and 1 reject it (and lose identical matches: test 95.5–95.9% found), seed 2 accepts it (and accepts all 404 such test case1 pairs).

**A fixed random row per rare token fixes it; giving numbers meaning does not.** The question for these pairs is identity (same number or not), not meaning. A random row per token gives the same vector to the same token and a near-orthogonal one to a different token. Of four alternatives, only this one rejected ≥ 90% of the val rare-number case1 pairs on every seed. word2vec with `min_count=1` gives every number a vector, but numbers on the same street share neighbours (cos 0.92 vs 0.57 for random number pairs), so the collision survives. fastText vectors (character n-grams) were close but failed on one seed. Stacking word2vec and fastText still left same-street numbers similar.

**Cost on the deciding slices: not detectable at 3 seeds.** On val, non-identical found and `case2s` rejected overlap between the two settings (non-identical 80.9–82.7 vs 77.8–81.8; `case2s` 76.1–80.0 vs 74.6–82.2). A cost of about 1 point can't be ruled out: three seeds and about 800–900 val rows per slice can't resolve it.

**Identical matches are not a target.** An exact-string check before the model handles them, which is why the decision rule ignored that slice. The rare-number case1 pairs differ as strings, so they still reach the model and still needed the fix.

**Step 4 vs Step 3: the cross-encoder bought nothing measurable.** Compared with the siamese model trained on the same `case2s` split (one seed): non-identical found 80.1 (val 83.1) vs ESIM `rand_mc2` 74.9–78.9 (val 77.8–81.8); `case2s` rejected 75.7 (74.5) vs 72.9–75.9 (74.6–82.2); case1 99.3 (99.3) vs 98.6–98.9 (99.0–99.2). The siamese number sits above ESIM's whole range on non-identical, on val and on test, but it is a single seed, and the siamese model's own seeds spanned 3.8 pts of val found in Step 3, so the honest ranking is "ESIM no better, possibly worse". `case2s` and case1 are level.

**Two changes at once, so the cause can't be assigned.** Step 3 → Step 4 changes both the architecture (bi-encoder → cross-attention) and the input unit (characters → words). Word-level input is a plausible cost on non-identical matches: a typo or an unusual abbreviation becomes a different token (a rare one gets an unrelated random row), where characters still share most of the string. Ramani & Borrajo (2024) report about +1 pt from adding character embeddings to ESIM on English addresses, consistent with this, but nothing here separates the two changes. A char-level ESIM, or a word-level siamese model, would. Step 5 (a subword transformer as bi- and cross-encoder) partly addresses this, since both variants then share one input unit.

**vs Lin et al. (2020): not directly comparable.** They report precision / recall / F1 of 0.97 on the Shenzhen Address Database (abstract only; the full text is paywalled, so their negatives, class balance and splits are unknown). Three things differ. Data: Chinese addresses from one city's address database vs US cross-source pairs from Overture. Negatives: ours include same-street case1 and look-alike-street `case2s` pairs at 1:9, built to be hard. Metric: one overall F1 vs accuracy per slice. A single overall number hides the hard slices here too: the same `mc2` runs score 97.6–98.1% overall accuracy on test while rejecting only 73.9–75.5% of `case2s`. So their 0.97 says nothing about how ESIM handles look-alike streets, and our slice numbers can't be read as falling short of it. (Written by Claude at David's request, 2026-11-09.)

## `case2s` — what training on look-alike street swaps changed

**Every learned model trades non-identical recall for `case2s` rejection.** Retrained on the `case2s` train split, each model rejects far more look-alike swaps and finds fewer true non-identical matches (test, old train → `case2s` train):

| Model | Non-identical found | `case2s` rejected |
|---|---|---|
| LogReg, 3 features | 79.2 → 59.2 | 59.6 |
| LogReg, 23 features | 79.0 → 50.5 | 74.4 |
| Random forest | 83.0 → 70.1 | 88.7 |
| XGBoost | 81.7 → 69.9 | 88.5 |
| Siamese BiLSTM (seed 0) | 83.6 → 80.1 | 22.2 → 75.7 |

The two slices pull in opposite directions: a look-alike negative and a rewritten true match are both near-misses on the street string, so a model pushed to reject the first also rejects more of the second. The feature-based models pay 13–28 pts of non-identical recall; the siamese model pays 3.5 for a 53-pt gain on `case2s`, the best exchange of the learned models. libpostal (no training) sits at the ends of the same trade-off: strict rejects 97.4% of `case2s` but finds 52.3% of non-identical matches.

**What this does not show.**
- *That the models learned "look-alike street".* `case2s` test rows are built by the same swap procedure as the training rows, so a gain there may be a construction trace rather than the concept. Real case2 (15 test rows) is the only independent check and is too small: the siamese model's 33.3 → 53.3 is 3 rows.
- *Why non-identical recall dropped.* "Less reliance on simple heuristics" and "the decision boundary moved toward rejecting" predict the same drop. The positives that flipped between the two siamese models would separate them; that check was not run.
- *A siamese spread.* One seed; Step 3's val seeds spanned 3.8 pts of non-identical found, so the 3.5-pt cost is within one seed's noise.

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

