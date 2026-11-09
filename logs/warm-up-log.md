# Warm-up log

One row per recall question, added **when the question is asked** (not at the end of the session), so a session that ends early still leaves a record. Before writing a warm-up, read this table: never re-ask a question already here unless its result is ✗ (a miss comes back, reworded, at a later warm-up). "Not recorded" = asked, but the answer wasn't written down; treat it as asked.

| Date | Concept | Question | Result |
|---|---|---|---|
| 2026-10-30 | Jaro-Winkler vs edit distance | What does JW reward that edit distance doesn't? | ✗ (2nd miss; repair: matches within a window + transpositions, prefix bonus ≤ 4 chars, shared ` ST` suffix lifts whole-string JW) |
| 2026-10-30 | `assign` + `pd.Series(list)` alignment | What happens when you assign a list-built Series after a filter? | ✓ |
| 2026-10-30 | Regex `\|` precedence | Why group alternatives before adding `\b`? | ✓ |
| 2026-10-30 | Merge key | What goes wrong merging without `split` in the key? | ✓ |
| 2026-10-30 | `clip` vs `<= k` | How do they differ? | ✓ |
| ≤ 2026-11-02 | Jaro-Winkler | `OAK ST` vs `ELM ST`: why high, which part doesn't help? | not recorded (David: already done) |
| ≤ 2026-11-02 | AUC | 90% accuracy but AUC 0.5: what does it say? | not recorded (David: already done) |
| ≤ 2026-11-02 | `axis=0` vs `axis=1` | Shapes when averaging word vectors in `address_vector` | not recorded (David: already done) |
| ≤ 2026-11-02 | Skip-gram vs CBOW | Which suits a word seen in 3 addresses, why? | not recorded (David: already done) |
| 2026-11-03 | Corpus dedup | What goes wrong if each address appears once per pair? | ✓ (sharpened: the extra copies add training updates and `min_count` weight, so frequency reflects pair construction) |
| 2026-11-05 | Early stopping checkpoint | You save `best = model.state_dict()` (no `deepcopy`) at the best epoch and keep training 5 more. What does `best` hold at the end, and why? | ✗ (effect right: ends as the final weights; mechanism missing. Repair: `state_dict()` returns references to the live tensors, optimiser updates them in place) |
| 2026-11-05 | Masked max pooling | In max pooling, what goes wrong if you set the pad positions to 0 instead of −inf? | ✓ |
| 2026-11-05 | Random forest decorrelation | Bootstrap already gives each tree different rows. Why also restrict each split to a random subset of features? | ✗ partial (had: weaker features get a chance. Repair: a dominant feature otherwise tops every tree → correlated trees → averaging cuts less variance) |
| 2026-11-05 | Shared siamese weights | In Step 3, what shortcut could two separate encoders (one per side) learn that one shared encoder can't? | ✓ (side-specific patterns, A = Overture / B = Foursquare) |
| 2026-11-06 | Two-stage stacking leakage | You fit a stage-2 model on stage-1 predictions made on the same pairs stage 1 trained on. What goes wrong at test time, and why? | ✗ (didn't recall the term. Repair: stage 1 = siamese score, stage 2 = e.g. LR on score + ZIP; train-pair scores are overconfident → stage 2 over-trusts stage 1 → fit stage 2 on pairs stage 1 never saw) |
| 2026-11-06 | Frozen vs fine-tuned embeddings | If ESIM fine-tuned the word2vec table, what would happen to a street name that appears only in test-ZIP addresses? | ✓ (train words move, test-only words stay at word2vec values → inconsistent space. Sharpened: cause = no gradient reaches rows never in a train batch) |
| 2026-11-06 | Threshold tuning | Your model misses case1 at threshold 0.5. Why can't moving the threshold alone fix that without costing something else? | ✗ partial (had: a move flips errors both ways. Wrong: 'model is optimised for the threshold' — BCE never sees it. Repair: scores are fixed; if case1 negatives score like true positives, any cut that rejects them rejects those positives too; only better ranking fixes it) |
| 2026-11-06 | Hard negatives in eval | Why oversample hard negatives in training but not in the headline evaluation number? | ✓ (headline should reflect the real population mix, mostly easy; hard slices reported separately) |
| 2026-11-08 | Id-based split before pairing | If you built all pairs first and then split the *pairs* randomly into train/test, what leaks, and why would test scores look better than they should? | ✗ partial (had: leakage. Repair: the same address id appears in train and test pairs, so test partly scores addresses the model has already seen; split ids first, then pair) |
| 2026-11-08 | Loss floors | On a 1:9 dataset, val loss sits at ≈ 0.325 from the first epoch and barely moves. What is the model doing, and what do you check first? | ✗ (said vanishing gradients. Repair: 0.325 = loss of always predicting the 0.1 base rate, so the model ignores its input; check the pipeline first: labels aligned, inputs not all PAD/constant) |
| 2026-11-08 | Duplicate features in tree importances | You add an exact copy of your strongest feature to a random forest. What happens to that feature's importance, and why? | ✗ (said it doubles. Repair: either copy can make the same split, so credit is shared; each shows about half, and the feature looks weaker than it is) |
| 2026-11-08 | Gradient boosting classification | In gradient boosting for regression a leaf outputs the mean residual. Why doesn't a classification leaf just do the same? | ✗ partial (had the ingredients, multiplied instead of divided. Repair: leaf = Σ(y − p) / Σ p(1 − p); residuals are in probability units, the model adds in log-odds, so the Newton step divides by the curvature p(1 − p) to convert) |
