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
