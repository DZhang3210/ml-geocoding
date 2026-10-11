# Overture Dataset Construction Walkthrough — Step 2b

How the real cross-source address-pair dataset (`Data/Overture/ct_pairs_{train,val,test}_2026-09-23.1.parquet`) was built, 2026-10-05 → 2026-10-11. For each stage: **what** was done, **why** (the thought process and the alternatives rejected), the **checks** that confirmed it, and the **pitfalls** hit along the way.

Code lives in two notebooks:
- `Data/grabbing_new_data.ipynb`: acquisition + address cleaning (stages 1–6)
- `Data/creating_dataset.ipynb`: flags, split, negatives, assembly (stages 7–14)

A self-check quiz is at the end. If you can answer it without looking back, you understand the pipeline.

---

## The big picture

**Goal.** Step 2's Stamford positives came from our own corruption function, so a model could learn to undo *our* corruption rather than real-world variation. Step 2b gets pairs where **two independent data providers wrote the same place's address**, so the formatting differences are real.

**Where real pairs come from.** Overture Maps *conflates* places: it merges records from several providers (Meta, Microsoft, Foursquare, …) into one place with a stable **GERS id**, publishes one provider's version, and lists every merged provider record in **bridge files**. Overture's published address (side **A**) + Foursquare's raw record for the same GERS id (side **B**) = a naturally occurring positive pair.

**Pipeline at a glance**

```
Overture places (S3) ──► ct_places (CT only, 193,270)
bridge files (S3)    ──► ct_bridge (GERS id → Foursquare/ATP record id, 58,694)
Foursquare OS (HF)   ──► fsq_ct
            3-way join ──► ct_positives (37,680)  →  clean  →  flag  →  ZIP clean  →  split by ZIP
                                                                                        │
                    positives (deduped 28,834) + Case 1 + Case 2 + easy  ◄──────────────┘
                                         │
                                         ▼
                  ct_pairs_train / val / test  (288,294 rows, 1 : 9 everywhere)
```

**Scope decisions made up front (2026-10-05)**
- **Address fields only.** Overture's own matcher also uses names, phones, websites and categories. We don't, because the project's task is address matching.
- **Stamford is kept** as the synthetic benchmark. Every model is reported on both datasets, and "trained on synthetic, tested on real" is a planned result.
- **Label-noise caveat.** Positives come from Overture's ML matcher, not from a human. Training on them means imitating that matcher, mistakes included. Hence the (deferred) hand-labelled test subset.

---

## Stage 1 — Overture places in Connecticut (`ct_places`)

**What.** A DuckDB query over Overture's places on S3, filtered to CT, keeping the first address (`freeform`, `locality`, `postcode`, `region`), `confidence`, `sources` and `bbox`.

**Why this way**
- **Bounding box first, then the exact filter.** The `bbox` filter lets DuckDB skip whole Parquet row groups, so only the CT parts of the ~11 GB theme are fetched. The exact CT test then uses `region`. This is the *coarse filter, then exact selection* pattern, the same idea as blocking.
- **`UPPER(region) = 'CT'`.** The data had both `CT` and `ct`.
- **NULL-region rows with a `06…` ZIP were kept** (393 places). The other NULL-region rows were NY/MA/RI or had no postcode.
- **`confidence` kept as a column, not filtered on.** It measures "does this place exist?", not address quality.

**Checks**
- 193,270 places, all ids unique.
- `sources[1].property = ''` for every row. That means the first source is the provider whose whole record was published, which matters for stage 2.

**Pitfalls**
- `AND` binds tighter than `OR`, so the region condition needed parentheses.
- 54 places have `freeform = ''` (an empty string, not NULL), so they passed the `IS NOT NULL` filter. They were caught later (stage 7).

**Writeup limitation:** Overture Places are **businesses / POIs only, with no residential addresses.**

---

## Stage 2 — Bridge links (`ct_bridge`)

**What.** The Foursquare and AllThePlaces bridge rows for CT places, stacked with `UNION ALL`. Columns: `gers_id`, `record_id`, `provider` (whose record this row points to → side B), `published_from` (whose address Overture published → side A).

**Why.** The published record keeps only the provider that "won" the merge. Merged-away providers appear **only** in the bridge files, so those files are what turn single records into pairs.

**Key idea: self-pairs.** When `provider = published_from`, both sides would be the same provider's text, so the pair carries no cross-source variation and is unusable.

**Checks**
- 58,694 rows (50,181 Foursquare + 8,513 AllThePlaces), with one record per provider per place.
- Self-pairs: 12,483 FSQ + 2,067 ATP, leaving **44,144 usable cross-source rows**.

**Pitfall.** `UNION ALL` matches columns **by position**, so both `SELECT`s must list their columns in the same order.

**Decision.** AllThePlaces was **deferred**. It's chain stores only (~6.4k pairs), and its advertised Parquet output returned 404, leaving only a 2.6 GB zip of GeoJSON files.

---

## Stage 3 — Foursquare records (`fsq_ct`)

**What.** Foursquare Open Source Places (a gated dataset on Hugging Face), read with DuckDB using a token from the git-ignored `.env`.

**Why bbox only, no region filter.** Some Foursquare places physically in CT carry wrong states (`KS`, `CA`, …). A region filter would silently drop them and their pairs. Whether a place is in CT was already decided on the Overture side, so this table is just a lookup and the join does the exact selection.

**Check: coverage.** `LEFT JOIN` the bridge's Foursquare ids to `fsq_place_id` and count NULLs. Only **31 of 50,181** were missing (99.94%), which proves the two id formats match.

**Pitfalls**
- Never print or commit the token: `.env` is ignored by git, while `.envrc` is tracked.
- A missing comma in a `SELECT` silently turns the next column into an alias (`postcode latitude` = `postcode AS latitude`).

---

## Stage 4 — Positive pairs (`ct_positives`)

**What.** A chain of joins: `ct_places ── gers_id ── ct_bridge ── record_id ── fsq_ct`, excluding self-pairs. **37,680 pairs.** Names are kept for reading pairs, never as features.

**Checks: what's actually in the positives**

| Category | Count | Decision |
|---|---|---|
| Identical address strings | 26,286 (70%) | keep; flag `identical_address` |
| Different leading house number | 1,778 | likely label noise; flag |
| Number missing on one side | 1,177 | ambiguous; keep, flag |
| NULL side B | 544 | drop |
| Genuine formatting variation | ~7,900 | the interesting part |

The eyeballed variation included units/suites, abbreviations, ranges (`16-18`), ZIP+4, junk (`\N\N`), real typos (`DANBURY EOAD`) and locality variants.

**Decision: flag, don't delete** each category, so the decision about what to drop stays visible and reversible.

**Key decision: the building-level match definition (2026-10-06).** Same street address means same location; **units are ignored**. Consequences:
- different businesses at the same address are matches;
- `STE 1` vs `STE 2` is a match;
- negatives must never pair two places whose clean street addresses are identical (the **exclusion rule**).

Unit-level matching is "future work".

**Pitfall.** `regexp_extract` returns `''`, not NULL, when nothing matches. That bit more than once (e.g. an `IS NULL` check that was always false).

---

## Stage 5 — Address cleaning (`ct_positives_clean`)

**What.** A 5-step DuckDB chain (CTEs) producing `address_a_clean` / `address_b_clean`:
1. ordinal floors at the end (`2ND FLOOR`);
2. designator + unit code (`STE 200`, `SUITE #320`, optionally two designators: `SPACE NUMBER 1029A`);
3. bare `#` units;
4. punctuation → space (not deleted, so `RD,SUITE` doesn't glue together; `-` kept);
5. collapse whitespace + trim.

**Why this order**
- Unit stripping needs `#` and `-`, so it has to run **before** punctuation removal.
- Floor before designator fixes `UNIT 3 2ND FLOOR`.

**Decisions**
- Floors stripped, since they're inside the building.
- `BLDG` kept: a different building means a different location.
- Road-number words (`ROUTE`, `RT`, `US`, `I`, …) and `BOX` are not designators.
- The designator list came from the data (the word before a digit-containing last token: `STE` 1,656, `UNIT` 262, …) plus USPS Pub 28.
- Bare unit codes after `RD`/`ST`: only 52 rows (most tokens there are directionals, which must stay), so no rule was written.

**Checks: a gap check.** Search the *cleaned* output for things that should be gone. Leftovers were under 0.3% per category and accepted as known gaps. Results:
- `same_street_address` 29,802 vs. 26,286 identical raw strings: **3,516 (32%)** of the non-identical pairs differed only by units or punctuation;
- sanity check: identical raw strings that were not the same after cleaning = 0.

**Pitfalls (regex)**
- `\d+` left letter suffixes glued onto the street (`AVEC`).
- An escaped `\[` turned the character class into a literal.
- Without the `'g'` flag, DuckDB replaces only the first match.
- The replacement argument is plain text, so `'\s'` there is not a space.

---

## Stage 6 — House-number flags (`ct_positives_flagged`, `creating_dataset` §1)

**What.**
- `number_mismatch`: both sides have a leading number and they differ (**1,778**).
- `number_missing`: exactly one side has a number (**999**). Both-missing is deliberately not flagged.

The 46 empty `address_a_clean` rows and the 544 NULL-B rows were dropped, leaving **37,090 rows**.

**Why on `_clean` columns.** Cleaning can change the start of a string (e.g. leading spaces). From here on, every comparison uses the clean columns.

**Checks.** The two flags are never both true, and the counts match the stage-4 analysis.

**Pitfalls**
- "Exactly one side" and "either side" are different conditions.
- `COUNT * FILTER(…)` parses as a multiplication and gives a confusing binder error.
- Extracting `regexp_extract` six times made the query unreadable. Use a CTE so the number is extracted once.

---

## Stage 7 — ZIP cleaning (`ct_positives_zip_cleaned`, §2–3)

**Why ZIPs got this much attention.** `postcode_a` is the **split key** (stage 8), so it has to be clean before anything is split.

**Health check of `postcode_a`**
- 10 NULL and 2 empty;
- **11,066 ZIP+4 values (~30%)**;
- 15 values not starting with `06`: 12 were 4-digit ZIPs that had lost their leading zero (`6371`), and 3 were junk or out of state.

**Fix.** `zip_5 = LPAD(postcode_a, 5, '0')`. DuckDB's `LPAD` **truncates** strings longer than the target, so this one call both pads `6371` → `06371` and cuts `06514-4033` → `06514`. That's an intentional side effect, and it's commented in the code. Pandas' `zfill` only pads; it never truncates.

**Checks**
- All 12 padded values were compared with side B's `postcode_b`, and **all agree** (one Foursquare value had lost its zero too).
- 37,075 rows kept (15 dropped); **8,931 raw `postcode_a` values collapse to 331 ZIPs**. Without normalizing, every ZIP+4 would have been its own "ZIP", and neighbourhoods would have been scattered across all splits.

**`zip_b` (added 2026-10-11).** Foursquare's ZIP, carried as a **feature**, so it never drops rows. The same `LPAD`, wrapped in a `CASE` that keeps only exactly-5-digit results. Junk becomes NULL: one `postcode_b` was a phone number (`203-874-6772`, a CT area code; Foursquare data is crowd-sourced). Out-of-state values are kept as real source disagreement. Checked: 2 NULLs, 0 values that aren't 5 digits.

**Pitfalls**
- A regex written `/d` instead of `\d` silently matched nothing, so "0 ZIP+4" looked like good news.
- An "empty" check written with `IS NULL` was identical to the NULL check.
- A filter on raw `postcode_a` starting with `^06` threw away the 4-digit rows before they could be padded.
- `ELSE THEN`, a nested `CASE`, a missing comma: SQL syntax.
- Over and over, **stale outputs**: the displayed table came from a previous version of the code. The give-away is a row count that doesn't match the file the code reads.

---

## Stage 8 — Train / val / test split by ZIP (§4)

**What.** Each whole ZIP goes into exactly one split. Every place then takes its ZIP's split. The ZIP → split mapping is saved on its own (`ct_positives_split_data`).

**Why split before generating any pairs.** Negatives pair place X with place Y. Generating pairs first and splitting afterwards would:
- **leak**: the same address string would appear in train and test (positive in one, part of a negative in the other);
- repeat the **Step 2 v1 bug**: dropping boundary-crossing pairs removes negatives quadratically faster than positives, which once gave 58% vs 77% label ratios across splits.

**Why ZIP, rather than random places.**
- Hard negatives are *neighbours* (same street). Splitting individual places at random would scatter each street across all three splits, and each split would lose most of its same-street pairs (the quadratic shrink again).
- The test set becomes **neighbourhoods the model has never seen**, a stronger claim.

**How.** Shuffle the 331 ZIPs with a seed, then walk down the list keeping a **running total of rows** and cut at 70% and 85% (`cumsum` + `pd.cut`). Rejected alternatives:
- cutting by ZIP *count*, or a round-robin pattern: both balance ZIPs rather than rows, and ZIP sizes range from 1 to 763 rows;
- `GroupShuffleSplit`: same problem, since it splits by group count;
- town or county splits: counties are too few to balance; towns are lumpier and gain little.
- The shuffle is done in pandas, because DuckDB's `random()` isn't reproducible under parallel execution.

**Checks**
- Biggest ZIP ≈ 2.1% of rows, so no single ZIP can swamp a split.
- Result: **70.0 / 14.8 / 15.2%** of rows, **228 / 47 / 56** ZIPs, 0 unlabelled.
- Merge back: 37,075 rows, 25,950 / 5,483 / 5,642.

**Pitfalls**
- Taking `cumsum` on the size-sorted list would put every big ZIP in train.
- `pd.cut` bins come from the arguments you pass. The toy example's `[0, .5, .8, 1]` would have given 50/30/20.
- Shuffling only one column of a DataFrame works only because of index alignment. It's fragile.
- A categorical `split` column triggers `observed=` warnings, and in a multi-key `groupby` it produces a zero-count row for every combination.
- An inner `merge` silently drops unmatched rows. Check the length or pass `validate=`.
- `== None` never matches NaN.

---

## Stage 9 — Positives and the `number_mismatch` holdout (§5)

**Decision.** `number_mismatch` rows are **excluded from all three splits**, not only train. They're ~5% of each split (1,277 / 238 / 262). If many are really different buildings, a correct model would "miss" them and lose up to ~5 points of match recall. Step 1b's models differed by only 0–2 F1 points, so noise of that size would swamp the comparison.

They're saved as `ct_positives_mismatch_holdout` (**1,777** rows) for the hand-labelled subset.

**Check.** Positives = **35,298** (24,673 / 5,245 / 5,380); the two files together give 37,075; `label` was already 1 on every row.

**Pitfalls**
- A copy-pasted filter (`== False` twice) would have saved the positives twice.
- A filename suggested the opposite of the file's contents.

---

## Stage 10 — Hard negatives, Case 1: same street, different number (§6)

**What.**
- Split `address_a_clean` into `number_a` and `street_a`: `^(\d+)` extracts the number, and replacing `^\d+\s+` with nothing leaves the street. 299 places have no number.
- **Self-join** the places on `(split, zip_5, street_a)` and keep pairs that are different places with different numbers, applying the exclusion rule.
- The negative row is **A of X + B of Y**.

**Why the group includes ZIP.** `MAIN ST` exists in dozens of towns. Pairing Main St in one town with Main St in another isn't a *hard* negative, and grouping on the street name alone would create one enormous group.

**Why A of X vs B of Y (and not A vs A).** Positives are always cross-source (Overture vs Foursquare formatting). If negatives were A vs A, formatting consistency alone would reveal the label. Mixing sources in both classes stops formatting from leaking the label.

**The size problem, and the cap**
- 8,492 street groups; 3,479 have 2 or more places. The biggest are commercial corridors (`POST RD E` 225, `BOSTON POST RD` 215).
- Uncapped n·(n−1) pairs: 812,888 / 199,234 / 161,932, about 3.7× the entire negative budget. One street alone (`POST RD E`) would give ~50k pairs.
- **Per-place cap k = 5**: each place X gets at most 5 random partners. That's chosen over Step 2's per-*group* cap, because every place gets used and big streets grow linearly instead of quadratically.

**Text-pair dedupe before the cap.** Several businesses at one building produce identical strings, and about a quarter of capped pairs at k = 4 were duplicates. Deduping first means each place's 5 slots hold distinct examples.

**Checks**
- After filters: 775,732 / 189,794 / 155,012, so only ~5% were removed (same-number and no-number pairs).
- Result: **95,432** (67,195 / 13,781 / 14,456), about 30% of each split's budget at that point.
- `gers_id` is unique per row, so it's a valid "different place" test.

**Pitfalls**
- `NaN != anything` is True in pandas, so no-number places counted as having a "different number" from everyone.
- `&` binds tighter than `!=`: wrap every comparison in parentheses.
- `len([df])` is always 1.
- `.size` (the cells of a DataFrame) vs `groupby().size()` (rows per group).
- `.str.replace` needs `regex=True` in pandas 2.
- `\d+?` is lazy and matches only one digit.

---

## Stage 11 — Hard negatives, Case 2: same number, similar street, far apart (§7)

**Intended.** Same number + same ZIP + a *different but similar* street (`100 N MAIN ST` vs `100 S MAIN ST`), with similarity measured by Jaro-Winkler on the **street only**. Full addresses share the house-number prefix, which inflates Jaro-Winkler's prefix bonus.

**What actually happened (the key finding).**
- 43,418 candidates; about a third scored JW > 0.8.
- The top-scoring pairs were **the same street spelled two ways**: `KILLINGWORTH TURNPIKE` / `TURNPIK`, `20A` / `20-A`, `CUTOFF` / `CUT-OFF`. Under the building-level definition those are **matches**, and labelling them 0 would teach the model the opposite of the truth.
- In Stamford, every street had one canonical spelling, so "similar street" really meant "different street". In real cross-source data it mostly means the same street.

**The tie-breaker: distance.** Strings can't separate "same street, different spelling" from "different street that looks alike", but a map can.
- Coordinates come from Overture's `bbox` (`ymin` = latitude, `xmin` = longitude).
- **Haversine distance** in meters, with trig in radians.
- Crosstab of JW band × distance band: **~99% of JW > 0.85 pairs are within 500 m**, while JW ≤ 0.8 pairs are 70% beyond 1 km.
- The 50–200 m peak is a mall or plaza: one street address, with each store's pin at its own storefront. Unit numbers were stripped in cleaning, so text can't tell those stores apart.

**Rule: JW > 0.80 and distance > 500 m**, then the exclusion rule, dedupe and k = 5. A seeded sample of what the rule keeps was mostly **directional variants** (`N MAIN ST` / `S MAIN ST`, `MAIN ST` / `W MAIN ST`). Those are exactly the false positives Step 1b complained about.

**Rejected alternative.** Taking Case 2 from the 50–200 m band. That's where same-property *matches* concentrate.

**Result: 242** (180 / 39 / 23). That's small, but correct.

**Not yet done.** The control-group check (same number + identical street, i.e. known same-address pairs: what's their distance distribution?) would justify the 500 m cutoff directly.

**Pitfalls**
- Forgetting radians gave plausible-looking but wrong distances (307 km instead of 4.5 km).
- `sin(x²)` vs `sin(x)²` is invisible at tiny angles, so a sanity check can't catch it.
- `merge` treats NaN keys as **equal**, unlike `!=`, so filter NaN numbers *before* the join.
- `range()` only takes integers (use `np.arange`).
- Overwriting `pairs` with an inspection result.
- Rerunning a cell that merges in place adds the columns twice.

**For the writeup:** the standard string-similarity rule for mining hard negatives, which works on synthetic data, **produces mislabelled negatives on real cross-source data**, and geography is what fixes it.

---

## Stage 12 — Dedupe positives (§8a)

**Finding.** **23% of positives repeated the same text pair**: many businesses at one big address, e.g. `7 BACKUS AVE | 7 BACKUS AVE` × 77 and `1 MOHEGAN SUN BLVD` × 40. The model only sees the strings, so those are one example repeated. That would overweight big sites in training and count one example dozens of times in test metrics.

**Decision.** Dedupe **within each split** on exactly what the model sees: both addresses + both ZIPs. Including the ZIPs means `100 MAIN ST` in two towns stays as two examples. **35,298 → 28,834** (20,261 / 4,187 / 4,386).

**Why before the easy negatives.** The negative budget is 9 × positives, so it must be computed from the deduped count.

---

## Stage 13 — Easy negatives, "Case 3" (§8a)

**What.**
- Budget per split = 9 × positives − Case 1 − Case 2.
- Draw X and Y at random **with replacement from the same split** (separate seeds) and place them side by side (`reset_index` + `add_suffix` + `concat(axis=1)`).
- Keep pairs that are different places, pass the exclusion rule, are not already a positive / Case 1 / Case 2 text pair, and are not duplicates.
- Trim to the exact budget. A 1.2× over-draw was enough.

**Why some easy negatives at all.** Real matchers mostly see obviously different candidates. A model trained only on near-misses can behave oddly on easy cases. The paper check showed that Comber & Arribas-Bel give no hard/easy ratio to copy: all their negatives are FEBRL-mutated near-duplicates, and their ZIP use was a blocking key.

**The `A|B` key.** To test whether a *pair* exists elsewhere, combine both columns into one key with a separator. Using `isin` on each column separately would remove pairs that never existed. `|` is safe because cleaning removed all punctuation except `-`. Alternatives: `merge(indicator=True)` (the textbook anti-join) or `MultiIndex.isin`.

**Result: 163,832** (114,974 / 23,863 / 24,995); only ~2% of easy pairs share a ZIP.

**Pitfalls**
- Sampling from **all** places instead of the split (cross-split leakage).
- `concat` defaulting to stacking rows.
- Filtering `left` instead of the combined frame.
- A key of A of X + **A** of Y: it can never match Case 1/2 keys, so the collision check silently passes everything.
- Building the positive key from raw instead of clean columns.
- Dedupe on the wrong subset.
- `.groupby().sample(5)` errors on small groups.
- Unassigned `concat` results.
- `apply` without `axis=1`.

---

## Stage 14 — Final assembly (§8b)

**What.** Stack positives (`neg_type = case0`) + Case 1 + Case 2 + easy (`case3`) into one table:

`address_a, address_b, zip_a, zip_b, split, gers_id_x, gers_id_y, label, neg_type`

**The two ZIPs follow the two addresses**
- `zip_a` is X's Overture ZIP (the old `zip_5`).
- `zip_b` is **Y's Foursquare ZIP**, looked up by `gers_id_y` in the **full** positives table. The deduped table would silently drop 10 Case 2 rows whose Y was deduped away.
- An earlier version used Y's *Overture* ZIP as `zip_b`. That's the same cross-source asymmetry the A-of-X / B-of-Y design exists to prevent.

**Contradiction removal.** 46 negatives had exactly the same text pair as some positive: a third place Z happened to have X's A and Y's B. The same strings labelled both 1 and 0 were removed.

**Is ZIP "too obvious"?** Not for hard negatives: Case 1/2 pairs share a ZIP by construction, just like positives. It only gives away the easy negatives, which are trivial anyway. The way to settle it with evidence is an ablation: train with and without the ZIP features and look at the results per `neg_type`.

**Final result (verified by reading the saved files back)**

| split | positives | Case 1 | Case 2 | easy | neg : pos |
|---|---|---|---|---|---|
| train | 20,261 | 67,178 | 172 | 114,974 | 9.00 |
| val | 4,187 | 13,781 | 34 | 23,863 | 9.00 |
| test | 4,386 | 14,448 | 15 | 24,995 | 9.00 |

**288,294 rows.** 0 duplicate model inputs, 0 positive/negative conflicts, 0 negatives with A == B, and the `split` column matches each file.

**Known caveats**
- `zip_b` has 17 NULLs, so feature code must handle missing values.
- A few hundred address strings appear in two splits. Almost all are the same string in different towns, but `500 W FARMS MALL` (on the Farmington / West Hartford line) is one place in both train and test. A ZIP split can't separate places that straddle ZIP boundaries.

**Still open for Step 2b:** the hand-labelled clean test subset (deferred) and re-scoring the Step 1/1b models.

---

## Cross-cutting lessons

1. **Split before pairing.** Leakage and quadratic retention both come from pairing first.
2. **Decide the match definition first** (building-level). The exclusion rule, cleaning, dedupe and Case 2 all follow from it.
3. **Keep cross-source formatting in both classes** (A of X vs B of Y everywhere, ZIPs included), or formatting leaks the label.
4. **Dedupe on what the model sees.** Different ids with the same strings are the same example.
5. **A filter rule that works on synthetic data can mislabel real data.** Check the top-scoring examples by eye, and use an independent signal (here, distance).
6. **Verify with numbers you predicted in advance**, e.g. "expect 37,075": row counts after every merge, read files back from disk, and per-split breakdowns.
7. **Stale outputs lie.** If a number doesn't match the file the code reads, the output is from an older run.

---

## Self-check quiz

Try each from memory, then check against the stage it points to.

1. Where do the positive pairs come from, and why is a self-pair useless? *(Stages 2, 4)*
2. Why are `number_mismatch` rows excluded from val and test, not just train? *(Stage 9)*
3. What does the building-level match definition imply for (a) two stores in one mall and (b) negative generation? *(Stage 4)*
4. Why does one `LPAD` call fix both 4-digit ZIPs and ZIP+4 in DuckDB, and why wouldn't pandas `zfill` do the same? *(Stage 7)*
5. Give two reasons to split by ZIP rather than by individual place, and one reason to split before generating pairs. *(Stage 8)*
6. Why cut the shuffled ZIP list by cumulative **rows** rather than ZIP count? *(Stage 8)*
7. Why is the Case 1 group `(split, zip_5, street_a)` and not just `street_a`? *(Stage 10)*
8. Why a per-place cap rather than a per-group cap, and why dedupe before capping? *(Stage 10)*
9. What went wrong with "same number + similar street" on real data, and how did distance fix it? Why is 50–200 m the wrong band for Case 2? *(Stage 11)*
10. Why must every negative be A of X vs **B** of Y, including the ZIPs? *(Stages 10, 14)*
11. Why dedupe positives, why include ZIPs in the dedupe key, and why must it happen before the easy budget? *(Stages 12, 13)*
12. Why does checking `isin` on each column separately give the wrong answer for "is this pair already a Case 1 pair?" *(Stage 13)*
13. Why must the `zip_b` lookup use the full positives table, not the deduped one? *(Stage 14)*
14. Does including ZIP as a feature make the task "too obvious"? How would you show it either way? *(Stage 14)*
15. Name one remaining leakage path through the ZIP split. *(Stage 14)*
