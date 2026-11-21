# Ramani & Borrajo (2024)

**Title:** "Methods for Matching English Language Addresses"
**PDF:** `papers/novelty_ramani-borrajo-2024_english-address-matching.pdf`
**Venue:** arXiv 2403.12092 (cs.IR), v1 14 Mar 2024, no journal listed · https://arxiv.org/abs/2403.12092
**Authors:** Keshav Ramani, Daniel Borrajo — J.P. Morgan AI Research (Borrajo on leave from Universidad Carlos III de Madrid)
**Role:** Step D novelty check — closest prior work (English addresses, generated pairs, distance baselines vs ESIM)

## Reading guide (one question per section)
| Section | Question for the novelty table |
|---|---|
| III Dataset generation | Real or synthetic pairs? Is "Building Redirection" = our case1, "Street Redirection" = our case2s? |
| IV Algorithms | Which baselines? Same ESIM as ours (word vs char embeddings)? |
| V Results | Overall numbers only, or per mismatch type? Seeds / variance? |
| VI Conclusions | Do they name the limits we'd fix (real data, open benchmark, transformers)? |

## David's notes

## Discussion takeaways (Claude)
*2026-11-13, side session with David. Verdicts are relative to this paper only; the other Step D papers (Cruz 2022, Quan 2024, Yin 2025) are checked in their own notes.*

**Guide questions, settled**
| Section | Answer |
|---|---|
| III Data | Fully synthetic: Faker numbers and streets + top-1,000 US cities, `<Building, Street, City, State>`, **no ZIP**. 10,000 base addresses → 10k match + 10k mismatch pairs, random 80/10/10 shuffle (no geographic split). Match noise is narrow: 5 two-word substitution groups, character noise only inside the street name, so "noise in street = match, changed number = mismatch" is a generator rule a model can learn. **Building Redirection ≈ our case1** (but a shifted number, not a second real address). **Street Redirection ≠ case2s**: random Faker street with no similarity requirement (`123 ABC Ct → 123 XYZ St`), close to our easy negatives. City Redirection is a non-factor under ZIP blocking. Only real data: 100 hand-labelled UK Companies House pairs, 3 matches. |
| IV Algorithms | 13 baselines, all **fixed rules with hand-tuned thresholds** (normalize / regex segmentation / tokens / 3-grams / tf-idf × simple / Jaccard / Levenshtein / Jaro-Winkler / cosine), one metric per rule, no learned classical model. ESIM = frozen 300-d GloVe, ± char-embedding branch concatenated per word (how word and char sequences align is not specified). Ours: 50-d word2vec / fastText trained on our addresses; characters only via fastText sub-words (`stack_mc1` is the closest analogue). |
| V Results | Overall precision / recall / accuracy only; **no breakdown by mismatch type**. ESIM: mean (sd) over 5 seeds; baselines deterministic. ESIM + Char test acc 0.95 vs ESIM 0.94; best rules ≈ 0.87–0.88. Companies House: acc 71.8% (sd 20.6), precision 15.3%, recall 80%. Table VI colours look swapped vs the caption (smooth → 0 curve labelled validation). |
| VI Conclusions | Named: BERT as future work (in place of GloVe), more realistic data generation, other granularities, more tuning, out-of-domain stability. Not named: releasing data or generator; per-type evaluation; learned classical baselines (David's critique). |

**Step D 2b candidates vs this paper**
| Candidate | Verdict | Evidence |
|---|---|---|
| Open real US cross-source benchmark (Overture ↔ Foursquare) | **New** | Their pairs are generated, not cross-source; no ZIP; the only real data is 100 UK pairs; no data or generator release mentioned. Our caveat stays: Overture labels = agreement with Overture's matcher (Step 2b). |
| Results per hard-negative slice | **Partly done** | They generate a case1-like type (Building Redirection) but never score it separately; nothing like case2s exists (Street Redirection is random). Per-slice reporting, and a look-alike-street slice at all, remain ours. |
| Rules, classical, siamese, ESIM, transformer on identical splits | **Partly done** | Rules + ESIM (± char) on one split, but no learned classical model, no siamese / bi-encoder, no transformer (BERT only as future work). |
| Failure modes: missing house number, ZIP-level variation | **New** | The generator never drops the building number (§III-C2) although their Table I pair 4 names the case; no ZIP field, so no spatial variation can be measured. |
