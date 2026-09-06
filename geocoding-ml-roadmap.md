# Geocoding ML Roadmap — Address Matching Project

**Task scope:** Address matching / entity resolution only (not parsing, not reverse geocoding, not toponym resolution).
**Sequence:** Depth-first on one task — each step builds on the last; only the last two steps branch outward.
**Time estimates assume full-time study: ~6-8 hours/day.**

---

## At a glance

| #   | Step                                  | Paper                                           | Est. time | Status |
| --- | ------------------------------------- | ----------------------------------------------- | --------- | ------ |
| 1   | Dumb baseline                         | Comber & Arribas-Bel (2019)                     | 1-2 days  | 🔶     |
| 1b  | Stronger feature-engineered baseline  | Lee, Claridades & Lee (2020)                    | 1-2 days  | ⬜      |
| 2   | Real dataset                          | —                                               | 2-3 days  | ⬜      |
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
- **Status note (2026-09-05):** Pipeline built and working end-to-end in `step1-baseline/` (dataset generator, feature extraction, logistic regression, eval) against a synthetic placeholder dataset — see `step1-baseline/results.md`. Test F1 = 0.9945, but this is inflated by easy synthetic negatives; the real benchmark number will come from rerunning `train.py` once Step 2's real dataset replaces the synthetic one. Don't cite 0.9945 as the baseline to beat.

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
- **Finish threshold:** A results-table row for this model sitting between Step 1's baseline and Step 3's hand-built siamese model, plus one paragraph on whether the added feature engineering + XGBoost meaningfully beat the Step 1 single-feature baseline on your data, and by how much.

### 2 — Assemble a real, small dataset
- **Est. time:** 2-3 days.
- **Task:** Pull address data from OpenAddresses (https://openaddresses.io) or Census TIGER (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html). Construct positive pairs (same location, different formatting) and hard negative pairs (similar-looking, different location).
- **Implementation instructions:**
  1. Download one city/region extract rather than a whole country — tens of thousands of addresses is plenty.
  2. Generate positive pairs by applying synthetic formatting noise to real addresses (abbreviate "Street" to "St", drop unit numbers, swap word order) so you know the ground truth.
  3. Generate hard negatives deliberately — addresses on the same street with different numbers, or same number on visually similar street names — not just random pairs, which are too easy and won't stress-test your model.
  4. Split into train/val/test by *location*, not by row, so the same address doesn't leak across splits.
- **Finish threshold:** A single dataset file (or a small set of them) with clearly labeled positive/negative pairs and a fixed train/val/test split you will reuse for every later step without re-splitting.

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