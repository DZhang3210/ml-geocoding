# Progress Log — Geocoding ML Project

**What this file is (since 2026-10-16):** the short index read at the start of every session: where each step stands, papers reviewed, every concept already explained (one line each), and the next-session plan. The full dated session entries, with all the detail, live in `logs/progress-history.md` (the complete log up to 2026-10-16, moved there unchanged; new full entries are appended there). Before explaining a concept that isn't listed here, grep `logs/progress-history.md` for it.

## Where each step stands
| Step | Status | One line | Detail in history |
|---|---|---|---|
| 1 — dumb baseline | ✅ 2026-09-21 | 3 Jaro-Winkler features + LogReg; re-run on the shared schema 2026-10-16 → Step 1 rows in `results.md` (old F1 0.81 not comparable) | "Roadmap steps completed", 2026-10-16 (2nd) |
| 1b — feature-engineered baseline | ✅ 2026-10-16 | 23 similarity features; on Overture test non-identical no better than Step 1's 3 (79.0 vs 79.2%), +5 pts on Stamford; Overture LR nearly ignores the street (weight 0.6 vs ZIP 16). Feature selection / street-type feature dropped | 2026-10-05, 2026-10-15, 2026-10-16 (2nd) |
| 1b side lab | ✅ 2026-10-04 | Decision tree, RF, GB (reg + clf), XGBoost (reg + clf) from scratch, each matched against sklearn/xgboost (`Algorithm_Reimplementation/`) | "Roadmap steps completed" |
| 2 — Stamford dataset | ✅ 2026-09-21 | Synthetic pairs from Stamford shapefile; split ids first, then pair; 1:9 pos:neg per split | "Roadmap steps completed" |
| 2b — Overture real pairs | ✅ 2026-10-16 (labelled subset deferred) | CT Overture ↔ Foursquare pairs, split by ZIP, case1/case2/easy negatives, 9:1 per split; hand-labelled test subset deferred indefinitely (David: too much time) → Overture numbers = agreement with Overture's matcher; case2 can't support claims | 2026-10-06 → 2026-10-13 |
| 2c — libpostal baseline | ✅ 2026-10-15 | Overture test non-identical found 52.3–71.9% vs 79–83% for Step 1b, but rejects 73–80% of case2 vs 7–47%; Stamford 39% | 2026-10-14, 2026-10-15 |
| 3 — siamese network | ✅ 2026-10-23 | Char-level siamese BiLSTM, no ZIP features: Overture test non-identical found 83.6 (val 87.1) vs Step 1 79.2 / RF 83.0 — beats Step 1 by ≈ seed noise, ties RF; pooling mean = max within seed noise; `results.md` row + paragraph | 2026-10-16, 2026-10-18 → 10-23 |
| 4 — ESIM reproduction | 🔶 | Spec decided 2026-10-24 (word2vec 200 + ESIM 200, frozen, pure first, char expansion); next: word2vec set-up, then the case2 data step | 2026-10-23 → 10-24 |

## Papers reviewed
| Paper | Date reviewed | One-line takeaway |
|---|---|---|
| Comber & Arribas-Bel (2019), "Machine learning innovations in address matching: A practical comparison of word2vec and CRFs" (Transactions in GIS 23(2):334-348) | 2026-09-13 | This is Step 1's actual paper -- CRF-based address parsing (Libpostal, pretrained on OSM) feeding Jaro-Winkler comparison vectors into logistic/RF/XGBoost (best: XGBoost, precision 0.955, recall 0.902), vs. swapping in word2vec cosine-similarity comparison vectors (best: XGBoost, precision 0.950, recall 0.870 -- competitive but doesn't win). Negative pairs generated synthetically via FEBRL character-level corruption of real matched pairs, not random pairing. Notably, their "baseline" is already parser+comparison-vector+classifier, not a single raw-string score -- worth knowing before implementing our own dumb baseline. |
| Lee, Claridades & Lee (2020), "Improving a Street-Based Geocoding Algorithm Using Machine Learning Techniques" (Applied Sciences 10(16):5628) | 2026-09-05 | Feature-engineered classical baseline (17 string-similarity metrics -> XGBoost) hits 96%+ accuracy on Korean street addresses without any embeddings/neural net; added to roadmap as Step 1b, a stronger classical floor to beat before Steps 3-5. |
| Reimers & Gurevych (2019), Sentence-BERT | 2026-10-16 | Bi-encoder: shared BERT + mean pooling, fine-tuned so cosine between sentence vectors is meaningful; training head `[u; v; \|u−v\|]`, dropped at inference. |
| Hadsell, Chopra & LeCun (2006), contrastive loss (DrLIM) | 2026-10-16 | Attract similar pairs, repel dissimilar ones only within margin m; the repel term prevents collapse to one point. |
| Neculoiu et al. (2016), siamese recurrent networks for job titles | 2026-10-16 | Closest Step 3 template: char-level 4-layer BiLSTM, mean pooling, dense, cosine, contrastive loss (L+ scaled ¼ for 4:1 neg:pos); negative-loss condition looks reversed as printed (open question). |
| Mudgal et al. (2018), DeepMatcher | 2026-10-16 | Embed → summarise → compare; fixed distances perform poorly, `\|u−v\|` into a classifier up to +40% F1; soft-alignment models win on small data at much higher training cost. |
| Lilian Weng (2021), "Contrastive Representation Learning" (blog) | 2026-10-16 | Reference for contrastive and triplet losses; the margin stops pushing satisfied negatives and prevents collapse. |
| Lin, Kang, Wu, Du & Liu (2020), "A deep learning architecture for semantic address matching" (IJGIS 34(3)) | 2026-10-23 (abstract + secondary sources; full text paywalled) | Step 4's paper: word2vec trained on the address corpus → ESIM (BiLSTM encode → soft-alignment cross-attention → `[a; ã; a−ã; a⊙ã]` → BiLSTM → avg+max pool → MLP) on Shenzhen addresses; P/R/F1 0.97. A cross-encoder, unlike Step 3's bi-encoder. Dataset details, baselines and settings unverified |
| Ramani & Borrajo (2024), "Methods for Matching English Language Addresses" (arXiv 2403.12092, J.P. Morgan AI Research) | 2026-10-23 | ESIM adapted to English (US synthetic, 20k pairs, house-number hard negatives; GloVe instead of word2vec): ≈ 0.94 test accuracy, + character embeddings ≈ 0.95 (5 seeds); low out-of-domain precision. Table numbers from a model extraction — check the PDF before citing |

## Concepts already explained (one line each; details and dates in `logs/progress-history.md`)
**pandas / Python mechanics**
- `.str` accessor vs indexing a Series (`.str[-1]`); `Series.map()` as lookup (2026-09-14)
- `&`/`|` vs `and`/`or`, and `&` binds tighter than `==`/`!=` — parenthesise each comparison (2026-09-14, recurring)
- `df.copy()` is deep; `.apply()` returning a Series → several columns; `sort_values` vs `sort_index`; `value_counts` sorted (2026-09-14)
- In-place methods return `None` (`insert`); `rename`/`drop`/`concat`/`drop_duplicates`/`assign` return new frames — reassign (2026-09-15 →)
- Notebook statefulness: stale outputs after edits; only the last top-level expression displays (use `display`) (2026-09-15, 09-16, recurring)
- Shadowing builtins (`list`) persists until kernel restart (2026-09-15)
- `list.index()` vs `str.find()`, `if index:` fails at 0 (2026-09-15)
- Row-as-Series dtype round-tripping; `iterrows()` yields `(index, Series)` (2026-09-15)
- `groupby` as iterator, `.apply(list).to_dict()`; list vs string keys give tuple vs scalar keys (2026-09-15, 09-16)
- `itertools.combinations`; dict iteration needs `.items()`; `defaultdict(list)` (2026-09-15)
- `merge` suffixes vs `concat(axis=1)` (positional by index); `merge(left_index, right_index)` joins on label equality — `reset_index(drop=True)` first (2026-09-16, 09-17)
- `.isin()` (set membership, order/duplicates lost) vs `set_index(col).loc[list]` (ordered, duplicates kept); `== list` vs `.isin` (2026-09-17)
- `set_index` / `reset_index(drop=…)` mechanics (2026-09-17)
- `random.choice` / `choices` (with replacement, `k=`) / `sample` (without) (2026-09-16, 09-17)
- f-string `=` and format specs; `int` vs `round` vs `:.0f` (2026-09-16)
- Python default arguments evaluated once (mutable `rng` default) (2026-10-04)
- `@dataclass`; `from __future__ import annotations` on Python 3.9; `X | None` needs 3.10+ (2026-10-03, 10-14)
- numpy basics reference: shapes, masks, `unique`/`bincount`, `argmin`, single dtype per array, avoid `np.append` (2026-09-23 → 10-03)
- Masks with `.values`, `.mean()` of booleans as a rate, `len(bool_series)` counts rows (2026-10-11, 10-14)
- Modules vs notebooks: `sys.path` import fix, `%autoreload 2` caveats, modules don't see notebook globals, filename-keyed caches go stale, Restart & Run All (2026-10-13)
- `DataFrame.size` counts cells; `pd.DataFrame(series, columns=…)` selects rather than renames; `.assign` alignment (2026-10-13, 10-14)
- Dedupe on model inputs; anti-joins three ways; `apply(axis=1)`; inner merge silently drops rows → `validate`/length checks (2026-10-11)
- pandas: ZIP+4 normalisation, `pd.cut`, MultiIndex Series, self-joins, `.str.extract` vs `regexp_extract`, `.str.replace(regex=True)`, method chaining (2026-10-09/10)

**SQL / DuckDB / regex** (all new to David)
- DuckDB basics: OLAP vs OLTP, Parquet reading/pruning, clause order, aliases, STRUCT/LIST access, NULL semantics, JOIN ↔ `pd.merge`, `COPY`, `UNION ALL` (2026-10-06)
- Regex and cleaning: `regexp_extract` groups, character classes vs alternation, whole-word matching, `'g'` flag; aggregates after `GROUP BY`; `CASE WHEN` ↔ `np.select`; CTEs; DuckDB conveniences; gap-checking a cleaning chain (2026-10-07)
- `regexp_extract` returns `''` not NULL; aggregates can't go in `WHERE`; `COUNT(*) FILTER` parse trap; verify flags with counts (2026-10-07)
- DuckDB `LPAD` truncation; `random()` not reproducible in parallel (2026-10-09/10)

**Data construction and evaluation**
- Rejection sampling to avoid ordering bias (2026-09-15)
- Blocking's two purposes: retrieval cost (deployment) vs finding hard negatives (training data) (2026-09-15)
- String-similarity families: Levenshtein, Damerau, Jaro-Winkler, token/set metrics (2026-09-15)
- Hard-negative oversampling for training but not evaluation → report per slice (2026-09-15)
- Quadratic (not exponential) pair growth; concentration via pair-weighted contribution, not histograms; quadratic shrinkage `f²` under subsampling and in split-then-filter (2026-09-16, 09-21, 10-11)
- Ablating a structurally constant component isn't a fair test (2026-09-16)
- Positive/negative vs easy/hard are orthogonal (2026-09-16)
- Id-based splitting to prevent leakage; split before pairing (2026-09-17, 09-21, 10-09/10)
- Independent sampling of both sides doesn't give corresponding pairs — join on the relation (2026-09-16)
- Edit-distance vs n-gram metrics fooled by opposite corruptions; `textdistance` `qval`; Tversky(1,1) = Jaccard; Dice/Overlap/Cosine equal for equal-size sets; Levenshtein = Hamming on fixed-length fields (2026-09-21)
- Group split methods (cumulative row share, round-robin, `GroupShuffleSplit`); per-place vs per-group caps (2026-10-09/10)
- Blocking key as a realism assumption; stress-test sets reported separately; multi-pass blocking; confidence intervals for small-n rates (2026-10-12)
- Headline metric dominated by easy rows → per-slice results; single-label slice "% correct" = recall or rejection rate (2026-10-14)
- Filename-keyed caches can't detect changed data → per-row content hash (`check_cache`) (2026-10-16)
- A settings grid needs a reproduction row; a setting that changes nothing needs explaining; known-answer tests must cover the reject direction; bare `except:` hides crashes; prediction caches keyed by setting + split (2026-10-14)
- Test run once, after every choice is fixed; val-vs-test per slice; uniform gap across unchosen settings = data, not overfitting; small slices and mirrored pairs make swings noise; fair comparison needs the same split (2026-10-15)
- A slice compares across datasets only if defined the same way; training mix of a slice changes the weights (2026-10-15)
- libpostal API: `expand_address`, `parse_address`, `postal.dedupe` enums, ZIP+4 vs ZIP5 (2026-10-14)

**Models and metrics**
- `LinearRegression` vs `LogisticRegression`; `predict` vs `predict_proba`, `classes_` (2026-09-21)
- `classification_report` columns, macro vs weighted average; accuracy misleading under 9:1 (2026-09-21, 09-23)
- Model selection on val, never test; built-in validation works within a family only (2026-09-23)
- Caching feature matrices to parquet (and staleness) (2026-09-23)
- Step 1b lineup (LR, SVC, RF, XGB) and what each tests; SVM scale sensitivity; `SVC(probability=True)` hidden CV; class weights vs threshold moving (2026-09-23)
- Tree importances split across duplicate features (2026-09-23)
- When linear/kernel/tree models tie and share errors, the features are the bottleneck (2026-10-05)
- Gini precisely; split mechanics (feature routes rows, Gini sees labels); midpoint thresholds; where Gini is used; gain ≠ prediction change; ties vs sklearn; `min_impurity_decrease` vs `score_cutoff`; tree function decomposition (2026-09-23 → 10-03)
- Regression trees use variance (2026-10-03, 10-04)
- Random forests: bias/variance, decorrelation via bootstrap + per-split feature subsets (2026-10-03)
- Gradient boosting for regression and classification (log-odds, Newton leaves); model state stored with the model (2026-10-04)
- XGBoost split objective: g/h, similarity, gain, λ, γ, `min_child_weight` (2026-10-04)
- Recurring bug pattern: silent parameter fallbacks and positional-argument swaps → pass by keyword (2026-10-04)
- Reading LR weights to explain a slice failure (Overture street weight 0.6 → case2); a 59-row slice can't rank models (2026-10-16, Claude-run)
- A feature easy for a hand-built model can be hard for a learned one (case1) (2026-10-12)
- libpostal vs LR: "same street *name*?" vs "similar *letters*?" (2026-10-15)

**Siamese / transformer (Step 3 prep)**
- BERT = context-dependent token vectors (WordPiece in, one vector per token out, no causal mask, masked-LM pretraining, head needed for an answer); raw BERT cosine isn't trained to be meaningful (2026-10-16)
- Cross-encoder vs bi-encoder; why bi-encoders can cache (2026-10-16)
- Contrastive loss: why separate L+/L− (opposite gradients, collapse); the margin's two jobs; triplet margin (2026-10-16)
- `[u; v; |u−v|]`: per-dimension differences let the head learn a weighted distance; symmetry and the swap test (2026-10-16)
- Head vs cosine: pair classification vs retrieval; both need a threshold chosen on val (2026-10-16)
- Mean pooling dilutes a one-digit change; a single-distance threshold can't separate case1 from a one-letter typo; the head can't recover what pooling dropped (2026-10-16)
- Masked pooling: `-inf` for max only, mean = mask-multiply ÷ true length; BiLSTM backward pass reads pads → `pack_padded_sequence`; `pad_packed_sequence` fills 0, `total_length` (2026-10-18)
- Why `predict` applies `sigmoid`: training applies it inside `BCEWithLogitsLoss` (fused, numerically stable); the model never sees `y`; `zero_grad` first in the loop (2026-10-20)
- A test only counts if a broken implementation would fail it (e.g. the pad must be a value that would change the answer); gradient must reach real positions, not just avoid pads (2026-10-19)
- Bidirectional output = concat of both directions (2×hidden); `torch.equal` for copies only, `allclose` for computed floats; LSTM parameter structure (4 stacked gates, `ih`/`hh`, two biases) (2026-10-19)
- PyTorch basics: registration (attribute assignment) ≠ `requires_grad`; `no_grad` = no graph recorded, also for manual updates; `eval()` only flips `training` (dropout/batchnorm), independent of `no_grad`; `collate_fn` gets a list of samples, returns one batch (2026-10-19)
- Vocab from train only + rare-char cutoff; two-stage stacking leakage (fit stage 2 on unseen pairs); ZIP-blocked hard negatives → ZIP weight comes from easy negatives (2026-10-18)
- Early stopping: patience (not val-loss slope), keep the best epoch not the last; `state_dict()` is live → `deepcopy`; a checkpoint needs vocab + config too (2026-10-21)
- Loss floors: 0.693 = coin flip, 0.325 = always predicting the 1:9 base rate; train ≪ val from epoch 0 ≠ overfitting (2026-10-21)
- Threshold tuning only slides along a model's trade-off; 0.5 is principled when train/val/test share the class mix (2026-10-21)
- Guardrail order: a below-guardrail first "best" blocks every later epoch; a broken-run check must be shown to flag a known-broken run (2026-10-22)
- 3 seeds → min–max range, not std; slice-size margins (n = 15 → ±25 pts at 50%) (2026-10-21)
- Unit tests sit directly under the function they test (David, 2026-10-22)
- Comparing settings over seeds: a gap counts only if bigger than the within-setting seed spread; tie-break rule stated before computing; best-of-epochs val is optimistic; a pre-set seed avoids sending the luckiest run to test (2026-10-23)
- Held-out 100% ≠ overfitting (overfitting = train ≫ held-out); val loss is dominated by easy rows (2026-10-23)
- Why shared weights: same input type → same string must give the same vector (identical pairs, symmetry); two encoders could learn side/source-specific shortcuts (here side A = Overture, side B = Foursquare); vs CLIP, where two modalities need two encoders (2026-10-23)
- A result that lands exactly on a no-learning value (90% accuracy at 1:9, 0.325 / 0.693 loss) → suspect the pipeline first (2026-10-23)

**Step 4 prep (ESIM / word2vec)**
- Decomposable attention (Parikh 2016): F → e_ij = F(a)·F(b) → soft-align (β_i, α_j) → compare G([a; β]) per word → aggregate by **sum** → H → class; no RNN = no word order (2026-10-24)
- ESIM (Chen 2017): BiLSTM = *intra*-sequence context, then *inter*-sequence soft alignment (plain dot product); enhancement `[a; ã; a−ã; a⊙ã]` (Step 3 head idea per word); composition BiLSTM (not attention); avg + max pooling instead of sum (length-independent) → MLP (2026-10-24)
- word2vec skip-gram with negative sampling is itself a binary classifier (real neighbour vs random negative) — same training shape as the matcher (2026-10-24)
- word2vec sees text, never labels: training it on all splits is a *coverage* question, not label leakage; caveat = test-town words get vectors only from test text (2026-10-24)
- Frozen vs fine-tuned embeddings: fine-tuning moves only tokens seen in train pairs → inconsistent spaces (2026-10-24)
- Word-level risk for case1: `123` / `124` share contexts → near-identical vectors; rare numbers → UNK; per-word fixed random OOV vectors (not zeros / one shared UNK); Parikh's hashed OOV buckets (2026-10-24)
- A guardrail only sets a floor: selecting on found above it still trades case1 away; report the metric's range across epochs to separate selection from model failure (2026-10-24)
- Embedding / hidden size: GloVe's 300 reflects billions of words, not a rule; hidden = embedding is a convention; nearest-neighbour check to validate word2vec size (2026-10-24)
- Soft alignment (ESIM §3.2 / Parikh "Attend"): score grid e (len_a × len_b), softmax per row/column, weighted sum of the other address's word vectors = a word's translation (d,); both directions needed (one-way misses the other side's unmatched words, breaks symmetry); softmax ≠ √d scaling. Note: `Geocoding ML NOTES/Notes/Soft alignment (ESIM, Parikh).md` (2026-10-23)

## Background already established
- Software engineer, ex-Ping Data Intelligence (Ping.Maps, tile server pipelines, SQL optimization, SOV text representation for ML pretraining)
- Strong GPT-2-level transformer/NN background coming in
- Working on address matching as the chosen narrow task (not parsing, not reverse geocoding)

## Next session plan (latest; earlier plans in the history)
- **Start (2026-10-27):** first recall warm-up (new start-of-session checklist item 6), then continue the Step 4 plan below.
- **Plan (2026-10-24):** Step 4 spec decided (table in the history, 2026-10-24). **Next:** word2vec set-up (tokeniser, gensim settings — window / min_count / sg vs CBOW / epochs —, nearest-neighbour check) → case2 data step (semi-synthetic street swap; re-score every model) → ESIM build, deciding symmetry first → timing → runs. Also pending: ZIP-text diagnostic.
