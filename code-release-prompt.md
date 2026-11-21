# Kick-off prompt: code release rework

Paste this into a new session in `ml-geocoding/`.

---

Read `CLAUDE.md` first. This session is the **code release rework** for the paper in `~/Downloads/Obsidian Vault/Machine Learning/Article Write Up.md` (first draft done 2026-11-19). This is reorganisation work, not learning work: per my memory note "Reorg is OK to do", you may move, rename, merge and delete files yourself. Don't change what any model computes.

**Goal:** a public repository that contains only what the paper uses, so a reader can rebuild the data and reproduce every table, and nothing else.

**Keep (paper-facing):**
1. Data pipeline: `geocoding_ml/Overture/01_merging_overture_with_fsq.ipynb` → `02_processing_overture_data.ipynb` → `03_case2_street_swap.ipynb` (case2s), plus the helpers they import (`helpers/pairs.py`, `helpers/features.py`, `helpers/check_model_performance.py`, `helpers/case2s_checks.py` if acceptance is kept as a test).
2. Models and training: `model_baseline/process_dumb_models.py` (LogReg / RF / XGB, sentinel and zero-fill via `get_processed_dfs(..., missing_number_as_zero=...)`), `siamese_model/model.py` + `training.py`, `esim/model.py` + `training.py`, `transformers/` (bi- and cross-encoder code).
3. One results notebook per family (or one overall) that produces exactly the paper's tables from saved predictions/probabilities: Tables 3–4 (classical, sentinel vs zero-fill, one-side-missing split), 5–7 (siamese mean/max, vs trees), 8–11 (ESIM settings, rare-number group, D11 AUC, vs siamese), 12–14 (DistilBERT 0.5 table, AUC by house-number group with RF and ESIM, cross-family AUC), Appendix A1–A2 (RF importance). Map each table to the cell that makes it.

**Drop or archive (not in the public repo):** Diagnostics sections, step-by-step build notebooks (`siamese_model/01_siamese_build.ipynb`, `esim/02_esim.ipynb`'s build sections), `scratch.ipynb`, the libpostal baseline (dropped from the paper), Stamford, `Algorithm_Reimplementation/`, `address_matcher/`, logs, notes, `dummy.py`. Keep them in this private repo; the public one is a clean copy or a new repo — ask me which before creating anything.

**Rules:**
- Before deleting or moving anything, show me a table: file → keep / drop / merge into, with one line of reason. Wait for my OK.
- Every table in the paper must be reproducible from the kept code; check each one by re-running from saved predictions (no retraining) and comparing to the numbers in the paper / `results.md`.
- Generated data and checkpoints stay git-ignored; the README explains how to rebuild them (Overture release `2026-09-23.1`, Foursquare release `2026-09-15`) and how long training takes (see the paper's Experimental Setup).
- README: what the project is (one paragraph, link to the paper), setup (`requirements.txt`, pinned), the run order, and the table → notebook map.
- Don't commit or push without asking. Licences: Overture and Foursquare data are openly licensed; check the attribution wording each requires and put it in the README.

**Done when:** the keep/drop table is approved, the public repo layout exists, every paper table is reproduced from it, and the README is written. Then give me the repo link line for the paper's *Data and Code Availability*.
