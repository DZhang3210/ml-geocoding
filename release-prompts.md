# Release prompts (2026-11-21)

Paste each into a fresh Claude session opened in `~/Downloads/ml-geocoding`. Run the code base first: the write-up quotes the repo's numbers.

---

## 1. Code base (public repo + Hugging Face)

> Update the public code release to match the paper's v2. The paper text is final: `~/Downloads/Obsidian Vault/Machine Learning/Article Write Up.md` (PDFs in `~/Downloads/address-matching-paper/`, `main_public.pdf`). Read it first; it is the source of truth for every number and label. Background: `logs/progress-history.md`, entry "2026-11-19 → 11-21".
>
> The repo is `~/Downloads/address-matching` (GitHub `DZhang3210/address-matching-by-pair-type`, three commits, v1). It currently runs both versions through env overrides: v2 = `AM_DATA=/Users/david/Downloads/ml-geocoding/geocoding_ml/Overture/Data AM_FEATURES=$PWD/classical/features_v2 AM_PREDICTIONS=$PWD/predictions_v2 AM_CHECKPOINTS=$PWD/checkpoints_v2`; v1 outputs are in `predictions/`, `checkpoints/`, `classical/features/`. Python: `/Users/david/Downloads/ml-geocoding/venv/bin/python`. Uncommitted work: `helpers/paths.py`, `results/paper_tables.ipynb`, `requirements.txt`, `.gitignore`, `esim/01_word2vec.ipynb`, plus new `siamese/train_seeds.py`, `tools/classical_seeds.py`, `distilbert/train.py`, `results/figures/`, `predictions_v2/`, `classical/features_v2/`.
>
> Done when:
> 1. v2 is the default: `predictions_v2` → `predictions`, `checkpoints_v2` → `checkpoints`, `features_v2` → `features`, the pipeline's pair files are the fixed ones (`pipeline/Data/ct_pairs_*_case2s.parquet` = the ml-geocoding copies). v1 outputs are removed from the repo (git history keeps them). No env overrides needed.
> 2. `results/paper_tables.ipynb` runs top to bottom and reproduces every table and figure of the paper exactly: pair-type labels as in the paper (Different number / Real look-alike street / Look-alike street / Random; code names only in Table 1), Tables 1–15, A1–A2, B1 (% of missing-number matches found, old vs new; keep the v1 numbers needed for B1 as a small saved file), Figures 1 (pipeline) and 3 (Youden's J), siamese as min–max over 3 seeds, XGBoost the reference tree, ESIM kept = fastText. Check each cell's output against the paper; list any mismatch instead of silently changing the paper.
> 3. The pipeline's case 1 fix is in the repo's pipeline code (both sides must have a house number), matching `geocoding_ml/Overture/`.
> 4. README: v2 results, the correction (one short paragraph pointing to Appendix B), how to reproduce, the seed scripts; plain wording, no hooks or intensifiers. `paper/` holds the new `main_public.pdf`.
> 5. Hugging Face: model repo `zhangdav/address-matching-by-pair-type` and dataset repo `datasets/zhangdav/address-matching-by-pair-type` updated to the v2 weights and pairs, with model/dataset cards matching the README. David is logged in to HF himself; never handle tokens.
>
> Ask David before every commit, push and HF upload. Keep his email out of everything you send.

---

## 2. Write-up (blog post + portfolio)

> Update my blog post on the address-matching study to match the paper's v2. The paper text is final: `~/Downloads/Obsidian Vault/Machine Learning/Article Write Up.md` (read it first; it is the source of truth). The repo `~/Downloads/address-matching` should already be on v2 (check its README; if it isn't, stop and tell me).
>
> Files: the post is `~/Downloads/DZhang3210.github.io/src/posts/address-matching.md` (built output in `posts/address-matching/`; find the build command in the repo). The portfolio card is in `~/Downloads/portfolio-site/lib/ml-projects-list.js` with the screenshot `public/address-matching-post-header.png` (a crop of the post's opening, same proportions as the Same Place? image).
>
> Done when:
> 1. Every number, table and claim in the post agrees with the paper v2: the main finding is now stated as shares found (tree models find 94–96% of matches with a missing house number, the neural models 63–76%); XGBoost is the reference tree; ESIM's kept setting is fastText; siamese has 3 seeds; pair types use the paper's names. Add a short note that an earlier version had a data error, fixed and retrained, linking the paper's Appendix B.
> 2. Writing style: plain, states what was done and found; no hooks, teasers, intensifiers or "gamey" framing; dry is fine. Keep the post's structure unless something no longer holds.
> 3. The post builds; check it in the browser pane. The portfolio card's description still holds (update it if not) and its screenshot shows the new opening.
>
> Show me the post diff before committing. Ask before every commit and push.
