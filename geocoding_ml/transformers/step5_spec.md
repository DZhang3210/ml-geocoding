# Step 5 spec — pretrained transformer (DistilBERT)

Deliberated 2026-11-09. Principle: **only the encoder (and the optimiser it forces) changes** relative to the from-scratch model each variant is compared with.

## Decided

| # | Item | 5a — bi-encoder | 5b — cross-encoder |
|---|---|---|---|
| 1 | Compared with | Step 3 siamese BiLSTM (`case2s`-trained run) | Step 4 ESIM (`rand_mc2`) |
| 2 | Pretrained model | `distilbert-base-uncased` (shared by both sides) | `distilbert-base-uncased` |
| 3 | Input | each address alone: `[CLS] A [SEP]` | one joined row: `[CLS] A [SEP] B [SEP]`, A = Overture first |
| 4 | Tokenisation | `max_length=64`, `padding=True` (pad to the batch's longest row); no row is cut (longest pair 35 tokens + 3 specials) | same |
| 5 | Summary vector | masked mean pooling (`attention_mask`) per side | masked mean + max pooling over the joined row |
| 6 | Head | `[\|u−v\|; u⊙v]` → Linear → 1 logit (symmetric) | MLP 200, tanh → 1 logit |
| 7 | Loss / threshold | `BCEWithLogitsLoss`, 0.5 | same |
| 8 | Train data | `overture_case2s` train | same |
| 9 | Optimiser | AdamW, two groups: encoder 2e-5, head 1e-3; weight decay 0.01; linear warm-up 10% then decay | same |
| 10 | Best epoch | highest val non-identical found with val case1 ≥ 98% (Step 3 rule) | same |
| 11 | Order symmetry | symmetric by construction | **deferred**: one order only; swap test later on saved models |

## Open (decided after the one-epoch timing)

| Item | Proposal | Depends on |
|---|---|---|
| Epoch budget / patience | **5a: 5 epochs, patience 2 (David, 2026-11-10)**; 5 epochs also sets the scheduler's total steps. Re-think if the best epoch is the last one. **Kept at 5 (David, 2026-11-11)** although seeds 0 and 1 picked their last epoch (val found still rising; lr decays to 0 at epoch 5); revisit for the paper — writeup caveat. **5b: 5 epochs, patience 2 (David, 2026-11-11)**; timing 16.1 min/epoch train on M1 Pro MPS (≈ 6.5 on Colab by 5a's 2.5× ratio) | 5a timing 2026-11-10: ≈ 25 min/epoch on MPS incl. val pass |
| Hardware | **Colab T4 (2026-11-10)**: ≈ 10 min/epoch vs ≈ 25 on M1 Pro MPS | — |
| Seeds | **3 (0, 1, 2)**, all scored on test, reported as min–max (Step 4 protocol) | decided 2026-11-10 |
| Batch size | **32** | decided 2026-11-10 |
| Head dropout (5b MLP) | **0.3 as in ESIM (built 2026-11-11)** | — |
| `[CLS]` head for 5b | optional extra config, 3 seeds | timing |

## Checks David runs

| Check | Why |
|---|---|
| Rare-number case1 group (Step 4 D5 table) must pass for 5a and 5b | WordPiece splits `4817` → `48 ##17`; does the model keep `##17` / `##19` apart? |
| Slice table as in `results.md` (non-identical, identical, case1, case2, `case2s`, easy), val then test | like-for-like with Steps 3/4 |
| Loss stuck at 0.325 / 0.693 from the start → suspect pipeline or a collapsed run (Mosbach) | high-lr / bad-init failure mode |
| Swap test on 5b (later) | 5b is order-dependent; compare flip rate with ESIM's 2.3–4.3% |
