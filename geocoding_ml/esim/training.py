"""ESIM runs: train and save one run, then read the saved runs back.

Reorganised 2026-11-14 from esim_model.py (run_training, save_run) and esim_diagnostics.py (health,
slice_table, rare_number_*), unchanged. The model itself: model.py. Notebook: 02_esim.ipynb.

Protocol (Step 3's): BCE, Adam 1e-3, best epoch by is_better (case1 guardrail 98), patience 3, 3 seeds.
Every reader below works from the saved files of each setting (history_seed_*.csv, preds_{split}_seed_*.npy);
no model is reloaded.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from gensim.models import Word2Vec

from esim.model import ESIM, CHECKPOINTS, tokenize
from helpers.check_model_performance import review_typical_slices
from siamese_model.training import predict, train_epoch
from siamese_model.training import evaluate, is_better, broken_run_problems, THRESHOLD

LR, MAX_EPOCHS, PATIENCE = 1e-3, 20, 3
SEEDS = [0, 1, 2]

# setting name -> run folder (absolute, so it works from any folder)
SETTINGS = {
    "mc2":       os.path.join(CHECKPOINTS, "esim_mc2"),        # word2vec min_count=2 (original)
    "mc1":       os.path.join(CHECKPOINTS, "esim_mc1"),        # word2vec min_count=1
    "ft_mc1":    os.path.join(CHECKPOINTS, "esim_ft_mc1"),     # fastText min_count=1
    "rand_mc2":  os.path.join(CHECKPOINTS, "esim_rand_mc2"),   # word2vec mc2 + a fixed random row per dropped token
    "stack_mc1": os.path.join(CHECKPOINTS, "esim_stack_mc1"),  # [word2vec mc1; fastText mc1]
}
# vocab that defines the rare-number group
W2V_PATH = os.path.join(CHECKPOINTS, "word2vec_50_mc2.model")


# ---------- train + save ----------

def run_training(matrix, loaders, val, val_label, seed):
    """Seeded run with early stopping on is_better -> (history DataFrame, best row, best state_dict)."""
    torch.manual_seed(seed)
    model = ESIM(matrix)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, since_best, history = None, None, 0, []
    for epoch in range(MAX_EPOCHS):
        row = {"epoch": epoch, "train_loss": train_epoch(model, loaders["train"], opt, loss_fn),
               **evaluate(model, loaders["val"], val, val_label)}
        print(row)
        history.append(row)
        if is_better(row, best):
            best, best_state, since_best = row, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    if best is None:                                  # no epoch reached the case1 guardrail -> keep the last
        best_state = copy.deepcopy(model.state_dict())
    return pd.DataFrame(history), best, best_state


def save_run(folder, seed, history, best, best_state, vocab, matrix, loaders, config):
    """Checkpoint + history + thresholded val/test predictions (esim_seed_{s}.pt, history_seed_{s}.csv, preds_*)."""
    os.makedirs(folder, exist_ok=True)
    history.to_csv(os.path.join(folder, f"history_seed_{seed}.csv"), index=False)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": config, "seed": seed, "best": best},
               os.path.join(folder, f"esim_seed_{seed}.pt"))
    model = ESIM(matrix)
    model.load_state_dict(best_state)
    for split in ["val", "test"]:
        preds = (predict(model, loaders[split]) >= THRESHOLD).int().numpy()
        np.save(os.path.join(folder, f"preds_{split}_seed_{seed}.npy"), preds)


# ---------- read saved runs ----------

def _available(settings):
    """(name, folder, seed) for every run whose val predictions exist."""
    return [(name, folder, seed) for name, folder in settings.items() for seed in SEEDS
            if os.path.exists(f"{folder}/preds_val_seed_{seed}.npy")]


def health(settings=SETTINGS):
    """One row per run: epochs trained, broken_run_problems ([] = healthy)."""
    rows = []
    for name, folder, seed in _available(settings):
        h = pd.read_csv(f"{folder}/history_seed_{seed}.csv")
        rows.append({"setting": name, "seed": seed, "epochs": len(h), "problems": broken_run_problems(h)})
    return pd.DataFrame(rows).set_index(["setting", "seed"])


def slice_table(split_df, labels, split, settings=SETTINGS):
    """% correct per slice (rows) for every (setting, seed) (columns)."""
    cols = {(name, seed): review_typical_slices(split_df, np.load(f"{folder}/preds_{split}_seed_{seed}.npy"), labels)
                          .set_index("slice")["% correct"]
            for name, folder, seed in _available(settings)}
    return pd.DataFrame(cols)


def rare_number_group(split_df, group_vocab=None, w2v_path=W2V_PATH):
    """D2-D4's group: both addresses start with a number unknown to the word2vec mc2 vocab,
    and the rest of the address is identical. Fixed vocab, so every setting is compared on the same rows."""
    if group_vocab is None:
        group_vocab = Word2Vec.load(w2v_path).wv.key_to_index
    strip_number = lambda s: s.str.replace(r"^(\d+\s+)", "", regex=True)
    grp = np.ones(len(split_df), dtype=bool)
    for side in ["a", "b"]:
        addr = split_df[f"address_{side}"]
        first_unknown = addr.apply(lambda x: (tokenize(x) or [None])[0] not in group_vocab)
        grp &= first_unknown.values & addr.str.contains(r"^\d+").values
    return grp & (strip_number(split_df["address_a"]) == strip_number(split_df["address_b"])).values


def rare_number_table(splits, settings=SETTINGS, w2v_path=W2V_PATH):
    """02's D5. splits: {"val": (df, labels), "test": ...}. Per (split, setting, seed):
    identical positives found / case1 negatives rejected inside the rare-number group, and errors outside it."""
    group_vocab = Word2Vec.load(w2v_path).wv.key_to_index
    rows = []
    for split, (split_df, labels) in splits.items():
        grp = rare_number_group(split_df, group_vocab)
        ident = (split_df["address_a"] == split_df["address_b"]).values & (labels == 1).values
        case1 = (split_df["neg_type"] == "case1").values & (labels == 0).values
        for name, folder, seed in _available(settings):
            p = np.load(f"{folder}/preds_{split}_seed_{seed}.npy")
            rows.append({
                "split": split, "setting": name, "seed": seed,
                "identical UNK: found":    f"{(p[ident & grp] == 1).sum()} / {(ident & grp).sum()}",
                "case1 UNK: rejected":     f"{(p[case1 & grp] == 0).sum()} / {(case1 & grp).sum()}",
                "identical other: missed": int((p[ident & ~grp] == 0).sum()),
                "case1 other: errors":     int((p[case1 & ~grp] == 1).sum()),
            })
    return pd.DataFrame(rows).set_index(["split", "setting", "seed"])
