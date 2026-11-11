"""ESIM diagnostics as functions: health check, per-slice table, and the rare-number group (D2-D5 of 02_esim.ipynb).
Written by Claude 2026-11-06 for 03_esim_extension.ipynb. Every function reads the saved files of each
setting (history_seed_*.csv, preds_{split}_seed_*.npy); no model is reloaded.
"""
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from esim.tokenizer import tokenize
from helpers.check_model_performance import review_typical_slices
from siamese_model.real_training import broken_run_problems

SEEDS = [0, 1, 2]
# defines the rare-number group; relative to esim/ -- from another folder pass e.g. "../esim/checkpoints/word2vec_50_mc2.model"
W2V_PATH = "checkpoints/word2vec_50_mc2.model"

# setting name -> run folder (relative to esim/)
SETTINGS = {
    "mc2":       "checkpoints/esim_mc2",        # word2vec min_count=2 (original)
    "mc1":       "checkpoints/esim_mc1",        # word2vec min_count=1
    "ft_mc1":    "checkpoints/esim_ft_mc1",     # fastText min_count=1
    "rand_mc2":  "checkpoints/esim_rand_mc2",   # option 5: word2vec mc2 + a fixed random row per dropped token
    "stack_mc1": "checkpoints/esim_stack_mc1",  # option 4: [word2vec mc1; fastText mc1]
}


def _available(settings):
    """(name, folder, seed) for every run whose val predictions exist."""
    return [(name, folder, seed) for name, folder in settings.items() for seed in SEEDS
            if os.path.exists(f"{folder}/preds_val_seed_{seed}.npy")]


def health(settings=SETTINGS):
    """One row per run: epochs trained, best epoch, broken_run_problems ([] = healthy)."""
    rows = []
    for name, folder, seed in _available(settings):
        h = pd.read_csv(f"{folder}/history_seed_{seed}.csv")
        rows.append({"setting": name, "seed": seed, "epochs": len(h), "problems": broken_run_problems(h)})
    return pd.DataFrame(rows).set_index(["setting", "seed"])


def slice_table(split_df, labels, split, settings=SETTINGS):
    """% correct per slice (rows) for every (setting, seed) (columns) -- 02's 14d."""
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
