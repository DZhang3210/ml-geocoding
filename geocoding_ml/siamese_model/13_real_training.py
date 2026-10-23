"""Step 3 real training: per-epoch evaluation, the early-stopping rule, broken-run checks,
and loading saved runs.

Used by 13_real_training.ipynb (training) and 14_training_analysis.ipynb (analysis). The functions
were written by David in 13_real_training.ipynb and moved here unchanged on 2026-10-23;
load_run / rebuild_model are the loading lines of his summary cell, wrapped as functions.
"""
from importlib import import_module

import pandas as pd
import torch
from torch import nn

from helpers.check_model_performance import review_typical_slices

_model_step = import_module("siamese_model.08_encoder_step")          # digit-prefixed filenames -> can't use `from ... import`
Encoder, SiameseMatcher = _model_step.Encoder, _model_step.SiameseMatcher
predict = import_module("siamese_model.12_training_step").predict

THRESHOLD = 0.5
CASE1_GUARDRAIL = 98.0      # % rejected on val case1; an epoch below it can't be "best"
COIN_FLIP_LOSS = 0.693      # -ln 0.5: a model that has learned nothing
BASE_RATE_LOSS = 0.325      # -(0.1 ln 0.1 + 0.9 ln 0.9): always predicting the 1:9 class mix
CHECKPOINT_DIR = "checkpoints"   # git-ignored, relative to siamese_model/


def evaluate(model, loader, df, labels):
    """One model on one split -> {"val_loss", "found", "case1", "easy"} (% correct per slice, as in results.md)."""
    probs = predict(model, loader)
    y_true = torch.tensor(labels.values, dtype=torch.float32)
    preds = (probs >= THRESHOLD).int().numpy()
    val_loss = nn.functional.binary_cross_entropy(probs, y_true).item()
    table = review_typical_slices(df, preds, labels).set_index("slice")
    pct = lambda name: float(100 * table.loc[name, "correct"] / table.loc[name, "n"])
    found = pct("positives: non-identical")
    case1 = pct("neg: case1 (same street, diff number)")
    easy = pct("neg: case3 (easy)")

    return {"val_loss": val_loss, "found": found, "case1": case1, "easy": easy}


def is_better(row, best):
    """Early-stopping rule: below the case1 guardrail never qualifies; then higher found; tie -> lower val loss."""
    if row["case1"] < CASE1_GUARDRAIL:
        return False
    if best is None:
        return True
    return (row["found"] > best["found"]) or (row["found"] == best["found"] and row["val_loss"] < best["val_loss"])


def broken_run_problems(history):
    """One run's history -> list of problems; [] = healthy."""
    problems = []
    if history.isna().any().any():
        problems.append("NaN in history")
    first_val_loss = history["val_loss"].iloc[0]
    if first_val_loss >= COIN_FLIP_LOSS:
        problems.append(f"val loss after epoch 0 = {first_val_loss:.3f} (coin flip = {COIN_FLIP_LOSS})")
    min_train_loss = history["train_loss"].min()
    if min_train_loss >= 0.30:
        problems.append(f"train loss never below 0.30 (min {min_train_loss:.3f}; base rate = {BASE_RATE_LOSS})")
    max_found = history["found"].max()
    if max_found == 0:
        problems.append("val found = 0% in every epoch")
    return problems


def load_run(seed, pooling, checkpoint_dir=CHECKPOINT_DIR):
    """One saved run -> (per-epoch history DataFrame, checkpoint dict with state_dict / vocab / config / seed / best)."""
    history = pd.read_csv(f"{checkpoint_dir}/history_seed_{seed}_pooling_{pooling}.csv")
    ckpt = torch.load(f"{checkpoint_dir}/overture_siamese_seed_{seed}_pooling_{pooling}.pt")
    return history, ckpt


def rebuild_model(ckpt):
    """Checkpoint -> the model that produced ckpt["best"], built from the checkpoint's own vocab + config."""
    model = SiameseMatcher(Encoder(len(ckpt["vocab"]), pooling=ckpt["config"]["pooling"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
