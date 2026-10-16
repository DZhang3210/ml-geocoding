from __future__ import annotations  # lets `list[str] | None` hints work on Python 3.9

import numpy as np
import numpy.typing as npt
import pandas as pd


# Predictions/labels may arrive as numpy arrays, Series or a one-column DataFrame
# (libpostal_baseline returns one). The slice code indexes with boolean masks and
# compares element-wise, which only behaves on flat arrays: a DataFrame would
# align on column labels instead. Flatten to int64 once at the entry points.
def _as_flat_int_array(x):
    return np.asarray(x).ravel().astype(np.int64)


# One table row for one slice: [name, n, correct, errors, % correct].
# Each slice should hold a single label (all positives or all negatives), so
# "% correct" is recall on a positive slice and rejection rate on a negative one.
def _compute_slice(preds, labels, desc, mask):
    n = mask.sum()
    correct = (preds[mask] == labels[mask]).sum()
    errors = (preds[mask] != labels[mask]).sum()
    return [desc, n, correct, errors, f"{(correct*100)/ n:.1f}%"]


# One model's results, one row per slice.
# slices: {slice name: boolean mask over the eval rows}, in display order.
# Returns the table (doesn't print) so several models can be combined later.
def review_slices(
    preds: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    slices: dict[str, npt.NDArray[np.bool_]],
) -> pd.DataFrame:
    preds, labels = _as_flat_int_array(preds), _as_flat_int_array(labels)
    rows = [_compute_slice(preds, labels, desc, mask) for desc, mask in slices.items()]
    return pd.DataFrame(rows, columns=["slice", "n", "correct", "errors", "% correct"])


# review_slices with the standard slices below.
# df is the pairs frame (address_a/b, neg_type), not the feature matrix.
def review_typical_slices(df, preds, labels):
    return review_slices(preds, labels, _typical_masks(df, labels))


# Standard slices: positives split by identical vs non-identical address strings,
# then one slice per neg_type (Overture only; Stamford has no neg_type).
def _typical_masks(df, labels):
    identical = df["address_a"] == df["address_b"]
    masks = {
        "positives: identical": (identical & (labels == 1)).values,
        "positives: non-identical": (~identical & (labels == 1)).values,
    }
    if "neg_type" in df.columns:
        masks["neg: case1 (same street, diff number)"] = (df["neg_type"] == "case1").values
        masks["neg: case2 (same number, similar street)"] = (df["neg_type"] == "case2").values
        masks["neg: case3 (easy)"] = (df["neg_type"] == "case3").values
    # Mixed-label row = plain accuracy; dominated by easy rows, don't quote it alone
    masks["overall (accuracy)"] = np.ones(len(df), dtype=bool)
    return masks


# Up to 10 random rows of df[mask], with the model's prediction and the true label attached.
# attributes: optional list of df columns to keep (predicted/label are always kept).
def find_random_sample_given_mask(
    df,
    preds: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    mask: npt.NDArray[np.bool_] | int | None,
    attributes: list[str] | None = None,
    title = None,
    seed: int | None = 0,   # fixed so a sample can be re-shown; None = new sample each call
):
  
    preds, labels = _as_flat_int_array(preds), _as_flat_int_array(labels)
    if isinstance(mask, (int, np.integer)):
        masks = _typical_masks(df, labels)
        if not 0 <= mask < len(masks.keys()):
            raise ValueError(f"mask must be 0-{len(masks) - 1}: {list(masks)}")
        title = list(masks.keys())[mask]
        mask = list(masks.values())[mask]
    elif mask is None:
        mask = np.ones(len(df), dtype = bool)
    df = df[mask]
    # .assign returns a copy; preds is an array (attaches by position),
    # labels a Series (attaches by index, so it must share df's index)
    df = df.assign(predicted=preds[mask], label=labels[mask])

    if attributes:
        df = df[[*attributes, "predicted", "label"]]

    if title:
        print(f"{'='*10}{title}{'='*10}")
    return df.sample(frac=1, random_state=seed).head(10)
