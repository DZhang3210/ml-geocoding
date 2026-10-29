"""Diagnostics for the case2s swap (Overture/05_case2s_diagnostics.ipynb): the queries behind D0-D5, as functions.

Each function returns the frame the notebook displays; the notebook's markdown keeps the question, the red flag and
the written-down result.
"""
import jellyfish
import pandas as pd

from helpers.pairs import add_number_street

SPLITS = ["train", "val", "test"]

# 02's Result table (2026-10-11), per split x neg_type: the counts D0 compares `pairs` against.
OLD_COUNTS_02 = pd.DataFrame(
    {"case0": [20261, 4187, 4386], "case1": [67178, 13781, 14448], "case2": [172, 34, 15], "case3": [114974, 23863, 24995]},
    index = SPLITS,
)


# D0: rows now minus the rows in 02's result table, per split x neg_type
def row_gap(pairs):
    now = pairs.groupby(["split", "neg_type"], observed = True).size().unstack().loc[SPLITS]
    return now - OLD_COUNTS_02


# D1: seeded sample per JW band, on the pairs the other filters would keep
JW_EDGES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

def filter_not_disq(street_pairs):
    return street_pairs[(~street_pairs["lib_match"]) & (~street_pairs["prefix_match"]) & (~street_pairs["dir_disq"])]

def jw_band_sample(street_pairs, seed, edges = JW_EDGES, n = 5):
    df = filter_not_disq(street_pairs)
    bins = pd.cut(df["jw_similarity"], edges, right = False)
    return (
        df[["street_a", "street_b", "jw_similarity"]]
        .sample(frac = 1, random_state = seed)
        .groupby(bins, observed = True).head(n)
        .sort_values("jw_similarity")
    )


# D2: real case2 with a JW on the clean streets (same recipe as the swaps' `jw`), and the quantile table
def real_case2_jw(pairs):
    real = add_number_street(pairs[pairs["neg_type"] == "case2"])
    real["jw"] = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(real["street_a"], real["street_b"])]
    return real

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

def jw_quantiles(real, swaps, qs = QUANTILES):
    return pd.DataFrame({"case2": real["jw"].quantile(qs), "case2s": swaps["jw"].quantile(qs)})


# D3: length and extra-text shares of the swapped B strings vs the B side of real positives
EXTRA_TEXT = r"\b(?:STE|SUITE|UNIT|APT|BLDG|BUILDING|FL|FLOOR)\b|#"   # suite / unit markers
STATE_ZIP = r"\b(?:CT|USA)\b|\b\d{5}\b"                              # state / country / ZIP text

def b_string_profile(swaps, pairs):
    sw = swaps["address_b"]
    c0 = pairs.loc[pairs["neg_type"] == "case0", "address_b"]
    return pd.DataFrame({
        "case2s": [sw.str.len().mean(), sw.str.len().median(), sw.str.contains(EXTRA_TEXT).mean(), sw.str.contains(STATE_ZIP).mean()],
        "case0 B": [c0.str.len().mean(), c0.str.len().median(), c0.str.contains(EXTRA_TEXT).mean(), c0.str.contains(STATE_ZIP).mean()],
    }, index = ["mean length", "median length", "share with suite / unit", "share with state / ZIP"])


# D4: rows per X and per street pair in the written case2s rows, one row per split
def concentration(written):
    rows = []
    for split in SPLITS:
        sw = add_number_street(written[(written["split"] == split) & (written["neg_type"] == "case2s")])
        per_x = sw.groupby("id_a").size().sort_values(ascending = False)
        per_pair = sw.groupby(["street_a", "street_b"]).size().sort_values(ascending = False)
        rows.append({
            "split": split,
            "swaps": len(sw),
            "distinct_x": len(per_x),
            "max_per_x": per_x.max(),
            "top10_x_share": per_x.head(10).sum() / len(sw),
            "street_pairs": len(per_pair),
            "max_per_pair": per_pair.max(),
            "top10_pair_share": per_pair.head(10).sum() / len(sw),
        })
    return pd.DataFrame(rows).set_index("split")


# D5: pairs that ONLY prefix_match removes (every other filter keeps them), and a seeded sample of them
def only_prefix(street_pairs):
    return street_pairs[
        street_pairs["prefix_match"] & ~street_pairs["lib_match"] & ~street_pairs["dir_disq"] & (street_pairs["jw_similarity"] > 0.8)
    ]

def only_prefix_sample(street_pairs, seed, n = 40):
    sub = only_prefix(street_pairs)
    return sub.sample(min(n, len(sub)), random_state = seed)[["street_a", "street_b", "jw_similarity", "swap_type"]]
