"""Acceptance checks for the case2s swap files (Overture/04_case2s_acceptance.ipynb).

`flag_issues` takes the full new pair set (with `number_*` / `street_*` columns from `add_number_street`) plus the
original `pairs` and returns a list of [message, sample of the offending rows]. Tests on hand-made frames live in the
notebook, directly under where the function is used.
"""
import pandas as pd
from postal.expand import expand_address

from helpers.pairs import add_number_street

SPLITS = ["train", "val", "test"]


# Every swap row must pass these; the 1:9 check (#6) compares the full set against the original `pairs`.
def flag_issues(df, pairs):
    issues = []  # [Error Message, Sample of Examples]
    checks = {}  # message -> mask of the rows that VIOLATE the check

    # 6. swaps REPLACE easy negatives: per split, positives and negatives after §13 must equal the counts in `pairs`
    #    (the set before §13). Exact counts, not a share: 400 val swaps added on top only move the share 0.900 -> 0.901.
    #    Runs on the full set, before the neg_type filter.
    before = pairs.groupby(["split", "label"], observed=True).size()
    after = df.groupby(["split", "label"], observed=True).size()
    for (split, label), n_before in before.items():
        n_after = after.get((split, label), 0)
        if n_after != n_before:
            kind = "positives" if label == 1 else "negatives"
            issues.append([f"#6 {split}: {n_after} {kind}, expected {n_before} (as in `pairs`)", df[df["split"] == split].head()])

    df = df[df["neg_type"].isin(["case2", "case2s"])].copy()

    # 1. ids / zip / number
    checks["#1 id_a equals id_b"] = df["id_a"] == df["id_b"]
    checks["#1 zip_a doesn't match zip_b"] = df["zip_a"] != df["zip_b"]
    has_number = df["number_a"].notna()  # only judge the number when A has one
    checks["#1 number_a doesn't match number_b"] = has_number & (df["number_a"] != df["number_b"])

    # 2. split: id -> split lookup from both id columns of `pairs`
    lookup = (
        pd.concat([
            pairs[["id_a", "split"]].rename(columns={"id_a": "id"}),
            pairs[["id_b", "split"]].rename(columns={"id_b": "id"}),
        ])
        .drop_duplicates()
        .set_index("id")["split"]
    )
    df["i_split_a"] = df["id_a"].map(lookup)
    df["i_split_b"] = df["id_b"].map(lookup)
    checks["#2 id_a or id_b is not in the row's split"] = ~((df["i_split_a"] == df["i_split_b"]) & (df["i_split_a"] == df["split"]))

    # 3. libpostal overlap
    unique_names = set(df["street_a"].unique().tolist() + df["street_b"].unique().tolist())
    encoding_libpostal = {name: set(expand_address(name, languages=["en"])) for name in unique_names}

    df["libpostal_overlap_a"] = df["street_a"].map(encoding_libpostal)
    df["libpostal_overlap_b"] = df["street_b"].map(encoding_libpostal)
    checks["#3 street_a and street_b share a libpostal expansion"] = pd.Series(
        [len(a & b) > 0 for a, b in zip(df["libpostal_overlap_a"], df["libpostal_overlap_b"])],
        index=df.index,
    )

    # 4. (address_a, address_b) unique
    checks["#4 (address_a, address_b) is duplicated"] = (df["address_a"] + "|" + df["address_b"]).duplicated(keep=False)

    # 5. extra whitespace
    checks["#5 address_b doesn't match ^\\d+ \\S"] = has_number & ~df["address_b"].str.contains(r"^\d+ \S", na=False)

    for message, bad in checks.items():
        if bad.any():
            issues.append([f"{message}. Rows mismatch: {bad.sum()}", df[bad].head()])

    return issues


# The written files (paths relative to Overture/, where the notebooks run).
def load_written():
    return pd.concat([pd.read_parquet(f"Data/ct_pairs_{split}_case2s.parquet") for split in SPLITS], ignore_index=True)


# Real case2 rows are not swaps and legitimately fail #1 number / ZIP, #3, #5: drop them from both frames, so #6 still balances.
def without_real_case2(df):
    return df[df["neg_type"] != "case2"]


def run_acceptance(written, pairs):
    return flag_issues(add_number_street(without_real_case2(written)), without_real_case2(pairs))
