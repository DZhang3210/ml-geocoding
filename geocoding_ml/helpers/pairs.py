import os

import pandas as pd

# Both datasets share one pair schema:
#   id_a, id_b, address_a, address_b, zip_a, zip_b, split, label, neg_type (case0 = positive, case1-3 = negatives)
# Overture: Overture/02_processing_overture_data.ipynb. Stamford: Stamford/03_convert_to_shared_schema.py.

_ROOT = os.path.join(os.path.dirname(__file__), "..")   # geocoding_ml/
OVERTURE_RELEASE = "2026-09-23.1"


def _path(dataset, split, release):
    if dataset == "overture":
        return os.path.join(_ROOT, "Overture", "Data", f"ct_pairs_{split}_{release}.parquet")
    if dataset == "stamford":
        return os.path.join(_ROOT, "Stamford", "Data", f"stamford_pairs_{split}.parquet")
    raise ValueError(f"Unknown dataset {dataset!r}: use 'overture' or 'stamford'")


# One split -> (pairs without label, label). Works from any notebook folder.
def load_pairs(dataset, split, release=OVERTURE_RELEASE):
    df = pd.read_parquet(_path(dataset, split, release))
    return df.drop(columns=["label"]), df["label"]


# Split each address into house number + street remainder, for the similarity features.
# Number token = starts with a digit, runs to the first space (12, 12A, 16-18, 20-A).
# extract and replace use the same token so the number never stays on the street.
# No number -> NaN in number_* -> number_missing (one side) / both_number_missing (both sides).
# Returns a copy; the pairs frame itself is left unchanged.
def add_number_street(df):
    df = df.copy()
    for side in "ab":
        df[f"number_{side}"] = df[f"address_{side}"].str.extract(r"^(\d[A-Z0-9a-z-]*)\s", expand=False)
        df[f"street_{side}"] = df[f"address_{side}"].str.replace(r"^(\d[A-Z0-9a-z-]*\s+)", "", regex=True)
    df["both_number_missing"] = df["number_a"].isna() & df["number_b"].isna()
    df["number_missing"] = (df["number_a"].isna() | df["number_b"].isna()) & ~df["both_number_missing"]
    return df
