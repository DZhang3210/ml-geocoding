# Stamford pairs -> the shared pair schema used by Overture (run from this folder, after 02).
#
#   Data/raw/{train,val,test}.parquet  (02's output, kept as-is: its corruption isn't seeded, so it can't be regenerated)
#   -> Data/stamford_pairs_{train,val,test}.parquet with columns
#      id_a, id_b, address_a, address_b, zip_a, zip_b, split, label, neg_type
#
# address = "number directional name suffix" -- the same street string the Step 1b features
# have always used (EXT is still dropped, as in 02's parsing).
#
# neg_type is derived from the CLEAN source rows (Data/output.csv, row number = id), because
# side b is corrupted and 02 never stored which generator a negative came from:
#   case0  label 1
#   case1  same street, different number
#   case2  same number, different street
#   case3  anything else (easy / random)
# Rule-based, so a random easy pair that happens to sit on the same street lands in case1.
# A negative whose clean rows have the same street AND number would be case3 here -- it is
# counted and printed below.

import pandas as pd

SPLITS = ["train", "val", "test"]

clean = pd.read_csv("Data/output.csv", dtype={"ZipCode": str, "Address": str})
clean_street = clean["StreetName"]
clean_number = clean["Address"]


def _street(df, side):
    parts = df[f"directional_{side}"].fillna("") + " " + df[f"StreetName_{side}"] + " " + df[f"street_extension_{side}"].fillna("")
    return parts.str.split().str.join(" ")   # collapses the double space left by a missing directional


def _neg_type(df):
    street_a, street_b = clean_street.loc[df["id_a"]].values, clean_street.loc[df["id_b"]].values
    number_a, number_b = clean_number.loc[df["id_a"]].values, clean_number.loc[df["id_b"]].values
    same_street, same_number = street_a == street_b, number_a == number_b

    neg_type = pd.Series("case3", index=df.index)
    neg_type[same_street & ~same_number] = "case1"
    neg_type[~same_street & same_number] = "case2"
    neg_type[df["label"] == 1] = "case0"

    same_place_negatives = ((df["label"] == 0) & same_street & same_number).sum()
    return neg_type, same_place_negatives


for split in SPLITS:
    raw = pd.read_parquet(f"Data/raw/{split}.parquet")
    out = pd.DataFrame({
        "id_a": raw["id_a"].astype(str),
        "id_b": raw["id_b"].astype(str),
        "address_a": raw["Address_a"].astype(str) + " " + _street(raw, "a"),
        "address_b": raw["Address_b"].astype(str) + " " + _street(raw, "b"),
        "zip_a": raw["ZipCode_a"],
        "zip_b": raw["ZipCode_b"],
        "split": split,
        "label": raw["label"],
    })
    out["neg_type"], same_place = _neg_type(raw)
    out.to_parquet(f"Data/stamford_pairs_{split}.parquet", index=False)

    print(split, len(out), out["neg_type"].value_counts().to_dict(), f"negatives with same clean street+number: {same_place}")
