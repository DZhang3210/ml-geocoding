# Dataset Construction Walkthrough — Step 2

Code-block-by-code-block notes on `stamford_analysis.ipynb`, in notebook order, with the reasoning behind each decision.

---

## Cell 1 — load + assign id

```python
import os
import pandas as pd
import numpy as np
import re
import random
import itertools
import jellyfish
from collections import defaultdict

csv = pd.read_csv("./output.csv", dtype = {"ZipCode": str})
csv.insert(0, "id",[i for i in range(len(csv))])

csv.head()
```

Needed one identifier that never changes, since pandas' actual index shifts on sort/filter/concat. `id` is the fixed reference that lets a corrupted row trace back to its exact clean source later.

---

## Cell 2 — suffix frequency

```python
suffix_df = csv.copy()
suffix_df["suffix"] = suffix_df["StreetName"].str.split().str[-1]
suffix_frequencies = suffix_df["suffix"].value_counts()
print(suffix_frequencies.keys())
suffix_df["suffix_freq"] = suffix_df["suffix"].map(suffix_frequencies)
suffix_df = suffix_df.drop_duplicates(subset = ["suffix"])
suffix_df = suffix_df[suffix_df["suffix"].isin(["N","S","E","W","NE","NW","SE","SW"])]
suffix_df = suffix_df.sort_values(by = "suffix_freq", ascending= False)
suffix_df
```

Didn't know upfront if directionals were a prefix or suffix, so checked both — this pass, then a separate prefix pass next. Directional letters (`S`/`W`/`E`/`N`) showed up very often here, which is what prompted checking suffix first.

---

## Cell 3 — prefix frequency

```python
prefix_df = csv.copy()
prefix_df["prefix"] = prefix_df["StreetName"].str.split().str[0]
prefix_frequencies = prefix_df["prefix"].value_counts()
print(prefix_frequencies.keys())
prefix_df["prefix_freq"] = prefix_df["prefix"].map(prefix_frequencies)
prefix_df = prefix_df.drop_duplicates(subset = ["prefix"])
prefix_df = prefix_df[prefix_df["prefix"].isin(["N","S","E","W","NE","NW","SE","SW"])]
prefix_df = prefix_df.sort_values(by = "prefix_freq", ascending= False)
prefix_df
```

Same check, other end of the string — confirming directionals also show up as a leading token, not just trailing.

---

## Cell 4 — prefix/suffix collision check

```python
## Check for collision where there are any prefixes or suffixes
suffix_and_prefix_df = csv.copy()
suffix_and_prefix_df["suffix"] = suffix_and_prefix_df["StreetName"].str.split().str[-1]
suffix_and_prefix_df["prefix"] = suffix_and_prefix_df["StreetName"].str.split().str[0]
suffix_and_prefix_df = suffix_and_prefix_df[
    suffix_and_prefix_df["suffix"].isin(["N","S","E","W","NE","NW","SE","SW"]) &
    suffix_and_prefix_df["prefix"].isin(["N","S","E","W","NE","NW","SE","SW"])
]

print(suffix_and_prefix_df)
#* Result shows that we don't need to worry about prefix/suffix collision
```

Explicit check for a row having a directional at *both* ends at once — empty result, so no precedence/tie-break rule was needed for extraction.

---

## Cell 5 — the parsing pipeline

```python
new_df = csv.copy()

def get_directional(text):
    valid_dirs = ["N","S","E","W"]
    directional = None
    split_txt = text.split()
    directional =  split_txt[0] if split_txt[0] in valid_dirs else directional
    directional = split_txt[-1] if split_txt[-1] in valid_dirs else directional
    return directional
new_df["directional"] = new_df["StreetName"].apply(get_directional)

def drop_directional(text):
    valid_dirs = ["N","S","E","W"]
    split_txt = text.split()
    split_txt = split_txt[1:] if split_txt[0] in valid_dirs else split_txt
    split_txt = split_txt[:-1] if split_txt[-1] in valid_dirs else split_txt
    return " ".join(split_txt)
new_df["StreetName"] = new_df["StreetName"].apply(drop_directional)


def get_ext(text):
    valid_ext = ["EXT"]
    ext = False
    split_txt = text.split()
    ext =  True if split_txt[0] in valid_ext else ext
    ext = True if split_txt[-1] in valid_ext else ext
    return ext
new_df["is_extensional"] = new_df["StreetName"].apply(get_ext)

def drop_ext(text):
    valid_ext = ["EXT"]
    split_txt = text.split()
    split_txt = split_txt[1:] if split_txt[0] in valid_ext else split_txt
    split_txt = split_txt[:-1] if split_txt[-1] in valid_ext else split_txt
    return " ".join(split_txt)
new_df["StreetName"] = new_df["StreetName"].apply(drop_ext)

known_suffixes = ["RD", "ST", "AVE", "DR", "LN", "PL", "CT", "CIR", "TER", "TRL", "BLVD", "WAY", "QUAY", "PARK", "TPKE", "HOLW", "GRV", "DOCK", "PATH", "RUN", "LNDG", "PLZ", "WALK", "PT"]
new_df["street_extension"] = new_df["StreetName"].apply(
    lambda x: x.split()[-1] if x.split()[-1] in known_suffixes else None
)
new_df["StreetName"] = new_df["StreetName"].apply(
    lambda x: " ".join(x.split()[:-1] if x.split()[-1] in known_suffixes else x.split())
) 

new_df
```

A pipeline, not independent extractions — directional stripped first, then EXT, then street-type suffix, each stage working off the already-stripped output of the last. `known_suffixes` — no confirmed external source; checking back, it matches your own suffix-frequency output exactly, so it looks dataset-derived rather than sourced from a document. The USPS-sourced table is a *separate* list (next cell) — link not confirmed, just "Pub 28, Appendix C1, pe.usps.com" from earlier notes.

---

## Cell 6 — USPS abbreviation lookup

```python
from pprint import pprint

suffix_abbrev = pd.read_csv("../usps_pub28_street_suffix_abbreviations.csv")

sfx_abrev_lookup = suffix_abbrev.groupby("standard_abbreviation")["commonly_used"].apply(list).to_dict()

for key, variants in sfx_abrev_lookup.items():
    index = variants.index(key) if key in variants else None
    if index is not  None:
        sfx_abrev_lookup[key] = sfx_abrev_lookup[key][:index] + sfx_abrev_lookup[key][index+1:]
sfx_abrev_lookup = {
    key:variants for key, variants in sfx_abrev_lookup.items() if len(variants) > 0
}

directional_lookup = {
    "N": ["NORTH"],
    "S": ["SOUTH"],
    "E": ["EAST"],
    "W": ["WEST"],
}
```

Filtered out any standard abbreviation that also appeared as its own "commonly used" variant, so a corruption swap is guaranteed to always actually change something.

---

## Cell 7 — character-level corruption

```python
def _generate_mask(street_name, pct_flip = 0.1):
    n = len(street_name)
    mask = [1 if random.random() <= pct_flip else 0 for _ in range(n)]
    while sum(mask) > n//2:
       mask = [1 if random.random() <= pct_flip else 0 for _ in range(n)] 

    #1 = Removal
    #2 = Replacement
    for i, res in enumerate(mask):
        if res == 1:
            if random.random() > 0.5: mask[i] = 2
    return mask

def update_string(street_name, pct2):
    new_street_name = ""
    for i,op in enumerate(_generate_mask(street_name, pct2)):
        c = street_name[i]
        if op == 0:
            new_street_name += street_name[i]
        elif op == 1:
            pass
        elif op == 2:
            if ord('A') <= ord(c) <= ord('Z'):
                new_street_name += random.choice([chr(c) for c in range(ord('A'), ord('Z') + 1)])
            if ord('a') <= ord(c) <= ord('z'):
                new_street_name += random.choice([chr(c) for c in range(ord('a'), ord('z') + 1)])
    return new_street_name
```

Capped total edits at ~half the string — past that it stops reading as a noisy version of the same word and becomes gibberish. Side benefit: half always survives untouched, so a short word can never get wiped to nothing. Generated the whole mask at once and rerolled if over-budget, rather than scanning and stopping early, so no character position gets biased toward corruption over another.

---

## Cell 8 — `_corrupt_row` / `corrupt_rows`

```python
def _corrupt_row(row, pct1=0.7, pct2=0.1, pct3=0.7):

    new_row = row.copy()
    for row_name, lookup in zip(["street_extension", "directional"], [sfx_abrev_lookup, directional_lookup]):
        if row[row_name] in lookup and random.random() < pct1:
            choices = lookup[row[row_name]]
            new_row[row_name] = random.choice(choices)
    new_row["StreetName"] = update_string(row["StreetName"], pct2)
    return new_row

def corrupt_rows(df):
    corrupted_rows = [_corrupt_row(row) for _, row in df.iterrows()]
    new_df = pd.DataFrame(corrupted_rows)
    new_df = new_df.rename(columns = {"id": "original_id"})
    return new_df

corrupted_df = corrupt_rows(new_df)

corrupted_df.head()
```

Each row independently rolls three things: swap `street_extension`, swap `directional`, corrupt `StreetName`'s characters. Renaming `id`→`original_id` is what lets a corrupted row be joined back to its clean source later when building positive pairs.

---

## Cell 9 — hard-negative candidates

```python
#Compiling negative cases

NUM_NEGATIVE = 4

jw_lookup = {}
debug_jw = []

for address, group_df in new_df.groupby("Address"):
    for row1,row2 in itertools.combinations(group_df.to_dict('records'), 2):
        a,b, aid, bid = row1["StreetName"], row2["StreetName"], row1["id"], row2["id"]    
        score = 0
        debug_jw_tmp = [aid, bid]
        def _add_score(amt, condition = True, default = 0):
             global score
             score += amt if condition else default
             debug_jw_tmp.append(amt if condition else default)
        _add_score(jellyfish.jaro_winkler_similarity(a, b))
        _add_score(0.05) #Due to same Address
        _add_score(0.05, row1["directional"] != None and row1["directional"] == row2["directional"]) 
        _add_score(0.05, row1["street_extension"] != None and row1["street_extension"] == row2["street_extension"])
        _add_score(0.05, row1["is_extensional"] == True and row1["is_extensional"] == row2["is_extensional"]) 
        jw_lookup[(min(aid, bid), max(aid, bid))] = score

        debug_jw_tmp.append(sum(debug_jw_tmp[2:]))
        debug_jw.append(debug_jw_tmp)        

def all_scores_above(thresh = 0.7):
    all_scores = [[res, a, b] for (a,b), res in jw_lookup.items()]
    return [[a,b] for res,a,b in all_scores if res >= thresh]

train_df_lst = []
for (StreetName, street_extension, directional), lst in new_df.groupby(["StreetName", "street_extension", "directional"], dropna = False)["id"].apply(list).to_dict().items():
    if len(lst) < 2: continue
    train_df_lst.extend(
        [[min(a, b),max(a,b)] for a,b in itertools.combinations(random.sample(lst, min(len(lst), NUM_NEGATIVE)), 2)]
    )
train_df_lst.extend(all_scores_above(0.85))
print("length w/ dupes", len(train_df_lst))
train_df_lst = list(set(map(tuple, train_df_lst)))
print("length w/o dupes", len(train_df_lst))
```

Three categories exist, named plainly: **positive** (same address, clean vs. corrupted), **easy negative** (two random unrelated addresses), **hard negative** (two different but deliberately confusable addresses), found two ways in this cell.

The `train_df_lst.extend([...for a,b in itertools.combinations(...)])` block is **method 1** — exact match on street/extension/directional: if two locations agree on all three, add every pairwise permutation. Capped per group (`NUM_NEGATIVE`) because permutations are `O(k²)` — a few oversized groups would otherwise dominate and blow up the dataset.

The `jw_lookup`/`all_scores_above` block is **method 2** — holistic similarity search, needed to surface harder cases than exact matching can find. Blocked on `Address` for two reasons: house number looked like an easy shortcut a model could lean on, so forcing same-house-number pairs makes the model discriminate on street name instead; and full pairwise comparison is `O(N²)`, while blocking within address groups drops it to a laptop-feasible `O(k²)`. Base score is Jaro-Winkler, plus bonuses for matching directional/extension/is_extensional — worth noting the character-corruption function and Jaro-Winkler aren't fully independent, they're sensitive to similar kinds of edits, which isn't ideal.

ids stored sorted (`min`/`max`) specifically so `set(map(tuple, ...))` dedupes correctly — unsorted, `(3,7)` and `(7,3)` are different tuples and wouldn't collapse.

---

## Cell 10 — diagnostics

```python
#* Checking Each Factor on Address Acceptance Rate
num_pass_normal = sum(
    1 if total_score > 0.85 else 0 for aid, bid, jaro_score, address_score, dir_score, str_score, is_ext_score, total_score in debug_jw
)/len(debug_jw) * 100
print(f"{num_pass_normal=:.3f}")

num_pass_wo_address = sum(
    1 if total_score-address_score > 0.85 else 0 for aid, bid, jaro_score, address_score, dir_score, str_score, is_ext_score, total_score in debug_jw
)/len(debug_jw) * 100
print(f"{num_pass_wo_address=:.3f}")

num_pass_wo_dir = sum(
    1 if total_score-dir_score > 0.85 else 0 for aid, bid, jaro_score, address_score, dir_score, str_score, is_ext_score, total_score in debug_jw
)/len(debug_jw) * 100
print(f"{num_pass_wo_dir=:.3f}")

num_pass_wo_str = sum(
    1 if total_score-str_score > 0.85 else 0 for aid, bid, jaro_score, address_score, dir_score, str_score, is_ext_score, total_score in debug_jw
)/len(debug_jw) * 100
print(f"{num_pass_wo_str=:.3f}")

num_pass_wo_ext = sum(
    1 if total_score-is_ext_score > 0.85 else 0 for aid, bid, jaro_score, address_score, dir_score, str_score, is_ext_score, total_score in debug_jw
)/len(debug_jw) * 100
print(f"{num_pass_wo_ext=:.3f}")

#* Matplot graph, looking for any extreme outliers in the candidates
# import matplotlib.pyplot as plt
# lst = [[address, len(group_df)] for address, group_df in new_df.groupby("Address")]
# lst.sort(key = lambda x: [x[1], x[0]], reverse=True)
# print(lst[:5])

# selected_addresses_list = [new_df[new_df["id"] == a].iloc[0]["Address"] for a,b in all_scores_above(0.85)]
# values, counts = np.unique(selected_addresses_list, return_counts=True)
# freq_list = [[c, v] for v,c in zip(values, counts)]
# freq_list.sort(reverse = True)
# print(freq_list[:5])
# print(sum(c for c, v in freq_list[:10]) / 15435) #0.04742468415937804
# plt.hist(counts, bins = 100)
# plt.yscale("log")
# plt.xlabel("value")
# plt.title("")
# plt.show()

#* Sanity check looking at some of the matches
# for _ in range(4):
#     a, b = random.choice(train_df_lst)
#     row1 = new_df[new_df["id"] == a].iloc[0]
#     row2 = new_df[new_df["id"] == b].iloc[0]
#     tmp_df = pd.DataFrame([row1, row2])
#     display(tmp_df.head())
```

Robustness checks on method 2's composite score: the `num_pass_wo_*` block ablates each bonus one at a time, checking how much the 0.85 pass rate shifts. The commented block below it is a concentration check (top-10 addresses ≈ 4.7% of matches — not dominated) plus an eye test, manually viewing sampled pairs to confirm they look like real mix-ups. On whether the concentration check should've also matched on `b`: no — method 2 blocks on `Address`, so `a` and `b` always share the same `Address` by construction; checking `b` would just repeat the same numbers.

---

## Cell 11 — the three generator functions

```python
def _generate_positive_dataset(amount = 10000):
    indexes = random.sample([i for i in range(len(new_df))], k=amount)
    good = new_df.iloc[indexes]
    bad = corrupted_df.iloc[indexes]
    combined = pd.merge(good, bad,
        left_index = True,
        right_index = True,
        suffixes = ("_a", "_b")
    )
    combined.rename(columns = {"id": "id_a", "original_id":"id_b"}, inplace = True)
    combined.drop(columns = ["X_COORD_a","Y_COORD_a", "X_COORD_b", "Y_COORD_b"], inplace = True)
    combined["label"] = 1

    return combined

def _generate_easy_negative_dataset(amount = 10000):
    good = new_df.iloc[
        random.choices([i for i in range(len(new_df))], k = amount)
    ].reset_index(drop = True)
    bad = corrupted_df.iloc[
        random.choices([i for i in range(len(corrupted_df))], k = amount)
    ].reset_index(drop = True)
    dataset = pd.merge(good, bad, left_index = True, right_index = True,  suffixes=("_a", "_b"))
    dataset.drop(columns = ["X_COORD_a","Y_COORD_a","X_COORD_b","Y_COORD_b"], inplace = True)
    dataset.rename(columns = {"id":"id_a", "original_id":"id_b"}, inplace = True)
    dataset["label"] = 0
    return dataset

def _generate_hard_negative_dataset(amount = 10000):
    chosen_neg_data = [random.sample(_, k = len(_)) for _ in random.sample(train_df_lst, k = amount)]
    clean_ids = [x[0] for x in chosen_neg_data]
    dirty_ids = [x[1] for x in chosen_neg_data]
    good = new_df.set_index("id").loc[clean_ids].reset_index()
    bad = corrupted_df.set_index("original_id").loc[dirty_ids].reset_index()
    dataset = pd.merge(good, bad, left_index = True, right_index = True, suffixes = ("_a", "_b"))
    dataset.drop(columns = ["X_COORD_a","Y_COORD_a","X_COORD_b","Y_COORD_b"], inplace = True)
    dataset.rename(columns = {"id":"id_a", "original_id":"id_b"}, inplace = True)
    dataset["label"] = 0

    return dataset

def _generate_training_data():
    positive_dataset = _generate_positive_dataset(14818)
    easy_neg_dataset = _generate_easy_negative_dataset(14818)
    hard_neg_dataset = _generate_hard_negative_dataset(14818)
    dataset = pd.concat([positive_dataset, easy_neg_dataset, hard_neg_dataset])
    dataset = dataset.sample(frac = 1).reset_index(drop = True)

    return dataset

dataset = _generate_training_data()
dataset
```

**Positive**: `random.sample` (no duplicates) — every draw should be a different address, maximizing coverage rather than repeating the same pair. No `reset_index` — you *want* the merge to match on id here, since that's literally what a positive pair is.

**Easy negative**: kept `random.choices` (duplicates fine, even good — repeated exposure with different random partners makes the model more robust). Needed `.reset_index(drop=True)`, since without it the merge would try to match on the old id-based index carried over from `.iloc[]`, not your actual random draw.

**Hard negative**: `random.sample` on the outer draw over `train_df_lst` — pulling from an already-curated pool of hard pairs, want full coverage, not duplicates.

**`_generate_training_data`**: shuffled after concatenating, since otherwise it's three solid blocks (all positives, then all easy, then all hard) back to back, not actually mixed. `reset_index(drop=True)` because this is a new combined table — real traceability lives in `id_a`/`id_b`, not the DataFrame's own index.

---

## Cell 12 — split + persist

```python
from sklearn.model_selection import train_test_split
train_ids, test_ids = train_test_split(range(len(new_df)), test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

train_dataset = dataset[dataset["id_a"].isin(train_ids) & dataset["id_b"].isin(train_ids)]
val_dataset = dataset[dataset["id_a"].isin(val_ids) & dataset["id_b"].isin(val_ids)] 
test_dataset = dataset[dataset["id_a"].isin(test_ids) & dataset["id_b"].isin(test_ids)]  

print(len(train_dataset), len(val_dataset), len(test_dataset))

train_dataset.to_parquet("./train.parquet", index = False)
val_dataset.to_parquet("./val.parquet", index = False)
test_dataset.to_parquet("./test.parquet", index = False)
```

Core risk: any test data leaking into training inflates scores artificially. Prevented by assigning every address id to exactly one split first, then keeping only rows where *both* `id_a` and `id_b` fall in the same split — full isolation. Saved to Parquet via `pyarrow` for built-in dtype handling, avoiding CSV's loose-format conversion quirks. `index=False` since the row index isn't worth persisting.
