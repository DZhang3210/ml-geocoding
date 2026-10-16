import os

import pandas as pd
import jellyfish
import textdistance

from helpers.pairs import CACHE_KEY, check_cache, pair_fingerprint

# Features for [[df, cache_path], ...] (train, val, test).
# If the cache file exists it is loaded, after checking it was built from the same pair rows
# (helpers.pairs.check_cache). Code changes aren't detected: delete the caches whenever the
# parsing or _evaluation_function changes, otherwise stale features are silently reused.
# Never pass a sample (e.g. .sample(2000)) through here: it would be cached as the full split.

def _evaluation_function(s):
    """One pair-row -> one row of similarity features.

    Expects columns street_a/b, number_a/b (house number), zip_a/b,
    number_missing, both_number_missing -- i.e. a pairs frame passed through
    helpers.pairs.add_number_street.
    House-number features (the 7 *_address / *_address_sim columns):
      - exactly one side has no number -> -1 (outside the 0-1 range of real similarities)
        and number_missing = 1;
      - both sides have no number (e.g. "I-95 SOUTHBOUND") -> 1, i.e. treated as agreeing
        numbers, and number_missing = 0 (2026-10-13: previously lumped with one-missing,
        which made every no-number identical positive a predicted non-match).
    number_missing (0/1) is also returned as a feature.
    """
    # --- edit-based: Jaro-Winkler, normalised Levenshtein, Hamming (ZIP only, fixed length 5) ---
    jw_street_sim = jellyfish.jaro_winkler_similarity(s["street_a"], s["street_b"])
    jw_zipcode_sim =  jellyfish.jaro_winkler_similarity(s["zip_a"], s["zip_b"])
    jw_address_sim =  (
        -1 if s["number_missing"] else
        1 if s["both_number_missing"] else 
        jellyfish.jaro_winkler_similarity(str(s["number_a"]), str(s["number_b"])))
    lv_street_sim = 1 - jellyfish.levenshtein_distance(s["street_a"], s["street_b"])/(max(len(s["street_a"]), len(s["street_b"])))
    lv_zipcode_sim = 1 - jellyfish.levenshtein_distance(s["zip_a"], s["zip_b"])/max((len(s["zip_a"]), len(s["zip_b"])))

    lv_address_sim = (-1 if s["number_missing"] else 
        1 if s["both_number_missing"] else
        1 - jellyfish.levenshtein_distance(str(s["number_a"]), str(s["number_b"]))/max(len(str(s["number_a"])), len(str(s["number_b"]))))
    hm_zipcode_sim = 1 - jellyfish.hamming_distance(s["zip_a"], s["zip_b"])/5

    # --- token-based: (street/ZIP metric = character bigrams, house-number metric = single characters) ---
    metrics = {
        "jaccard": (textdistance.Jaccard(qval = 2), textdistance.Jaccard(qval = 1)),
        "sorenson":(textdistance.Sorensen(qval = 2), textdistance.Sorensen(qval = 1)),
        "overlap": (textdistance.Overlap(qval=2), textdistance.Overlap(qval=1)),
        "tversky": (textdistance.Tversky(qval=2), textdistance.Tversky(qval=1)),
        "cosine": (textdistance.Cosine(qval = 2),textdistance.Cosine(qval = 1) )
    }

    n_gram_results = {}  # e.g. jaccard_street, jaccard_zip, jaccard_address, ...
    # Street guard: a string under 2 chars has no bigrams (Overlap divides by zero) -> 0.
    # Only hit by the 18 test addresses that are literally "1".
    for option, (initial, for_zip) in metrics.items():
        street_score = initial.normalized_similarity(s["street_a"], s["street_b"]) if (len(s["street_a"])>=2) and (len(s["street_b"]) >= 2) else 0
        zip_score = initial.normalized_similarity(s["zip_a"], s["zip_b"])
        address_score = (
            -1 if s["number_missing"] else
            1 if s["both_number_missing"] else 
            for_zip.normalized_similarity(str(s["number_a"]), str(s["number_b"])))
        n_gram_results[f"{option}_street"] = street_score
        n_gram_results[f"{option}_zip"] = zip_score
        n_gram_results[f"{option}_address"] = address_score

    return pd.Series(
        {
            **n_gram_results,
            "jw_street_sim": jw_street_sim,
            "jw_zipcode_sim": jw_zipcode_sim,
            "jw_address_sim": jw_address_sim,
            "lv_street_sim": lv_street_sim,
            "lv_zipcode_sim": lv_zipcode_sim,
            "lv_address_sim": lv_address_sim,
            "hm_zipcode_sim": hm_zipcode_sim,
            "number_missing": s["number_missing"],
        })

# Row-wise over a whole pairs frame (~0.3 ms/row -> ~1.5 min for all Overture pairs)
def _process_df(df):
    return pd.DataFrame([_evaluation_function(row) for _,row in df.iterrows()])

def get_processed_dfs(parquet_eval):
    ret = []
    for df,filepath in parquet_eval:
        if os.path.exists(filepath):
            curr = check_cache(pd.read_parquet(filepath), df, filepath)
        else:
            curr = _process_df(df)
            curr.assign(**{CACHE_KEY: pair_fingerprint(df)}).to_parquet(filepath, index = False)
        ret.append(curr)
    if len(ret) != 3:
        raise IndexError("Did you modify parquet eval?")
    return ret