import numpy as np
import pandas as pd
from postal.parser import parse_address
from postal.dedupe import is_street_duplicate, is_house_number_duplicate, duplicate_status
from typing import Literal

LANGUAGES = ["en"]


COMPARE_ADDR_ATTR = [["house_number", is_house_number_duplicate],
                     ["road", is_street_duplicate],
                     ]

def _compare_parsed_addrs(addr1, addr2, zip1, zip2, 
                          road_rule: Literal["strict", "loose"] = "loose", 
                          zip_rule: Literal["strict", "off"] = "strict",
                          number_rule: Literal["strict", "loose"] = "strict"
                          ):
    for item, func in COMPARE_ADDR_ATTR:
        one, two = addr1.get(item), addr2.get(item)
        if item == "house_number":
            #If one side is missing and "strict" -> bad 
            if number_rule == "strict":
                if not one or not two:
                    if one != two: return False
                elif func(one, two, languages=LANGUAGES) != duplicate_status.EXACT_DUPLICATE: return False
            elif (one and two) and func(one, two, languages=LANGUAGES) == duplicate_status.NON_DUPLICATE: #If both available check same
                return False
        elif item == "road" and (one and two):
            if road_rule == "loose":
                if func(one, two, languages=LANGUAGES) == duplicate_status.NON_DUPLICATE: return False
            elif road_rule == "strict":
                if func(one, two, languages=LANGUAGES) != duplicate_status.EXACT_DUPLICATE: return False
            else:
                raise ValueError(f"Did you give an invalid road rule? {road_rule}")
    # ZIP comes from the zip columns, not the parser; skip if either is missing
    if zip_rule == "strict":
        if pd.notna(zip1) and pd.notna(zip2) and str(zip1)[:5] != str(zip2)[:5]: 
            return False
    elif zip_rule == "off":
        return True 
    else: 
        raise ValueError(f"Did you give an invalid zip rule? {zip_rule}")
    return True

def _create_parsed_addresses(addr):
    return {cat: val for val, cat in parse_address(addr, language="en", country="us")}

def libpostal_baseline(df, 
                        road_rule: Literal["strict", "loose"] = "loose", 
                        zip_rule: Literal["strict", "off"] = "strict",
                        number_rule: Literal["strict", "loose"] = "strict"
                    ):
    ret = []
    for row in df.itertuples():
        a = _create_parsed_addresses(row.address_a)
        b = _create_parsed_addresses(row.address_b)
        # dedupe functions expand abbreviations internally, so no expand_address here
        ret.append(_compare_parsed_addrs(a, b, row.zip_a, row.zip_b, road_rule, zip_rule, number_rule))
    # int (not bool) to match the old array; index=df.index keeps rows alignable with df
    return pd.DataFrame(np.array(ret, dtype=np.int64), columns=["pred"], index=df.index)
