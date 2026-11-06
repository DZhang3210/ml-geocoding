import collections
import re

DEFAULT_MAX_STR_LENGTH = 85
MIN_CHR_FREQUENCY = 20
ADDRESS_COLUMN_NAMES = ["address_a", "address_b"]
PAD_ID = 0
UNK_ID = 1


def normalize(s, max_length=DEFAULT_MAX_STR_LENGTH):
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace (incl. \n), drop the ends
    s = s.lower()
    s = s[:max_length]                  # truncate after stripping, so the cut never lands on padding spaces
    return s


def build_vocab(df_train, min_count=MIN_CHR_FREQUENCY):
    vocab_dct = collections.Counter()
    for col_name in ADDRESS_COLUMN_NAMES:
        vocab_dct.update("".join(df_train[col_name].map(normalize)))
    # Sort by count (descending), ties broken by the character itself -> ids depend only on the counts
    vocab_dct = dict(sorted(vocab_dct.items(), key=lambda kv: (-kv[1], kv[0])))
    # remove items not satisfying min_count
    vocab_dct = {k: v for k, v in vocab_dct.items() if v >= min_count}

    chr_to_id = {"<pad>": PAD_ID, "<unk>": UNK_ID}

    for i, char in enumerate(vocab_dct.keys()):
        chr_to_id[char] = i + 2

    return chr_to_id
