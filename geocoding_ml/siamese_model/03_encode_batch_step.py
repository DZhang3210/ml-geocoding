from importlib import import_module

import torch

_vocab_step = import_module("siamese_model.01_vocab_step")   # digit-prefixed filename -> can't use `from ... import`
normalize = _vocab_step.normalize
PAD_ID, UNK_ID = _vocab_step.PAD_ID, _vocab_step.UNK_ID

MAX_BATCH_FEATURE_SIZE = _vocab_step.DEFAULT_MAX_STR_LENGTH   # 85, same cut as normalize


def encode_batch(strings, vocab, max_len=MAX_BATCH_FEATURE_SIZE):
    strings = [normalize(s, max_len) for s in strings]   # same normalisation the vocab was built with (lowercase, truncate)
    batch_len = max(len(string) for string in strings)
    ret = []
    for string in strings:
        res = [vocab.get(c, UNK_ID) for c in string]     # unseen characters -> unk, never pad
        res += [PAD_ID] * (batch_len - len(res))
        ret.append(res)
    lengths = [len(string) for string in strings]
    return torch.tensor(ret, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
