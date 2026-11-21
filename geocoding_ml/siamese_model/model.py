"""Step 3 siamese bi-encoder, all of it in one place: address string -> match logit.

The single copy of every Step 3 component (reorganised 2026-11-18 from vocab_step.py, encode_batch_step.py,
data_plumbing.py and encoder_step.py, which held the same code, unchanged; the functions and layers are David's).
Layers keep the same attribute names, so every checkpoint in checkpoints/ and checkpoints_case2s/ loads unchanged.

Reading order = data flow:
  1. vocab             normalize, build_vocab: string -> lowercase characters -> id table (train split only)
  2. encode_batch      strings -> padded ids (B, T) + lengths (B,)
  3. data plumbing     PairDataset (one pair as raw strings + label), make_collate (batch -> tensors)
  4. layers            CharEmbedder -> BiLSTM -> MaskedPooling
  5. models            Encoder (one string -> one vector), SiameseMatcher (pair -> one logit)

Training and reading saved runs: training.py. Tests: 01_siamese_build.ipynb.
"""
import collections
import re

import torch
from torch import nn
from torch.utils.data import Dataset


# ---------- 1. vocab ----------

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


# ---------- 2. encode_batch ----------

MAX_BATCH_FEATURE_SIZE = DEFAULT_MAX_STR_LENGTH   # 85, same cut as normalize


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


# ---------- 3. data plumbing ----------
# PairDataset[i] -> one pair as raw strings + label; make_collate(vocab) -> the DataLoader's collate_fn, which
# encodes each side per batch (padding only to that batch's longest string).

class PairDataset(Dataset):
    def __init__(self, df, labels):
        # plain lists: list indexing is fast, df.iloc[i] would be called 200k times per epoch
        self.address_a_lookup = df["address_a"].tolist()
        self.address_b_lookup = df["address_b"].tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        address_a = self.address_a_lookup[i]
        address_b = self.address_b_lookup[i]
        label = self.labels[i]
        return address_a, address_b, label      # e.g. ("12 MAIN ST", "12 MAIN STREET", 1)


def make_collate(vocab):
    # returns a closure: DataLoader calls collate(samples) with one argument, so vocab is baked in here
    def collate(samples):
        # samples = [("12 main st", "12 main street", 1), ("4 elm ave", "9 oak rd", 0), ...]
        addr_a, addr_b, labels = zip(*samples)                   # unzip into three tuples
        ids_a, len_a = encode_batch(list(addr_a), vocab)         # (B, T_a), (B,)
        ids_b, len_b = encode_batch(list(addr_b), vocab)         # (B, T_b), (B,)
        y = torch.tensor(labels, dtype=torch.float32)            # (B,) float, as BCEWithLogitsLoss wants
        return ids_a, len_a, ids_b, len_b, y
    return collate


# ---------- 4. layers ----------

class CharEmbedder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)   # pad row stays zero and never trains

    def forward(self, ids):
        return self.emb(ids)   # (B, T) -> (B, T, emb_dim)


class BiLSTM(nn.Module):
    def __init__(self, input_size=32, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        # pack so the LSTM never sees pads (otherwise the backward direction reads them first)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)   # h_n / c_n unused: we pool over every position instead
        # total_length keeps h as wide as the input, so it lines up with the pad mask
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return h   # (B, T, input_size) -> (B, T, 2 * hidden); pad positions are 0


class MaskedPooling(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        if mode not in ("mean", "max"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode   # no layers -> no parameters

    def forward(self, h, lengths):
        lengths = lengths.to(h.device)
        mask = torch.arange(h.size(1), device=h.device)[None, :] < lengths[:, None]   # (B, T), True = real position
        m = mask.unsqueeze(-1)                                                         # (B, T, 1), broadcasts over features
        if self.mode == "mean":
            return (h * m).sum(1) / m.sum(1)                                   # pads add 0 and don't count: (B, T, D) -> (B, D)
        return h.masked_fill(~m, float("-inf")).max(dim=1).values              # pads can never win: (B, T, D) -> (B, D)


# ---------- 5. models ----------

class Encoder(nn.Module):
    """One address string (as ids) -> one vector. Shared by both sides of the siamese model."""

    def __init__(self, vocab_size, emb_dim=32, hidden=128, pooling="mean"):
        super().__init__()
        self.embedder = CharEmbedder(vocab_size, emb_dim)
        self.bilstm = BiLSTM(emb_dim, hidden)
        self.pool = MaskedPooling(pooling)
        self.output_dim = 2 * hidden

    def forward(self, ids, lengths):
        x = self.embedder(ids)          # (B, T) -> (B, T, emb_dim)
        h = self.bilstm(x, lengths)     # -> (B, T, 2 * hidden)
        return self.pool(h, lengths)    # -> (B, 2 * hidden)


class SiameseMatcher(nn.Module):
    """Pair of address ids -> one match logit. ONE encoder, called on both sides (weight sharing)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(2 * encoder.output_dim, 1)

    def forward(self, ids_a, len_a, ids_b, len_b):
        a_out = self.encoder(ids_a, len_a)                  # (B, T_a) -> (B, enc_dim)
        b_out = self.encoder(ids_b, len_b)                  # same encoder: (B, T_b) -> (B, enc_dim)
        abs_diff = (a_out - b_out).abs()
        multiple = a_out * b_out
        combined = torch.cat([abs_diff, multiple], dim=1)   # both order-free -> score(a, b) == score(b, a); (B, 2 * enc_dim)
        res = self.head(combined)                           # (B, 1)
        return res.squeeze(-1)                              # logits (B,); no sigmoid: BCEWithLogitsLoss applies it
