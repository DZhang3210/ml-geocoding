"""ESIM, all of it in one place: address string -> match logit.

The single copy of every ESIM component (reorganised 2026-11-14 from tokenizer.py + esim_model.py, which held
the same code; the layers are David's from 02_esim.ipynb, with the table and sizes as arguments so one class
serves all five embedding settings). Layers are created in the same order and under the same attribute names
as the saved checkpoints, so every run in checkpoints/esim_*/ loads unchanged.

Reading order = data flow:
  1. tokenize                 string -> words
  2. embedding tables         word -> row id -> frozen vector (one builder per setting)
  3. batches                  strings -> padded ids + lengths; DataLoaders
  4. layers                   embedding -> context BiLSTM -> align -> enhance -> composition BiLSTM -> pool -> head
  5. ESIM                     the assembled model

Training and reading saved runs: training.py. Tests and results: 02_esim.ipynb.
"""
import os
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from gensim.models import FastText, Word2Vec

from siamese_model.model import PairDataset
from siamese_model.model import BiLSTM

PAD_ID, UNK_ID = 0, 1
MAX_LEN = 85                         # tokens kept per address (02's MAX_BATCH_FEATURE_SIZE)
EMB_DIM, HIDDEN, DROPOUT = 200, 200, 0.3
BATCH_SIZE = 64
CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")   # esim/checkpoints, from any folder


# ---------- 1. tokenize ----------

def tokenize(s):
    # the tokenisation word2vec was trained with (01_word2vec.ipynb); ESIM must use the same one
    return re.findall(r"[A-Za-z]+|\d+", s.upper())


# ---------- 2. embedding tables: setting -> (vocab, matrix) ----------
# vocab: token -> row id; matrix: (rows, emb_dim) float tensor, rows 0 / 1 = PAD / UNK (zeros).

def _load(kind, min_count):
    path = os.path.join(CHECKPOINTS, f"{kind}_50_mc{min_count}.model")
    return (FastText if kind == "fasttext" else Word2Vec).load(path)


def _table(keys, vectors):
    vocab = {"PAD": PAD_ID, "UNK": UNK_ID, **{k: i + 2 for i, k in enumerate(keys)}}
    matrix = torch.cat([torch.zeros(2, vectors.shape[1]), torch.tensor(vectors, dtype=torch.float32)])
    return vocab, matrix


def plain_table(kind, min_count):
    """One model's vectors (settings mc2, mc1, ft_mc1): vocab id = word index + 2."""
    wv = _load(kind, min_count).wv
    return _table(wv.index_to_key, wv.vectors)


def random_rows_table(all_addresses, min_count=2, seed=0):
    """Setting rand_mc2: word2vec mc2 rows, then one fixed random row per token word2vec dropped.
    Random rows ~ normal with the word2vec rows' per-dimension mean and std; dropped tokens sorted
    so the same token always gets the same row."""
    wv = _load("word2vec", min_count).wv
    seen = {t for s in all_addresses for t in tokenize(s)}
    dropped = sorted(seen - set(wv.key_to_index))
    rng = np.random.default_rng(seed)
    random_rows = rng.normal(wv.vectors.mean(axis=0), wv.vectors.std(axis=0), size=(len(dropped), wv.vector_size))
    vocab, matrix = _table(wv.index_to_key + dropped, np.vstack([wv.vectors, random_rows]))
    return vocab, matrix, dropped


def stacked_table(min_count=1):
    """Setting stack_mc1: each word's row = [word2vec vector; fastText vector], 400-d, in word2vec's word order."""
    w2v, ft = _load("word2vec", min_count).wv, _load("fasttext", min_count).wv
    assert set(w2v.key_to_index) == set(ft.key_to_index), "word2vec and fastText vocabularies differ"
    keys = w2v.index_to_key
    ft_rows = ft.vectors[[ft.key_to_index[k] for k in keys]]
    return _table(keys, np.hstack([w2v.vectors, ft_rows]))


# ---------- 3. batches ----------

def encode_batch(strings, vocab, max_len=MAX_LEN):
    """Strings -> (ids (B, T) padded with PAD_ID, lengths (B,)); unknown tokens -> UNK_ID."""
    token_lists = [tokenize(s)[:max_len] for s in strings]
    batch_len = max(len(t) for t in token_lists)
    ids = [[vocab.get(w, UNK_ID) for w in t] + [PAD_ID] * (batch_len - len(t)) for t in token_lists]
    lengths = [len(t) for t in token_lists]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def make_collate(vocab):
    def collate(samples):
        # samples = [(address_a, address_b, label), ...] -> encode each side separately
        addr_a, addr_b, labels = zip(*samples)
        ids_a, len_a = encode_batch(list(addr_a), vocab)
        ids_b, len_b = encode_batch(list(addr_b), vocab)
        return ids_a, len_a, ids_b, len_b, torch.tensor(labels, dtype=torch.float32)
    return collate


def make_loaders(vocab, splits):
    """splits: {"train": (df, labels), "val": ..., "test": ...} -> {name: DataLoader}; only train shuffles."""
    return {name: DataLoader(PairDataset(df, labels), batch_size=BATCH_SIZE, shuffle=(name == "train"),
                             collate_fn=make_collate(vocab))
            for name, (df, labels) in splits.items()}


# ---------- 4. layers ----------

class EmbeddingLayer(nn.Module):
    """Frozen table + one trainable UNK vector swapped in where id == UNK_ID."""
    def __init__(self, matrix):
        super().__init__()
        self.frozen = nn.Embedding.from_pretrained(matrix, freeze=True)
        self.unk = nn.Parameter(torch.zeros(matrix.shape[1]))

    def forward(self, ids):
        out = self.frozen(ids)                                         # (B, T, emb_dim)
        return torch.where((ids == UNK_ID).unsqueeze(-1), self.unk, out)


class ContextEncoder(nn.Module):
    """dropout -> BiLSTM (shared by both sides); output 2 * hidden per word."""
    def __init__(self, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bilstm = BiLSTM(input_size=emb_dim, hidden=hidden)

    def forward(self, vectors, lengths):
        return self.bilstm(self.dropout(vectors), lengths)


def attention_weights(A, B, len_B):
    e = A @ B.transpose(1, 2)                                                   # (rows, len_a, len_b)
    mask = torch.arange(B.size(1), device=B.device) < len_B[:, None]           # (rows, len_b)
    return e.masked_fill(~mask[:, None, :], float("-inf")).softmax(dim=-1)


def align(A, B, len_B):
    return attention_weights(A, B, len_B) @ B                                  # (rows, len_a, 2 * hidden)


def enhance(A, len_A, B, len_B):
    """Per word: [a; ã; a − ã; a ⊙ ã], both sides."""
    tilde_A, tilde_B = align(A, B, len_B), align(B, A, len_A)
    return (torch.cat([A, tilde_A, A - tilde_A, A * tilde_A], dim=2),
            torch.cat([B, tilde_B, B - tilde_B, B * tilde_B], dim=2))


class Composition(nn.Module):
    """Composition: projection 8*hidden -> hidden, ReLU, dropout, BiLSTM."""
    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.linear = nn.Linear(4 * 2 * hidden, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.BiLSTM = BiLSTM(hidden, hidden)

    def forward(self, vectors, lengths):
        return self.BiLSTM(self.dropout(self.relu(self.linear(vectors))), lengths)


class MeanMaxPooling(nn.Module):
    """Masked mean + max over real positions: (B, T, D) -> (B, 2D)."""
    def forward(self, h, lengths):
        lengths = lengths.to(h.device)
        m = (torch.arange(h.size(1), device=h.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean_pool = (h * m).sum(1) / m.sum(1)
        max_pool = h.masked_fill(~m, float("-inf")).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)


class ClassifierHead(nn.Module):
    """[v_a; v_b] (8 * hidden) -> MLP (hidden, tanh, dropout) -> 1 logit."""
    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.linear1 = nn.Linear(hidden * 4 * 2, hidden)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, 1)

    def forward(self, vectors):
        return self.linear2(self.dropout(self.tanh(self.linear1(vectors)))).squeeze(-1)


# ---------- 5. the assembled model ----------

class ESIM(nn.Module):
    # attribute names are the state_dict keys of the saved checkpoints: renaming one breaks load_state_dict
    def __init__(self, matrix, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.embedding = EmbeddingLayer(matrix)
        self.context_encoder = ContextEncoder(matrix.shape[1], hidden, dropout)
        self.encode_attention = Composition(hidden, dropout)
        self.pooling = MeanMaxPooling()
        self.head = ClassifierHead(hidden, dropout)

    def forward(self, ids_a, len_a, ids_b, len_b):
        a = self.context_encoder(self.embedding(ids_a), len_a)
        b = self.context_encoder(self.embedding(ids_b), len_b)
        A_res, B_res = enhance(a, len_a, b, len_b)
        pa = self.pooling(self.encode_attention(A_res, len_a), len_a)
        pb = self.pooling(self.encode_attention(B_res, len_b), len_b)
        return self.head(torch.cat([pa, pb], dim=1))
