"""Step 3 data plumbing: DataFrame rows -> batches of tensors for SiameseMatcher.

PairDataset[i] -> one pair as raw strings + label; make_collate(vocab) -> the
DataLoader's collate_fn, which encodes each side per batch (padding only to
that batch's longest string).
"""
from importlib import import_module

import torch
from torch.utils.data import Dataset

encode_batch = import_module("siamese_model.03_encode_batch_step").encode_batch   # digit-prefixed filename -> can't use `from ... import`


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
