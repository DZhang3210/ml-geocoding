"""Step 3 model: the final copy of every stage, composed into `Encoder`, plus `SiameseMatcher`.

Stages were built and tested one notebook each (05 char embedding, 06 BiLSTM,
07 masked pooling); those notebooks now import their class from here, so the
tests run on exactly the code that gets trained.
"""
from importlib import import_module

import torch
from torch import nn

PAD_ID = import_module("siamese_model.01_vocab_step").PAD_ID   # digit-prefixed filename -> can't use `from ... import`


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
