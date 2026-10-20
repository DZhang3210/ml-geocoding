"""Step 3 training loop: one epoch of updates, and prediction without updates."""
import torch
from torch import nn


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss, n_pairs = 0.0, 0
    for ids_a, len_a, ids_b, len_b, y in loader:
        opt.zero_grad()                                              # first: clear anything left from before
        logits = model(ids_a, len_a, ids_b, len_b)                   # the model never sees y
        loss = loss_fn(logits, y)                                    # BCEWithLogitsLoss applies the sigmoid itself
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # between backward and step: edits the gradients step uses
        opt.step()
        total_loss += loss.item() * len(y)                           # weight by batch size -> average per pair, not per batch
        n_pairs += len(y)
    return total_loss / n_pairs


def predict(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():                                            # no graph recorded, no updates
        for ids_a, len_a, ids_b, len_b, y in loader:                 # y unused
            logits = model(ids_a, len_a, ids_b, len_b)
            probs.append(torch.sigmoid(logits))                      # logits -> probabilities
    return torch.cat(probs)                                          # (N,), in loader order: val/test loaders must not shuffle
