import pandas as pd
def dis_pct_suc_on_mask(labels, preds, mask, title = None):
    if title:
        print(f"{'='*20}{title}{'='*20}")
    if len(labels) != len(preds) or len(preds) != len(mask):
        raise KeyError("DF/Mask/preds must all be the same length")
    # else:
    #     print(f"mask: {mask}")
    n = mask.sum()
    good = len(labels[mask] == preds[mask])
    bad = len(labels[mask] == preds[mask])

    n1 = (~mask).sum()
    good1 = len(labels[~mask] == preds[~mask])
    bad1 = len(labels[~mask] == preds[~mask])
    new_df = pd.DataFrame(
        {
            f"Mask[Good] (n={good})": [f"{good/n:.2f}"],
            f"Mask[Bad] (n={bad})": [f"{bad/n:.2f}"],
            f"No Mask[Good] (n={good1})": [f"{good1/n1:.2f}"],
            f"No Mask[Bad] (n={bad1})": [f"{bad1/n1:.2f}"],
        }
    )
    print(new_df)
