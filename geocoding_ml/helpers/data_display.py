import pandas as pd
def disp_pct_succ_on_mask(labels, preds, mask, title = None, show_other = True):
    if title:
        print(f"{'='*20}{title}{'='*20}")
    if len(labels) != len(preds) or len(preds) != len(mask):
        raise ValueError("DF/Mask/preds must all be the same length")
    # else:
    #     print(f"mask: {mask}")
    n = mask.sum()
    good = (labels[mask] == preds[mask]).sum()
    bad = (labels[mask] != preds[mask]).sum()

    n1 = (~mask).sum()
    good1 = (labels[~mask] == preds[~mask]).sum()
    bad1 = (labels[~mask] != preds[~mask]).sum()
    new_df = pd.DataFrame(
        {
            f"Mask[Good](n={n})": [f"{good/n:.2f}(n = {good})"],
            f"Mask[Bad](n={n})": [f"{bad/n:.2f}(n = {bad})"],

            **({f"No Mask[Good](n={n1})": [f"{good1/n1:.2f}(n = {good1})"],
            f"No Mask[Bad](n={n1})": [f"{bad1/n1:.2f}(n = {bad1})"]} if show_other else {})
        }
    )
    print(new_df)
