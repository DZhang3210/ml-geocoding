from sklearn.metrics import classification_report

# Print metrics + misclassified pairs for each model.
# val_df here is the original pairs frame (addresses, neg_type), not the feature matrix; val_label is its labels.
def review_model_peformance(ret, val_df, val_label):
    for i, ((model),(preds, probs)) in enumerate(ret.items()):
        print(f"{'='*20} {model} {'='*20}")
        print(classification_report(val_label, preds))
        
        # Per-negative-type breakdown (Overture only; Stamford has no neg_type)
        if "neg_type" in val_df.columns:
            for neg_type in ["case1", "case2", "case3"]:
                print(f"{'='*20}{neg_type}{'='*20}")
                mask = (val_df["neg_type"] == neg_type).values
                print((preds[mask] == 0).mean())

        # Misclassified pairs, with the label the model wrongly gave them
        # (neg_type only exists for Overture)
        cols = ["StreetName_a", "Address_a","ZipCode_a", "StreetName_b", "Address_b", "ZipCode_b", "neg_type"]
        cols = [c for c in cols if c in val_df.columns]
        wrong = preds != val_label.values
        comparison_df = val_df[wrong][cols].assign(
            correct_label = val_label.values[wrong],
            incorrect_label = preds[wrong],
        ).sort_values(by = "correct_label")
        print(
            comparison_df
        )
