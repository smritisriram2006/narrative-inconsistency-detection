from datasets import load_dataset

# Load MNLI from Hugging Face (official source)
mnli = load_dataset("nyu-mll/multi_nli")

# Access splits
train_ds = mnli["train"]                       # Not used (no training)
val_matched_ds = mnli["validation_matched"]    # Primary validation
val_mismatched_ds = mnli["validation_mismatched"]  # Robustness check

# Label mapping (MNLI standard)
label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

# Sanity check
print("MNLI loaded successfully\n")
print("Train size:", len(train_ds))
print("Validation (matched):", len(val_matched_ds))
print("Validation (mismatched):", len(val_mismatched_ds))

# View one sample
sample = val_matched_ds[0]
print("\nSample:")
print("Premise:", sample["premise"])
print("Hypothesis:", sample["hypothesis"])
print("Label:", label_map[sample["label"]])
