import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
PAIR_PATH = Path("data/pairs/sentence_pairs.csv")
OUTPUT_PATH = Path("experiments/results/nli_results.csv")
MODEL_NAME = "cross-encoder/nli-roberta-base"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------


def main():
    print(f"[INFO] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    df = pd.read_csv(PAIR_PATH)
    print(f"[INFO] Loaded {len(df)} sentence pairs\n")

    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i + BATCH_SIZE]

        encodings = tokenizer(
            batch["premise"].tolist(),
            batch["hypothesis"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)

        for row_idx, row in enumerate(batch.itertuples(index=False)):
            entailment = probs[row_idx][2].item()
            neutral = probs[row_idx][1].item()
            contradiction = probs[row_idx][0].item()

            results.append({
                "video_id": row.video_id,
                "premise": row.premise,
                "hypothesis": row.hypothesis,
                "entailment": entailment,
                "neutral": neutral,
                "contradiction": contradiction,
                "is_contradiction": contradiction > max(entailment, neutral)
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\n[INFO] NLI inference completed.")
    print(f"[INFO] Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
