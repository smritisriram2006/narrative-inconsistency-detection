import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
SENTENCE_DIR = Path("data/sentences")
OUTPUT_DIR = Path("data/pairs")
PAIR_STRATEGY = "windowed"   
WINDOW_SIZE = 5  # Compare each sentence to the next 5 sentences
# ---------------------------------------

def generate_windowed_pairs(sentences, window_size):
    """
    Generates pairs within a specific look-ahead window.
    Example (window=2): (S1,S2), (S1,S3), (S2,S3), (S2,S4)...
    """
    pairs = []
    n = len(sentences)
    for i in range(n):
        # Look ahead up to the window size, but stay within list bounds
        for j in range(i + 1, min(i + 1 + window_size, n)):
            pairs.append((sentences[i], sentences[j]))
    return pairs

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    sentence_files = list(SENTENCE_DIR.glob("*_sentences.txt"))
    print(f"[INFO] Found {len(sentence_files)} sentence files\n")

    for file in sentence_files:
        video_id = file.stem.replace("_sentences", "")
        print(f"[PROCESS] Pairing sentences (Window: {WINDOW_SIZE}) for: {video_id}")

        sentences = [
            line.strip()
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if len(sentences) < 2:
            continue

        # Use the new windowed logic
        pairs = generate_windowed_pairs(sentences, WINDOW_SIZE)

        for p, h in pairs:
            all_pairs.append({
                "video_id": video_id,
                "premise": p,
                "hypothesis": h,
                "pair_type": f"windowed_{WINDOW_SIZE}"
            })

    df = pd.DataFrame(all_pairs)
    out_path = OUTPUT_DIR / "sentence_pairs.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[INFO] Saved {len(df)} sentence pairs to {out_path}")

if __name__ == "__main__":
    main()