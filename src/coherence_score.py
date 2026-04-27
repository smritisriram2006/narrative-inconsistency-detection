import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
NLI_RESULTS_PATH = Path("experiments/results/nli_results.csv")
OUTPUT_PATH = Path("experiments/results/coherence_scores.csv")
# ---------------------------------------


def main():
    print("[INFO] Loading NLI results...")
    df = pd.read_csv(NLI_RESULTS_PATH)
    print(f"[INFO] Loaded {len(df)} sentence-pair predictions\n")

    grouped = df.groupby("video_id")

    results = []

    for video_id, group in grouped:
        total_pairs = len(group)
        contradiction_count = group["is_contradiction"].sum()

        inconsistency_score = contradiction_count / total_pairs
        coherence_score = 1.0 - inconsistency_score

        results.append({
            "video_id": video_id,
            "total_pairs": total_pairs,
            "contradictions": int(contradiction_count),
            "inconsistency_score": round(inconsistency_score, 4),
            "coherence_score": round(coherence_score, 4)
        })

    result_df = pd.DataFrame(results).sort_values("coherence_score")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("[INFO] Coherence scoring completed.")
    print(f"[INFO] Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
