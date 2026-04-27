import pandas as pd
from pathlib import Path
import spacy
import os

# ---------------- CONFIG ----------------
DEFAULT_METADATA_PATH = Path("data/metadata.csv")
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DEFAULT_METADATA_PATH))
OUTPUT_DIR = Path("data/sentences")
SPACY_MODEL = "en_core_web_sm"
MIN_SENT_LEN = 5   # characters
# ---------------------------------------


def segment_sentences(nlp, text):
    """
    Splits text into clean sentences using spaCy.
    """
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) >= MIN_SENT_LEN:
            sentences.append(s)

    return sentences


def main():
    print("[INFO] Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)
    print("[INFO] spaCy loaded\n")

    metadata = pd.read_csv(METADATA_PATH)
    print(f"[INFO] Loaded metadata: {len(metadata)} entries\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in metadata.iterrows():
        video_id = row["video_id"]
        transcript_path = Path(row["transcript_path"])
        out_path = OUTPUT_DIR / f"{video_id}_sentences.txt"

        if not transcript_path.exists():
            print(f"[SKIP] Transcript not found: {transcript_path}")
            continue

        if out_path.exists():
            print(f"[SKIP] Sentences already exist: {out_path}")
            continue

        print(f"[PROCESS] Segmenting sentences: {video_id}")

        text = transcript_path.read_text(encoding="utf-8").strip()
        sentences = segment_sentences(nlp, text)

        with open(out_path, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

    print("\n[INFO] Sentence preprocessing completed.")


if __name__ == "__main__":
    main()
