import os
from pathlib import Path

FFMPEG_BINARY = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
if FFMPEG_BINARY.exists():
    os.environ["FFMPEG_BINARY"] = str(FFMPEG_BINARY)
    os.environ["PATH"] = str(FFMPEG_BINARY.parent) + os.pathsep + os.environ.get("PATH", "")

import pandas as pd
import whisper

# ---------------- CONFIG ----------------
DEFAULT_METADATA_PATH = Path("data/metadata.csv")
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DEFAULT_METADATA_PATH))
WHISPER_MODEL_SIZE = "base"   # base is stable & fast
LANGUAGE = "en"               # force English
# ---------------------------------------


def transcribe_audio(model, audio_path, transcript_path):
    """
    Transcribes a single audio file using Whisper.
    """
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    result = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        fp16=False  # CPU-safe
    )

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())


def main():
    print("[INFO] Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    print("[INFO] Whisper model loaded\n")

    metadata = pd.read_csv(METADATA_PATH)
    print(f"[INFO] Loaded metadata: {len(metadata)} entries\n")

    for _, row in metadata.iterrows():
        video_id = row["video_id"]
        audio_path = Path(row["audio_path"])
        transcript_path = Path(row["transcript_path"])

        if not audio_path.exists():
            print(f"[SKIP] Audio not found: {audio_path}")
            continue

        if transcript_path.exists():
            print(f"[SKIP] Transcript already exists: {transcript_path}")
            continue

        print(f"[PROCESS] Transcribing: {video_id}")

        try:
            transcribe_audio(model, audio_path, transcript_path)
        except Exception as e:
            print(f"[ERROR] Transcription failed for {video_id}: {e}")
            continue

    print("\n[INFO] Transcription completed.")


if __name__ == "__main__":
    main()
