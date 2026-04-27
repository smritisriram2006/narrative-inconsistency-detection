import subprocess
import pandas as pd
from pathlib import Path
import os

# ---------------- CONFIG ----------------
DEFAULT_METADATA_PATH = Path("data/metadata.csv")
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DEFAULT_METADATA_PATH))
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
SAMPLE_RATE = "16000"   # Required for Whisper
CHANNELS = "1"          # Mono audio
# ---------------------------------------


def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file using ffmpeg.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", SAMPLE_RATE,
        "-ac", CHANNELS,
        str(audio_path)
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def main():
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata: {len(metadata)} entries\n")

    for _, row in metadata.iterrows():
        video_id = row["video_id"]
        video_path = Path(row["video_path"])
        audio_path = Path(row["audio_path"])

        if not video_path.exists():
            print(f"[SKIP] Video not found: {video_path}")
            continue

        if audio_path.exists():
            print(f"[SKIP] Audio already exists: {audio_path}")
            continue

        print(f"[PROCESS] Extracting audio for: {video_id}")
        try:
            extract_audio(video_path, audio_path)
        except subprocess.CalledProcessError:
            print(f"[ERROR] ffmpeg failed for: {video_path}")

    print("\nAudio extraction completed.")


if __name__ == "__main__":
    main()
