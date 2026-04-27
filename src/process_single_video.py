import pandas as pd
from pathlib import Path
import subprocess
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
FFMPEG_BINARY = Path(r"C:\ffmpeg\bin\ffmpeg.exe")

def process_video(video_path):

    video_path = Path(video_path)
    video_id = video_path.stem

    base_data = Path("data")
    base_data.mkdir(parents=True, exist_ok=True)
    audio_path = base_data / "audio" / f"{video_id}.wav"
    transcript_path = base_data / "transcripts" / f"{video_id}.txt"
    sentence_path = base_data / "sentences" / f"{video_id}_sentences.txt"

    metadata_path = base_data / "temp_metadata.csv"

    # Create temporary metadata
    df = pd.DataFrame([{
        "video_id": video_id,
        "label": "unknown",
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path)
    }])

    df.to_csv(metadata_path, index=False)

    env = os.environ.copy()
    env["METADATA_PATH"] = str(metadata_path)
    if FFMPEG_BINARY.exists():
        env["FFMPEG_BINARY"] = str(FFMPEG_BINARY)
        env["PATH"] = str(FFMPEG_BINARY.parent) + os.pathsep + env.get("PATH", "")

    steps = [
        "extract_audio.py",
        "transcribe.py",
        "preprocess.py",
        "pair_sentences.py",
        "nli_inference.py",
        "coherence_score.py"
    ]

    for step in steps:
        step_path = Path("src") / step
        result = subprocess.run(
            [PYTHON, str(step_path)],
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            details = "\n".join(
                part for part in (
                    result.stdout.strip(),
                    result.stderr.strip(),
                ) if part
            )
            raise RuntimeError(f"{step} failed.\n{details}")

    if not transcript_path.exists() or not transcript_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            "Transcription did not produce any text for the uploaded video. "
            "Whisper usually needs ffmpeg available on PATH."
        )

    if not sentence_path.exists():
        raise RuntimeError("Sentence segmentation did not produce any output for the uploaded video.")

    sentence_count = sum(
        1 for line in sentence_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if sentence_count < 2:
        raise RuntimeError(
            "The uploaded video produced fewer than 2 sentences, so there was nothing to compare."
        )

    # Read result
    results_path = Path("experiments/results/coherence_scores.csv")
    if not results_path.exists():
        raise RuntimeError("Coherence scoring did not produce a results file.")

    results = pd.read_csv(results_path)
    result_rows = results[results["video_id"] == video_id]

    if result_rows.empty:
        raise RuntimeError("The pipeline completed, but no coherence score was generated for this upload.")

    return result_rows
