import subprocess
import sys
from pathlib import Path
import os

# ---------------- CONFIG ----------------
PYTHON = sys.executable

PIPELINE_STEPS = [
    ("Extract Audio", "extract_audio.py"),
    ("Transcribe Audio", "transcribe.py"),
    ("Sentence Segmentation", "preprocess.py"),
    ("Sentence Pairing", "pair_sentences.py"),
    ("NLI Inference", "nli_inference.py"),
    ("Coherence Scoring", "coherence_score.py"),
]
# ---------------------------------------

def run_step(step_name, script_name):
    # Ensure we are looking in the 'src' folder
    script_path = Path("src") / script_name

    if not script_path.exists():
        print(f"[SKIP] {script_name} not found at {script_path}")
        return

    print(f"\n>> Executing: {step_name} ({script_name})")
    
    # We run the script and let it print its own errors to your console
    result = subprocess.run(
        [PYTHON, str(script_path.absolute())],
        check=False,
        # This ensures the script runs as if you were in the project root
        cwd=os.getcwd() 
    )

    if result.returncode != 0:
        print(f"\n[!!!] CRITICAL ERROR in {step_name}")
        print(f"[!!!] Script {script_name} exited with code {result.returncode}")
        # We exit so we don't try to process empty/broken data in the next step
        sys.exit(1)

def main():
    print("=== Narrative Inconsistency Detection Pipeline ===")
    print(f"System: HP Laptop | CPU: 11th Gen i3 | Interpreter: {PYTHON}")

    for step_name, script_name in PIPELINE_STEPS:
        run_step(step_name, script_name)

    print("\n[SUCCESS] Pipeline completed successfully.")

if __name__ == "__main__":
    main()