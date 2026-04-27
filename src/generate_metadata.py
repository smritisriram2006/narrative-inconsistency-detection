import pandas as pd
from pathlib import Path
import os

DEFAULT_METADATA_PATH = Path("data/metadata.csv")
METADATA_PATH = Path(os.environ.get("METADATA_PATH", DEFAULT_METADATA_PATH))
PROJECT_ROOT = METADATA_PATH.resolve().parents[1]

def fix_paths():
    df = pd.read_csv(METADATA_PATH)

    def to_relative(path_str):
        path = Path(path_str)

        # If already relative, keep it
        if not path.is_absolute():
            return path_str.replace("\\", "/")

        # Convert absolute path -> relative to project root
        try:
            rel_path = path.resolve().relative_to(PROJECT_ROOT)
            return str(rel_path).replace("\\", "/")
        except ValueError:
            return path_str.replace("\\", "/")

    df["video_path"] = df["video_path"].apply(to_relative)
    df["audio_path"] = df["audio_path"].apply(to_relative)
    df["transcript_path"] = df["transcript_path"].apply(to_relative)

    df.to_csv(METADATA_PATH, index=False)

    print(f"[SUCCESS] Fixed paths for {len(df)} entries in metadata.csv")


if __name__ == "__main__":
    fix_paths()
