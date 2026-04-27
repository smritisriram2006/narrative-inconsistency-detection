import hashlib
from pathlib import Path

import streamlit as st

from src.process_single_video import process_video

st.title("AI Narrative Inconsistency Detector")

st.write(
"""
Upload a video containing a narrative statement.
The system will detect internal contradictions and compute a coherence score.
"""
)

uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","mkv"])

if uploaded_file is not None:

    video_dir = Path("data/videos/uploaded")
    video_dir.mkdir(parents=True, exist_ok=True)

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha1(file_bytes).hexdigest()[:8]
    original_path = Path(uploaded_file.name)
    stored_name = f"{original_path.stem}_{file_hash}{original_path.suffix.lower()}"
    video_path = video_dir / stored_name

    with open(video_path, "wb") as f:
        f.write(file_bytes)

    st.success("Video uploaded successfully")

    if st.button("Analyze Narrative"):

        with st.spinner("Processing video..."):
            try:
                result = process_video(video_path)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        if len(result) > 0:

            row = result.iloc[0]

            st.subheader("Analysis Result")

            st.metric(
                "Narrative Coherence Score",
                round(row["coherence_score"],3)
            )

            st.metric(
                "Contradictions Detected",
                int(row["contradictions"])
            )

            st.metric(
                "Sentence Comparisons",
                int(row["total_pairs"])
            )

        else:
            st.error("No results found")
