# AI-Based Narrative Inconsistency Detection for Criminology Using NLP

This project analyzes spoken narrative statements and flags possible internal contradictions using Natural Language Processing (NLP) and transformer-based Natural Language Inference (NLI). It is designed for criminology and forensic-linguistics use cases where long-form narratives such as witness statements, suspect interviews, or recorded testimony need a fast consistency check.

The current system supports two workflows:

- A `Streamlit` app for uploading and analyzing a single video
- A batch pipeline for processing all videos listed in `data/metadata.csv`

## What the system does

Given a narrative video, the pipeline:

1. Extracts audio from the video with `ffmpeg`
2. Transcribes the audio with OpenAI Whisper
3. Splits the transcript into sentences with `spaCy`
4. Builds sentence pairs using a windowed comparison strategy
5. Runs NLI inference with `cross-encoder/nli-roberta-base`
6. Computes an inconsistency score and final coherence score

The final coherence score is:

```text
coherence_score = 1 - (contradictions / total_pairs)
```

A lower coherence score suggests more internal contradiction across the narrative.

## Project structure

```text
.
|-- app.py
|-- src/
|   |-- extract_audio.py
|   |-- transcribe.py
|   |-- preprocess.py
|   |-- pair_sentences.py
|   |-- nli_inference.py
|   |-- coherence_score.py
|   |-- process_single_video.py
|   |-- run_pipeline.py
|   |-- generate_metadata.py
|   `-- load_mnli.py
|-- data/
|   |-- metadata.csv
|   |-- videos/
|   |-- audio/
|   |-- transcripts/
|   |-- sentences/
|   `-- pairs/
|-- experiments/
|   |-- results/
|   `-- plots/
`-- docs/
```

## Core modules

- `app.py`: Streamlit interface for uploading a single video and displaying results
- `src/process_single_video.py`: Creates temporary metadata for one uploaded file and runs the full pipeline
- `src/run_pipeline.py`: Runs the full batch pipeline step by step
- `src/extract_audio.py`: Converts video to mono 16 kHz WAV using `ffmpeg`
- `src/transcribe.py`: Transcribes audio using Whisper
- `src/preprocess.py`: Segments transcripts into sentences with `spaCy`
- `src/pair_sentences.py`: Creates windowed sentence pairs for comparison
- `src/nli_inference.py`: Scores each pair for entailment, neutral, or contradiction
- `src/coherence_score.py`: Aggregates pairwise contradiction results into a narrative-level score

## Data format

The batch workflow expects `data/metadata.csv` in this format:

```csv
video_id,label,video_path,audio_path,transcript_path
trial_truth_001,truthful,data/videos/truthful/trial_truth_001.mp4,data/audio/trial_truth_001.wav,data/transcripts/trial_truth_001.txt
```

Current labels in the dataset include `truthful` and `deceptive`, but the scoring pipeline itself focuses on narrative contradiction detection rather than classification training.

## Models and tools used

- OpenAI Whisper for speech-to-text transcription
- `en_core_web_sm` from `spaCy` for sentence segmentation
- `cross-encoder/nli-roberta-base` from Hugging Face for contradiction detection
- `ffmpeg` for audio extraction
- `pandas`, `PyTorch`, `transformers`, and `tqdm` for data handling and inference

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

`requirements.txt` is currently empty, so install the packages used by the code directly:

```powershell
pip install streamlit pandas openai-whisper spacy torch transformers tqdm datasets
python -m spacy download en_core_web_sm
```

### 3. Install ffmpeg

The code currently expects `ffmpeg` at:

```text
C:\ffmpeg\bin\ffmpeg.exe
```

If your installation is somewhere else, update:

- `src/extract_audio.py`
- `src/process_single_video.py`

## Running the app

Launch the Streamlit interface:

```powershell
streamlit run app.py
```

In the app:

1. Upload a video file (`.mp4`, `.mov`, or `.mkv`)
2. Click `Analyze Narrative`
3. Review:
   - Narrative coherence score
   - Number of contradictions detected
   - Number of sentence comparisons

Uploaded files are stored in `data/videos/uploaded/`.

## Running the batch pipeline

To process every video listed in `data/metadata.csv`:

```powershell
python src/run_pipeline.py
```

This executes the following steps in order:

1. `extract_audio.py`
2. `transcribe.py`
3. `preprocess.py`
4. `pair_sentences.py`
5. `nli_inference.py`
6. `coherence_score.py`

## Outputs

Main outputs are written to:

- `data/audio/`: extracted WAV files
- `data/transcripts/`: Whisper transcripts
- `data/sentences/`: sentence-level text files
- `data/pairs/sentence_pairs.csv`: generated sentence pairs
- `experiments/results/nli_results.csv`: pairwise NLI predictions
- `experiments/results/coherence_scores.csv`: final narrative scores

Example fields in `coherence_scores.csv`:

- `video_id`
- `total_pairs`
- `contradictions`
- `inconsistency_score`
- `coherence_score`

## Current methodology

The current pairing strategy is window-based:

- Each sentence is compared with the next 5 sentences
- This reduces the number of comparisons versus all-pairs matching
- It keeps comparisons more local and narratively relevant

A contradiction is counted when the model's contradiction probability is higher than both entailment and neutral probabilities for a sentence pair.

## Limitations

- `ffmpeg` is hardcoded to a Windows path
- `requirements.txt` has not been populated yet
- Most files in `docs/` are still placeholders
- The current system reports contradiction-based coherence, but does not yet provide rich explanation or forensic interpretation
- Sentence pairing is heuristic and may miss long-distance contradictions

## Future improvements

- Make `ffmpeg` path configurable through environment variables
- Fill in `requirements.txt`
- Add contradiction explanations and highlighted sentence pairs in the UI
- Add evaluation metrics for truthful vs deceptive narratives
- Improve sentence pairing with discourse-aware or retrieval-based strategies
- Export investigator-friendly reports

## Abstract

Narrative inconsistency detection is an emerging area within criminology and forensic linguistics, aimed at improving the analysis of textual and spoken evidence in criminal investigations. This project presents an AI-assisted system that uses NLP and transformer-based NLI to automatically identify contradictions and logical gaps in criminal narratives. By combining audio transcription, sentence segmentation, semantic inference, and coherence scoring, the system provides interpretable signals that can help investigators review narrative reliability more efficiently.
