# Bangla Speech-to-Summary

A web app that turns Bangla speech into a written transcript and a short summary. Upload an audio clip, and the pipeline runs automatically:

**Audio → Speech recognition (fine-tuned Whisper) → Transcript → Summarization (fine-tuned BanglaT5) → Summary**

Built as part of a thesis project on a Bengali-English code-switched speech dataset for generating meeting minutes.

## Features

- Web interface for uploading audio (MP3, WAV, WEBM, OGG, up to 25 MB)
- Handles audio of any length: speech is split into short chunks at natural pauses before transcription
- Long transcripts are summarized with a map-reduce strategy
- REST API (FastAPI) that the web page and other clients can call
- Evaluation script that measures WER, CER and ROUGE on a test set

## Project structure

```
app.py                      FastAPI server (serves the web page and the API)
inference.py                Loads both models and runs the transcribe → summarize pipeline
evaluate.py                 Measures accuracy (WER, CER, ROUGE) on a test folder
index.html                  Web interface
requirements.txt            Python dependencies
whisper-bangla-trained/     Fine-tuned Whisper model (not in repo, see below)
banglat5-summary-trained/   Fine-tuned BanglaT5 model (not in repo, see below)
```

## Requirements

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) installed and available in PATH (used to convert uploaded audio)
  - Windows: `winget install Gyan.FFmpeg`, then restart the terminal
- About 4 GB of free RAM. Runs on CPU; an NVIDIA GPU is used automatically if available

## Models

The trained model folders are too large for GitHub (about 2 GB in total), so they are not included in this repository.

Download them from: **[add model download link here]**

Place both folders directly in the project root:

```
whisper-bangla-trained/
banglat5-summary-trained/
```

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source venv/bin/activate           # macOS / Linux

# 2. Install dependencies
python -m pip install -r requirements.txt
```

If PowerShell blocks the activate script, run this first (affects the current terminal only):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the web app

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (model loading takes about a minute), then open **http://127.0.0.1:8000** in your browser.

On CPU, processing takes roughly 5–6× the audio length.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Server status and device (cpu / cuda) |
| POST | `/api/transcribe-summarize` | Form field `audio` (file upload). Returns `transcript`, `summary`, `duration_sec`, `processing_time_sec` |

## Evaluation

Prepare a test folder with matching filenames:

```
test_set/
    speech/    1.mp3  2.mp3  ...
    text/      1.txt  2.txt  ...   (word-for-word reference transcripts)
    summary/   1.txt  2.txt  ...   (reference summaries)
```

```powershell
python evaluate.py "path\to\test_set"                  # full evaluation
python evaluate.py "path\to\test_set" --limit 3        # quick test on 3 clips
python evaluate.py "path\to\test_set" --no-summary     # transcripts only
python evaluate.py "path\to\test_set" --summary-only   # summarizer only (no ASR, fast)
```

Results are printed and saved as a CSV. Text is Unicode-normalized and punctuation is removed before scoring. ROUGE is computed with whitespace tokenization, because common ROUGE libraries drop non-English characters.

## Configuration

Settings at the top of `inference.py`:

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_MAX_S` | 10.0 | Maximum audio length per ASR chunk (seconds) |
| `SILENCE_TOP_DB` | 30 | Pause-detection sensitivity |
| `ASR_NUM_BEAMS` | 4 | Lower to 1–2 for faster transcription on CPU |
| `SUMMARY_MIN_RATIO` / `SUMMARY_MAX_RATIO` | 0.35 / 0.60 | Summary length relative to the transcript |
| `SUMMARY_LENGTH_PENALTY` | 1.5 | Values above 1.0 favor longer summaries |

## Known limitations

- Transcripts are often phonetically correct but misspelled, especially for conjuncts (যুক্তাক্ষর) and vowel signs
- Summaries can be short or start mid-sentence, and forcing longer summaries can introduce invented content
- Processing is slow on CPU