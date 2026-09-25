# Bangla Speech-to-Summary

A web app that turns Bangla speech into a written transcript and a short summary. Upload an audio clip, and the pipeline runs automatically:

**Audio → Speech recognition (fine-tuned Whisper-small) → Transcript → Summarization (fine-tuned BanglaT5) → Summary**

Built as part of a thesis project on a Bengali-English code-switched speech dataset for generating meeting minutes.

## Features

- Web interface for uploading audio (MP3, WAV, WEBM, OGG, up to 25 MB)
- Handles audio of any length: speech is split into short chunks at natural pauses before transcription
- Long transcripts are summarized with a map-reduce strategy
- REST API (FastAPI) that the web page and other clients can call
- Evaluation script that measures WER, CER and ROUGE on a test set

## Project structure

```
app.py                              FastAPI server (serves the web page and the API)
inference.py                        Loads both models and runs the transcribe -> summarize pipeline
evaluate.py                         Measures accuracy (WER, CER, ROUGE) on a test folder
check_dataset.py                    Checks a speech/text/summary dataset before training
split_dataset.py                    Creates the fixed 80/10/10 train/val/test split
index.html                          Web interface
requirements.txt                    Python dependencies
whisper-bangla-trained/             Fine-tuned Whisper model (not in repo, see below)
banglat5-summary-retrained-base/    Fine-tuned BanglaT5 summarizer (not in repo, see below)
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
banglat5-summary-retrained-base/
```

The summarizer is [BanglaT5](https://huggingface.co/csebuetnlp/banglat5) fine-tuned on 928 transcript–summary pairs from the project dataset (80/10/10 train/validation/test split). Its folder must contain `spiece.model`; if it is missing, copy it from the original BanglaT5:

```powershell
python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download('csebuetnlp/banglat5', 'spiece.model'); shutil.copy(p, 'banglat5-summary-retrained-base/spiece.model')"
```

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source venv/bin/activate           # macOS / Linux

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Install the BanglaT5 text normalizer (required by the summarizer).
#    --no-deps skips its outdated pinned dependencies; compatible versions
#    of emoji, ftfy and regex come from requirements.txt.
python -m pip install --no-deps git+https://github.com/csebuetnlp/normalizer
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
| `MT5_DIR` | `banglat5-summary-retrained-base` | Summarizer model folder |
| `SUMMARY_USE_NORMALIZER` | True | Apply the BanglaT5 normalizer before summarizing (required for the retrained model) |
| `SUMMARY_NUM_BEAMS` | 4 | Beam search width for summaries |

## Results

Summarizer, on the 116 held-out test clips (reference transcripts as input):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. summary words |
|---|---|---|---|---|
| Previous summarizer | 15.7% | 4.4% | 14.5% | 6.5 |
| BanglaT5, retrained | 30.4% | 11.1% | 25.3% | 21.4 |

Reference summaries average 24.4 words.

## Known limitations

- Transcripts are often phonetically correct but misspelled, especially for conjuncts (যুক্তাক্ষর) and vowel signs
- Summaries can swap names and numbers for similar-looking ones and can repeat words; ROUGE does not capture factual errors, so outputs should be checked
- Processing is slow on CPU