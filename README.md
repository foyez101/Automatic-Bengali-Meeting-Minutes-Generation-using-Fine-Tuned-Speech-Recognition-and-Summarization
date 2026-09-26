# Bangla Speech-to-Summary

A web app that turns Bangla speech into a written transcript and a short summary. Upload an audio clip, and the pipeline runs automatically:

**Audio → Speech recognition (fine-tuned Whisper-small) → Transcript → Summarization (fine-tuned BanglaT5) → Summary**

Built as part of the thesis *"Automatic Bengali Meeting Minutes Generation using Fine-Tuned Speech Recognition and Abstractive Summarization"*.

## Features

- Web interface for uploading audio (MP3, WAV, WEBM, OGG, up to 25 MB)
- Handles audio of any length: speech is split into chunks of up to 20 seconds at natural pauses before transcription
- Long transcripts are summarized with a map-reduce strategy
- REST API (FastAPI) that the web page and other clients can call
- Evaluation script that measures WER, CER and ROUGE on a test set

## How the models were developed

The project went through two versions:

1. **Version 1: Whisper-small + mT5-small.** Whisper-small was fine-tuned for Bangla speech recognition and mT5-small for summarization. The results were not good enough. The Whisper model stopped transcribing after about 10 seconds of speech, so longer recordings had to be cut into very short pieces, and many words came out misspelled (test WER 55.0%). The mT5-small summaries were far too short (about 6.5 words on average, against 24.4 words in the reference summaries) and scored low (ROUGE-L 14.5%).
2. **Version 2: Whisper-small + BanglaT5 (current).** The summarizer was replaced with BanglaT5, a model pretrained specifically on Bangla text, and fine-tuned on the same data. Whisper-small was fine-tuned further on full-length segments of up to 28 seconds with their complete transcripts. Both models improved substantially (see [Results](#results)).

## Models

| Task | Base model | Folder |
|---|---|---|
| Speech recognition | [Whisper-small](https://huggingface.co/openai/whisper-small) | `whisper-bangla-v2/` |
| Summarization | [BanglaT5](https://huggingface.co/csebuetnlp/banglat5) | `banglat5-summary-retrained-base/` |

Both models were fine-tuned on the project dataset of 1160 Bangla recordings with transcripts and summaries, using a fixed 80/10/10 train/validation/test split (928 / 116 / 116 recordings).

- **Whisper-small:** the recordings were cut into 3713 segments of up to 28 seconds, each paired with its matching part of the transcript. The model was trained on the 2917 training segments (about 13 hours of audio) for 6 epochs (learning rate 1e-5, 100 warmup steps, effective batch size 16), keeping the checkpoint with the lowest validation WER.
- **BanglaT5:** fine-tuned on the 928 training transcript–summary pairs for up to 20 epochs (learning rate 3e-4, early stopping on validation ROUGE-L).

The trained models are too large for the repository and are published as zip files on the **[v1.0 release page](https://github.com/foyez101/Automatic-Bengali-Meeting-Minutes-Generation-using-Fine-Tuned-Speech-Recognition-and-Summarization/releases/tag/v1.0)** (about 1.8 GB in total).

## Run it locally

The steps below are for Windows (PowerShell). Commands for macOS / Linux are shown where they differ.

### 1. Install the prerequisites

- **Python 3.10 or newer**: [python.org/downloads](https://www.python.org/downloads/)
- **Git**: [git-scm.com](https://git-scm.com/)
- **FFmpeg** (used to convert uploaded audio):
  - Windows: `winget install Gyan.FFmpeg`, then close and reopen the terminal
  - macOS: `brew install ffmpeg`
  - Ubuntu / Debian: `sudo apt install ffmpeg`
- About **4 GB of free RAM** and **4 GB of disk space**. The app runs on CPU; an NVIDIA GPU is used automatically if available.

Check that FFmpeg works:

```powershell
ffmpeg -version
```

### 2. Clone the repository

```powershell
git clone https://github.com/foyez101/Automatic-Bengali-Meeting-Minutes-Generation-using-Fine-Tuned-Speech-Recognition-and-Summarization.git
cd Automatic-Bengali-Meeting-Minutes-Generation-using-Fine-Tuned-Speech-Recognition-and-Summarization
```

### 3. Create a virtual environment and install the dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS / Linux: source venv/bin/activate

python -m pip install -r requirements.txt

# BanglaT5 text normalizer (required by the summarizer).
# --no-deps skips its outdated pinned dependencies; compatible versions
# of emoji, ftfy and regex come from requirements.txt.
python -m pip install --no-deps git+https://github.com/csebuetnlp/normalizer
```

If PowerShell blocks the activate script, run this first (it only affects the current terminal):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The Whisper model is saved in the `transformers` 5 format, so `transformers` 5.x is required.

### 4. Download the models

Download both zip files from the [v1.0 release page](https://github.com/foyez101/Automatic-Bengali-Meeting-Minutes-Generation-using-Fine-Tuned-Speech-Recognition-and-Summarization/releases/tag/v1.0) and extract them into the project folder, or run:

```powershell
$base = "https://github.com/foyez101/Automatic-Bengali-Meeting-Minutes-Generation-using-Fine-Tuned-Speech-Recognition-and-Summarization/releases/download/v1.0"
curl.exe -L -o whisper-bangla-v2.zip "$base/whisper-bangla-v2.zip"
curl.exe -L -o banglat5-summary-retrained-base.zip "$base/banglat5-summary-retrained-base.zip"
tar -xf whisper-bangla-v2.zip
tar -xf banglat5-summary-retrained-base.zip
```

On macOS / Linux, use `curl` instead of `curl.exe`, `unzip` instead of `tar -xf`, and set the variable with `base="..."`.

The project folder should now contain:

```
whisper-bangla-v2/
banglat5-summary-retrained-base/
```

The zip files can be deleted afterwards.

### 5. Start the web app

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (loading the models takes about a minute), then open **http://127.0.0.1:8000** in your browser, choose an audio file and click **Transcribe & Summarize**.

Processing is much faster on a GPU. On CPU it takes several times the audio length.

To test the pipeline without the web page:

```powershell
python inference.py "path\to\audio.mp3"
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `FFmpeg is not installed or not available in PATH` | Install FFmpeg (step 1), then close and reopen the terminal |
| `Summarizer model folder not found` or a Whisper loading error | Check that both model folders sit directly in the project folder, not nested one level deeper (e.g. `whisper-bangla-v2/whisper-bangla-v2/`) |
| `The Bangla normalizer is not installed` | Run the normalizer install command from step 3 inside the activated venv |
| PowerShell refuses to run `Activate.ps1` | Run the `Set-ExecutionPolicy` command from step 3 |

## Project structure

```
app.py                              FastAPI server (serves the web page and the API)
inference.py                        Loads both models and runs the transcribe -> summarize pipeline
evaluate.py                         Measures accuracy (WER, CER, ROUGE) on a test folder
check_dataset.py                    Checks a speech/text/summary dataset before training
split_dataset.py                    Creates the fixed 80/10/10 train/val/test split
index.html                          Web interface
requirements.txt                    Python dependencies
whisper-bangla-v2/                  Fine-tuned Whisper-small (download, see above)
banglat5-summary-retrained-base/    Fine-tuned BanglaT5 (download, see above)
```

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
| `WHISPER_DIR` | `whisper-bangla-v2` | Speech recognition model folder |
| `CHUNK_MAX_S` | 20.0 | Maximum audio length per ASR chunk (seconds); 20 s gave the lowest validation WER |
| `CUT_SEARCH_S` | 8.0 | If there is no clear pause, cut at the quietest point in the last N seconds before the limit |
| `SILENCE_TOP_DB` | 30 | Pause-detection sensitivity |
| `ASR_NUM_BEAMS` | 1 | Greedy decoding (same as the reported evaluation); raise to 4 for slightly different output at ~3–4× the CPU time |
| `MT5_DIR` | `banglat5-summary-retrained-base` | Summarizer model folder |
| `SUMMARY_USE_NORMALIZER` | True | Apply the BanglaT5 normalizer before summarizing (required for this model) |
| `SUMMARY_NUM_BEAMS` | 4 | Beam search width for summaries |

## Results

All results are on the 116 held-out test recordings, which were not used for training or for choosing settings.

**Version 1 vs version 2:**

| Component | Metric | Version 1 | Version 2 (current) |
|---|---|---|---|
| Speech recognition (Whisper-small) | WER | 55.0% | **17.3%** |
| | CER | 22.6% | **6.0%** |
| Summarization (v1: mT5-small, v2: BanglaT5) | ROUGE-1 | 15.7% | **30.4%** |
| | ROUGE-2 | 4.4% | **11.1%** |
| | ROUGE-L | 14.5% | **25.3%** |
| | Avg. summary words | 6.5 | **21.4** |

Speech recognition was scored on full recordings; summarization was scored with the reference transcripts as input, so the two parts are measured separately.

**Speech recognition details** (version 2, full recordings split into chunks of up to 20 s):

| Metric | Result |
|---|---|
| WER | **17.3%** |
| CER | **6.0%** |

The chunk length was chosen on the validation recordings (WER with 10 s chunks: 20.9%, 20 s: 18.8%, 28 s: 25.6%). On the 361 test segments of up to 28 s, WER is 16.0% and CER 5.0%.

**Full pipeline** (fine-tuned BanglaT5 summaries compared with the reference summaries):

| Summary made from | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. summary words |
|---|---|---|---|---|
| Reference transcript (upper limit) | 29.0% | 10.7% | 24.5% | 24.2 |
| Whisper transcript (audio → summary) | **29.3%** | **10.3%** | **24.6%** | 23.9 |

Summaries generated directly from speech are as good as summaries generated from the reference transcripts. Reference summaries average 24.4 words.

## Known limitations

- The training data is read, news-style Bangla speech from one speaker at a time. Spontaneous conversation, several overlapping speakers and English technical terms are transcribed less accurately.
- Transcripts are sometimes misspelled, especially for conjuncts (যুক্তাক্ষর), vowel signs and numbers.
- Summaries can swap names and numbers or state facts that are not in the transcript. ROUGE does not capture factual errors, so outputs should be checked.
- Speaker diarization (who said what) and structured meeting minutes are not implemented yet (see Future work).
- Processing is slow on CPU.

## Future work

- **Speaker diarization:** identify who is speaking when, so transcripts of multi-speaker meetings can be labelled by speaker.
- **Meeting minutes generation:** turn a speaker-labelled meeting transcript into structured minutes (participants, discussion points, decisions and action items) instead of a single free-text summary.
- **More varied training data:** add spontaneous, multi-speaker meeting recordings and Bangla speech mixed with English terms, to improve accuracy on real meetings.