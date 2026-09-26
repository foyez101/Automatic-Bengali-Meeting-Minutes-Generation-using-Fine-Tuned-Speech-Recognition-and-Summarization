"""
inference.py
------------
Loads the fine-tuned Whisper (ASR) and BanglaT5 (summarization) models ONCE at
process startup, and exposes a single function that runs the full
transcribe -> summarize pipeline. Import this module from your web server
(FastAPI, Flask, Django, etc.) instead of reloading the models per request.

Directory layout expected (adjust the paths below):
    MODEL_ROOT/
        whisper-bangla-v2/
        banglat5-summary-retrained-base/
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import librosa

try:
    from normalizer import normalize as bn_normalize
except ImportError:
    bn_normalize = None

from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# ============================================================
# Configuration
# ============================================================
# Model folders sit directly in the project root. Run the app from this
# folder, or set the MODEL_ROOT environment variable to point elsewhere.
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "."))
WHISPER_DIR = MODEL_ROOT / "whisper-bangla-v2"
# Summarizer: BanglaT5 retrained on the full dataset (Kaggle, v2 notebook).
# Change to "banglat5-summary-trained" to go back to the old model
# (and set SUMMARY_USE_NORMALIZER = False for it).
MT5_DIR = MODEL_ROOT / "banglat5-summary-retrained-base"
SUMMARY_USE_NORMALIZER = True   # the retrained BanglaT5 was trained on normalized text

SAMPLE_RATE = 16000

# ASR chunking: the retrained Whisper (v2) was trained on segments up to 28 s.
# 20 s chunks gave the lowest validation WER (10 s: 20.9%, 20 s: 18.8%, 28 s: 25.6%).
# Audio is split at natural pauses (silences) so words are not cut in half.
CHUNK_MAX_S = 20.0      # max chunk length fed to Whisper
CHUNK_MIN_S = 3.0       # a leftover chunk shorter than this is merged into the previous one
SILENCE_TOP_DB = 30     # how quiet (dB below peak) counts as a clear pause; lower = more pauses found
CUT_SEARCH_S = 8.0      # if no clear pause, cut at the quietest point in the last N seconds before the limit
ASR_NUM_BEAMS = 4       # lower to 1-2 for faster (slightly less accurate) transcription on CPU

# Summary generation: the retrained model learned summary length from the
# data, so no minimum length is forced (forcing it caused invented content).
SUMMARY_MAX_TOKENS = 256
SUMMARY_NUM_BEAMS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FFmpeg is found automatically from PATH, so this works on any PC
# where FFmpeg is installed (no hardcoded user-specific path).
FFMPEG_PATH = shutil.which("ffmpeg")
if FFMPEG_PATH is None:
    raise RuntimeError(
        "FFmpeg is not installed or not available in PATH. "
        "Install it (e.g. `winget install Gyan.FFmpeg`) and restart your terminal."
    )

# ============================================================
# Load models once (module-level singletons)
# ============================================================
print(f"[inference] Loading models on device: {DEVICE}")
print(f"[inference] Using FFmpeg at: {FFMPEG_PATH}")

asr_model = WhisperForConditionalGeneration.from_pretrained(str(WHISPER_DIR)).to(DEVICE)
asr_feature_extractor = WhisperFeatureExtractor.from_pretrained(str(WHISPER_DIR))
asr_tokenizer = WhisperTokenizer.from_pretrained(str(WHISPER_DIR))
asr_processor = WhisperProcessor(feature_extractor=asr_feature_extractor, tokenizer=asr_tokenizer)
asr_model.eval()

if SUMMARY_USE_NORMALIZER and bn_normalize is None:
    raise RuntimeError(
        "The Bangla normalizer is not installed. In the venv run:\n"
        '  python -m pip install "emoji==1.7.0" ftfy regex\n'
        "  python -m pip install --no-deps git+https://github.com/csebuetnlp/normalizer"
    )
if not MT5_DIR.exists():
    raise RuntimeError(f"Summarizer model folder not found: {MT5_DIR.resolve()}")

summ_model = AutoModelForSeq2SeqLM.from_pretrained(str(MT5_DIR)).to(DEVICE)
summ_tokenizer = AutoTokenizer.from_pretrained(str(MT5_DIR), use_fast=False)
summ_model.eval()

print("[inference] Models loaded and ready.")


def clean_mt5_output(text: str) -> str:
    """Remove mT5 sentinel tokens like <extra_id_0> that survive decoding."""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Converts any uploaded audio (mp3/wav/webm/ogg) to 16 kHz mono via FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + ".wav"

    try:
        result = subprocess.run(
            [
                FFMPEG_PATH, "-y",
                "-i", tmp_in_path,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.decode(errors='ignore')[-500:]}")

        y, _ = librosa.load(tmp_out_path, sr=SAMPLE_RATE, mono=True)
    finally:
        for p in (tmp_in_path, tmp_out_path):
            if os.path.exists(p):
                os.remove(p)

    return y.astype(np.float32)


def _quietest_point(audio: np.ndarray, lo: int, hi: int) -> int:
    """Returns the sample index of the quietest 50 ms moment in audio[lo:hi].
    Used as a cut point when no clear pause is detected (e.g. background
    music or room noise) - the gap between words is still the quietest spot."""
    lo, hi = max(0, lo), min(len(audio), hi)
    frame = int(0.05 * SAMPLE_RATE)
    hop = int(0.01 * SAMPLE_RATE)
    if hi - lo <= frame:
        return (lo + hi) // 2
    rms = librosa.feature.rms(y=audio[lo:hi], frame_length=frame, hop_length=hop, center=False)[0]
    return lo + int(np.argmin(rms)) * hop + frame // 2


def _split_on_silence(audio: np.ndarray) -> list:
    """Splits audio into (start, end) sample ranges no longer than
    CHUNK_MAX_S. Prefers cutting at clear pauses; if there is none, cuts at
    the quietest moment near the limit so words are not sliced in half.
    A too-short final piece is re-balanced with the previous chunk so no
    chunk ever exceeds CHUNK_MAX_S. Chunks with no speech are dropped."""
    n = len(audio)
    max_len = int(CHUNK_MAX_S * SAMPLE_RATE)
    min_len = int(CHUNK_MIN_S * SAMPLE_RATE)
    search = int(CUT_SEARCH_S * SAMPLE_RATE)

    speech = librosa.effects.split(audio, top_db=SILENCE_TOP_DB)
    if len(speech) == 0:
        return []
    if n <= max_len:
        return [(0, n)]

    # Clear pauses: the midpoint between consecutive speech regions
    pauses = [int((speech[i][1] + speech[i + 1][0]) // 2) for i in range(len(speech) - 1)]

    segments = []
    seg_start = 0
    while n - seg_start > max_len:
        limit = seg_start + max_len
        in_range = [p for p in pauses if seg_start + min_len <= p <= limit]
        if in_range:
            cut = max(in_range)                      # latest clear pause before the limit
        else:
            cut = _quietest_point(audio, max(seg_start + min_len, limit - search), limit)
        segments.append((seg_start, cut))
        seg_start = cut
    segments.append((seg_start, n))

    # Re-balance a too-short final piece: split the last two pieces into two
    # roughly equal halves at a pause / quiet point near the middle
    if len(segments) > 1 and segments[-1][1] - segments[-1][0] < min_len:
        s, e = segments[-2][0], n
        mid = (s + e) // 2
        window = int(1.5 * SAMPLE_RATE)
        near = [p for p in pauses if mid - window <= p <= mid + window]
        cut = min(near, key=lambda p: abs(p - mid)) if near else _quietest_point(audio, mid - window, mid + window)
        segments[-2:] = [(s, cut), (cut, e)]

    def has_speech(s, e):
        return any(ss < e and s < se for ss, se in speech)

    return [(s, e) for s, e in segments if has_speech(s, e)]


def transcribe(audio: np.ndarray, num_beams: int = ASR_NUM_BEAMS,
               max_new_tokens: int = 440) -> str:
    """Transcribes audio of ANY length by splitting it into short chunks
    (cut at natural pauses), transcribing each chunk separately, and
    joining the results."""
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"[inference] Full audio duration: {duration_sec:.1f}s")

    segments = _split_on_silence(audio)
    if not segments:
        print("[inference] No speech detected.")
        return ""

    transcripts = []
    for idx, (s, e) in enumerate(segments, 1):
        print(f"[inference] Transcribing chunk {idx}/{len(segments)} "
              f"({s / SAMPLE_RATE:.1f}s - {e / SAMPLE_RATE:.1f}s)...")
        chunk_text = _transcribe_chunk(audio[s:e], num_beams, max_new_tokens)
        if chunk_text:
            transcripts.append(chunk_text)

    full_transcript = re.sub(r"\s+", " ", " ".join(transcripts)).strip()
    print(f"[inference] Full transcript assembled from {len(segments)} chunk(s).")
    return full_transcript


def _transcribe_chunk(audio_chunk: np.ndarray, num_beams: int, max_new_tokens: int) -> str:
    """Runs Whisper generation on a single audio chunk (<= ~30s)."""
    inputs = asr_processor(
        audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        pred_ids = asr_model.generate(
            inputs,
            language="bengali",
            task="transcribe",
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
    text = asr_processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return re.sub(r"\s+", " ", text).strip()


def summarize(text: str, num_beams: int = SUMMARY_NUM_BEAMS, max_input_tokens: int = 480) -> str:
    """Summarizes text of ANY length. The model accepts ~512 input tokens at
    once, so long transcripts are split into chunks, each chunk is
    summarized, and if the combined chunk-summaries are still too long they
    are summarized again (a 'map-reduce' strategy)."""

    if SUMMARY_USE_NORMALIZER:
        text = bn_normalize(text)

    token_count = len(summ_tokenizer(text)["input_ids"])
    print(f"[inference] Transcript token count: {token_count}")

    # Short enough for a single pass
    if token_count <= max_input_tokens:
        return _summarize_chunk(text, num_beams)

    # Long transcript: split into word-based chunks that stay under the token limit
    print("[inference] Transcript too long for one pass - chunking for summarization...")
    words = text.split()
    chunks = []
    current_chunk_words = []

    for word in words:
        current_chunk_words.append(word)
        candidate = " ".join(current_chunk_words)
        if len(summ_tokenizer(candidate)["input_ids"]) >= max_input_tokens:
            # back off the last word so this chunk stays under the limit
            current_chunk_words.pop()
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    print(f"[inference] Split into {len(chunks)} chunk(s) for summarization.")

    # Map step: summarize each chunk independently
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[inference] Summarizing chunk {i}/{len(chunks)}...")
        chunk_summary = _summarize_chunk(chunk, num_beams)
        if chunk_summary:
            chunk_summaries.append(chunk_summary)

    combined = " ".join(chunk_summaries)

    # Reduce step: if the combined chunk-summaries are still long, summarize
    # them again into one final summary
    combined_token_count = len(summ_tokenizer(combined)["input_ids"])
    if combined_token_count > max_input_tokens:
        print("[inference] Combined chunk summaries still long - running final reduce pass...")
        return _summarize_chunk(combined, num_beams)
    return combined


def _summarize_chunk(text: str, num_beams: int) -> str:
    """Runs generation on a single chunk of text (<= ~512 tokens).
    Same settings as the evaluation in the training notebook."""
    prompt = "summarize: " + text
    inputs = summ_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

    with torch.no_grad():
        summ_ids = summ_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=SUMMARY_MAX_TOKENS,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    summary = summ_tokenizer.decode(summ_ids[0], skip_special_tokens=True)
    return clean_mt5_output(summary)


def transcribe_and_summarize(audio_bytes: bytes) -> dict:
    """Full pipeline entry point: raw audio bytes -> {transcript, summary}."""
    audio = load_audio_bytes(audio_bytes)
    duration_sec = len(audio) / SAMPLE_RATE

    transcript = transcribe(audio)
    summary = summarize(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "duration_sec": round(duration_sec, 2),
    }


if __name__ == "__main__":
    # Quick local smoke test: python inference.py path/to/audio.mp3
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        result = transcribe_and_summarize(f.read())
    print(result)