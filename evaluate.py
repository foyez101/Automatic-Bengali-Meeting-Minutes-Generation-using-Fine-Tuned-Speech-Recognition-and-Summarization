"""
evaluate.py
-----------
Runs every audio clip in a test folder through the same pipeline the web app
uses (inference.py) and measures:

  ASR (transcript)   : WER and CER against the reference transcript
  Summarization      : ROUGE-1, ROUGE-2, ROUGE-L (F1) against the reference summary
                       - "end-to-end": summary made from the model's own transcript
                       - "oracle":     summary made from the CORRECT transcript
                         (shows how good the summarizer is on its own, without ASR errors)

Expected folder layout (files matched by name, e.g. 1.mp3 <-> 1.txt):
    DATA_DIR/
        speech/   1.mp3  2.wav  ...
        text/     1.txt  2.txt  ...   (reference transcripts)
        summary/  1.txt  2.txt  ...   (reference summaries, optional)

Run from the project folder (so the model folders are found):
    python evaluate.py "D:\\Updated"
    python evaluate.py "D:\\Updated" --no-summary      (ASR only, faster)
    python evaluate.py "D:\\Updated" --limit 3         (quick test on 3 clips)
    python evaluate.py "D:\\Updated" --summary-only    (summarizer only, from the correct
                                                     transcripts - no ASR, runs in minutes)

Results are printed and saved to eval_results_<date_time>.csv (opens in Excel).
"""

import argparse
import csv
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import jiwer

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".opus", ".wma", ".mp4"}


# ============================================================
# Text normalization (applied to BOTH reference and model output)
# ============================================================
# Bangla text can store the same visible letter in different Unicode forms
# (e.g. য় as one character or as য + ়). Without normalizing, identical-looking
# words count as errors. Punctuation is removed so only words are scored.
_PUNCT = r"[।॥,\.\?\!;:\"'“”‘’\(\)\[\]\{\}\-–—…/\\|*~`@#$%^&_=+<>]"

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")   # zero-width (non-)joiners
    text = re.sub(_PUNCT, " ", text)
    text = text.lower()                                        # affects English words only
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# ROUGE (implemented here because the common `rouge-score` library
# silently deletes all non-English characters, which gives Bangla a score of 0)
# ============================================================
def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def _f1(overlap, ref_count, hyp_count):
    if overlap == 0 or ref_count == 0 or hyp_count == 0:
        return 0.0
    p, r = overlap / hyp_count, overlap / ref_count
    return 2 * p * r / (p + r)

def rouge_n(ref: str, hyp: str, n: int) -> float:
    from collections import Counter
    r, h = Counter(_ngrams(ref.split(), n)), Counter(_ngrams(hyp.split(), n))
    overlap = sum((r & h).values())
    return _f1(overlap, sum(r.values()), sum(h.values()))

def rouge_l(ref: str, hyp: str) -> float:
    a, b = ref.split(), hyp.split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return _f1(prev[-1], len(a), len(b))


def read_text(path: Path) -> str:
    # utf-8-sig also handles files saved by Notepad with a BOM
    return path.read_text(encoding="utf-8-sig").strip()


def sort_key(p: Path):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def pct(x):
    return f"{x * 100:5.1f}%"


def summary_only(audio_files, text_dir, summ_dir):
    """Summarizes the CORRECT transcripts (no ASR) and scores them with ROUGE."""
    print(f"Summary-only mode: {len(audio_files)} clip(s). Loading models...")
    import inference

    rows = []
    start = time.time()
    for i, audio_path in enumerate(audio_files, 1):
        stem = audio_path.stem
        t_path, s_path = text_dir / f"{stem}.txt", summ_dir / f"{stem}.txt"
        if not t_path.exists() or not s_path.exists():
            print(f"[{i}/{len(audio_files)}] {stem}: missing text or summary - skipped")
            continue
        transcript = read_text(t_path)
        ref_sum = normalize(read_text(s_path))
        summary = inference.summarize(transcript)
        s_n = normalize(summary)
        row = {
            "file": audio_path.name,
            "ref_summary_words": len(ref_sum.split()),
            "summary_words": len(s_n.split()),
            "R1": round(rouge_n(ref_sum, s_n, 1), 4),
            "R2": round(rouge_n(ref_sum, s_n, 2), 4),
            "RL": round(rouge_l(ref_sum, s_n), 4),
            "summary": summary,
            "reference_summary": read_text(s_path),
        }
        rows.append(row)
        print(f"[{i}/{len(audio_files)}] {audio_path.name}: {row['summary_words']} words "
              f"(reference {row['ref_summary_words']})  ROUGE-L {pct(row['RL'])}")

    if not rows:
        sys.exit("Nothing was evaluated.")
    avg = lambda k: sum(r[k] for r in rows) / len(rows)
    print("\n" + "=" * 64)
    print(f"{'File':<14}{'Words':>8}{'Ref words':>11}{'R-1':>9}{'R-2':>9}{'R-L':>9}")
    print("-" * 64)
    for r in rows:
        print(f"{r['file']:<14}{r['summary_words']:>8}{r['ref_summary_words']:>11}"
              f"{pct(r['R1']):>9}{pct(r['R2']):>9}{pct(r['RL']):>9}")
    print("-" * 64)
    print(f"Average ROUGE-1/2/L : {pct(avg('R1'))} / {pct(avg('R2'))} / {pct(avg('RL'))}")
    print(f"Total time          : {(time.time() - start) / 60:.1f} min")

    out = Path(f"summary_results_{datetime.now():%Y%m%d_%H%M}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved detailed results to {out.resolve()}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate the Bangla ASR + summarization pipeline.")
    ap.add_argument("data_dir", help='Folder containing speech/, text/ and summary/ (e.g. "D:\\Updated")')
    ap.add_argument("--no-summary", action="store_true", help="Only evaluate transcripts (faster)")
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate the first N clips")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip ASR; summarize the reference transcripts and score against reference summaries")
    args = ap.parse_args()

    data = Path(args.data_dir)
    speech_dir, text_dir, summ_dir = data / "speech", data / "text", data / "summary"
    if not speech_dir.is_dir() or not text_dir.is_dir():
        sys.exit(f"Could not find 'speech' and 'text' folders inside {data}")

    audio_files = sorted([p for p in speech_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS], key=sort_key)
    if args.limit:
        audio_files = audio_files[:args.limit]
    if not audio_files:
        sys.exit(f"No audio files found in {speech_dir}")

    do_summary = not args.no_summary and summ_dir.is_dir()
    if args.summary_only:
        if not summ_dir.is_dir():
            sys.exit(f"--summary-only needs a 'summary' folder inside {data}")
        return summary_only(audio_files, text_dir, summ_dir)

    print(f"Found {len(audio_files)} audio file(s). Loading models (this takes a minute)...")
    import inference   # loads both models once

    rows, all_refs, all_hyps = [], [], []
    total_start = time.time()

    for i, audio_path in enumerate(audio_files, 1):
        stem = audio_path.stem
        ref_path = text_dir / f"{stem}.txt"
        if not ref_path.exists():
            print(f"[{i}/{len(audio_files)}] {audio_path.name}: no text/{stem}.txt - skipped")
            continue

        print(f"\n[{i}/{len(audio_files)}] {audio_path.name}")
        t0 = time.time()
        audio = inference.load_audio_bytes(audio_path.read_bytes())
        duration = len(audio) / inference.SAMPLE_RATE
        hyp_raw = inference.transcribe(audio)
        asr_time = time.time() - t0

        ref_raw = read_text(ref_path)
        ref_n, hyp_n = normalize(ref_raw), normalize(hyp_raw)
        wer = jiwer.wer(ref_n, hyp_n) if hyp_n else 1.0
        cer = jiwer.cer(ref_n, hyp_n) if hyp_n else 1.0
        all_refs.append(ref_n)
        all_hyps.append(hyp_n if hyp_n else "<empty>")

        row = {
            "file": audio_path.name,
            "duration_sec": round(duration, 1),
            "ref_words": len(ref_n.split()),
            "hyp_words": len(hyp_n.split()),
            "WER": round(wer, 4),
            "CER": round(cer, 4),
            "asr_time_sec": round(asr_time, 1),
        }
        print(f"    WER {pct(wer)}   CER {pct(cer)}   ({duration:.0f}s audio, {asr_time:.0f}s to transcribe)")

        if do_summary and (summ_dir / f"{stem}.txt").exists():
            ref_sum = normalize(read_text(summ_dir / f"{stem}.txt"))
            sum_e2e = inference.summarize(hyp_raw) if hyp_n else ""
            sum_oracle = inference.summarize(ref_raw)
            e2e_n, orc_n = normalize(sum_e2e), normalize(sum_oracle)
            row.update({
                "R1_e2e": round(rouge_n(ref_sum, e2e_n, 1), 4),
                "R2_e2e": round(rouge_n(ref_sum, e2e_n, 2), 4),
                "RL_e2e": round(rouge_l(ref_sum, e2e_n), 4),
                "R1_oracle": round(rouge_n(ref_sum, orc_n, 1), 4),
                "R2_oracle": round(rouge_n(ref_sum, orc_n, 2), 4),
                "RL_oracle": round(rouge_l(ref_sum, orc_n), 4),
                "summary_e2e": sum_e2e,
                "summary_oracle": sum_oracle,
            })
            print(f"    ROUGE-L  end-to-end {pct(row['RL_e2e'])}   oracle {pct(row['RL_oracle'])}")

        row["reference"] = ref_raw
        row["transcript"] = hyp_raw
        rows.append(row)

    if not rows:
        sys.exit("Nothing was evaluated.")

    # ---------------- Summary report ----------------
    n = len(rows)
    corpus_wer = jiwer.wer(all_refs, all_hyps)
    corpus_cer = jiwer.cer(all_refs, all_hyps)
    avg = lambda k: sum(r[k] for r in rows if k in r) / max(1, sum(1 for r in rows if k in r))

    print("\n" + "=" * 64)
    print(f"{'File':<14}{'Dur(s)':>8}{'WER':>9}{'CER':>9}", end="")
    print(f"{'RL e2e':>10}{'RL orac':>10}" if do_summary else "")
    print("-" * 64)
    for r in rows:
        print(f"{r['file']:<14}{r['duration_sec']:>8}{pct(r['WER']):>9}{pct(r['CER']):>9}", end="")
        print(f"{pct(r['RL_e2e']):>10}{pct(r['RL_oracle']):>10}" if "RL_e2e" in r else "")
    print("-" * 64)
    print(f"Clips evaluated        : {n}")
    print(f"Overall WER (corpus)   : {pct(corpus_wer)}    average per clip: {pct(avg('WER'))}")
    print(f"Overall CER (corpus)   : {pct(corpus_cer)}    average per clip: {pct(avg('CER'))}")
    if do_summary and any("RL_e2e" in r for r in rows):
        print(f"ROUGE-1/2/L end-to-end : {pct(avg('R1_e2e'))} / {pct(avg('R2_e2e'))} / {pct(avg('RL_e2e'))}")
        print(f"ROUGE-1/2/L oracle     : {pct(avg('R1_oracle'))} / {pct(avg('R2_oracle'))} / {pct(avg('RL_oracle'))}")
    print(f"Total time             : {(time.time() - total_start) / 60:.1f} min")

    # ---------------- Save CSV ----------------
    out = Path(f"eval_results_{datetime.now():%Y%m%d_%H%M}.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig so Excel shows Bangla
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        w.writerow({"file": "OVERALL (corpus)", "WER": round(corpus_wer, 4), "CER": round(corpus_cer, 4)})
    print(f"Saved detailed results to {out.resolve()}")


if __name__ == "__main__":
    main()