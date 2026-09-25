"""
check_dataset.py
----------------
Checks a speech / text / summary dataset BEFORE training. Changes nothing.

Reports:
  - clips missing a transcript or summary (and orphan text files)
  - empty files and files that are not valid UTF-8
  - length statistics for transcripts and summaries
  - SUSPICIOUS pairs: transcript shorter than or about the same length as
    its summary (a sign the transcript is actually a summary / condensed text,
    or that the two files were swapped)
  - how many transcripts would exceed the summarizer's 512-token input limit
    (rough estimate)

Run from the project folder (venv active):
    python check_dataset.py "D:\\path\\to\\dataset"
The dataset folder must contain speech/, text/ and summary/.
"""

import sys
import re
from pathlib import Path
from statistics import mean, median

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".opus", ".wma", ".mp4"}


def words(s):
    return len(re.sub(r"[।॥,\.\?\!;:\"'“”‘’\(\)\-–—…]", " ", s).split())


def read(path, bad):
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        bad.append(path.name)
        return None


def stats(name, vals):
    if not vals:
        return
    v = sorted(vals)
    print(f"  {name:<22} min {v[0]:>5}   median {median(v):>7.0f}   mean {mean(v):>7.0f}   max {v[-1]:>5}")


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python check_dataset.py "D:\\path\\to\\dataset"')
    root = Path(sys.argv[1])
    sp, tx, sm = root / "speech", root / "text", root / "summary"
    for d in (sp, tx, sm):
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}")

    audio = {p.stem: p for p in sp.iterdir() if p.suffix.lower() in AUDIO_EXTS}
    texts = {p.stem: p for p in tx.glob("*.txt")}
    summs = {p.stem: p for p in sm.glob("*.txt")}

    print(f"Audio files     : {len(audio)}")
    print(f"Transcript files: {len(texts)}")
    print(f"Summary files   : {len(summs)}")

    no_text = sorted(set(audio) - set(texts))
    no_summ = sorted(set(audio) - set(summs))
    orphan = sorted((set(texts) | set(summs)) - set(audio))
    complete = sorted(set(audio) & set(texts) & set(summs))
    print(f"Complete sets   : {len(complete)}")

    def show(label, items):
        if items:
            print(f"\n{label} ({len(items)}): {', '.join(items[:20])}{' ...' if len(items) > 20 else ''}")

    show("Audio without transcript", no_text)
    show("Audio without summary", no_summ)
    show("Text/summary files without audio", orphan)

    bad, empty, suspicious = [], [], []
    t_words, s_words, ratios, long_inputs = [], [], [], 0
    for stem in complete:
        t = read(texts[stem], bad)
        s = read(summs[stem], bad)
        if t is None or s is None:
            continue
        if not t or not s:
            empty.append(stem)
            continue
        tw, sw = words(t), words(s)
        t_words.append(tw)
        s_words.append(sw)
        ratios.append(sw / tw)
        if tw <= sw * 1.2:
            suspicious.append(f"{stem} (text {tw} words, summary {sw} words)")
        if len(t) > 1300:          # rough: ~1300 Bangla characters is about 512 tokens
            long_inputs += 1

    show("Not valid UTF-8", bad)
    show("Empty transcript or summary", empty)

    print("\nLength statistics (words):")
    stats("Transcripts", t_words)
    stats("Summaries", s_words)
    if ratios:
        print(f"  Summary / transcript length ratio: median {median(ratios):.2f}")

    print(f"\nTranscripts likely longer than the summarizer's 512-token input: ~{long_inputs}")

    if suspicious:
        print(f"\nSUSPICIOUS - transcript not clearly longer than its summary ({len(suspicious)}):")
        for s in suspicious[:30]:
            print("  " + s)
        if len(suspicious) > 30:
            print(f"  ... and {len(suspicious) - 30} more")
        print("  -> These may be condensed texts or swapped files. Check them before training.")
    else:
        print("\nNo suspicious pairs: every transcript is clearly longer than its summary.")


if __name__ == "__main__":
    main()