"""
split_dataset.py
----------------
Splits the dataset into fixed train / validation / test sets (80 / 10 / 10)
and prepares files for training. Your original dataset is NOT changed.

Creates (in the output folder, default: .\\splits):
  train.txt, val.txt, test.txt   clip IDs in each set (keep these - the same
                                 split is used for BOTH the summarizer and Whisper)
  summarizer_data.jsonl          id, split, text, summary for every clip
                                 (small file - this is what you upload to train the summarizer)

With --test-folder, it also copies the TEST clips (speech + text + summary)
into a separate folder, ready for evaluate.py.

Run from the project folder (venv active):
    python split_dataset.py "D:\\NEW DATASET" --test-folder "D:\\TestSet"

The split is random but fixed (seed 42): running it again gives the same split.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".opus", ".wma", ".mp4"}


def sort_key(s):
    return (0, int(s)) if s.isdigit() else (1, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help='Folder with speech/, text/, summary/ (e.g. "D:\\NEW DATASET")')
    ap.add_argument("--out", default="splits", help="Output folder for split files (default: splits)")
    ap.add_argument("--test-folder", default=None, help='Copy test clips here for evaluate.py (e.g. "D:\\TestSet")')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset)
    sp, tx, sm = root / "speech", root / "text", root / "summary"
    audio = {p.stem: p for p in sp.iterdir() if p.suffix.lower() in AUDIO_EXTS}
    ids = sorted((set(audio) & {p.stem for p in tx.glob("*.txt")} & {p.stem for p in sm.glob("*.txt")}),
                 key=sort_key)
    if not ids:
        sys.exit("No complete speech/text/summary sets found.")

    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * 0.10)
    n_val = round(n * 0.10)
    splits = {
        "test": sorted(shuffled[:n_test], key=sort_key),
        "val": sorted(shuffled[n_test:n_test + n_val], key=sort_key),
        "train": sorted(shuffled[n_test + n_val:], key=sort_key),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        (out / f"{name}.txt").write_text("\n".join(members) + "\n", encoding="utf-8")

    split_of = {i: name for name, members in splits.items() for i in members}
    with open(out / "summarizer_data.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            rec = {
                "id": i,
                "split": split_of[i],
                "text": (tx / f"{i}.txt").read_text(encoding="utf-8-sig").strip(),
                "summary": (sm / f"{i}.txt").read_text(encoding="utf-8-sig").strip(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Total clips : {n}")
    for name in ("train", "val", "test"):
        print(f"{name:<6}: {len(splits[name])}")
    print(f"Saved split lists and summarizer_data.jsonl to {out.resolve()}")

    if args.test_folder:
        dest = Path(args.test_folder)
        for sub in ("speech", "text", "summary"):
            (dest / sub).mkdir(parents=True, exist_ok=True)
        for i in splits["test"]:
            shutil.copy2(audio[i], dest / "speech" / audio[i].name)
            shutil.copy2(tx / f"{i}.txt", dest / "text" / f"{i}.txt")
            shutil.copy2(sm / f"{i}.txt", dest / "summary" / f"{i}.txt")
        print(f"Copied {len(splits['test'])} test clips to {dest.resolve()}")


if __name__ == "__main__":
    main()