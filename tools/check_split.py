"""
READ-ONLY audit of the imported IIIT splits (explains e.g. val = 11,598 vs official 11,597). Changes nothing.

  .venv\\Scripts\\python tools\\check_split.py --data data\\real\\iiit_hw --src external\\IIIT-INDIC-HW-WORDS-Tamil

Reports: rows per split in the source parquet vs the manifests, exact-duplicate images (by content hash)
within and across splits, repeated source file names, and unusual labels (spaces, Latin, digits only).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OFFICIAL = {"train": 75736, "val": 11597, "test": 16184}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", default=None, help="original parquet folder (optional, for row counts)")
    a = ap.parse_args()
    data = Path(a.data)

    if a.src:
        import pyarrow.parquet as pq
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.import_hf_parquet import find_shards, split_of
        src = Path(a.src)
        rows = defaultdict(int)
        print("source parquet shards:")
        for s in find_shards(src):
            n = pq.ParquetFile(s).metadata.num_rows
            rows[split_of(s, src)] += n
            print(f"  {s.relative_to(src)}  rows={n}")
        print("source rows per split:", dict(rows))

    by_hash, by_orig = defaultdict(list), defaultdict(list)
    for split in ("train", "val", "test"):
        mf = data / f"manifest_{split}.jsonl"
        if not mf.exists():
            continue
        entries = [json.loads(l) for l in open(mf, encoding="utf-8") if l.strip()]
        odd = []
        for i, e in enumerate(entries):
            h = hashlib.md5((data / e["image"]).read_bytes()).hexdigest()
            by_hash[h].append((split, i, e["image"], e["label"], e.get("orig", "")))
            if e.get("orig"):
                by_orig[(split, e["orig"])].append((i, e["image"], e["label"]))
            lab = e["label"]
            if " " in lab or re.search(r"[A-Za-z]", lab) or re.fullmatch(r"[0-9௦-௯.,\-]+", lab):
                odd.append((i, e["image"], lab))
        print(f"\n{split}: manifest {len(entries)}  (official {OFFICIAL[split]}, difference {len(entries) - OFFICIAL[split]:+d})")
        for i, img, lab in odd[:10]:
            print(f"  unusual label  row {i}  {img}  '{lab}'")

    dups = [v for v in by_hash.values() if len(v) > 1]
    within = [v for v in dups if len({x[0] for x in v}) == 1]
    across = [v for v in dups if len({x[0] for x in v}) > 1]
    print(f"\nidentical image files: {len(within)} groups within one split, {len(across)} groups across splits")
    # Every cross-split group is printed (they are possible train/test leakage); within-split groups are capped. Before
    # 2026-10-10 one cap of 15 covered both and the within-split groups filled it, so no cross-split group was ever shown.
    for v in across:
        print("  ACROSS SPLITS  " + " | ".join(f"{s} row {i} {img} '{lab}' orig={o}" for s, i, img, lab, o in v))
    for v in within[:15]:
        print("  within split   " + " | ".join(f"{s} row {i} {img} '{lab}' orig={o}" for s, i, img, lab, o in v))
    if len(within) > 15:
        print(f"  ... {len(within) - 15} more within-split groups not shown")
    rep = [(k, v) for k, v in by_orig.items() if len(v) > 1]
    print(f"repeated source file names inside a split: {len(rep)}")
    for (s, o), v in rep[:10]:
        print(f"  {s} orig={o}: " + ", ".join(f"row {i} '{lab}'" for i, _, lab in v))


if __name__ == "__main__":
    main()
