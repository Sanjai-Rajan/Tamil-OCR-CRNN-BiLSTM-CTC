"""
Import a HuggingFace-style dataset (parquet shards with an image column + a text column) into
our format: image files + one manifest_<split>.jsonl per split  {"image": ..., "label": ..., "source": ...}.

  # 1) look first (prints shards, columns, first rows; writes nothing)
  .venv\\Scripts\\python tools\\import_hf_parquet.py --src external\\IIIT-INDIC-HW-WORDS-Tamil --inspect

  # 2) import
  .venv\\Scripts\\python tools\\import_hf_parquet.py --src external\\IIIT-INDIC-HW-WORDS-Tamil ^
      --out data\\real\\iiit_hw --name iiit_hw

Image bytes are written unchanged (no re-encoding). Labels are NFC-normalised; labels with characters
outside src/charset.py are kept out of the manifest and listed in import_report.txt.
"""
import argparse
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import charset as cs  # noqa: E402

TEXT_NAMES = ("text", "label", "labels", "transcription", "ground_truth", "gt", "word", "sentence", "caption")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "dev": "val", "test": "test", "testing": "test"}
OFFICIAL = {"train": 75736, "val": 11597, "test": 16184}     # IIIT-INDIC-HW-WORDS Tamil


def find_shards(src: Path):
    return sorted(p for p in src.rglob("*.parquet") if ".cache" not in p.parts)


def split_of(path: Path, src: Path) -> str:
    parts = [p.lower() for p in path.relative_to(src).parts]
    for part in reversed(parts):
        stem = part.split(".")[0]
        for tok in re.split(r"[-_/]", stem):
            if tok in SPLIT_ALIASES:
                return SPLIT_ALIASES[tok]
    return "train"


def pick_columns(schema, want_img=None, want_txt=None):
    names = schema.names
    img = want_img
    if img is None:
        for n in names:
            t = schema.field(n).type
            if str(t).startswith("struct") and "bytes" in str(t):
                img = n; break
        if img is None:
            img = next((n for n in names if str(schema.field(n).type) in ("binary", "large_binary")), None)
    txt = want_txt
    if txt is None:
        txt = next((n for n in names if n.lower() in TEXT_NAMES), None)
        if txt is None:
            txt = next((n for n in names if n != img and str(schema.field(n).type) in ("string", "large_string")), None)
    return img, txt


def image_bytes(v):
    if v is None:
        return None, None
    if isinstance(v, dict):
        return v.get("bytes"), v.get("path")
    return v, None


def ext_of(b: bytes, hint: str | None):
    if hint and "." in hint:
        e = hint.rsplit(".", 1)[1].lower()
        if e in ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"):
            return e
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    return "png"


def inspect(src: Path, img_col, txt_col):
    shards = find_shards(src)
    print(f"{len(shards)} parquet shards under {src}")
    for s in shards[:30]:
        md = pq.ParquetFile(s).metadata
        print(f"  [{split_of(s, src):5}] {s.relative_to(src)}  rows={md.num_rows}")
    if not shards:
        others = Counter(p.suffix.lower() for p in src.rglob("*") if p.is_file() and ".cache" not in p.parts)
        print("no parquet found; file types present:", dict(others))
        return
    pf = pq.ParquetFile(shards[0])
    schema = pf.schema_arrow
    print("\nschema:")
    for f in schema:
        print(f"  {f.name}: {f.type}")
    img, txt = pick_columns(schema, img_col, txt_col)
    print(f"\nimage column -> {img}\ntext column  -> {txt}")
    tbl = pf.read_row_group(0).slice(0, 5).to_pylist()
    for r in tbl:
        show = {k: (f"<{len(image_bytes(v)[0] or b'')} bytes>" if k == img else v) for k, v in r.items()}
        print("  ", show)
    tot = defaultdict(int)
    for s in shards:
        tot[split_of(s, src)] += pq.ParquetFile(s).metadata.num_rows
    print("\nrows per split:", dict(tot))
    for k, v in OFFICIAL.items():
        if k in tot:
            print(f"  {k}: {tot[k]} (official IIIT Tamil {v}) {'OK' if tot[k] == v else 'DIFFERENT'}")


def do_import(src: Path, out: Path, name: str, img_col, txt_col, limit=0):
    shards = find_shards(src)
    if not shards:
        sys.exit(f"no parquet shards under {src}")
    out.mkdir(parents=True, exist_ok=True)
    manifests, counts = {}, Counter()
    bad_chars, bad_examples, empty, n_bad, lengths = Counter(), [], 0, 0, defaultdict(Counter)
    uniq = defaultdict(set)
    for s in shards:
        split = split_of(s, src)
        pf = pq.ParquetFile(s)
        img, txt = pick_columns(pf.schema_arrow, img_col, txt_col)
        if img is None or txt is None:
            sys.exit(f"could not find image/text columns in {s}: {pf.schema_arrow.names} (use --img-col/--txt-col)")
        if split not in manifests:
            (out / split).mkdir(exist_ok=True)
            manifests[split] = open(out / f"manifest_{split}.jsonl", "w", encoding="utf-8")
        for batch in pf.iter_batches(batch_size=2048, columns=[img, txt]):
            for r in batch.to_pylist():
                if limit and counts[split] >= limit:
                    break
                b, hint = image_bytes(r[img])
                raw = r[txt] if isinstance(r[txt], str) else (r[txt][0] if r[txt] else "")
                label = cs.normalize((raw or "").strip())
                if not b or not label:
                    empty += 1; continue
                if not cs.is_encodable(label):
                    n_bad += 1
                    miss = [c for c in label if not cs.is_encodable(c)]
                    bad_chars.update(miss)
                    if len(bad_examples) < 40:
                        bad_examples.append((label, " ".join(f"U+{ord(c):04X}({unicodedata.name(c, '?')})" for c in set(miss))))
                    continue
                i = counts[split]
                rel = f"{split}/{i:06d}.{ext_of(b, hint)}"
                (out / rel).write_bytes(b)
                manifests[split].write(json.dumps({"image": rel, "label": label, "source": name,
                                                   "orig": hint or ""}, ensure_ascii=False) + "\n")
                counts[split] += 1
                lengths[split][min(len(label), 20)] += 1
                uniq[split].add(label)
                if counts[split] % 10000 == 0:
                    print(f"  {split}: {counts[split]}", flush=True)
    for f in manifests.values():
        f.close()

    lines = [f"import of {src} -> {out}", ""]
    for sp in sorted(counts):
        n = counts[sp]
        L = lengths[sp]
        mean = sum(k * v for k, v in L.items()) / max(1, n)
        lines.append(f"{sp:5}: {n:6d} images, {len(uniq[sp]):6d} unique words, mean length {mean:.1f} chars, "
                     f">=12 chars: {sum(v for k, v in L.items() if k >= 12)}"
                     + (f"   (official {OFFICIAL[sp]})" if sp in OFFICIAL else ""))
    if "train" in uniq:
        for sp in ("val", "test"):
            if sp in uniq:
                ov = len(uniq[sp] & uniq["train"]) / max(1, len(uniq[sp]))
                lines.append(f"{sp} words also in train: {ov*100:.1f}%")
    lines.append(f"skipped empty rows: {empty}")
    lines.append(f"skipped labels with unsupported characters: {n_bad}")
    for c, k in bad_chars.most_common(20):
        lines.append(f"   U+{ord(c):04X} {unicodedata.name(c, '?'):40} x{k}")
    for lab, why in bad_examples[:15]:
        lines.append(f"   e.g. {lab}   {why}")
    rep = "\n".join(lines)
    (out / "import_report.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--name", default="iiit_hw")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--img-col"); ap.add_argument("--txt-col")
    ap.add_argument("--limit", type=int, default=0, help="debug: max rows per split")
    a = ap.parse_args()
    src = Path(a.src)
    if a.inspect or not a.out:
        inspect(src, a.img_col, a.txt_col)
    else:
        do_import(src, Path(a.out), a.name, a.img_col, a.txt_col, a.limit)


if __name__ == "__main__":
    main()
