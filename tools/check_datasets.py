"""
Check locally prepared datasets against data/manifests/datasets.json (sizes, SHA-256, line counts, image counts).

    python tools/check_datasets.py                  every dataset of the manifest
    python tools/check_datasets.py --only word_list iiit_hw_converted
    python tools/check_datasets.py --quick          sizes and counts only (no hashing of the 5 GB of Parquet files)
    python tools/check_datasets.py --root <folder>  check a data tree kept elsewhere (same relative layout); only read

The datasets are NOT part of the repository (DATASETS.md explains why and how to prepare them). This tool reads only
the files at the paths named in the manifest; it never downloads anything. Per dataset it prints one of:
    OK          every file present with the recorded size and hash (and line / image counts)
    MISSING     a file is not there (not prepared yet)
    DIFFERENT   present, but size, hash or count differs: a different copy or version than the project used
Exit code: 0 if every checked dataset is OK, 1 if any is DIFFERENT, 2 if any is MISSING (and none DIFFERENT).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "datasets.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check(name: str, d: dict, quick: bool, root: Path = ROOT) -> tuple:
    problems, missing = [], []
    for f in d["files"]:
        p = root / f["file"]
        if not p.is_file():
            missing.append(f["file"])
            continue
        if p.stat().st_size != f["bytes"]:
            problems.append(f"{f['file']}: {p.stat().st_size} bytes, expected {f['bytes']}")
            continue
        if "lines" in f:
            n = sum(1 for _ in open(p, "rb"))
            if n != f["lines"]:
                problems.append(f"{f['file']}: {n} lines, expected {f['lines']}")
        if not quick and sha(p) != f["sha256"]:
            problems.append(f"{f['file']}: SHA-256 differs from the recorded one")
    img = d.get("images")
    if isinstance(img, dict) and "dir" in img:
        q = root / img["dir"]
        if not q.is_dir():
            missing.append(img["dir"] + "/")
        else:
            n = sum(1 for _ in q.glob("*.png"))
            if n != img["count"]:
                problems.append(f"{img['dir']}: {n} images, expected {img['count']}")
    status = "DIFFERENT" if problems else "MISSING" if missing else "OK"
    return status, problems, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding data/ and external/ (default: this repository)")
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    worst = 0
    for name, d in man["datasets"].items():
        if a.only and name not in a.only:
            continue
        status, problems, missing = check(name, d, a.quick, a.root.resolve())
        print(f"{status:9s} {name:20s} {d['required_for']}")
        for m in missing[:5]:
            print(f"          missing: {m}")
        if len(missing) > 5:
            print(f"          ... and {len(missing) - 5} more missing")
        for p in problems:
            print(f"          {p}")
        worst = max(worst, {"OK": 0, "MISSING": 2, "DIFFERENT": 3}[status])
    print("see DATASETS.md for how to prepare each dataset")
    return {0: 0, 2: 2, 3: 1}[worst]


if __name__ == "__main__":
    sys.exit(main())
