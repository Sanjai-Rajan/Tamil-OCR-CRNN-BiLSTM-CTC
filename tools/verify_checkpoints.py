"""
Verify the registered checkpoints (wordocr/checkpoints.py) on this computer: present, with the registered SHA-256,
and loadable by the project's own model code with weights_only=True.

    python tools/verify_checkpoints.py            all registered checkpoints
    python tools/verify_checkpoints.py --id 002a  one of them
    python tools/verify_checkpoints.py --no-load  hashes only (no PyTorch import)

Checkpoints are not stored in Git (each is about 127 MB; see MODEL_CARD.md and DATASETS.md). Put an authorised copy at
the registered path, then run this. Nothing is downloaded, and no other file is tried in place of a missing one.
Exit code: 0 if every checked checkpoint is present and verified, 1 if any has the wrong hash or fails to load,
2 if any is missing (and none is wrong).
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wordocr import checkpoints  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", nargs="+")
    ap.add_argument("--no-load", action="store_true")
    a = ap.parse_args()
    worst = 0
    for c in checkpoints.REGISTRY:
        if a.id and c.id not in a.id:
            continue
        p = c.file()
        if not p.is_file():
            print(f"MISSING  {c.id:9s} {c.path}")
            worst = max(worst, 2)
            continue
        got = sha(p)
        if got != c.sha256:
            print(f"WRONG    {c.id:9s} {c.path}: SHA-256 {got[:16]}..., registered {c.sha256[:16]}...")
            worst = 3
            continue
        note = "hash verified"
        if not a.no_load:
            try:
                from pageocr.engine import OcrEngine
                OcrEngine(p, device="cpu", expected_sha256=c.sha256)
                note += ", loads into CRNNv2 (weights_only=True, CPU)"
            except Exception as e:  # noqa: BLE001 - reported, not hidden
                print(f"LOADFAIL {c.id:9s} {c.path}: {type(e).__name__}: {str(e)[:160]}")
                worst = 3
                continue
        print(f"OK       {c.id:9s} {c.path}: {note}")
    return {0: 0, 2: 2, 3: 1}[worst]


if __name__ == "__main__":
    sys.exit(main())
