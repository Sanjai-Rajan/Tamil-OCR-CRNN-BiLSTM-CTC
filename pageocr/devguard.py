"""
The development guard: no recognizer output on a development page before the human-verified set is frozen.

The development pages are being verified by a person, and the recognizer must not be run on them until that work is
finished and frozen; otherwise its output could influence the reference it is later scored against. This module is
the one place that decides whether that moment has come. Every path that could run the recognizer on a development
page asks it: pageocr.pipeline.recognize, the page application, and the evaluation tools.

    status()                  every condition, with what is missing
    assert_ready()            raise DevelopmentBlocked unless every condition holds
    assert_ocr_allowed(sha)   raise DevelopmentBlocked if the page with this SHA-256 is a development page and the
                              conditions do not hold; pages that are not development pages are not this guard's concern
    development_page_for_pixels(gray)
                              the master's SHA-256 if a decoded array has exactly a development master's pixels (so an
                              array or a lossless re-encode is identified too), else ""

It reads files and hashes them. It loads no model and writes nothing. There is no switch that turns it off.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from pageocr import ROOT, protected

DEV = ROOT / "development_pages"
VERIFIED = DEV / "annotations" / "human_verified01"
QUEUE = DEV / "annotations" / "human_verify01" / "queue.json"
CHECKPOINT = ROOT / "runs" / "experiment_002a" / "best.pth"
FROZEN_002A_SHA256 = "fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1"
VERIFIED_FILES = ("verified_records.jsonl", "manifest.csv", "summary.json", "README.md", "integrity.json")


class DevelopmentBlocked(RuntimeError):
    """The recognizer may not be run on development pages yet. The message lists exactly what is missing."""


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def development_master_hashes() -> dict:
    """SHA-256 -> name of every development master: the fingerprints in pageocr/protected_manifest.json, and every image
    in development_pages/original if that folder is present (excluded pages too: they are development pages).
    Raises pageocr.protected.GuardUnavailable if the fingerprints cannot be read (fail closed)."""
    d = DEV / "original"
    out = {h: "development master (fingerprint)" for h in protected.manifest()["development_masters"]}
    if d.is_dir():
        out.update({sha_file(p): p.name for p in sorted(d.iterdir()) if p.is_file()})
    return out


def is_development_page(page_sha256: str) -> bool:
    return page_sha256 in development_master_hashes()


_PIXEL_INDEX: dict = {}


def pixel_digest(gray) -> str:
    """SHA-256 of a decoded grayscale array (shape and pixels), independent of the file it came from."""
    import numpy as np
    a = np.ascontiguousarray(gray)
    return hashlib.sha256(f"{a.shape}|{a.dtype}|".encode() + a.tobytes()).hexdigest()


def development_page_for_pixels(gray) -> str:
    """The file SHA-256 of the development master whose decoded pixels equal `gray`, or "" if none does.

    This identifies a development page that reaches the recognizer WITHOUT its original file bytes: as a bare array
    (cv2.imread of the master) or as a lossless re-encode (the master saved again as PNG). A lossy re-encode or any
    edit of the pixels is not recognised. The masters are decoded once per process with the project's page loader.
    """
    listed = protected.manifest()["development_master_pixels"].get(pixel_digest(gray), "")     # fail closed: may raise
    if listed:
        return listed
    d = DEV / "original"
    key = tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir() if p.is_file())) if d.is_dir() else ()
    if _PIXEL_INDEX.get("key") != key:
        from pageocr import imageio
        index = {}
        for p in sorted(d.iterdir()) if d.is_dir() else []:
            if not p.is_file():
                continue
            try:
                page = imageio.load_page(p)
            except Exception:              # e.g. the binarized DEV-P012: the page loader refuses it, so nothing reaches OCR
                continue
            index[pixel_digest(page.gray)] = page.sha256
        _PIXEL_INDEX.clear()
        _PIXEL_INDEX.update(key=key, index=index)
    return _PIXEL_INDEX["index"].get(pixel_digest(gray), "")


def _lock_differences() -> list:
    tool = DEV / "human_verification_tool"
    if str(tool) not in sys.path:
        sys.path.insert(0, str(tool))
    try:
        import hv_records
        return hv_records.verify_lock()
    except Exception as e:                                     # the lock cannot be checked: that is a failed condition, not a pass
        return [f"the protocol lock could not be checked ({type(e).__name__}: {e})"]


def status() -> dict:
    """Every condition for running the recognizer on development pages. {"ready": bool, "conditions": [...], "missing": [...]}"""
    cond = []

    def add(name, ok, missing):
        cond.append({"condition": name, "ok": bool(ok), "missing": "" if ok else missing})

    add("the verified set exists", VERIFIED.is_dir(), f"{VERIFIED} does not exist: the human review has not been frozen")
    have = VERIFIED.is_dir() and all((VERIFIED / n).is_file() for n in VERIFIED_FILES + ("SHA256SUMS.txt",))
    add("the verified manifest and its files exist", have, "one or more of " + ", ".join(VERIFIED_FILES + ("SHA256SUMS.txt",)) + " is missing")
    total = None
    try:
        total = len(json.loads(QUEUE.read_text(encoding="utf-8"))["items"])
    except Exception:
        pass
    n_rec, frozen_ok, complete = None, False, False
    if have:
        try:
            sums = dict(reversed(l.split("  ", 1)) for l in (VERIFIED / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip())
            frozen_ok = set(sums) == set(VERIFIED_FILES) and all(sha_file(VERIFIED / n) == h for n, h in sums.items())
            recs = [json.loads(l) for l in (VERIFIED / "verified_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            with open(VERIFIED / "manifest.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            n_rec = len(recs)
            integ = json.loads((VERIFIED / "integrity.json").read_text(encoding="utf-8"))
            complete = (total is not None and n_rec == total == len(rows) == integ.get("records") and len({r["region_id"] for r in recs}) == n_rec
                        and all(r.get("reviewer_id") and r.get("final_state") in ("ACCEPT", "CORRECT", "AMBIGUOUS", "UNREADABLE") for r in recs))
        except Exception:
            frozen_ok, complete = False, False
    add("the review is complete", complete, f"the verified set must hold one record for each of the {total if total is not None else '?'} queued regions (found {n_rec if n_rec is not None else 'none'})")
    add("the verified set is frozen", frozen_ok, "the files of the verified set do not match its SHA256SUMS.txt")
    diff = _lock_differences()
    add("the protocol lock still matches", not diff, "; ".join(diff[:6]))
    ck = CHECKPOINT.is_file() and sha_file(CHECKPOINT) == FROZEN_002A_SHA256
    add("the Experiment 002-A checkpoint is the frozen one", ck, f"{CHECKPOINT} is missing or its SHA-256 is not {FROZEN_002A_SHA256}")
    missing = [c["missing"] for c in cond if not c["ok"]]
    return {"ready": not missing, "conditions": cond, "missing": missing}


def explain(st: dict = None) -> str:
    st = st or status()
    if st["ready"]:
        return "Every condition holds: the recognizer may be run on the development pages."
    return ("The recognizer may NOT be run on the development pages yet. Missing:\n" + "\n".join("  - " + m for m in st["missing"])
            + "\nThe one step that produces what is missing is the human review: development_pages\\human_verification_tool\\run_human_verification.cmd")


def assert_ready() -> None:
    st = status()
    if not st["ready"]:
        raise DevelopmentBlocked(explain(st))


def assert_ocr_allowed(page_sha256: str) -> None:
    if is_development_page(page_sha256):
        assert_ready()
