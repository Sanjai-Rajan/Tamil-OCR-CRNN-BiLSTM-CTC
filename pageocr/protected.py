"""
Which files are protected research data: one definition, independent of the user's home folder.

The recognizer must not be run on the project's frozen research data (the R1 pilot material) or on its development
pages. Those files are not part of this package. To recognise them anyway when someone uploads one, the package
carries their fingerprints:

    pageocr/protected_manifest.json     for every protected file: its size and SHA-256 (no names, no content);
                                        for every development master: its file SHA-256 and the digest of its decoded
                                        pixels (so a lossless re-encode is recognised too)

and, in addition, it indexes folders that hold protected files on this computer:

    <package>/real_docs, <package>/development_pages        if they exist (they are not shipped)
    every folder named in the environment variable TAMIL_OCR_PROTECTED_ROOTS (separated by the path-list separator)

Nothing here depends on HOME or USERPROFILE. FAIL CLOSED: if the manifest is missing, unreadable or altered, or if a
folder named in TAMIL_OCR_PROTECTED_ROOTS is not an existing folder, GuardUnavailable is raised and the caller must
refuse the input: an input is never accepted because it could not be classified. There is no switch that turns the
guard off. The manifest is written by release_tools/build_protected_manifest.py and is read-only here.

LIMITATION (unchanged): only exact file bytes, and for development masters exact decoded pixels, are recognised. A
lossy re-encode, a re-crop or any edit of protected material is not. Files added to the protected folders after the
manifest was built are covered only on a computer where those folders are indexed (see above).
"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from pageocr import ROOT

MANIFEST = Path(__file__).with_name("protected_manifest.json")
SCHEMA = "tamil-ocr-protected-manifest-1"
ENV_ROOTS = "TAMIL_OCR_PROTECTED_ROOTS"
LOCAL_ROOTS = (ROOT / "real_docs", ROOT / "development_pages")       # indexed when present; not shipped with the release
MANIFEST_HIT = "<protected manifest>"


class GuardUnavailable(RuntimeError):
    """Protected data cannot be recognised right now, so nothing may be accepted. The message says what is wrong."""


def body_digest(body: dict) -> str:
    """SHA-256 over the canonical JSON of the manifest without its own digest."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def read_manifest(path) -> dict:
    """Parse and verify a manifest file. Raises GuardUnavailable with the reason; never returns a partial manifest."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GuardUnavailable(f"the protected-data manifest is missing ({Path(path).name})") from None
    except Exception as e:
        raise GuardUnavailable(f"the protected-data manifest cannot be read ({type(e).__name__})") from None
    try:
        if doc.get("schema") != SCHEMA:
            raise ValueError("unknown schema")
        digest = doc.pop("digest")
        if body_digest(doc) != digest:
            raise ValueError("its digest does not match its content")
        files = {int(size): frozenset(hashes) for size, hashes in doc["files"].items()}
        masters = frozenset(doc["development_masters"])
        pixels = dict(doc["development_master_pixels"])
        if sum(len(v) for v in files.values()) != doc["file_count"] or not all(len(h) == 64 for v in files.values() for h in v):
            raise ValueError("the entry count does not match")
        if not masters <= frozenset(h for v in files.values() for h in v) or not set(pixels.values()) <= masters:
            raise ValueError("the development masters are not among the listed files")
    except GuardUnavailable:
        raise
    except Exception as e:
        raise GuardUnavailable(f"the protected-data manifest is not valid ({e})") from None
    return {"files": files, "development_masters": masters, "development_master_pixels": pixels, "file_count": doc["file_count"],
            "digest": digest, "created": doc.get("created", ""), "sources": doc.get("sources", [])}


@lru_cache(maxsize=1)
def _cached(path: str, size: int, mtime_ns: int) -> dict:
    return read_manifest(path)


def manifest() -> dict:
    """The verified manifest (cached while the file is unchanged). Raises GuardUnavailable."""
    try:
        st = MANIFEST.stat()
    except OSError:
        raise GuardUnavailable(f"the protected-data manifest is missing ({MANIFEST.name})") from None
    return _cached(str(MANIFEST), st.st_size, st.st_mtime_ns)


def manifest_match(data: bytes) -> bool:
    """True if a protected file with exactly these bytes is listed. Raises GuardUnavailable."""
    candidates = manifest()["files"].get(len(data))
    return bool(candidates) and hashlib.sha256(data).hexdigest() in candidates


def configured_roots() -> tuple:
    """Folders named in TAMIL_OCR_PROTECTED_ROOTS. Each must be an existing folder; otherwise GuardUnavailable."""
    raw = os.environ.get(ENV_ROOTS, "")
    roots = []
    for item in raw.split(os.pathsep):
        item = item.strip().strip('"')
        if not item:
            continue
        p = Path(item)
        if not p.is_absolute() or not p.is_dir():
            raise GuardUnavailable(f"{ENV_ROOTS} names a folder that is not an existing absolute folder path "
                                   f"(entry {len(roots) + 1}); protected files there cannot be recognised")
        roots.append(p)
    return tuple(roots)


def protected_roots() -> tuple:
    """Every folder indexed on this computer: the local project folders and the configured ones."""
    return LOCAL_ROOTS + configured_roots()


def describe() -> dict:
    """What the guard knows, for a status display. Never raises: a failure is reported in "error"."""
    try:
        m = manifest()
        roots = configured_roots()
        return {"available": True, "error": "", "manifest_files": m["file_count"], "development_masters": len(m["development_masters"]),
                "manifest_digest": m["digest"], "manifest_created": m["created"], "configured_roots": len(roots),
                "local_roots_present": [p.name for p in LOCAL_ROOTS if p.is_dir()]}
    except GuardUnavailable as e:
        return {"available": False, "error": str(e)}
