"""
Redistribution rights of the development pages (CH-05; owner decision D-07, 2026-10-09).

The page manifest gives every development page a redistribution_status. Pages marked internal_development_only
(DEV-P009, DEV-P010, DEV-P016, DEV-P017) may be viewed and evaluated inside the project, but their IMAGES - whole
pages, crops, overlays, thumbnails - must not be put into anything that is shared: a report, a figure, a slide, an
export. Their METRICS remain reportable: the restriction is on the picture, not on the number, and excluding these pages
from the evaluation would change the estimand.

    redistribution_status(page_id)          the manifest's status; an unknown page raises (no default is assumed)
    image_shareable(page_id)                True only for a status on the SHAREABLE list; anything else, unknown pages and
                                            missing or misspelt statuses included, is not shareable (fail closed)
    assert_image_shareable(page_id, what)   raise RightsError before an image of the page goes into a shareable artifact
    assert_image_bytes_shareable(data)      raise if the bytes are an exact copy of a master image that is not shareable
    local_viewing_allowed(page_id)          True for every page of the manifest: viewing inside the project's own tools is not sharing
    metric_reportable(page_id)              True for every known page (the restriction does not reach metrics)

Every writer of a shareable artifact that embeds or exports a page image must call assert_image_shareable with the page
the image came from. A crop cannot be recognised from its bytes, so the caller has to say where the image came from;
assert_image_bytes_shareable catches only an unchanged master. Local viewing inside the project's own tools is not
sharing. Reads the manifest only; writes nothing.
"""
import csv
import hashlib
from functools import lru_cache

from pageocr import ROOT

MANIFEST = ROOT / "development_pages" / "manifests" / "development_pages_manifest.csv"
ORIGINAL = ROOT / "development_pages" / "original"
RESTRICTED = "internal_development_only"
SHAREABLE = frozenset({"public_domain", "open_license"})          # the statuses whose images may be shared


class RightsError(PermissionError):
    """An image of an internal-only page was about to go into a shareable artifact."""


@lru_cache(maxsize=None)
def _table() -> dict:
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return {r["page_id"]: {"status": (r.get("redistribution_status") or "").strip(), "filename": r["filename"]} for r in csv.DictReader(f)}


def redistribution_status(page_id: str) -> str:
    t = _table()
    if page_id not in t or not t[page_id]["status"]:
        raise KeyError(f"{page_id}: no redistribution status in the page manifest; none is assumed")
    return t[page_id]["status"]


def image_shareable(page_id: str) -> bool:
    """Fail closed: only a known page with a status on the SHAREABLE list."""
    try:
        return redistribution_status(page_id) in SHAREABLE
    except KeyError:
        return False


def assert_image_shareable(page_id: str, what: str = "an image") -> None:
    if not image_shareable(page_id):
        try:
            status = redistribution_status(page_id)
        except KeyError:
            status = "no recorded redistribution status"
        raise RightsError(f"Refused: {what} of {page_id} would go into a shareable artifact, but its status is '{status}', which is not "
                          f"on the shareable list {sorted(SHAREABLE)} (owner decision D-07: images of internal-only pages stay inside the "
                          "project; their metrics may be reported).")


def local_viewing_allowed(page_id: str) -> bool:
    redistribution_status(page_id)                               # a page outside the manifest is not a development page
    return True


@lru_cache(maxsize=None)
def _restricted_master_hashes() -> dict:
    out = {}
    for pid, r in _table().items():
        if r["status"] not in SHAREABLE and (ORIGINAL / r["filename"]).is_file():
            out[hashlib.sha256((ORIGINAL / r["filename"]).read_bytes()).hexdigest()] = pid
    return out


def assert_image_bytes_shareable(data: bytes, what: str = "an image") -> None:
    pid = _restricted_master_hashes().get(hashlib.sha256(data).hexdigest())
    if pid:
        assert_image_shareable(pid, what)


def metric_reportable(page_id: str) -> bool:
    redistribution_status(page_id)                               # unknown pages still raise
    return True


def restricted_pages() -> list:
    return sorted(p for p, r in _table().items() if r["status"] == RESTRICTED)
