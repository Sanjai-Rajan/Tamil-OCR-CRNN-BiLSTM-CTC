"""
Research-data guard for the word frontend: no recognizer output on frozen research data or on a development page.

The project forbids running the recognizer on
  * frozen research data: real_docs/ and the R1 review folders (fingerprints in pageocr/protected_manifest.json, plus
    the folders indexed on this computer: pageocr.protected), and
  * development pages before their human-verified set is frozen (pageocr.devguard; every path that could run the
    recognizer on a development page must ask it).
This module applies both rules to every upload, at upload time and again right before recognition:

  check_bytes(data)          refuses a file byte-identical to any file under PROTECTED_ROOTS or development_pages/
                             (masters, previews, crops, sheets): pageocr.imageio.protected_match
  check_image(sha256, gray)  pageocr.devguard.assert_ocr_allowed on the file hash and on the decoded pixels
                             (devguard.development_page_for_pixels: a bare array or lossless re-encode of a master)

LIMITATION, as for the page tools: a lossy re-encode, a re-crop or any edit of protected material is not recognised.
Do not upload research data. Nothing here loads a model or writes a file; there is no switch that turns it off.

FAIL CLOSED: if the fingerprints cannot be read, or a folder configured in TAMIL_OCR_PROTECTED_ROOTS does not exist,
every upload is refused with GuardUnavailableError. The guard does not depend on HOME / USERPROFILE.
"""
from functools import lru_cache

from pageocr import ROOT, devguard, protected
from pageocr import imageio as pio

ROOTS = tuple(pio.PROTECTED_ROOTS) + (ROOT / "development_pages",)


class ProtectedDataError(ValueError):
    """The upload is protected research data. str(error) is a complete, user-facing message."""


class GuardUnavailableError(ProtectedDataError):
    """Protected data cannot be recognised right now, so the upload is refused. str(error) is user-facing."""


def _unavailable(e) -> GuardUnavailableError:
    return GuardUnavailableError("The research-data guard cannot check this file, so it is not accepted: "
                                 f"{e}. Nothing was recognized.")


@lru_cache(maxsize=8)
def _index_for(configured: tuple) -> dict:
    return pio.protected_size_index(ROOTS + configured)     # file sizes only; files of a matching size are hashed on demand


def _size_index() -> dict:
    return _index_for(protected.configured_roots())         # raises GuardUnavailable if a configured folder is missing


def check_bytes(data: bytes) -> None:
    try:
        hit = pio.protected_match(data, _size_index())
    except protected.GuardUnavailable as e:
        raise _unavailable(e) from None
    if hit is not None:
        raise ProtectedDataError("This file is part of the project's protected research data (real_docs, the R1 review "
                                 "folders or development_pages) and is not accepted here. Nothing was recognized.")


def check_image(sha256: str, gray) -> None:
    try:
        devguard.assert_ocr_allowed(sha256)
        devguard.assert_ocr_allowed(devguard.development_page_for_pixels(gray))
    except protected.GuardUnavailable as e:
        raise _unavailable(e) from None
    except devguard.DevelopmentBlocked:
        raise ProtectedDataError("This image is a development page of the project's research. The recognizer may not "
                                 "be run on it until the human verification of the development pages is complete and "
                                 "frozen. Nothing was recognized.") from None
