"""
Synthetic printed Tamil pages for the full-page OCR tests: real Tamil words rendered with the project's own renderer
(src/synth: HarfBuzz shaping + the fonts in assets/fonts), placed on a page with known word boxes and text.

    page = make_tamil_page(seed=0, columns=1, lines=12)
    page.gray            2-D uint8 page (dark ink on light paper, anti-aliased)
    page.words           [TruthWord(block, line, index, text, box=(x1, y1, x2, y2))] in reading order
    page.text            the reference text: words joined by spaces, lines by newlines, blocks by a blank line

Generated in memory; nothing is read from data/, real_docs/ or development_pages/. The word list is a fixed set of
common Tamil words written here, not a corpus or a test set.
"""
import importlib.util
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RENDERING = all(importlib.util.find_spec(m) is not None for m in ("uharfbuzz", "freetype", "fontTools"))
SKIP_REASON = "the page fixtures need the rendering libraries (uharfbuzz, freetype-py); run these tests with the project's .venv"

WORDS = ("தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்றாகும் இந்த நூல் மாணவர்களுக்கு எளிய முறையில் "
         "பாடங்களைக் கற்பிக்கிறது பள்ளிக்கூடம் ஆசிரியர் புத்தகம் கணினி அறிவியல் கணக்கு வரலாறு புவியியல் "
         "நாம் அனைவரும் ஒன்றாகப் படிக்கிறோம் மழை பெய்தது மரங்கள் வளர்ந்தன நீர் நிலைகள் நிரம்பின "
         "விவசாயிகள் மகிழ்ச்சி அடைந்தனர் நமது நாடு பல மொழிகள் பேசும் மக்களைக் கொண்டது அறம் பொருள் இன்பம் "
         "திருக்குறள் வள்ளுவர் எழுதிய நூல் கல்வி கற்க வேண்டும் சுற்றுச்சூழல் பாதுகாப்பு நமது கடமை "
         "அரசு அலுவலகம் மருத்துவமனை சாலை பேருந்து நிலையம் தொடர்வண்டி விமானம் கப்பல் துறைமுகம்").split()


@dataclass(frozen=True)
class TruthWord:
    block: int
    line: int
    index: int
    text: str
    box: tuple          # (x1, y1, x2, y2) exclusive, tight to the rendered ink


@dataclass(frozen=True)
class TamilPage:
    gray: np.ndarray
    words: tuple
    text: str
    font: str
    size: int


@lru_cache(maxsize=1)
def _fonts():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.synth.fonts import discover_fonts
    return {Path(s.path).stem: s for s in discover_fonts([ROOT / "assets" / "fonts"])}


def render_word(text: str, font: str = "NotoSansTamil-Regular", size: int = 34) -> np.ndarray:
    """uint8 coverage mask (255 = ink), tight around the shaped word."""
    from src.synth.render import render_text_mask
    specs = _fonts()
    spec = specs.get(font) or next(v for k, v in sorted(specs.items()) if k.lower().startswith(font.lower().split("-")[0]))
    return render_text_mask(text, spec, size)


def font_names() -> list:
    return sorted(_fonts())


def make_tamil_page(seed: int = 0, columns: int = 1, lines: int = 12, size: int = 34, font: str = "NotoSansTamil-Regular",
                    width: int = 1600, margin: int = 90, word_space: float = 0.42, line_pitch: float = 2.0,
                    heading: bool = False, skew_deg: float = 0.0, noise: float = 3.0, blur: bool = True,
                    ink: int = 30, paper: int = 238, text_lines=None, heading_size: int | None = None,
                    figure: bool = False) -> TamilPage:
    """`text_lines`: the lines to set (strings of space-separated words) instead of random words (one column).
    `heading_size`: font size of the heading line. `figure`: a dark picture box in the right third, text beside it."""
    rng = np.random.default_rng(seed)
    pitch = int(size * line_pitch)
    col_gap = int(size * 1.6)
    if text_lines is not None:
        columns, lines = 1, len(text_lines)
    fig_w = int(width * 0.3) if figure else 0
    col_w = (width - 2 * margin - (columns - 1) * col_gap - fig_w - (col_gap if figure else 0)) // columns
    height = margin * 2 + pitch * (lines + (2 if heading else 0)) + size
    cover = np.zeros((height, width), np.float32)
    truth, block, y0 = [], 0, margin

    def set_line(x0, x1, y, n_words, blk, li, fixed=None, fsize=None):
        x, out = x0, []
        fsize = fsize or size
        for wi in range(len(fixed) if fixed is not None else n_words):
            t = fixed[wi] if fixed is not None else WORDS[int(rng.integers(len(WORDS)))]
            m = render_word(t, font, fsize)
            mh, mw = m.shape
            if x + mw > x1:
                break
            top = y + size - int(mh * 0.62)                    # a rough common baseline
            top = max(0, min(height - mh, top))
            region = cover[top:top + mh, x:x + mw]
            np.maximum(region, m.astype(np.float32) / 255.0, out=region)
            ys, xs = np.nonzero(m)
            out.append(TruthWord(blk, li, len(out), t, (x + int(xs.min()), top + int(ys.min()), x + int(xs.max()) + 1, top + int(ys.max()) + 1)))
            x += mw + int(fsize * word_space * float(rng.uniform(0.85, 1.25)))
        return out

    if heading:
        truth += set_line(margin + col_w // 3, width - margin, y0, 3, block, 0, fsize=heading_size)
        block += 1
        y0 += pitch * 2
    for c in range(columns):
        x0 = margin + c * (col_w + col_gap)
        for li in range(lines):
            truth += set_line(x0, x0 + col_w, y0 + li * pitch, 12, block, li,
                              fixed=text_lines[li].split() if text_lines is not None else None)
        block += 1
    img = paper - cover * (paper - ink)
    if figure:                                        # a picture: dark frame, mid-gray fill, a light disc
        fx1, fy1 = width - margin - fig_w, margin
        fx2, fy2 = width - margin, min(height - margin, margin + fig_w)
        cv2.rectangle(img, (fx1, fy1), (fx2, fy2), 60, -1)
        cv2.rectangle(img, (fx1 + 6, fy1 + 6), (fx2 - 6, fy2 - 6), 150, -1)
        cv2.circle(img, ((fx1 + fx2) // 2, (fy1 + fy2) // 2), fig_w // 4, 215, -1)
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
    img = img + rng.normal(0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    if skew_deg:
        m = cv2.getRotationMatrix2D((width / 2, height / 2), skew_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (width, height), flags=cv2.INTER_LINEAR, borderValue=paper)
    text_blocks = {}
    for w in truth:
        text_blocks.setdefault(w.block, {}).setdefault(w.line, []).append(w.text)
    text = "\n\n".join("\n".join(" ".join(ws) for _, ws in sorted(lines_.items())) for _, lines_ in sorted(text_blocks.items()))
    return TamilPage(gray=gray, words=tuple(truth), text=text, font=font, size=size)


def degraded_word(seed: int, text: str | None = None, font: str | None = None, size: int | None = None) -> np.ndarray:
    """A word crop made the way the training generator makes them (src/synth/degrade: modern or old print, loose or
    tight margins with pieces of neighbouring lines): realistic single-word inputs for the page detector."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.synth import degrade
    from src.synth.render import render_text_mask
    names = ["NotoSansTamil[wdth,wght]", "MuktaMalar-Regular", "HindMadurai-Regular", "TiroTamil-Regular",
             "Kavivanar-Regular", "Coiny-Regular", "Arima[wght]", "MeeraInimai-Regular"]
    rng = random.Random(1000 + seed)
    spec = _fonts()[font or names[seed % len(names)]]
    mask = render_text_mask(text or WORDS[seed % len(WORDS)], spec, size or [24, 32, 40, 48, 64][seed % 5], None, 0, None)
    fn = degrade.modern_print if (seed // 5) % 2 == 0 else degrade.old_print
    return fn(degrade.crop_with_margins(mask, rng, tight=(seed // 10) % 2 == 0), rng)


def png(gray) -> bytes:
    return cv2.imencode(".png", gray)[1].tobytes()
