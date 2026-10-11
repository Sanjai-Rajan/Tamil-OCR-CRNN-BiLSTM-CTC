"""
Test inputs for the word frontend: synthetic printed Tamil words rendered by the project's own renderer
(src/synth: HarfBuzz shaping, a bundled font, the modern-print degradation) with fixed seeds. No dataset image, no
held-out test image and no development or R1 page is used. These are smoke-test inputs: a correct reading of them
shows that the pipeline is wired correctly, not how accurate the model is.
"""
import importlib.util
import io
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# The renderer needs uharfbuzz and freetype-py (in .venv; not in the Store Python that runs the R1 pytest command).
RENDERING = all(importlib.util.find_spec(m) is not None for m in ("uharfbuzz", "freetype", "fontTools"))
SKIP_REASON = ("the synthetic word fixtures need the rendering libraries (uharfbuzz, freetype-py); "
               "run these tests with the project's .venv")
WORDS = ("தமிழ்", "கணினி", "புத்தகம்", "வணக்கம்", "௨௦௨௬")


@lru_cache(maxsize=1)
def _font():
    from src.synth.fonts import discover_fonts
    specs = [s for s in discover_fonts([str(ROOT / "assets" / "fonts")]) if s.family.lower().startswith("hindmadurai")]
    if not specs:
        raise RuntimeError("the bundled HindMadurai font was not found under assets/fonts")
    return specs[0]


@lru_cache(maxsize=32)
def word_gray(text: str, seed: int = 7, size: int = 48) -> np.ndarray:
    from src.synth import degrade
    from src.synth.render import render_text_mask
    rng = random.Random(seed)
    mask = render_text_mask(text, _font(), size, None, 0, None)
    return degrade.modern_print(degrade.crop_with_margins(mask, rng, tight=False), rng)


def encode(arr, fmt: str = "PNG", **kw) -> bytes:
    img = arr if isinstance(arr, Image.Image) else Image.fromarray(arr)
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def word_png(text: str = "வணக்கம்", fmt: str = "PNG", **kw) -> bytes:
    return encode(word_gray(text), fmt, **kw)
