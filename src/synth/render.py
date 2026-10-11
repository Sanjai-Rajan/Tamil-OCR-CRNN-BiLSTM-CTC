"""
Clean text rendering with correct Tamil shaping.

Tamil requires complex text layout: vowel signs such as ெ ே ை are typed AFTER the
consonant but drawn BEFORE it, and ொ/ோ/ௌ split around it. Without a shaping engine
(HarfBuzz via libraqm) the rendered image does not match the label, which would
silently recreate the label-noise problem. Two engines are supported:

  * "raqm"     - Pillow's built-in layout engine (needs Pillow compiled with libraqm)
  * "harfbuzz" - uharfbuzz for shaping + freetype-py for rasterising (pure pip wheels,
                 used automatically when raqm is unavailable, e.g. on some Windows builds)
"""
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from .fonts import FontSpec

RAQM_AVAILABLE = features.check("raqm")


def pick_variation(spec: FontSpec, rng: random.Random) -> dict:
    v = {}
    for tag, (lo, default, hi) in spec.axes.items():
        if tag == "wght":
            # bias towards normal text weights; bold and light still appear
            v[tag] = float(np.clip(rng.gauss(default, (hi - lo) / 5), lo, hi))
        elif tag == "wdth":
            v[tag] = float(np.clip(rng.gauss(default, (hi - lo) / 6), lo, hi))
        elif tag == "ital":
            v[tag] = lo
        else:
            v[tag] = default
    return v


# ------------------------------------------------------------------ raqm engine
_pil_cache: dict = {}


def _pil_font(spec: FontSpec, size: int, variation: dict):
    key = (spec.path, spec.index, size, tuple(sorted(variation.items())))
    f = _pil_cache.get(key)
    if f is None:
        f = ImageFont.truetype(spec.path, size, index=spec.index, layout_engine=ImageFont.Layout.RAQM)
        if variation:
            try:
                names = [a["name"] if isinstance(a, dict) else a for a in f.get_variation_axes()]
                # Pillow orders axes as in fvar; map by tag order stored in spec.axes
                f.set_variation_by_axes([variation[t] for t in spec.axes])
            except Exception:
                pass
        if len(_pil_cache) > 512:
            _pil_cache.clear()
        _pil_cache[key] = f
    return f


def _render_raqm(text, spec, size, variation, stroke, features=None):
    font = _pil_font(spec, size, variation)
    feats = [k for k, v in (features or {}).items() if v] or None
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font, stroke_width=stroke, language="ta", features=feats)
    # Reported boxes can under-estimate italic overhangs and some vowel-sign extents,
    # so draw on a generously padded canvas and crop to the real ink afterwards.
    pad = size + stroke * 2
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=255,
                             stroke_width=stroke, stroke_fill=255, language="ta", features=feats)
    return _ink_crop(np.asarray(img))


def _ink_crop(a, margin=2):
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return a[:1, :1].copy()
    y0, y1 = max(0, ys.min() - margin), min(a.shape[0], ys.max() + 1 + margin)
    x0, x1 = max(0, xs.min() - margin), min(a.shape[1], xs.max() + 1 + margin)
    return np.ascontiguousarray(a[y0:y1, x0:x1])


# -------------------------------------------------------------- harfbuzz engine
_hb_cache: dict = {}


def _hb_objects(spec: FontSpec):
    import freetype
    import uharfbuzz as hb
    key = (spec.path, spec.index)
    o = _hb_cache.get(key)
    if o is None:
        with open(spec.path, "rb") as fh:
            data = fh.read()
        face = hb.Face(data, spec.index)
        o = (hb.Font(face), freetype.Face(spec.path, index=spec.index), face.upem)
        _hb_cache[key] = o
    return o


def _render_harfbuzz(text, spec, size, variation, stroke, features=None, letterpress=None, rng=None):
    """letterpress: dict(baseline=px_sd, spacing=px_sd, ink=(lo,hi)) simulates hand-set metal type:
    each glyph sits slightly off the baseline, spacing is uneven and each sort inks differently."""
    import freetype
    import uharfbuzz as hb
    hb_font, ft_face, upem = _hb_objects(spec)
    if variation:
        hb_font.set_variations(variation)
        try:
            ft_face.set_var_design_coords([variation[t] for t in spec.axes])
        except Exception:
            pass
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    buf.language = "ta"
    feats = {"kern": True, "liga": True}
    feats.update(features or {})
    hb.shape(hb_font, buf, feats)
    lp = letterpress or {}
    jit_y = lp.get("baseline", 0.0); jit_x = lp.get("spacing", 0.0); ink_rng = lp.get("ink")
    rng = rng or random.Random(0)
    # glyphs of one cluster (consonant + its marks) must move together, so jitter per cluster
    cluster_off = {}
    ft_face.set_pixel_sizes(0, size)
    scale = size / upem
    pen_x = 0.0
    glyphs = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_DEFAULT)
        ft_face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)
        bmp = ft_face.glyph.bitmap
        arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width) if bmp.rows else None
        off = cluster_off.get(info.cluster)
        if off is None:
            off = (rng.gauss(0, jit_x) if jit_x else 0.0, rng.gauss(0, jit_y) if jit_y else 0.0,
                   rng.uniform(*ink_rng) if ink_rng else 1.0)
            cluster_off[info.cluster] = off
        if arr is not None and off[2] != 1.0:
            arr = (arr.astype(np.float32) * off[2]).astype(np.uint8)
        x = pen_x + pos.x_offset * scale + ft_face.glyph.bitmap_left + off[0]
        y = -pos.y_offset * scale - ft_face.glyph.bitmap_top + off[1]
        glyphs.append((arr, x, y))
        pen_x += pos.x_advance * scale + (max(-1.0, rng.gauss(0, jit_x * 0.5)) if jit_x else 0.0)
    placed = [(a, x, y) for a, x, y in glyphs if a is not None and a.size]
    if not placed:
        return np.zeros((size, size), np.uint8)
    x0 = min(x for _, x, _ in placed); y0 = min(y for _, _, y in placed)
    x1 = max(x + a.shape[1] for a, x, _ in placed); y1 = max(y + a.shape[0] for a, _, y in placed)
    W, H = int(np.ceil(x1 - x0)) + 4, int(np.ceil(y1 - y0)) + 4
    canvas = np.zeros((H, W), np.uint8)
    for a, x, y in placed:
        cx, cy = int(round(x - x0)) + 2, int(round(y - y0)) + 2
        region = canvas[cy:cy + a.shape[0], cx:cx + a.shape[1]]
        np.maximum(region, a[: region.shape[0], : region.shape[1]], out=region)
    if stroke > 0:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stroke + 1, 2 * stroke + 1))
        canvas = cv2.dilate(canvas, k)
    return canvas


def render_text_mask(text: str, spec: FontSpec, size: int, variation: dict | None = None,
                     stroke: int = 0, engine: str | None = None, features: dict | None = None,
                     letterpress: dict | None = None, rng=None) -> np.ndarray:
    """Return a uint8 coverage mask (255 = ink) tightly around the shaped text.
    `letterpress` jitter needs per-glyph control, so it always uses the HarfBuzz engine."""
    variation = variation or {}
    if letterpress:
        engine = "harfbuzz"
    engine = engine or ("raqm" if RAQM_AVAILABLE else "harfbuzz")
    if engine == "raqm":
        return _render_raqm(text, spec, size, variation, stroke, features)
    return _render_harfbuzz(text, spec, size, variation, stroke, features, letterpress, rng)
