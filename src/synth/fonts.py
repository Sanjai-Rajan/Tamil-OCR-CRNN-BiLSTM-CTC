"""
Font registry: discovers .ttf/.otf/.ttc files, checks Tamil glyph coverage, and
exposes weight/width variation for variable fonts.
"""
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

from src import charset as cs

# Fonts are grouped so the sampler can balance styles (a family with 7 weights
# should not dominate a family with 1).
MIN_TAMIL_COVERAGE = 0.95


@dataclass
class FontSpec:
    path: str
    family: str
    index: int = 0                      # face index inside .ttc
    axes: dict = field(default_factory=dict)   # tag -> (min, default, max)
    cmap: frozenset = frozenset()
    tags: tuple = ()                    # e.g. ("serif",) ("display",) ("handwriting-like",)
    trad_feature: str | None = None     # OpenType feature that switches on pre-1978 letter forms
    trad_default: bool = False          # font draws pre-1978 forms without any feature

    def supports(self, text: str) -> bool:
        return all(ord(c) in self.cmap for c in text)


# style hints used by the old-print mode (serif / book faces look more like old print)
STYLE_HINTS = {
    "notoseriftamil": ("serif", "book"),
    "tirotamil": ("serif", "book"),
    "pavanam": ("book",),
    "meerainimai": ("book",),
    "latha": ("book",),
    "vijaya": ("serif", "book"),
    "coiny": ("display",),
    "baloothambi2": ("display",),
    "kavivanar": ("calligraphic",),
    "arima": ("display",),
}


# Pre-1978 ("old script") Tamil drew ணா றா னா லை ளை னை ணை as single special glyphs.
# Unicode text is identical; only the font shapes change. We detect fonts that can draw them.
_TRAD_TEST = "ணா றா னா லை ளை னை ணை"
_TRAD_CANDIDATE_FEATURES = ("ss01", "ss02", "ss03", "ss04", "ss05", "dlig", "salt", "hist")


def _glyph_count(path, index, features):
    try:
        import uharfbuzz as hb
    except ImportError:
        return None
    with open(path, "rb") as fh:
        face = hb.Face(fh.read(), index)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(_TRAD_TEST)
    buf.guess_segment_properties()
    hb.shape(font, buf, features)
    return len(buf.glyph_infos)


def detect_traditional(path, index, gsub_tags):
    """Return (feature_or_None, draws_traditional_by_default)."""
    n_syll = len(_TRAD_TEST.split()) * 2 - 1          # 7 syllables + 6 spaces when fully ligated
    base = _glyph_count(path, index, {})
    if base is None:
        return None, False
    if base <= n_syll:
        return None, True
    for tag in _TRAD_CANDIDATE_FEATURES:
        if tag in gsub_tags and _glyph_count(path, index, {tag: True}) <= n_syll:
            return tag, False
    return None, False


def _family_key(path: Path) -> str:
    stem = path.stem.split("[")[0].split("-")[0]
    return stem.lower().replace(" ", "")


def _load_faces(path: Path):
    if path.suffix.lower() == ".ttc":
        coll = TTCollection(str(path), lazy=True)
        return list(enumerate(coll.fonts))
    return [(0, TTFont(str(path), lazy=True))]


def discover_fonts(font_dirs, required_chars=None) -> list[FontSpec]:
    required = required_chars or (cs.TAMIL_VOWELS + cs.TAMIL_CONSONANTS + cs.TAMIL_SIGNS)
    specs = []
    for d in font_dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            try:
                faces = _load_faces(p)
            except Exception:
                continue
            for idx, tt in faces:
                try:
                    cmap = frozenset(tt.getBestCmap().keys())
                except Exception:
                    continue
                cov = sum(ord(c) in cmap for c in required) / len(required)
                if cov < MIN_TAMIL_COVERAGE:
                    continue
                axes = {}
                if "fvar" in tt:
                    for a in tt["fvar"].axes:
                        axes[a.axisTag] = (a.minValue, a.defaultValue, a.maxValue)
                fam = _family_key(p)
                gsub = set()
                if "GSUB" in tt:
                    try:
                        gsub = {fr.FeatureTag for fr in tt["GSUB"].table.FeatureList.FeatureRecord}
                    except Exception:
                        pass
                trad_feat, trad_def = detect_traditional(str(p), idx, gsub)
                specs.append(FontSpec(str(p), fam, idx, axes, cmap, STYLE_HINTS.get(fam, ()),
                                      trad_feat, trad_def))
    return specs


def group_by_family(specs):
    fams = {}
    for s in specs:
        fams.setdefault(s.family, []).append(s)
    return fams
