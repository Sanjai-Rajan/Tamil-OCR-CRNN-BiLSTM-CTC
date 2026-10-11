"""
Single source of truth for the recognition character set (v2).

Labels are handled as Unicode code points after NFC normalisation, which matches
how CTC treats Tamil text in this project (one output class per code point).
Index 0 is the CTC blank.
"""
import unicodedata

# --- Tamil block ---------------------------------------------------------
TAMIL_VOWELS = list("அஆஇஈஉஊஎஏஐஒஓஔ")
TAMIL_AYTHAM = ["ஃ"]
TAMIL_CONSONANTS = list("கஙசஞடணதநபமயரலவழளறன")
TAMIL_GRANTHA = list("ஜஶஷஸஹ")
TAMIL_SIGNS = list("ாிீுூெேைொோௌ்ௗ")          # vowel signs, virama, au length mark
TAMIL_DIGITS = list("௦௧௨௩௪௫௬௭௮௯")
TAMIL_SYMBOLS = list("ௐ௰௱௲௳௴௵௶௷௸௹௺")

# --- Common characters that appear inside printed "words" -----------------
ASCII_DIGITS = list("0123456789")
PUNCT = list(".,;:!?'\"-()[]/%&*+=")

CHARSET = (TAMIL_VOWELS + TAMIL_AYTHAM + TAMIL_CONSONANTS + TAMIL_GRANTHA + TAMIL_SIGNS
           + TAMIL_DIGITS + TAMIL_SYMBOLS + ASCII_DIGITS + PUNCT)
assert len(CHARSET) == len(set(CHARSET)), "duplicate characters in CHARSET"

BLANK = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARSET) + 1  # + blank

# Unicode confusables that appear in real labels -> canonical form
_REPLACE = {
    "‌": "", "‍": "", "﻿": "", " ": "",   # joiners / BOM / nbsp
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
}


def normalize(text: str) -> str:
    """Canonical label form: NFC, joiners removed, typographic quotes/dashes folded."""
    text = unicodedata.normalize("NFC", text)
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.strip()


def is_encodable(text: str) -> bool:
    return len(text) > 0 and all(c in CHAR_TO_IDX for c in text)


def encode(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text]


def decode(indices) -> str:
    return "".join(IDX_TO_CHAR.get(int(i), "") for i in indices if int(i) != BLANK)


def ctc_greedy_collapse(indices) -> list[int]:
    out, prev = [], None
    for i in indices:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out
