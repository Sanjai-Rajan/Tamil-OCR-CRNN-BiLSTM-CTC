"""
Text sources for synthetic word rendering.

Mixes four kinds of strings so the recognizer learns glyphs rather than memorising a lexicon:
  * corpus words sampled by smoothed frequency (common words appear often, rare words still appear)
  * corpus words sampled uniformly from the long tail, with extra weight on long words
  * pseudo-words built from valid Tamil syllables (no dictionary bias)
  * numbers / words with attached punctuation (what a word segmenter actually produces on print)
"""
import json
import random
from pathlib import Path

from src import charset as cs

VOWEL_SIGNS = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
CONSONANTS = cs.TAMIL_CONSONANTS + cs.TAMIL_GRANTHA
VIRAMA = "்"


def _syllable(rng: random.Random, first: bool) -> str:
    r = rng.random()
    if first and r < 0.18:
        return rng.choice(cs.TAMIL_VOWELS)
    c = rng.choice(CONSONANTS)
    r = rng.random()
    if r < 0.30:
        return c + VIRAMA                      # pure consonant (very frequent in Tamil)
    if r < 0.55:
        return c                                # inherent 'a'
    return c + rng.choice(VOWEL_SIGNS[1:])


def pseudo_word(rng: random.Random, min_syl=1, max_syl=9) -> str:
    n = rng.randint(min_syl, max_syl)
    return "".join(_syllable(rng, i == 0) for i in range(n))


def number_string(rng: random.Random, tamil_digit_prob: float = 0.2) -> str:
    if rng.random() < tamil_digit_prob:          # Tamil numerals (common in old books: page/verse numbers)
        return "".join(rng.choice(cs.TAMIL_DIGITS) for _ in range(rng.randint(1, 4)))
    kind = rng.random()
    if kind < 0.4:
        return str(rng.randint(0, 99999))
    if kind < 0.6:
        return f"{rng.randint(1, 31)}.{rng.randint(1, 12)}.{rng.randint(1900, 2030)}"
    return f"{rng.randint(1, 999)}-{rng.randint(1, 999)}"


class TextSampler:
    def __init__(self, vocab_json: str | None = None, extra_wordlists: list[str] | None = None,
                 min_freq: int = 2, max_len: int = 40, seed: int = 0,
                 mix=(0.50, 0.20, 0.15, 0.05, 0.10)):
        """
        vocab_json: {word: frequency} (e.g. data/text_corpus/tamil_correction_vocabulary.json)
        extra_wordlists: plain text files, one word per line (e.g. handwritten-set labels, Wikipedia words)
        mix: probabilities of (frequent word, long-tail word, pseudo word, number, word+punctuation)
        """
        self.rng = random.Random(seed)
        self.max_len = max_len
        self.mix = mix
        self.tamil_digit_prob = 0.2
        words, freqs = [], []
        if vocab_json:
            p = Path(vocab_json)
            if not p.exists():
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_json}")
            with open(p, encoding="utf-8") as f:
                vocab = json.load(f)
            for w, c in vocab.items():
                w = cs.normalize(w)
                if c >= min_freq and cs.is_encodable(w) and len(w) <= max_len:
                    words.append(w)
                    freqs.append(c)
        for wl in extra_wordlists or []:
            with open(wl, encoding="utf-8") as f:
                for line in f:
                    w = cs.normalize(line.split("\t")[0])
                    if cs.is_encodable(w) and len(w) <= max_len:
                        words.append(w)
                        freqs.append(3)
        if not words:  # tiny built-in fallback so the module is testable anywhere
            words = ["தமிழ்", "மொழி", "உலகம்", "கொண்டு", "செய்யப்பட்டது", "அறிவியல்", "பள்ளிக்கூடம்",
                     "நூலகத்திலிருந்து", "ஒவ்வொருவருக்கும்", "கௌரவம்", "ஸ்ரீ", "ஜனநாயகம்"]
            freqs = [5] * len(words)
        self.words = words
        # smoothed frequency weights (freq ** 0.5) -> cumulative for fast sampling
        w = [f ** 0.5 for f in freqs]
        self.cum_freq = list(_accumulate(w))
        # long-tail weights favour longer words to counter the short-word bias seen in the old model
        lw = [1.0 + max(0, len(x) - 8) * 0.35 for x in words]
        self.cum_long = list(_accumulate(lw))

    def _pick(self, cum):
        x = self.rng.random() * cum[-1]
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return self.words[lo]

    def sample(self) -> str:
        r = self.rng.random()
        p = self.mix
        if r < p[0]:
            s = self._pick(self.cum_freq)
        elif r < p[0] + p[1]:
            s = self._pick(self.cum_long)
        elif r < p[0] + p[1] + p[2]:
            s = pseudo_word(self.rng)
        elif r < p[0] + p[1] + p[2] + p[3]:
            s = number_string(self.rng, self.tamil_digit_prob)
        else:
            s = self._pick(self.cum_freq)
            q = self.rng.random()
            if q < 0.55:
                s = s + self.rng.choice([",", ".", ";", ":", "?", "!", "-"])
            elif q < 0.8:
                s = self.rng.choice(["(", '"', "'"]) + s + self.rng.choice([")", '"', "'"])
            else:
                s = s + "-" + self._pick(self.cum_freq)
        s = cs.normalize(s)[: self.max_len]
        return s if cs.is_encodable(s) else "தமிழ்"


def _accumulate(xs):
    t = 0.0
    for x in xs:
        t += x
        yield t
