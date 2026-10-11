"""
Synthetic printed Tamil word generator (modern print and old print).

Usage (inside a PyTorch Dataset, or standalone):

    gen = SynthWordGenerator(font_dirs=["assets/fonts"],
                             vocab_json="data/text_corpus/tamil_correction_vocabulary.json")
    img, text, meta = gen.sample(seed=123)      # img: uint8 grayscale HxW, text: NFC label

Images are generated on the fly, so no disk space is needed. The same seed always
produces the same image, which makes fixed validation sets reproducible.
"""
import random

import numpy as np

from . import degrade
from .fonts import discover_fonts, group_by_family
from .render import pick_variation, render_text_mask
from .text_sampler import TextSampler


class SynthWordGenerator:
    def __init__(self, font_dirs, vocab_json=None, extra_wordlists=None,
                 old_print_prob=0.3, engine=None, text_seed=0,
                 font_size_range=(28, 72), exclude_families=()):
        specs = [s for s in discover_fonts(font_dirs) if s.family not in exclude_families]
        if not specs:
            raise RuntimeError(f"No Tamil-capable fonts found in {font_dirs}")
        self.families = group_by_family(specs)
        self.family_names = sorted(self.families)
        self.book_families = [f for f in self.family_names
                              if "book" in self.families[f][0].tags or "serif" in self.families[f][0].tags]
        self.trad_families = [f for f in self.family_names
                              if any(sp.trad_feature or sp.trad_default for sp in self.families[f])]
        self.sampler = TextSampler(vocab_json, extra_wordlists, seed=text_seed)
        self.old_print_prob = old_print_prob
        self.engine = engine
        self.font_size_range = font_size_range
        self.rejected = 0          # samples where every degraded attempt failed the legibility check

    def _pick_font(self, rng, text, old):
        names = self.family_names
        if old:
            r = rng.random()
            if self.trad_families and r < 0.35:           # pre-1978 letter forms
                names = self.trad_families
            elif self.book_families and r < 0.75:
                names = self.book_families
        for _ in range(10):
            fam = rng.choice(names)                       # balance by family, not by file count
            spec = rng.choice(self.families[fam])
            if spec.supports(text):
                return spec
        for fam in self.family_names:                     # fallback: any face that covers the text
            for spec in self.families[fam]:
                if spec.supports(text):
                    return spec
        return None

    def sample(self, seed: int):
        rng = random.Random(seed)
        self.sampler.rng = random.Random(seed ^ 0x5F3759DF)
        old = rng.random() < self.old_print_prob
        self.sampler.tamil_digit_prob = 0.5 if old else 0.15
        for _ in range(5):
            text = self.sampler.sample()
            spec = self._pick_font(rng, text, old)
            if spec is not None:
                break
        else:
            text, spec = "தமிழ்", self.families[self.family_names[0]][0]
        size = rng.randint(*self.font_size_range)
        variation = pick_variation(spec, rng)
        stroke = 1 if (not spec.axes and rng.random() < 0.12) else 0
        features, letterpress, trad = None, None, spec.trad_default
        if spec.trad_feature and rng.random() < (0.6 if old else 0.05):
            features, trad = {spec.trad_feature: True}, True
        if old and rng.random() < 0.6:                    # hand-set metal type
            letterpress = {"baseline": size * rng.uniform(0.008, 0.035),
                           "spacing": size * rng.uniform(0.0, 0.04),
                           "ink": (rng.uniform(0.45, 0.8), 1.0)}
        mask = render_text_mask(text, spec, size, variation, stroke, self.engine,
                                features=features, letterpress=letterpress, rng=rng)
        if old and rng.random() < 0.3:
            mask = degrade.page_curl(mask, rng)
        mask = degrade.geometric(mask, rng, strength=1.4 if old else 1.0)
        mask = degrade.crop_with_margins(mask, rng, tight=rng.random() < 0.25)
        mask = degrade.add_segmentation_artifacts(mask, rng)
        if old and rng.random() < 0.35:
            mask = degrade.line_interference(mask, rng)
        # Degrade, then verify the word is still legible; unreadable crops would be label noise.
        for attempt in range(4):
            img = degrade.old_print(mask, rng) if old else degrade.modern_print(mask, rng)
            ok, _ = degrade.legibility(img, mask)
            if ok:
                break
        else:  # give up on heavy effects for this sample
            img = degrade.modern_print(mask, random.Random(seed + 1))
            self.rejected += 1
        meta = {"font": spec.family, "size": size, "style": "old" if old else "modern",
                "variation": variation, "traditional_forms": trad, "letterpress": bool(letterpress)}
        return img, text, meta


try:  # optional torch wrapper (torch is not needed to preview/export samples)
    from torch.utils.data import Dataset

    class SynthWordDataset(Dataset):
        """On-the-fly dataset. `length` samples per epoch; `base_seed` separates train from val."""

        def __init__(self, generator: SynthWordGenerator, length: int, base_seed: int = 0,
                     transform=None, fixed=False):
            self.gen, self.length, self.base_seed = generator, length, base_seed
            self.transform, self.fixed = transform, fixed
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            seed = self.base_seed + idx if self.fixed else hash((self.base_seed, self.epoch, idx)) & 0x7FFFFFFF
            img, text, meta = self.gen.sample(seed)
            if self.transform:
                img = self.transform(img)
            return img, text, meta["style"]
except ImportError:  # pragma: no cover
    pass
