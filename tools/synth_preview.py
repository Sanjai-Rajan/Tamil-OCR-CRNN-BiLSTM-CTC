"""
Preview / benchmark the synthetic word generator.

  python tools/synth_preview.py --vocab data/text_corpus/tamil_correction_vocabulary.json --n 48 --out synth_preview.png
  python tools/synth_preview.py --style old --n 48 --out synth_old.png
  python tools/synth_preview.py --bench 500            # samples/second on this machine

Each tile shows the image with its label rendered underneath, so label/image agreement
can be checked by eye before any training run.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synth.fonts import discover_fonts  # noqa: E402
from src.synth.generator import SynthWordGenerator  # noqa: E402
from src.synth.render import render_text_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="assets/fonts")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--style", choices=["mixed", "modern", "old"], default="mixed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="synth_preview.png")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--engine", choices=["raqm", "harfbuzz"], default=None)
    a = ap.parse_args()

    p_old = {"mixed": 0.3, "modern": 0.0, "old": 1.0}[a.style]
    gen = SynthWordGenerator([a.fonts], vocab_json=a.vocab, old_print_prob=p_old, engine=a.engine)
    print(f"families: {len(gen.family_names)} {gen.family_names}")
    print(f"book/serif families (old-print bias): {gen.book_families}")
    print(f"corpus words loaded: {len(gen.sampler.words)}")

    if a.bench:
        t = time.time()
        for i in range(a.bench):
            gen.sample(a.seed * 100000 + i)
        dt = time.time() - t
        print(f"{a.bench / dt:.1f} samples/s per CPU process")
        return

    # Captions must be shaped too (plain Pillow without raqm scrambles Tamil vowel signs).
    label_spec = next(s for s in discover_fonts([a.fonts]) if s.family == "notosanstamil")
    info_font = ImageFont.load_default()
    tiles = []
    for i in range(a.n):
        img, text, meta = gen.sample(a.seed * 100000 + i)
        h = 64
        w = max(8, int(img.shape[1] * h / img.shape[0]))
        tile = Image.fromarray(img).resize((min(w, 460), h))
        canvas = Image.new("L", (470, h + 26), 255)
        canvas.paste(tile, (5, 2))
        cap = 255 - render_text_mask(text, label_spec, 18)
        cap_img = Image.fromarray(cap)
        if cap_img.width > 300:
            cap_img = cap_img.resize((300, max(1, int(cap_img.height * 300 / cap_img.width))))
        canvas.paste(cap_img, (5, h + 3))
        info = f"[{meta['font']}, {meta['style']}{', pre-1978' if meta.get('traditional_forms') else ''}{', letterpress' if meta.get('letterpress') else ''}]"
        ImageDraw.Draw(canvas).text((min(310, cap_img.width + 12), h + 8), info, font=info_font, fill=0)
        tiles.append(canvas)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("L", (cols * 470, rows * 90), 200)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 470, (k // cols) * 90))
    sheet.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
