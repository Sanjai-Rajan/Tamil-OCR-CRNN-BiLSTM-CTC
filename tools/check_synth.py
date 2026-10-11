"""
One-shot environment check for the synthetic generator. Run this first on a new machine:

  .venv\\Scripts\\python tools\\check_synth.py --vocab data\\text_corpus\\tamil_correction_vocabulary.json

It reports: which shaping engine is used, whether raqm and HarfBuzz produce the same shapes,
how many font families were found, generator speed, and writes check_synth.png
(12 modern + 12 old-print samples with their labels) for a visual check.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synth.fonts import discover_fonts, group_by_family  # noqa: E402
from src.synth.render import RAQM_AVAILABLE, render_text_mask  # noqa: E402

TEST_WORDS = ["கொண்டுவந்தோம்", "கௌரவம்", "பெண்ணைப்", "ஸ்ரீ", "நூற்றுக்கணக்கானவர்களை"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", nargs="+", default=[str(ROOT / "assets" / "fonts")])
    ap.add_argument("--vocab", default=None)
    a = ap.parse_args()

    print(f"[1] Pillow raqm available: {RAQM_AVAILABLE} -> engine = {'raqm' if RAQM_AVAILABLE else 'harfbuzz'}")
    specs = discover_fonts(a.fonts)
    fams = group_by_family(specs)
    print(f"[2] Tamil font faces: {len(specs)}  families: {len(fams)}  {sorted(fams)}")
    if len(fams) < 8:
        print("    WARNING: fewer than 8 families - add more fonts for variety.")

    try:
        import uharfbuzz  # noqa: F401
        agree = []
        for s in [f[0] for f in fams.values()]:
            for w in TEST_WORDS:
                h = render_text_mask(w, s, 40, {}, 0, "harfbuzz")
                if RAQM_AVAILABLE:
                    r = render_text_mask(w, s, 40, {}, 0, "raqm")
                    agree.append(abs(r.shape[1] - h.shape[1]) <= max(6, 0.03 * r.shape[1]))
        if agree:
            print(f"[3] raqm vs harfbuzz width agreement: {sum(agree)}/{len(agree)} (should be all or nearly all)")
        else:
            print("[3] harfbuzz engine renders OK (raqm not present, harfbuzz will be used)")
    except ImportError:
        if not RAQM_AVAILABLE:
            print("[3] ERROR: neither raqm nor uharfbuzz available -> pip install uharfbuzz freetype-py")
            return

    t = time.time()
    subprocess.run([sys.executable, str(ROOT / "tools" / "synth_preview.py"), "--bench", "300",
                    "--fonts", a.fonts[0]] + (["--vocab", a.vocab] if a.vocab else []), check=True)
    print(f"[4] benchmark finished in {time.time() - t:.1f}s")
    for style, out in (("modern", "check_modern.png"), ("old", "check_old.png")):
        subprocess.run([sys.executable, str(ROOT / "tools" / "synth_preview.py"), "--style", style, "--n", "12",
                        "--fonts", a.fonts[0], "--out", out] + (["--vocab", a.vocab] if a.vocab else []), check=True)
    print("[5] wrote check_modern.png and check_old.png - open them and confirm labels match images.")


if __name__ == "__main__":
    main()
