"""
Export a FIXED synthetic set to disk (for validation/test, not for training).
Training uses on-the-fly generation; validation must be identical across runs, so we freeze it.

  python tools/synth_export.py --vocab data/text_corpus/tamil_correction_vocabulary.json ^
         --out data/synth_val --n-modern 3000 --n-old 3000 --seed-base 900000000

Writes <out>/images/*.png and <out>/manifest.jsonl  ({"image","label","style","font"} per line).
Seeds are far from the training seed range, so no exported sample is ever seen in training.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synth.generator import SynthWordGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", nargs="+", default=["assets/fonts"])
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-modern", type=int, default=3000)
    ap.add_argument("--n-old", type=int, default=3000)
    ap.add_argument("--seed-base", type=int, default=900_000_000)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for style, n, p_old, offset in (("modern", a.n_modern, 0.0, 0), ("old", a.n_old, 1.0, 50_000_000)):
            gen = SynthWordGenerator(a.fonts, vocab_json=a.vocab, old_print_prob=p_old, text_seed=a.seed_base + offset)
            for i in range(n):
                img, text, meta = gen.sample(a.seed_base + offset + i)
                name = f"{style}_{i:06d}.png"
                cv2.imwrite(str(out / "images" / name), img)
                mf.write(json.dumps({"image": f"images/{name}", "label": text, "style": style,
                                     "font": meta["font"]}, ensure_ascii=False) + "\n")
            print(f"{style}: {n} images")
    print("done ->", out)


if __name__ == "__main__":
    main()
