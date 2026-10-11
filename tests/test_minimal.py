"""
Minimal smoke tests of the core modules: charset, preprocessing, metrics, synthetic generator.

Replaces an earlier version (2026-10-10) that was written against APIs this project never had (TamilCharset,
preprocess_image, cer/wer functions, SynthWordGenerator() without arguments, generate_sample): its first three tests
always skipped on an ImportError and the fourth errored, so it tested nothing. The original is kept in
reports/autonomous_testing_frontend_2026-10-10/original_files/tests/test_minimal.py. The same four areas are covered
here against the real APIs and the real contracts (for example, the tensor range is [-1, 1], not [0, 1]).

    .venv\\Scripts\\python -B -m unittest -v tests/test_minimal.py
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import charset as cs  # noqa: E402
from src import preprocess as pp  # noqa: E402
from src.metrics import Scoreboard  # noqa: E402

HAS_SHAPING = all(importlib.util.find_spec(m) is not None for m in ("uharfbuzz", "freetype", "fontTools"))


class TestMinimal(unittest.TestCase):
    def test_charset(self):
        text = "தமிழ்"
        self.assertTrue(cs.is_encodable(text))
        self.assertEqual(cs.decode(cs.encode(text)), text)
        self.assertEqual(cs.ctc_greedy_collapse([1, 1, 0, 2, 2, 0, 3]), [1, 2, 3])
        self.assertEqual(cs.ctc_greedy_collapse([1, 0, 1]), [1, 1])          # a blank separates a doubled letter
        self.assertEqual(cs.ctc_greedy_collapse([0, 0, 0]), [])
        self.assertEqual(cs.NUM_CLASSES, len(cs.CHARSET) + 1)
        self.assertNotIn(" ", cs.CHAR_TO_IDX)                                # one word per image: no space class
        self.assertEqual(cs.normalize("தமி‍ழ் "), "தமிழ்")

    def test_preprocess(self):
        img = Image.new("RGB", (100, 50), color="white")
        tensor, pad = pp.prepare(img)
        self.assertEqual(tuple(tensor.shape), (1, pp.TARGET_H, 96))           # 100 * 48 / 50
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertAlmostEqual(pad, 1.0)                                      # white paper pads as white
        wide, _ = pp.prepare(np.full((10, 5000), 255, np.uint8))
        self.assertEqual(wide.shape[2], pp.MAX_W)

    def test_metrics(self):
        b = Scoreboard()
        b.add("test", "test")
        b.add("test", "tast")
        s = b.summary()["all"]
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["cer"], 1 / 8)                               # micro: 1 edit / 8 reference characters
        self.assertAlmostEqual(s["wer"], 0.5)
        empty = Scoreboard().summary()
        self.assertEqual(empty, {})

    @unittest.skipUnless(HAS_SHAPING, "the synthetic generator needs uharfbuzz, freetype-py and fontTools")
    def test_synth(self):
        from src.synth.generator import SynthWordGenerator
        gen = SynthWordGenerator([str(ROOT / "assets" / "fonts")], text_seed=1)
        img, label, meta = gen.sample(123)
        again, label2, _ = gen.sample(123)
        self.assertEqual((img.ndim, img.dtype), (2, np.uint8))
        self.assertTrue(len(label) > 0 and cs.is_encodable(label))
        self.assertEqual(label, label2)
        np.testing.assert_array_equal(img, again)                             # same seed, same image
        self.assertIn(meta["style"], ("modern", "old"))


if __name__ == "__main__":
    unittest.main()
