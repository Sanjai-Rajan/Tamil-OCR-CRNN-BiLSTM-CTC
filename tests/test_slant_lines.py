"""
Tests of the EXPERIMENTAL angle-aware line finding (pageocr/slant_lines.py, SegParams.exp_angle_lines, off by default).

    python -B -m unittest -v tests/test_slant_lines.py

Pages are generated (tests/slant_fixtures.py, dev_tools/synthetic_conditions.py); word truth comes from the generator.
No recognizer is needed here: these tests are about lines, word boxes and crops.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "dev_tools"))

import numpy as np  # noqa: E402

import fullpage_fixtures as fx  # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)

import slant_fixtures as sf  # noqa: E402
import synthetic_conditions as sc  # noqa: E402
from pageocr import pipeline, slant_lines  # noqa: E402
from pageocr.layout import OcrImage, SegParams, build_layout_image  # noqa: E402
from wordocr import fullpage  # noqa: E402

OFF = fullpage.PageSettings().seg_params()
ON = fullpage.PageSettings(exp_angle_lines=True).seg_params()


def matched(truth, boxes):
    """Truth words matched one-to-one by a box at IoU >= 0.5 (greedy by IoU)."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u else 0.0
    pairs = sorted(((iou(t, p), i, j) for i, t in enumerate(truth) for j, p in enumerate(boxes)), reverse=True)
    ti, pj = set(), set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in ti and j not in pj:
            ti.add(i); pj.add(j)
    return len(ti)


def words(seg):
    return [(w.id, w.box, w.box_original, w.kind, w.gap_before) for w in seg.words]


class OffByDefault(unittest.TestCase):
    def test_flag_is_off_and_not_recorded(self):
        self.assertFalse(SegParams().exp_angle_lines)
        s = fullpage.PageSettings()
        self.assertFalse(s.exp_angle_lines)
        self.assertNotIn("exp_angle_lines", s.experimental())
        self.assertNotIn("exp_angle_lines", s.as_dict())
        self.assertFalse(s.seg_params().exp_angle_lines)

    def test_recorded_when_on(self):
        s = fullpage.PageSettings(exp_angle_lines=True)
        self.assertEqual(s.experimental(), {"exp_angle_lines": True})
        self.assertTrue(s.seg_params().exp_angle_lines)
        seg = pipeline.segment(OcrImage(gray=sf.make_slanted_page(seed=2, angles=[0, 2, -2]).gray), s.seg_params())
        self.assertTrue(any(st.startswith("EXPERIMENTAL angle-aware lines") for st in seg.steps))
        self.assertTrue(all("line_source" in d for ds in seg.word_diagnostics for d in ds))


class SameAsDefaultWhereBandsAreKept(unittest.TestCase):
    """Where the gate keeps the horizontal bands, the experimental path gives exactly the default words."""

    def test_standard_pages_are_identical(self):
        pages = {c: sc.make(c, 1000).gray for c in ("clean", "two_columns", "tight_spacing", "small_print", "skew_3deg", "low_contrast")}
        pages["flat_slant_fixture"] = sf.make_slanted_page(seed=3, angles=[0] * 6).gray
        pages["uneven_spacing"] = sf.make_slanted_page(seed=3, angles=[1] * 6, pitch=[60, 95, 70, 120, 64]).gray
        for name, gray in pages.items():
            with self.subTest(page=name):
                a = pipeline.segment(OcrImage(gray=gray), OFF)
                b = pipeline.segment(OcrImage(gray=gray), ON)
                self.assertEqual(words(b), words(a))
                self.assertEqual([[(ln.y1, ln.y2, ln.components) for ln in ls] for ls in b.lines],
                                 [[(ln.y1, ln.y2, ln.components) for ln in ls] for ls in a.lines])
                self.assertTrue(all(d["line_source"] == "bands" for ds in b.word_diagnostics for d in ds))

    def test_fragmented_print_keeps_the_bands(self):
        p = sc.make("low_contrast", 1000)
        lay = build_layout_image(OcrImage(gray=p.gray), OFF)
        seg = pipeline.segment(OcrImage(gray=p.gray), ON)
        b = seg.blocks[0]
        from pageocr.lines import find_lines
        band = find_lines(lay.ink, lay.components, b, OFF.line_split_factor, OFF.line_thin_factor)
        use, why = slant_lines.prefer_tracked(lay.components, band, slant_lines.find_lines_tracked(lay.components, b, lay.h), lay.h)
        self.assertFalse(use)
        self.assertIn("fragments", why)


class MixedAngles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = sf.make_slanted_page(seed=3, angles=[0, 1.5, -1.5, 3, -2.5, 1])
        cls.off = pipeline.segment(OcrImage(gray=cls.page.gray), OFF)
        cls.on = pipeline.segment(OcrImage(gray=cls.page.gray), ON)

    def test_tracked_lines_replace_merged_bands(self):
        n_off = sum(len(ls) for ls in self.off.lines)
        n_on = sum(len(ls) for ls in self.on.lines)
        self.assertLess(n_off, len(self.page.lines))                         # the default merges the lines (the problem)
        self.assertGreaterEqual(n_on, len(self.page.lines))
        self.assertTrue(any(d["line_source"] == "tracked" for ds in self.on.word_diagnostics for d in ds))

    def test_more_words_found_than_with_bands(self):
        truth = [b for _, b in self.page.words]
        m_off = matched(truth, [w.box_original for w in self.off.words])
        m_on = matched(truth, [w.box_original for w in self.on.words])
        self.assertEqual(m_off, 0)
        self.assertGreaterEqual(m_on, 25, (m_off, m_on))                    # measured 28 of 44 when written

    def test_every_component_on_exactly_one_line_and_one_word(self):
        seen = [c for ls in self.on.lines for ln in ls for c in ln.components]
        self.assertEqual(len(seen), len(set(seen)))
        lay = build_layout_image(self.on.ocr, ON)
        in_blocks = {i for i, c in enumerate(lay.components) for b in self.on.blocks
                     if b.x1 <= (c[0] + c[2]) / 2.0 < b.x2 and c[3] > b.y1 and c[1] < b.y2}
        self.assertEqual(set(seen), in_blocks)                              # no component of a block is lost or doubled
        self.assertEqual(len({w.id for w in self.on.words}), len(self.on.words))

    def test_word_boxes_are_the_union_of_original_component_boxes(self):
        lay = build_layout_image(self.on.ocr, ON)
        H, W = self.on.ocr.gray.shape
        for w in self.on.words:
            x1, y1, x2, y2 = w.box
            self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H)
        # find_words on the levelled copy, then boxes rebuilt from ORIGINAL boxes: no word box is levelled
        tracked = [t for b in self.on.blocks for t in slant_lines.find_lines_tracked(lay.components, b, lay.h)]
        lev = slant_lines.levelled_components(lay.components, tracked)
        for ln, s, xr in tracked:
            ws, _ = slant_lines.words_of_tracked(lay.components, lev, ln, ON, None)
            for w in ws:
                c = lay.components[np.array(w.components)]
                self.assertEqual((w.x1, w.y1, w.x2, w.y2), (int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 2].max()), int(c[:, 3].max())))

    def test_crops_are_plain_slices_of_the_original_grayscale(self):
        crops = pipeline.ocr_crops(self.on.ocr, self.on.words, 4)
        H, W = self.on.ocr.gray.shape
        for w, c in zip(self.on.words, crops):
            x1, y1, x2, y2 = w.box
            self.assertTrue(np.array_equal(c, self.on.ocr.gray[max(0, y1 - 4):min(H, y2 + 4), max(0, x1 - 4):min(W, x2 + 4)]))


class Levelling(unittest.TestCase):
    def test_levelling_moves_rows_only_and_keeps_heights(self):
        comps = np.array([[0, 100, 20, 120, 50], [200, 110, 230, 128, 60], [400, 120, 420, 140, 40]], int)
        from pageocr.lines import Line
        lev = slant_lines.levelled_components(comps, [(Line(100, 140, (0, 1, 2), False), 0.05, 0.0)])
        self.assertTrue(np.array_equal(lev[:, [0, 2, 4]], comps[:, [0, 2, 4]]))
        self.assertTrue(np.array_equal(lev[:, 3] - lev[:, 1], comps[:, 3] - comps[:, 1]))
        self.assertLess(np.ptp((lev[:, 1] + lev[:, 3]) / 2.0), np.ptp((comps[:, 1] + comps[:, 3]) / 2.0))
        self.assertTrue(np.array_equal(comps[:, 1], [100, 110, 120]))      # the input is not modified


if __name__ == "__main__":
    unittest.main()
