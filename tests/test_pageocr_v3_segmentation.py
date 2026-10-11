"""
Tests for the V3 automatic segmentation: the two image paths, layout cleaning, deskew, columns, lines, words,
reading order, export of diagnostics, the development guard and the page application.

    .venv\\Scripts\\python -B -m unittest discover -s tests -p "test_pageocr_*.py"

Every page here is generated in memory from invented glyphs (tests/pageocr_synth_pages.py). These are engineering
tests of geometry. They are not OCR accuracy evidence, no real page is segmented, and no development page is read
(the guard tests use only the SHA-256 of the master files).
"""
import ast
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pageocr import boxes as bx, columns, deskew, devguard, geometry, layout, pipeline, reading_order, words as pwords     # noqa: E402
from pageocr_synth_pages import make_page                                                                                                    # noqa: E402

CHECKPOINT = ROOT / "runs" / "experiment_002a" / "best.pth"
NEW_MODULES = ("layout", "deskew", "columns", "lines", "words", "reading_order", "pipeline", "geometry", "devguard")


def compare(page, seg):
    return geometry.compare([w["box"] for w in page.words], [w.box_original for w in seg.words])


class TwoImagePathTests(unittest.TestCase):
    """The recognizer may only ever receive original grayscale. Enforced by type."""

    @classmethod
    def setUpClass(cls):
        cls.page = make_page(seed=1)
        cls.seg = pipeline.segment(cls.page.gray)

    def test_ocr_image_refuses_a_mask_and_a_binarized_page(self):
        ink = layout.build_layout_image(layout.OcrImage(gray=self.page.gray)).ink
        with self.assertRaises(layout.ImagePathError):
            layout.OcrImage(gray=ink)                                         # a boolean mask
        with self.assertRaises(layout.ImagePathError):
            layout.OcrImage(gray=(ink * 255).astype(np.uint8))                # the same mask dressed as a gray image
        with self.assertRaises(layout.ImagePathError):
            layout.OcrImage(gray=np.where(self.page.gray > 128, 255, 0).astype(np.uint8))     # a thresholded page
        with self.assertRaises(layout.ImagePathError):
            layout.OcrImage(gray=cv2.cvtColor(self.page.gray, cv2.COLOR_GRAY2BGR))            # colour
        with self.assertRaises(layout.ImagePathError):
            layout.OcrImage(gray=self.page.gray.astype(np.float32))

    def test_layout_image_holds_no_gray_values(self):
        lay = layout.build_layout_image(self.seg.ocr)
        self.assertEqual(lay.ink.dtype, np.bool_)
        self.assertFalse(hasattr(lay, "gray"))
        with self.assertRaises(layout.ImagePathError):
            layout.LayoutImage(ink=self.page.gray, components=np.zeros((0, 5), np.int64), h=1.0)

    def test_crops_are_cut_only_from_an_ocr_image(self):
        lay = layout.build_layout_image(self.seg.ocr)
        for wrong in (lay, lay.ink, (lay.ink * 255).astype(np.uint8), self.page.gray):
            with self.assertRaises(TypeError, msg=type(wrong).__name__):
                pipeline.ocr_crops(wrong, self.seg.words)
        with self.assertRaises(layout.ImagePathError):
            pipeline.segment(lay)

    def test_crops_are_the_original_pixels(self):
        crops = pipeline.ocr_crops(self.seg.ocr, self.seg.words)
        self.assertEqual(len(crops), len(self.seg.words))
        for w, c in zip(self.seg.words[:40], crops[:40]):
            x1, y1, x2, y2 = w.box
            np.testing.assert_array_equal(c, self.page.gray[y1:y2, x1:x2])    # no skew on this page: the OCR image is the page
            self.assertEqual(c.dtype, np.uint8)
            self.assertGreater(len(np.unique(c)), 2)                           # not a two-level crop

    def test_the_ocr_image_of_an_unskewed_page_is_the_page_itself(self):
        np.testing.assert_array_equal(self.seg.ocr.gray, self.page.gray)
        self.assertEqual(self.seg.ocr.deskew_deg, 0.0)
        self.assertFalse(self.seg.ocr.gray.flags.writeable)

    def test_padding_is_applied_at_crop_time_and_kept_inside_the_page(self):
        w = self.seg.words[0]
        tight = pipeline.ocr_crops(self.seg.ocr, [w], pad_px=0)[0]
        padded = pipeline.ocr_crops(self.seg.ocr, [w], pad_px=5)[0]
        self.assertEqual(padded.shape, (tight.shape[0] + 10, tight.shape[1] + 10))
        edge = reading_order.PlacedWord("e", 0, 0, 0, (0, 0, 20, 20), (0, 0, 20, 20), "word", -1, False, "single")
        self.assertEqual(pipeline.ocr_crops(self.seg.ocr, [edge], pad_px=9)[0].shape, (29, 29))

    def test_no_new_module_imports_the_recognizer_or_a_network_library(self):
        banned = {"torch", "requests", "urllib", "http", "socket", "streamlit", "tensorflow", "easyocr", "pytesseract"}
        for name in NEW_MODULES:
            tree = ast.parse((ROOT / "pageocr" / f"{name}.py").read_text(encoding="utf-8"))
            top = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    top |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    top.add(node.module.split(".")[0])
            self.assertFalse(top & banned, f"pageocr/{name}.py imports {sorted(top & banned)}")
        # the only place the recognizer is named is pipeline.recognize, by a local import after the guard
        src = (ROOT / "pageocr" / "pipeline.py").read_text(encoding="utf-8")
        self.assertLess(src.index("devguard.assert_ocr_allowed(seg.ocr.page_sha256)"), src.index("from pageocr.engine import"))


class LayoutTests(unittest.TestCase):
    def test_uneven_light_border_bleed_and_specks_do_not_change_the_words(self):
        base = make_page(seed=4)
        clean = pipeline.segment(base.gray)
        self.assertEqual(compare(base, clean).recall, 1.0)
        for kw in (dict(uneven_light=True), dict(border=True), dict(bleed_through=True), dict(specks=150)):
            p = make_page(seed=4, **kw)
            s = pipeline.segment(p.gray)
            c = compare(p, s)
            self.assertGreaterEqual(c.recall, 0.98, kw)
            self.assertGreaterEqual(c.precision, 0.97, kw)
            self.assertEqual(sum(len(l) for l in s.lines), p.lines, kw)

    def test_cleaning_is_recorded(self):
        s = pipeline.segment(make_page(seed=2, border=True, specks=200, bleed_through=True).gray)
        self.assertGreater(s.removed["page border or scan edge"], 0)
        self.assertGreater(s.removed["rule line"], 0)
        self.assertTrue(any("Sauvola" in st for st in s.steps))
        self.assertTrue(any("illumination" in st for st in s.steps))

    def test_blank_and_tiny_pages(self):
        blank = np.full((400, 300), 230, np.uint8)
        blank[::7, ::5] = 228                                                 # paper texture, so it is not two-level
        blank[3, 3] = 226
        s = pipeline.segment(blank)
        self.assertEqual(len(s.words), 0)
        self.assertIn("no text found", " ".join(s.flags))

    def test_size_units_follow_the_print(self):
        small, big = make_page(seed=3, h=16, height=1000, width=800, lines=14), make_page(seed=3, h=40, height=2600, width=2000, lines=14)
        a, b = pipeline.segment(small.gray), pipeline.segment(big.gray)
        self.assertLess(a.h, b.h)
        self.assertGreaterEqual(compare(small, a).recall, 0.97)
        self.assertGreaterEqual(compare(big, b).recall, 0.97)


class DeskewTests(unittest.TestCase):
    def test_measured_angle(self):
        for angle in (-4.0, -2.0, -0.7, 0.0, 0.6, 1.5, 3.5):
            p = make_page(seed=6, skew_deg=angle)
            s = pipeline.segment(p.gray)
            self.assertAlmostEqual(s.skew["angle_deg"], -angle, delta=0.25, msg=f"page skewed by {angle}")
            self.assertEqual(s.skew["applied"], abs(angle) >= 0.3)

    def test_words_of_a_skewed_page_are_found_and_mapped_back_to_the_original(self):
        for angle in (2.5, -3.0):
            p = make_page(seed=7, layout="two_columns", skew_deg=angle)
            s = pipeline.segment(p.gray)
            c = compare(p, s)                                                 # the truth is in ORIGINAL page pixels
            self.assertGreaterEqual(c.recall, 0.98, angle)
            self.assertEqual(len(s.blocks), 2)
            self.assertNotEqual(s.ocr.deskew_deg, 0.0)
            H, W = p.gray.shape
            for w in s.words:
                x1, y1, x2, y2 = w.box_original
                self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H)

    def test_the_deskewed_ocr_image_is_still_grayscale_from_the_original(self):
        p = make_page(seed=8, skew_deg=3.0)
        s = pipeline.segment(p.gray)
        self.assertEqual(s.ocr.gray.dtype, np.uint8)
        self.assertGreater(len(np.unique(s.ocr.gray)), 50)
        self.assertGreaterEqual(s.ocr.gray.shape[0], p.gray.shape[0])        # the canvas grew; nothing was cut off
        self.assertEqual(s.ocr.original_shape, p.gray.shape)

    def test_estimate_skew_takes_a_mask_only(self):
        with self.assertRaises(TypeError):
            deskew.estimate_skew(make_page(seed=1).gray)


class ColumnTests(unittest.TestCase):
    def test_one_column(self):
        for seed in range(1, 6):
            for kw in (dict(), dict(justify=True), dict(word_gap=1.3)):       # wide and justified word spaces are not gutters
                p = make_page(seed=seed, **kw)
                s = pipeline.segment(p.gray)
                self.assertEqual(len(s.blocks), 1, (seed, kw))
                self.assertEqual(s.blocks[0].kind, "single")

    def test_two_columns(self):
        for seed in range(1, 6):
            p = make_page(seed=seed, layout="two_columns")
            s = pipeline.segment(p.gray)
            self.assertEqual([b.kind for b in s.blocks], ["column", "column"], seed)
            self.assertLess(s.blocks[0].x2, s.blocks[1].x1)                   # left column first

    def test_heading_above_two_columns(self):
        for seed in range(1, 6):
            p = make_page(seed=seed, layout="heading_two_columns")
            s = pipeline.segment(p.gray)
            self.assertEqual([b.kind for b in s.blocks], ["single", "column", "column"], seed)
            self.assertLess(s.blocks[0].y2, s.blocks[1].y1 + 5)
            self.assertTrue(any("mixed layout" in f for f in s.flags))

    def test_runs_and_strips(self):
        self.assertEqual(columns.runs(np.array([0, 1, 1, 0, 0, 1, 0], bool)), [(1, 3), (5, 6)])
        ink = np.zeros((20, 10), bool)
        ink[2:5] = True; ink[9:12] = True
        self.assertEqual(columns.strips(ink, 0, 20), [(2, 5), (9, 12)])
        self.assertEqual(columns.strips(ink, 0, 20, min_gap=6), [(2, 12)])
        with self.assertRaises(TypeError):
            columns.find_blocks(np.zeros((5, 5), np.uint8), 10.0)


class LineTests(unittest.TestCase):
    def test_line_count_with_marks_above_the_letters(self):
        for seed in range(1, 6):
            p = make_page(seed=seed, marks=True)                              # rows of pulli must not become lines
            s = pipeline.segment(p.gray)
            self.assertEqual(sum(len(l) for l in s.lines), p.lines, seed)

    def test_lines_are_top_to_bottom_and_do_not_overlap(self):
        s = pipeline.segment(make_page(seed=3, layout="two_columns").gray)
        for ls in s.lines:
            for a, b in zip(ls, ls[1:]):
                self.assertLessEqual(a.y2, b.y1)

    def test_touching_lines_are_cut_and_flagged(self):
        p = make_page(seed=2, touching_lines=True)
        s = pipeline.segment(p.gray)
        self.assertEqual(sum(len(l) for l in s.lines), p.lines)
        self.assertTrue(any(ln.forced_split for ls in s.lines for ln in ls))
        self.assertTrue(any("touching lines" in f for f in s.flags))


class WordTests(unittest.TestCase):
    def test_ordinary_gaps(self):
        for seed in range(1, 6):
            p = make_page(seed=seed)
            c = compare(p, pipeline.segment(p.gray))
            self.assertEqual((c.recall, c.precision), (1.0, 1.0), seed)
            self.assertGreater(c.mean_iou_of_matches, 0.9)

    def test_narrow_and_wide_gaps(self):
        for gap, floor in ((0.45, 0.93), (1.3, 0.98), (2.0, 0.98)):
            for seed in (1, 2, 3):
                p = make_page(seed=seed, word_gap=gap)
                c = compare(p, pipeline.segment(p.gray))
                self.assertGreaterEqual(c.recall, floor, (gap, seed))

    def test_pulli_and_superscript_stay_inside_their_word(self):
        p = make_page(seed=5, marks=True, punctuation=False)
        s = pipeline.segment(p.gray)
        c = compare(p, s)
        self.assertEqual(c.recall, 1.0)
        self.assertEqual(c.split, ())                                         # no mark was left as a box of its own
        self.assertGreater(c.mean_iou_of_matches, 0.93)                       # the boxes include the marks above the letters

    def test_trailing_punctuation_joins_its_word(self):
        p = make_page(seed=5, marks=False, punctuation=True)
        s = pipeline.segment(p.gray)
        c = compare(p, s)
        self.assertGreaterEqual(c.recall, 0.99)
        self.assertEqual(sum(w.kind == "punctuation" for w in s.words), 0)

    def test_a_line_set_solid_stays_one_run(self):
        p = make_page(seed=4, solid_lines=True)
        s = pipeline.segment(p.gray)
        self.assertEqual(len(s.words), p.lines)                               # one unit per line, not one per letter
        self.assertEqual(compare(p, s).recall, 1.0)

    def test_no_character_segmentation(self):
        p = make_page(seed=1)
        s = pipeline.segment(p.gray)
        self.assertEqual(len(s.words), len(p.words))                          # words, never letters
        self.assertGreater(np.median([w.box[2] - w.box[0] for w in s.words]), 2 * s.h)

    def test_boxes_are_tight_to_the_ink(self):
        p = make_page(seed=2, marks=False, punctuation=False)
        s = pipeline.segment(p.gray)
        truth = {tuple(w["box"]) for w in p.words}
        close = sum(any(max(abs(a - b) for a, b in zip(w.box, t)) <= 3 for t in truth) for w in s.words)
        self.assertGreaterEqual(close / len(s.words), 0.97)

    def test_line_threshold_rule(self):
        self.assertIsNone(pwords.line_threshold([3, 4], 30, 1.6, 0.16))                       # too few gaps
        self.assertIsNone(pwords.line_threshold([3, 4, 3, 4, 5, 4], 30, 1.6, 0.16))           # one size only
        self.assertAlmostEqual(pwords.line_threshold([3, 4, 3, 20, 22, 4, 21], 30, 1.6, 0.16), 12.0)      # half-way between 4 and 20
        self.assertIsNone(pwords.line_threshold([1, 1, 1, 3, 3], 30, 1.6, 0.16))              # the wide class is below the floor


class ReadingOrderTests(unittest.TestCase):
    def check_order(self, p, s):
        c = compare(p, s)
        by_det = {j: i for i, j, _ in c.matches}
        seq = [by_det[j] for j in range(len(s.words)) if j in by_det]
        self.assertEqual(seq, sorted(seq))                                    # detected words come in the truth's reading order
        got = [(w.block_index, w.line_index, w.word_index) for w in s.words]
        self.assertEqual(got, sorted(got))
        self.assertGreaterEqual(c.recall, 0.98)

    def test_single_column(self):
        p = make_page(seed=3)
        self.check_order(p, pipeline.segment(p.gray))

    def test_two_columns_left_column_first(self):
        p = make_page(seed=3, layout="two_columns")
        s = pipeline.segment(p.gray)
        self.check_order(p, s)
        last_left = max(i for i, w in enumerate(s.words) if w.block_index == 0)
        first_right = min(i for i, w in enumerate(s.words) if w.block_index == 1)
        self.assertLess(last_left, first_right)

    def test_heading_then_columns_and_skew(self):
        p = make_page(seed=3, layout="heading_two_columns", skew_deg=1.5)
        self.check_order(p, pipeline.segment(p.gray))

    def test_printed_line_breaks_are_kept(self):
        p = make_page(seed=2)
        s = pipeline.segment(p.gray)
        per_line = {}
        for w in s.words:
            per_line.setdefault((w.block_index, w.line_index), []).append(w)
        self.assertEqual(len(per_line), p.lines)
        truth = {}
        for w in p.words:
            truth[(w["block"], w["line"])] = truth.get((w["block"], w["line"]), 0) + 1
        self.assertEqual({k: len(v) for k, v in per_line.items()}, truth)     # no word moved to another line
        blocks = reading_order.as_text_lines(list(s.words), [w.id for w in s.words])
        self.assertEqual(sum(len(lines) for _, lines in blocks), p.lines)

    def test_export_uses_the_existing_box_convention(self):
        p = make_page(seed=2, layout="two_columns")
        s = pipeline.segment(p.gray)
        boxes = pipeline.boxes_for_export(s, "original")
        bx.validate_boxes(boxes, p.gray.shape[1], p.gray.shape[0])            # inside the page, unique ids
        self.assertEqual(bx.reading_order(boxes), list(range(len(boxes))))    # already in the project's reading order
        again = bx.loads_json(bx.dumps_json(boxes, p.gray.shape[1], p.gray.shape[0], "0" * 64))
        self.assertEqual(list(again.boxes), boxes)
        with self.assertRaises(ValueError):
            reading_order.to_boxes(list(s.words), "preview")


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_are_json_and_hold_no_recognizer_output(self):
        s = pipeline.segment(make_page(seed=1, layout="heading_two_columns", skew_deg=1.0).gray)
        d = json.loads(json.dumps(pipeline.diagnostics(s), ensure_ascii=False))
        self.assertEqual(d["counts"]["words"], len(s.words))
        self.assertEqual(d["statement"], pipeline.STATEMENT)
        self.assertIn("original grayscale", d["image_paths"]["ocr_image"])
        self.assertIn("Sauvola", d["image_paths"]["layout_image"])
        keys = set()
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.add(k); walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
        self.assertFalse({"text", "confidence", "prediction", "transcription"} & keys)
        self.assertEqual(d["parameters"]["word_pad_px"], 0)

    def test_segmentation_is_deterministic(self):
        p = make_page(seed=9, layout="two_columns", skew_deg=1.2, uneven_light=True)
        a, b = pipeline.diagnostics(pipeline.segment(p.gray)), pipeline.diagnostics(pipeline.segment(p.gray))
        a.pop("seconds"); b.pop("seconds")
        self.assertEqual(a, b)

    def test_overlay_is_a_picture_of_the_ocr_image(self):
        s = pipeline.segment(make_page(seed=1).gray)
        img = pipeline.draw_overlay(s)
        self.assertEqual(img.shape, s.ocr.gray.shape + (3,))


class DevelopmentGuardTests(unittest.TestCase):
    """No recognizer output on a development page before the human-verified set is frozen."""

    def test_guard_names_what_is_missing(self):
        st = devguard.status()
        self.assertEqual(len(st["conditions"]), 6)
        if not st["ready"]:
            text = devguard.explain(st)
            self.assertIn("may NOT be run", text)
            self.assertIn("run_human_verification.cmd", text)
            with self.assertRaises(devguard.DevelopmentBlocked):
                devguard.assert_ready()

    def test_recognize_refuses_a_development_page_before_any_model_is_touched(self):
        masters = devguard.development_master_hashes()
        self.assertGreaterEqual(len(masters), 16)
        if devguard.status()["ready"]:
            self.skipTest("the verified set is frozen: the guard is open")
        sha = sorted(masters)[0]
        self.assertTrue(devguard.is_development_page(sha))
        seg = pipeline.segment(layout.OcrImage(gray=make_page(seed=1).gray, page_sha256=sha))   # generated pixels, a development page's hash

        class Trap:
            def recognize_batch(self, *a, **k):
                raise AssertionError("the recognizer was called on a development page")
        with self.assertRaises(devguard.DevelopmentBlocked):
            pipeline.recognize(seg, Trap())

    def test_other_pages_are_not_the_guard_s_concern(self):
        devguard.assert_ocr_allowed("0" * 64)                                 # not a development page: no exception

    def test_recognize_on_a_generated_page_returns_one_result_per_word(self):
        from pageocr.engine import OcrEngine
        before = devguard.sha_file(CHECKPOINT)
        p = make_page(seed=1, lines=3)
        seg = pipeline.segment(p.gray)
        recs = pipeline.recognize(seg, OcrEngine(CHECKPOINT, device="cpu"))
        self.assertEqual(len(recs), len(seg.words))
        self.assertTrue(all(r is not None and isinstance(r.text, str) for r in recs))
        self.assertEqual(devguard.sha_file(CHECKPOINT), before)              # the checkpoint was only read
        self.assertEqual(before, devguard.FROZEN_002A_SHA256)

    def test_the_guard_has_no_off_switch(self):
        src = (ROOT / "pageocr" / "devguard.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("assert_ready", "assert_ocr_allowed", "status")):
            names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
            self.assertFalse(any(k in n.lower() for n in names for k in ("force", "skip", "bypass", "allow", "override")), fn.name)
        self.assertNotIn("os.environ", src)


class AutoAppTests(unittest.TestCase):
    """app_page_auto.py through streamlit.testing: upload, segmentation shown, export offered, OCR disabled by the guard."""

    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except Exception as e:                                               # pragma: no cover
            raise unittest.SkipTest(f"streamlit.testing is not available: {e}")
        cls.AppTest = AppTest
        cls.page = make_page(seed=2, layout="two_columns", lines=8, height=900)
        cls.png = cv2.imencode(".png", cls.page.gray)[1].tobytes()

    def start(self):
        at = self.AppTest.from_file(str(ROOT / "app_page_auto.py"), default_timeout=300)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_statement_and_disabled_ocr_before_upload(self):
        at = self.start()
        self.assertTrue(any(pipeline.STATEMENT in i.value for i in at.info))
        if not devguard.status()["ready"]:
            self.assertTrue(any("OCR is disabled" in w.value for w in at.warning))

    def test_upload_segments_and_keeps_ocr_disabled(self):
        at = self.start()
        at.file_uploader(key="page_upload").upload("generated_page.png", self.png, "image/png").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        seg = at.session_state["seg"]
        self.assertEqual(len(seg.blocks), 2)
        self.assertGreaterEqual(geometry.compare([w["box"] for w in self.page.words], [w.box_original for w in seg.words]).recall, 0.97)
        self.assertTrue(any("2 blocks" in c.value for c in at.caption))
        if not devguard.status()["ready"]:
            run = next(b for b in at.button if b.label == "Run OCR")
            self.assertTrue(run.disabled)
            self.assertIsNone(at.session_state["ocr"])
            self.assertEqual(len(at.text_area), 0)                            # no recognizer text anywhere on the page


if __name__ == "__main__":
    unittest.main()
