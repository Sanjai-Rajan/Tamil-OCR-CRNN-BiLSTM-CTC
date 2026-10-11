"""
Tests of the word-mode page check (wordocr.pagedetect) and of full-page OCR (wordocr.fullpage, wordocr.page_export).

    .venv\\Scripts\\python -B -m unittest -v tests/test_fullpage_ocr.py

Every image is GENERATED in memory (tests/fullpage_fixtures.py: real Tamil words rendered with the project's fonts;
training-style degraded word crops). No dataset, held-out test, development, R1 or real_docs image is used. Where a test
must show that the recognizer is NOT run, the recognizer is a trap that fails the test when called. The real-model
tests use the registered, hash-verified checkpoints; reading generated words correctly shows that the pipeline is
connected, not how accurate it is on real pages.
"""
import csv
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fullpage_fixtures as fx                                    # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)

import torch                                                      # noqa: E402

from pageocr import devguard                                      # noqa: E402
from wordocr import fullpage, guard, inputs, page_export, pagedetect, service   # noqa: E402

GUARD_OPEN = devguard.status()["ready"]


class Trap:
    """A recognizer that must not be called. It carries the real engine's identity, so result records can be built."""
    def __init__(self, info):
        self._info = info

    def info(self):
        return dict(self._info)

    def recognize_batch(self, *a, **k):
        raise AssertionError("the recognizer was called")


def trap_model():
    m = model("cpu")
    return service.LoadedModel(spec=m.spec, engine=Trap(m.engine.info()), device="cpu", load_seconds=0.0)


def decoded(gray, name="page.png"):
    return inputs.decode_upload(name, fx.png(gray))


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


_MODELS = {}


def model(device="cpu"):
    if device not in _MODELS:
        _MODELS[device] = service.load_model("002a", device)
    return _MODELS[device]


# ---------------------------------------------------------------- A. word mode: a page is never read as one word

class WordModePageCheckTests(unittest.TestCase):
    def test_the_reported_problem_a_dense_textbook_page_is_not_read_as_a_word(self):
        """The owner's report: a whole printed page in the word tab came out as one character. It must be refused
        before the recognizer runs, with a pointer to Full-page OCR, and never yield a one-character "page result"."""
        page = fx.make_tamil_page(seed=31, size=26, lines=40, width=1240, line_pitch=1.5)
        d = decoded(page.gray)
        r = service.recognize_one(trap_model(), d)                     # the trap fails the test if it is reached
        self.assertFalse(r.ok)
        self.assertEqual(r.text, "")
        self.assertIn("Full-page OCR", r.error)
        self.assertIn("ONE cropped word", r.error)

    def test_a_landscape_page_is_refused_too_orientation_is_not_the_criterion(self):
        page = fx.make_tamil_page(seed=32, lines=8)                    # wider than tall: no "taller than wide" warning
        d = decoded(page.gray)
        self.assertFalse(any("taller than it is wide" in w for w in d.warnings))
        a = pagedetect.assess(d.gray)
        self.assertEqual(a.kind, "page")
        self.assertFalse(service.recognize_one(trap_model(), d).ok)

    def test_pages_of_every_layout_and_polarity_are_detected(self):
        for kw in (dict(seed=1), dict(seed=4, columns=2), dict(seed=5, columns=2, heading=True), dict(seed=41, lines=2),
                   dict(seed=42, lines=3), dict(seed=11, line_pitch=1.35), dict(seed=45, size=16, lines=30)):
            g = fx.make_tamil_page(**kw).gray
            for img in (g, 255 - g):
                self.assertEqual(pagedetect.assess(img).kind, "page", kw)

    def test_a_page_verdict_cannot_be_overridden(self):
        d = decoded(fx.make_tamil_page(seed=1).gray)
        self.assertFalse(pagedetect.assess_cached(d.sha256, d.gray).overridable)
        self.assertFalse(service.recognize_one(trap_model(), d, confirmed_single_word=True).ok)

    def test_a_normal_word_is_accepted_and_read(self):
        m = model("cpu")
        g = (255 - np.pad(fx.render_word("தமிழ்"), 10).astype(np.float32) * 0.85).astype(np.uint8)
        d = decoded(g, "word.png")
        self.assertEqual(pagedetect.assess(d.gray).kind, "word")
        r = service.recognize_one(m, d)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "தமிழ்")

    def test_a_tall_word_crop_is_still_read(self):
        g = (255 - np.pad(fx.render_word("கணினி"), ((120, 120), (6, 6))).astype(np.float32) * 0.85).astype(np.uint8)
        d = decoded(g, "tall.png")
        self.assertTrue(any("taller than it is wide" in w for w in d.warnings))
        self.assertEqual(pagedetect.assess(d.gray).kind, "word")
        self.assertTrue(service.recognize_one(model("cpu"), d).ok)

    def test_realistic_single_word_crops_are_never_called_pages(self):
        """Training-style degraded crops (blur, noise, broken strokes, pieces of neighbouring lines, both polarities).
        A 1,920-crop run during development gave 0 "page" and 8 "multiline" verdicts; 240 are checked here."""
        kinds = [pagedetect.assess(fx.degraded_word(seed)).kind for seed in range(240)]
        self.assertEqual(kinds.count("page"), 0)
        self.assertLessEqual(kinds.count("multiline"), 3)

    def test_light_text_on_a_dark_background_is_handled(self):
        g = (255 - np.pad(fx.render_word("புத்தகம்"), 12).astype(np.float32) * 0.85).astype(np.uint8)
        inv, flipped = pagedetect.dark_on_light(255 - g)
        self.assertTrue(flipped)
        np.testing.assert_array_equal(inv, g)                        # exact inversion
        self.assertEqual(pagedetect.assess(255 - g).kind, "word")
        self.assertFalse(pagedetect.dark_on_light(g)[1])

    def test_one_line_of_words_is_a_warning_not_a_refusal(self):
        a = pagedetect.assess(fx.make_tamil_page(seed=40, lines=1).gray)
        self.assertEqual(a.kind, "line")
        self.assertFalse(a.is_page)

    def test_a_multiline_verdict_can_be_overridden_only_explicitly(self):
        weak = pagedetect.Assessment("multiline", 2, 4, 1, "test", 0.0)
        g = (255 - np.pad(fx.render_word("மொழி"), 10).astype(np.float32) * 0.85).astype(np.uint8)
        d = decoded(g, "w.png")
        with mock.patch.object(pagedetect, "assess_cached", return_value=weak):
            self.assertFalse(service.recognize_one(trap_model(), d).ok)
            self.assertIn("confirm", service.recognize_one(trap_model(), d).error)
            r = service.recognize_one(model("cpu"), d, confirmed_single_word=True)
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "மொழி")

    def test_a_page_in_a_batch_fails_alone(self):
        word = (255 - np.pad(fx.render_word("கல்வி"), 10).astype(np.float32) * 0.85).astype(np.uint8)
        items = [decoded(word, "w.png"), decoded(fx.make_tamil_page(seed=1).gray, "page.png")]
        out = service.recognize(model("cpu"), items)
        self.assertEqual([r.ok for r in out], [True, False])
        self.assertEqual(out[0].text, "கல்வி")
        self.assertIn("Full-page OCR", out[1].error)


# ---------------------------------------------------------------- B. segmentation of a page into word crops

class PageSegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = fx.make_tamil_page(seed=1)
        cls.d = decoded(cls.page.gray)
        cls.seg = fullpage.segment_page(cls.d)

    def test_lines_and_every_word_are_found(self):
        self.assertEqual(sum(len(ls) for ls in self.seg.lines), 12)
        self.assertEqual(len(self.seg.words), len(self.page.words))
        for tw in self.page.words:
            self.assertTrue(any(iou(tw.box, w.box_original) >= 0.6 for w in self.seg.words), tw)

    def test_boxes_lie_inside_the_page_and_crops_are_valid(self):
        H, W = self.page.gray.shape
        crops = fullpage.crop_images(self.seg)
        self.assertEqual(len(crops), len(self.seg.words))
        for w, c in zip(self.seg.words, crops):
            x1, y1, x2, y2 = w.box_original
            self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, w)
            self.assertEqual(c.ndim, 2)
            self.assertEqual(c.dtype, np.uint8)
            self.assertGreaterEqual(min(c.shape), 2)

    def test_crops_are_the_page_pixels_at_their_boxes(self):
        m = fullpage.PageSettings().margin_px(self.seg.h)
        crops = fullpage.crop_images(self.seg)
        H, W = self.page.gray.shape
        for w, c in zip(self.seg.words, crops):                        # no deskew on this page: boxes are original pixels
            x1, y1, x2, y2 = w.box
            np.testing.assert_array_equal(c, self.d.gray[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)])

    def test_marks_above_and_below_are_inside_the_word_boxes(self):
        for tw in self.page.words:                                    # the truth box is the rendered ink, marks included
            best = max(self.seg.words, key=lambda w: iou(tw.box, w.box_original))
            bx = best.box_original
            self.assertTrue(bx[0] <= tw.box[0] + 1 and bx[1] <= tw.box[1] + 1 and bx[2] >= tw.box[2] - 1 and bx[3] >= tw.box[3] - 1,
                            (tw, bx))

    def test_reading_order_is_stable(self):
        again = fullpage.segment_page(self.d)
        self.assertEqual([w.id for w in again.words], [w.id for w in self.seg.words])
        self.assertEqual([w.box for w in again.words], [w.box for w in self.seg.words])
        ys = [(w.line_index, w.box[0]) for w in self.seg.words]
        self.assertEqual(ys, sorted(ys))                             # lines top to bottom, words left to right

    def test_two_columns_are_read_left_column_first(self):
        page = fx.make_tamil_page(seed=4, columns=2, lines=14)
        seg = fullpage.segment_page(decoded(page.gray))
        self.assertEqual(len(seg.blocks), 2)
        xs = [w.box_original[0] for w in seg.words]
        split = next(i for i, w in enumerate(seg.words) if w.block_index == 1)
        self.assertLess(max(xs[:split]), min(xs[split:]))             # every left-column word comes before the right column

    def test_a_skewed_page_is_deskewed_and_boxes_map_back(self):
        page = fx.make_tamil_page(seed=19, skew_deg=3.0)
        seg = fullpage.segment_page(decoded(page.gray))
        self.assertTrue(seg.skew["applied"])
        self.assertAlmostEqual(seg.skew["angle_deg"], -3.0, delta=0.3)
        H, W = page.gray.shape
        self.assertTrue(all(0 <= w.box_original[0] < w.box_original[2] <= W and 0 <= w.box_original[1] < w.box_original[3] <= H
                            for w in seg.words))

    def test_a_blank_page_is_not_inverted_into_false_regions(self):
        """Defect found while building this: equal noise tails could flip the polarity of a blank light page, and the
        inverted (dark) page then produced page-sized "word regions"."""
        for shape in ((700, 500), (800, 600), (500, 900)):
            for seed in range(6):
                rng = np.random.default_rng(seed)
                blank = np.clip(235 + rng.normal(0, 3, shape), 0, 255).astype(np.uint8)
                self.assertFalse(pagedetect.dark_on_light(blank)[1], (shape, seed))
                self.assertEqual(len(fullpage.segment_page(decoded(blank)).words), 0, (shape, seed))
                self.assertEqual(pagedetect.assess(blank).kind, "word")

    def test_text_too_small_to_read_is_reported(self):
        with mock.patch.object(fullpage, "MIN_TEXT_HEIGHT", 500.0):       # any page's text is "too small" for this test
            with self.assertRaisesRegex(fullpage.PageError, "too small to read"):
                fullpage.segment_page(self.d)

    def test_a_blank_page_yields_no_regions_and_no_text(self):
        rng = np.random.default_rng(0)
        blank = np.clip(235 + rng.normal(0, 3, (800, 600)), 0, 255).astype(np.uint8)
        d = decoded(blank)
        seg = fullpage.segment_page(d)
        self.assertEqual(len(seg.words), 0)
        r = fullpage.recognize_page(trap_model(), d, seg)            # nothing to read: the trap is never reached
        self.assertEqual((r.words, r.text), ([], ""))

    def test_an_unreadable_crop_fails_alone(self):
        page = fx.make_tamil_page(seed=2, lines=3)
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        real = fullpage.pipeline.ocr_crops

        def one_binarized(ocr, words, pad):
            crops = real(ocr, words, pad)
            crops[1] = np.where(crops[1] > 128, 255, 0).astype(np.uint8)          # two gray levels: refused by the engine
            return crops
        with mock.patch.object(fullpage.pipeline, "ocr_crops", side_effect=one_binarized):
            r = fullpage.recognize_page(model("cpu"), d, seg)
        self.assertEqual([w.status for w in r.words].count("failed"), 1)
        self.assertEqual(r.words[1].status, "failed")
        self.assertIn("two gray levels", r.words[1].error)
        self.assertTrue(all(w.ok for i, w in enumerate(r.words) if i != 1))

    def test_a_recognizer_failure_on_one_region_does_not_lose_the_others(self):
        page = fx.make_tamil_page(seed=3, lines=2)
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        m = model("cpu")
        calls = {"n": 0}
        real = m.engine.recognize_batch

        def flaky(crops, batch_size=32):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated failure")
            return real(crops, batch_size=batch_size)
        with mock.patch.object(m.engine, "recognize_batch", side_effect=flaky):
            r = fullpage.recognize_page(m, d, seg)
        self.assertEqual(r.words[2].status, "failed")
        self.assertIn("RuntimeError", r.words[2].error)
        self.assertEqual(sum(w.ok for w in r.words), len(r.words) - 1)

    def test_the_settings_have_their_stated_effect(self):
        page = fx.make_tamil_page(seed=6, skew_deg=2.0)
        d = decoded(page.gray)
        base = fullpage.segment_page(d)
        self.assertTrue(base.skew["applied"])
        self.assertFalse(fullpage.segment_page(d, fullpage.PageSettings(deskew=False)).skew["applied"])
        merged = fullpage.segment_page(d, fullpage.PageSettings(gap_floor_h=0.9))   # every gap below 0.9 line heights joins
        self.assertLess(len(merged.words), len(base.words))
        self.assertEqual(fullpage.PageSettings(margin=0).margin_px(base.h), 0)
        self.assertEqual(fullpage.PageSettings(margin=9).margin_px(base.h), 9)
        self.assertEqual(fullpage.PageSettings().margin_px(28.0), 7)
        faint = fx.make_tamil_page(seed=7, ink=150, paper=200, noise=2)
        df = decoded(faint.gray)
        self.assertGreaterEqual(len(fullpage.segment_page(df, fullpage.PageSettings(sauvola_k=0.10)).words),
                                len(fullpage.segment_page(df, fullpage.PageSettings(sauvola_k=0.45)).words))

    def test_the_overlay_marks_every_region_on_the_image_the_crops_come_from(self):
        img = fullpage.overlay(self.seg)
        self.assertEqual(img.shape, self.seg.ocr.gray.shape + (3,))
        red = (img[:, :, 0] > 180) & (img[:, :, 1] < 60) & (img[:, :, 2] < 60)
        for w in self.seg.words:                                      # each word box has its red outline
            x1, y1, x2, y2 = w.box
            self.assertTrue(red[y1, x1:x2].mean() > 0.5 and red[y2 - 1, x1:x2].mean() > 0.5, w.id)
        np.testing.assert_array_equal(self.seg.ocr.gray, self.d.gray)                 # the overlay did not touch the page

    def test_an_inverted_page_is_segmented(self):
        page = fx.make_tamil_page(seed=8)
        seg = fullpage.segment_page(decoded(255 - page.gray))
        self.assertTrue(fullpage.segmentation_summary(seg)["polarity_inverted"])
        self.assertEqual(len(seg.words), len(page.words))


# ---------------------------------------------------------------- C. the real model on the page's word crops

class RealModelPageTests(unittest.TestCase):
    def check_page(self, device):
        page = fx.make_tamil_page(seed=5, columns=2, heading=True)
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        r = fullpage.recognize_page(model(device), d, seg)
        self.assertEqual(r.counts["failed"], 0)
        self.assertEqual(r.text, page.text)                          # words, spaces, line breaks and blocks exactly
        self.assertEqual(r.model["device"], device)
        self.assertTrue(r.model["sha256_verified"])
        for w in r.words:                                            # each prediction belongs to its own region
            tw = max(page.words, key=lambda t: iou(t.box, w.box))
            self.assertGreater(iou(tw.box, w.box), 0.5)
            self.assertEqual(w.text, tw.text)
        return r

    def test_cpu(self):
        r = self.check_page("cpu")
        self.assertIn("\n\n", r.text)                                # blocks are separated by a blank line

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA GPU")
    def test_cuda(self):
        self.check_page("cuda")

    def test_a_heading_above_the_body_is_set_apart_by_a_blank_line(self):
        page = fx.make_tamil_page(seed=52, heading=True, heading_size=64, lines=8)       # one column: one block
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        self.assertEqual(len(seg.blocks), 1)
        r = fullpage.recognize_page(model("cpu"), d, seg)
        self.assertEqual(r.text, page.text)                          # the reference has a blank line after the heading

    def test_a_figure_beside_the_text_is_left_out(self):
        page = fx.make_tamil_page(seed=53, figure=True, lines=10)
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        self.assertIn("large non-text component removed: figure, ornament or frame", seg.flags)
        self.assertEqual(fullpage.recognize_page(model("cpu"), d, seg).text, page.text)

    def test_a_light_on_dark_page_is_read(self):
        page = fx.make_tamil_page(seed=21, lines=6)
        d = decoded(255 - page.gray)                                  # light text on a dark page
        seg = fullpage.segment_page(d)
        r = fullpage.recognize_page(model("cpu"), d, seg)
        self.assertTrue(r.segmentation["polarity_inverted"])
        self.assertEqual(r.text, page.text)

    def test_a_word_reads_the_same_alone_and_inside_the_page(self):
        page = fx.make_tamil_page(seed=9, lines=4)
        d = decoded(page.gray)
        seg = fullpage.segment_page(d)
        m = model("cpu")
        r = fullpage.recognize_page(m, d, seg)
        crops = fullpage.crop_images(seg)
        for w, crop in list(zip(r.words, crops))[:12]:
            alone = m.engine.recognize_batch([crop], batch_size=1)[0].text
            self.assertEqual(alone, w.text)

    def test_unicode_is_preserved_in_text_and_exports(self):
        page = fx.make_tamil_page(seed=10, lines=3)
        d = decoded(page.gray, "பாடம் 1.png")
        r = fullpage.recognize_page(model("cpu"), d, fullpage.segment_page(d))
        self.assertEqual(page_export.to_txt(r).decode("utf-8"), r.text + "\n")
        doc = json.loads(page_export.to_json(r))
        self.assertEqual(doc["text"], r.text)
        self.assertEqual([w["text"] for w in doc["words"]], [w.text for w in r.words])
        self.assertEqual(doc["page"]["sha256"], d.sha256)
        self.assertEqual(doc["model"]["checkpoint_sha256"], r.model["checkpoint_sha256"])
        rows = list(csv.DictReader(io.StringIO(page_export.to_csv(r).decode("utf-8-sig"))))
        self.assertEqual(len(rows), len(r.words))
        self.assertEqual([x["text"] for x in rows], [w.text for w in r.words])
        self.assertEqual([int(x["x1"]) for x in rows], [w.box[0] for w in r.words])
        self.assertTrue(page_export.download_name(r, "txt").isascii())


# ---------------------------------------------------------------- text reconstruction and export details

def pw(i, block, line, word, text, kind="word", ok=True):
    return fullpage.PageWord(index=i, id=f"b{block:02d}_l{line:03d}_w{word:02d}", block=block, line=line, word=word, kind=kind,
                             box=(0, 0, 1, 1), box_ocr=(0, 0, 1, 1), ok=ok, text=text if ok else "", error="" if ok else "x")


class ReconstructionTests(unittest.TestCase):
    def test_spaces_line_breaks_blocks_and_punctuation(self):
        words = [pw(0, 0, 0, 0, "தமிழ்"), pw(1, 0, 0, 1, "மொழி"), pw(2, 0, 0, 2, ",", "punctuation"), pw(3, 0, 1, 0, "நூல்"),
                 pw(4, 1, 0, 0, "கல்வி"), pw(5, 1, 0, 1, "", ok=False), pw(6, 1, 0, 2, "அறம்")]
        self.assertEqual(fullpage.page_text(words), "தமிழ் மொழி,\nநூல்\n\nகல்வி அறம்")

    def test_a_larger_step_between_lines_starts_a_paragraph(self):
        import dataclasses
        ys = [0, 40, 80, 200, 240]                                    # the fourth line stands 3 pitches below the third
        words = [dataclasses.replace(pw(i, 0, i, 0, t), box_ocr=(0, y, 50, y + 20)) for i, (y, t) in enumerate(zip(ys, "அ ஆ இ ஈ உ".split()))]
        self.assertEqual(fullpage.page_text(words), "அ\nஆ\nஇ\n\nஈ\nஉ")
        self.assertEqual(fullpage.page_text(words[:2]), "அ\nஆ")      # too few lines to know the pitch: no paragraph

    def test_csv_cells_cannot_start_a_formula(self):
        r = fullpage.PageResult(name="=cmd.png", sha256="0" * 64, width=10, height=10, words=[pw(0, 0, 0, 0, "-1")], text="-1",
                                settings={}, margin_px=0, segmentation={}, model={})
        rows = list(csv.reader(io.StringIO(page_export.to_csv(r).decode("utf-8-sig"))))
        self.assertEqual(rows[1][0], "'=cmd.png")
        self.assertEqual(rows[1][12], "'-1")
        self.assertEqual(page_export.to_txt(r), b"-1\n")              # TXT keeps the text exactly


# ---------------------------------------------------------------- research guard

@unittest.skipIf(GUARD_OPEN, "the human-verified set is frozen: the guard is open")
class GuardTests(unittest.TestCase):
    """The page is GENERATED; the guard is made to treat its hash or pixels as protected. The recognizer is a trap."""

    def setUp(self):
        self.page = fx.make_tamil_page(seed=12, lines=3)
        self.data = fx.png(self.page.gray)

    def test_a_development_page_hash_is_refused_at_upload(self):
        sha = __import__("hashlib").sha256(self.data).hexdigest()
        real = devguard.development_master_hashes()
        with mock.patch.object(devguard, "development_master_hashes", return_value={**real, sha: "generated.png"}):
            with self.assertRaises(inputs.InputError):
                inputs.decode_upload("page.png", self.data)

    def test_development_pixels_are_refused_before_any_crop_is_read(self):
        d = inputs.decode_upload("page.png", self.data)
        seg = fullpage.segment_page(d)
        master = sorted(devguard.development_master_hashes())[0]
        with mock.patch.object(devguard, "development_page_for_pixels", return_value=master):
            with self.assertRaises(guard.ProtectedDataError):
                fullpage.recognize_page(trap_model(), d, seg)

    def test_a_protected_file_is_refused_at_upload(self):
        with mock.patch.object(guard, "_size_index", return_value={len(self.data): ["<protected twin>"]}), \
                mock.patch.object(guard.pio, "protected_match", return_value="<protected twin>"):
            with self.assertRaises(inputs.InputError):
                inputs.decode_upload("page.png", self.data)


if __name__ == "__main__":
    unittest.main()
