"""
Tests of the angle-aware line geometry (pageocr/line_geometry.py) and its use in the diagnostic view.

    python -B -m unittest -v tests/test_line_geometry.py

Pages are generated with KNOWN line geometry (tests/slant_fixtures.py): per-line angles, uneven spacing, ruling,
curved lines, perspective. The truth is the generator's own geometry, never a model output. The geometry is display
only: these tests also check that it changes no segmentation, crop or recognition.
"""
import io
import math
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import fullpage_fixtures as fx  # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)

import slant_fixtures as sf  # noqa: E402
from pageocr import line_geometry as lg  # noqa: E402
from pageocr import pipeline  # noqa: E402
from pageocr.layout import OcrImage  # noqa: E402
from wordocr import checkpoints, fullpage, inputs, service  # noqa: E402
from wordocr import diagnostics as dg  # noqa: E402

TOL_DEG = 0.5
P = fullpage.PageSettings().seg_params()
_CACHE = {}


def seg_of(name, **kw):
    """(page, segmentation, geometry) of a generated page; cached per case."""
    if name not in _CACHE:
        params = kw.pop("params", P)
        page = sf.make_slanted_page(seed=kw.pop("seed", 3), **kw)
        seg = pipeline.segment(OcrImage(gray=page.gray), params)
        _CACHE[name] = (page, seg, lg.line_geometry(seg))
    return _CACHE[name]


def match(geo, truth_line):
    """The geometry line whose original-frame polygon contains the truth line's centre (None if none does)."""
    c = tuple(float(v) for v in truth_line.centre)
    for r in geo["lines"]:
        if cv2.pointPolygonTest(np.array(r["polygon_original"], np.float32), c, False) >= 0:
            return r
    return None


class AngleAccuracy(unittest.TestCase):
    """Straight lines with known angles: every line gets a reliable local angle within TOL_DEG of the truth."""

    CASES = {
        "skew_2deg": dict(angles=[2] * 6),
        "skew_4deg": dict(angles=[4] * 6),
        "skew_minus_3deg": dict(angles=[-3] * 6),
        "uneven_spacing": dict(angles=[1] * 6, pitch=[60, 95, 70, 120, 64]),
        "ruled_below": dict(angles=[0, 1, -1, 2, 0, -2], rule="below"),
        "perspective": dict(angles=[0] * 6, perspective=0.12),
    }

    def test_angles_within_tolerance(self):
        for name, kw in self.CASES.items():
            page, seg, geo = seg_of(name, **dict(kw))
            with self.subTest(case=name):
                self.assertEqual(len(geo["lines"]), len(page.lines))
                for tl in page.lines:
                    r = match(geo, tl)
                    self.assertIsNotNone(r, f"no line box holds the centre of truth line {tl.index}")
                    self.assertTrue(r["reliable"], (tl.index, r["warnings"]))
                    self.assertEqual(r["source"], "local")
                    self.assertLessEqual(abs(r["angle_original_deg"] - tl.angle_deg), TOL_DEG, (tl.index, r["angle_original_deg"], tl.angle_deg))

    def test_perspective_lines_have_different_angles(self):
        page, seg, geo = seg_of("perspective", angles=[0] * 6, perspective=0.12)
        truth = [tl.angle_deg for tl in page.lines]
        self.assertGreater(max(truth) - min(truth), 3.0)                  # the fixture really fans the lines out
        got = [match(geo, tl)["angle_original_deg"] for tl in page.lines]
        self.assertGreater(max(got) - min(got), 3.0)
        self.assertEqual(np.sign(np.diff(got)).tolist(), np.sign(np.diff(truth)).tolist())

    def test_deskewed_page_angle_is_local_in_the_ocr_frame_and_global_in_the_original(self):
        page, seg, geo = seg_of("skew_4deg", angles=[4] * 6)
        self.assertTrue(seg.skew["applied"])
        self.assertTrue(geo["global"]["applied"])
        for r in geo["lines"]:
            self.assertLessEqual(abs(r["angle_ocr_deg"]), TOL_DEG)
            self.assertLessEqual(abs(r["angle_original_deg"] - 4.0), TOL_DEG)


class Honesty(unittest.TestCase):
    """Where no single trustworthy angle exists, none is claimed."""

    def test_flat_page_has_no_false_curvature(self):
        page, seg, geo = seg_of("flat", angles=[0] * 6)
        self.assertEqual(geo["summary"]["curved"], 0)
        self.assertTrue(all(r["reliable"] for r in geo["lines"]))

    def test_curved_lines_are_drawn_as_polylines_without_an_angle(self):
        page, seg, geo = seg_of("curved", angles=[0] * 5, curve_amp=20)
        poly = [r for r in geo["lines"] if r["shape"] == "polyline"]
        self.assertGreaterEqual(len(poly), 4)
        for r in poly:
            self.assertFalse(r["reliable"])
            self.assertIsNone(r["angle_original_deg"])
            self.assertTrue(any("curved" in w for w in r["warnings"]))
            self.assertGreaterEqual(len(r["polygon_ocr"]), 18)

    def test_merged_lines_of_different_angles_claim_no_angle(self):
        # lines at different angles: the horizontal-band segmentation merges them; the geometry must say so, not guess
        page, seg, geo = seg_of("mixed", angles=[0, 1.5, -1.5, 3, -2.5, 1])
        self.assertLess(len(geo["lines"]), len(page.lines))
        for r in geo["lines"]:
            self.assertFalse(r["reliable"], r)
            self.assertNotEqual(r["source"], "local")
            self.assertTrue(r["warnings"])

    def test_every_reliable_angle_is_right_on_every_case(self):
        cases = {"ruled_through_rule_removal": dict(angles=[0, 1, -1, 2, 0, -2], rule="through", params=fullpage.PageSettings(exp_rule_removal=True).seg_params()),
                 "mixed_wide": dict(angles=[0, 3, -3, 4, -4, 2]), "curved_14": dict(angles=[0] * 5, curve_amp=14)}
        for name, kw in cases.items():
            page, seg, geo = seg_of(name, **kw)
            with self.subTest(case=name):
                for r in geo["lines"]:
                    if not r["reliable"]:
                        continue
                    poly = np.array(r["polygon_original"], np.float32)
                    inside = [tl for tl in page.lines if cv2.pointPolygonTest(poly, tuple(float(v) for v in tl.centre), False) >= 0]
                    self.assertEqual(len(inside), 1, f"{r['id']} claims an angle but holds {len(inside)} truth lines")
                    if inside[0].angle_deg is not None:
                        self.assertLessEqual(abs(r["angle_original_deg"] - inside[0].angle_deg), TOL_DEG)


class Polygons(unittest.TestCase):
    def test_polygons_are_valid_in_their_frames_and_hold_their_words(self):
        for name, kw in {"skew_4deg": dict(angles=[4] * 6), "uneven_spacing": dict(angles=[1] * 6, pitch=[60, 95, 70, 120, 64]),
                         "curved": dict(angles=[0] * 5, curve_amp=20)}.items():
            page, seg, geo = seg_of(name, **kw)
            Ho, Wo = seg.ocr.original_shape
            H, W = seg.ocr.gray.shape
            with self.subTest(case=name):
                for r in geo["lines"]:
                    po, pr = np.array(r["polygon_ocr"]), np.array(r["polygon_original"])
                    self.assertTrue((po[:, 0] >= -2).all() and (po[:, 0] <= W + 2).all() and (po[:, 1] >= -2).all() and (po[:, 1] <= H + 2).all())
                    self.assertTrue((pr[:, 0] >= -2).all() and (pr[:, 0] <= Wo + 2).all() and (pr[:, 1] >= -2).all() and (pr[:, 1] <= Ho + 2).all())
                    self.assertGreater(cv2.contourArea(po.astype(np.float32)), 0)
                    # every word region of this line has its centre inside the line box (OCR frame)
                    for w in seg.words:
                        if (w.block_index, w.line_index) == (r["block"], r["line"]):
                            c = ((w.box[0] + w.box[2]) / 2.0, (w.box[1] + w.box[3]) / 2.0)
                            self.assertGreaterEqual(cv2.pointPolygonTest(po.astype(np.float32), c, True), -1.0, (r["id"], w.id))

    def test_original_polygon_is_the_exact_affine_image_of_the_ocr_polygon(self):
        page, seg, geo = seg_of("skew_4deg", angles=[4] * 6)
        A = np.array(seg.ocr.to_original)
        for r in geo["lines"]:
            po = np.array(r["polygon_ocr"])
            expect = np.c_[po, np.ones(len(po))] @ A.T
            self.assertLess(np.abs(expect - np.array(r["polygon_original"])).max(), 0.2)   # rounding to 0.1 px only

    def test_close_lines_stay_distinguishable(self):
        page, seg, geo = seg_of("uneven_spacing", angles=[1] * 6, pitch=[60, 95, 70, 120, 64])
        for r in geo["lines"]:
            area = cv2.contourArea(np.array(r["polygon_ocr"], np.float32))
            self.assertLess(r["overlap_with_next_px"], 0.05 * area, r["id"])

    def test_ids_follow_the_segmentation_lines(self):
        page, seg, geo = seg_of("skew_2deg", angles=[2] * 6)
        ids = [f"b{bi:02d}_l{li:03d}" for bi, ls in enumerate(seg.lines) for li, _ in enumerate(ls)]
        self.assertEqual([r["id"] for r in geo["lines"]], ids)
        self.assertEqual([r["band_ocr"][1:4:2] for r in geo["lines"]], [[ln.y1, ln.y2] for ls in seg.lines for ln in ls])


class DisplayOnly(unittest.TestCase):
    def test_geometry_changes_nothing_in_the_segmentation(self):
        page = sf.make_slanted_page(seed=5, angles=[3] * 4)
        a = pipeline.segment(OcrImage(gray=page.gray), P)
        before = [(w.id, w.box, w.box_original) for w in a.words]
        crops = [c.tobytes() for c in pipeline.ocr_crops(a.ocr, a.words, 5)]
        lg.line_geometry(a)
        self.assertEqual([(w.id, w.box, w.box_original) for w in a.words], before)
        self.assertEqual([c.tobytes() for c in pipeline.ocr_crops(a.ocr, a.words, 5)], crops)
        b = pipeline.segment(OcrImage(gray=page.gray), P)
        self.assertEqual([(w.id, w.box) for w in b.words], [(w.id, w.box) for w in a.words])

    def test_rebuilt_components_match_the_segmentation(self):
        page, seg, geo = seg_of("skew_4deg", angles=[4] * 6)
        comps = lg.rebuild_components(seg)
        for ls in seg.lines:
            for ln in ls:
                c = comps[np.array(ln.components)]
                share = np.minimum(c[:, 3], ln.y2) - np.maximum(c[:, 1], ln.y1)
                self.assertTrue((share > 0).all())          # every member overlaps its own band, as find_lines assigned it


class DiagnosticIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = service.load_model(checkpoints.DEFAULT_ID, "cpu")
        cls.page = sf.make_slanted_page(seed=7, angles=[3] * 4)
        data = fx.png(cls.page.gray)
        cls.d = inputs.decode_upload("slanted.png", data)
        cls.data = data
        cls.seg = fullpage.segment_page(cls.d, fullpage.PageSettings())
        cls.res = fullpage.recognize_page(cls.model, cls.d, cls.seg, fullpage.PageSettings())
        cls.diag = dg.analyse(cls.d, cls.seg, cls.res, fullpage.PageSettings())

    def test_record_has_geometry_with_the_same_line_ids(self):
        g = self.diag["line_geometry"]
        self.assertNotIn("error", g)
        self.assertEqual([r["id"] for r in g["lines"]], [ln["id"] for ln in self.diag["lines"]])

    def test_horizontal_overlay_is_unchanged_and_still_the_default(self):
        old = dg._draw(self.seg.ocr.gray, [(r["box_ocr"], r["number"], bool(r["warnings"])) for r in self.diag["regions"]], self.seg.h)
        for ln in self.diag["lines"]:
            cv2.rectangle(old, (ln["x1_ocr"], ln["y1_ocr"]), (ln["x2_ocr"] - 1, ln["y2_ocr"] - 1), (0, 150, 0), 1)
        self.assertTrue(np.array_equal(dg.overlay_ocr(self.seg, self.diag), old))
        self.assertTrue(np.array_equal(dg.overlay_ocr(self.seg, self.diag, True, "horizontal"), old))
        plain = dg._draw(self.d.gray, [(r["box_original"], r["number"], bool(r["warnings"])) for r in self.diag["regions"]], self.seg.h)
        self.assertTrue(np.array_equal(dg.overlay_original(self.d, self.diag, self.seg.h), plain))

    def test_both_line_modes_draw_in_both_frames(self):
        for mode in ("angle", "horizontal"):
            o = dg.overlay_original(self.d, self.diag, self.seg.h, mode, self.seg)
            c = dg.overlay_ocr(self.seg, self.diag, True, mode)
            self.assertEqual(o.shape[:2], self.d.gray.shape)
            self.assertEqual(c.shape[:2], self.seg.ocr.gray.shape)
        a = dg.overlay_original(self.d, self.diag, self.seg.h, "angle", self.seg)
        h = dg.overlay_original(self.d, self.diag, self.seg.h, "horizontal", self.seg)
        self.assertFalse(np.array_equal(a, h))

    def test_bundle_carries_geometry_and_angle_overlay(self):
        z = zipfile.ZipFile(io.BytesIO(dg.bundle_zip(self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, True, self.data)))
        names = set(z.namelist())
        self.assertIn("line_geometry.json", names)
        self.assertIn("overlay_original_frame_angle_lines.png", names)
        self.assertIn("line_geometry.json", z.read("SHA256SUMS.txt").decode())

    def test_recognition_is_the_same_with_or_without_diagnostics(self):
        again = fullpage.recognize_page(self.model, self.d, fullpage.segment_page(self.d, fullpage.PageSettings()), fullpage.PageSettings())
        self.assertEqual([(w.id, w.box, w.text, w.uncalibrated_score) for w in again.words],
                         [(w.id, w.box, w.text, w.uncalibrated_score) for w in self.res.words])

    def test_a_geometry_failure_is_recorded_not_raised(self):
        real = lg.line_geometry

        def fail(seg, comps=None):
            raise RuntimeError("simulated failure")
        lg.line_geometry = fail
        try:
            diag = dg.analyse(self.d, self.seg, self.res, fullpage.PageSettings())
        finally:
            lg.line_geometry = real
        self.assertIn("simulated failure", diag["line_geometry"]["error"])
        self.assertEqual([r["text"] for r in diag["regions"]], [r["text"] for r in self.diag["regions"]])
        self.assertTrue(np.array_equal(dg.overlay_ocr(self.seg, diag), dg.overlay_ocr(self.seg, self.diag)))   # horizontal still works
        with self.assertRaises(ValueError):
            dg.overlay_ocr(self.seg, diag, True, "angle")
        z = zipfile.ZipFile(io.BytesIO(dg.bundle_zip(self.d, self.seg, self.res, fullpage.PageSettings(), diag, False, None)))
        self.assertNotIn("overlay_original_frame_angle_lines.png", z.namelist())

    def test_rebuild_refuses_a_segmentation_it_does_not_match(self):
        other = pipeline.segment(OcrImage(gray=sf.make_slanted_page(seed=8, angles=[0] * 3).gray), P)
        fake = type(self.seg).__new__(type(self.seg))
        for f in self.seg.__dataclass_fields__:
            object.__setattr__(fake, f, getattr(self.seg, f))
        object.__setattr__(fake, "ocr", other.ocr)                  # lines of one page, pixels of another
        with self.assertRaises(RuntimeError):
            lg.rebuild_components(fake)


if __name__ == "__main__":
    unittest.main()
