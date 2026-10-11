"""
Tests of the diagnostic workspace additions: wordocr/diagnostics.py, wordocr/diagnostics_view.py (through app.py),
wordocr/crop_compare.py, pageocr/rules.py and the experimental options of wordocr/fullpage.py.

    python -B -m unittest -v tests/test_diagnostics.py

All pages are generated (tests/fullpage_fixtures.py, dev_tools/synthetic_conditions.py); labels come from the
generator. The checkpoint is the registered Experiment 002-A file, loaded on the CPU.
"""
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "dev_tools"))

import numpy as np  # noqa: E402

import fullpage_fixtures as fx  # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)

import synthetic_conditions as sc  # noqa: E402
from pageocr import pipeline, rules  # noqa: E402
from wordocr import checkpoints, crop_compare, fullpage, inputs, service  # noqa: E402
from wordocr import diagnostics as dg  # noqa: E402

MODEL = None


def model():
    global MODEL
    if MODEL is None:
        MODEL = service.load_model(checkpoints.DEFAULT_ID, "cpu")
    return MODEL


def run(gray, settings=fullpage.PageSettings(), name="page.png"):
    data = fx.png(gray)
    d = inputs.decode_upload(name, data)
    seg = fullpage.segment_page(d, settings)
    res = fullpage.recognize_page(model(), d, seg, settings)
    return data, d, seg, res


class DefaultsUnchanged(unittest.TestCase):
    def test_default_settings_look_exactly_as_before(self):
        s = fullpage.PageSettings()
        self.assertEqual(set(s.as_dict()), {"margin", "sauvola_k", "gap_floor_h", "deskew"})
        self.assertEqual(s.experimental(), {})
        p = s.seg_params()
        self.assertFalse(p.exp_rule_removal or p.exp_contrast_clahe)

    def test_default_run_has_no_experimental_trace_and_crops_are_the_pipeline_crops(self):
        page = fx.make_tamil_page(seed=21, lines=3)
        _, d, seg, res = run(page.gray)
        self.assertFalse(any(k.startswith("exp_") for k in res.segmentation["parameters"]))
        self.assertFalse(any("EXPERIMENTAL" in s for s in res.segmentation["steps"]))
        a = fullpage.crop_images(seg, fullpage.PageSettings())
        b = pipeline.ocr_crops(seg.ocr, seg.words, res.margin_px)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertEqual(res.text, page.text)

    def test_experimental_keys_are_recorded_when_switched_on(self):
        s = fullpage.PageSettings(exp_rule_removal=True)
        self.assertEqual(s.as_dict()["exp_rule_removal"], True)
        page = sc.make("ruled_through", 31)
        _, d, seg, res = run(page.gray, s)
        self.assertTrue(res.segmentation["parameters"]["exp_rule_removal"])
        self.assertTrue(any("EXPERIMENTAL" in x for x in res.segmentation["steps"]))


class DiagnosticRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = fx.make_tamil_page(seed=22, columns=2, lines=4, heading=True)
        cls.data, cls.d, cls.seg, cls.res = run(cls.page.gray)
        cls.diag = dg.analyse(cls.d, cls.seg, cls.res, fullpage.PageSettings())

    def test_regions_are_in_reading_order_with_stable_ids(self):
        r = self.diag["regions"]
        self.assertEqual([x["number"] for x in r], list(range(1, len(r) + 1)))
        self.assertEqual([x["id"] for x in r], [w.id for w in self.res.words])
        self.assertEqual(len({x["id"] for x in r}), len(r))
        self.assertEqual([x["text"] for x in r], [w.text for w in self.res.words])

    def test_crop_boxes_reproduce_the_recognizer_crops_exactly(self):
        H, W = self.seg.ocr.gray.shape
        crops = pipeline.ocr_crops(self.seg.ocr, self.seg.words, self.diag["margin_px"])
        for r, c in zip(self.diag["regions"], crops):
            x1, y1, x2, y2 = r["crop_box_ocr"]
            self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H)
            self.assertTrue(np.array_equal(self.seg.ocr.gray[y1:y2, x1:x2], c))
            self.assertEqual((r["crop_width"], r["crop_height"]), (c.shape[1], c.shape[0]))
            self.assertEqual(r["crop_sha256"], hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest())
            bx1, by1, bx2, by2 = r["box_ocr"]
            self.assertTrue(x1 <= bx1 < bx2 <= x2 and y1 <= by1 < by2 <= y2)

    def test_lines_list_every_region_once(self):
        listed = sorted(i for ln in self.diag["lines"] for i in ln["regions"])
        self.assertEqual(listed, list(range(len(self.diag["regions"]))))

    def test_a_clean_page_raises_no_warning(self):
        self.assertEqual(self.diag["regions_with_any_warning"], 0, self.diag["warning_counts"])

    def test_unavailable_metadata_is_declared_not_invented(self):
        self.assertIn("ground_truth", self.diag["unavailable"])
        self.assertNotIn("accuracy", json.dumps(self.diag["counts"]))

    def test_a_result_of_another_segmentation_is_refused(self):
        other = fx.make_tamil_page(seed=23, lines=2)
        _, d2, seg2, res2 = run(other.gray)
        with self.assertRaises(ValueError):
            dg.analyse(self.d, self.seg, res2, fullpage.PageSettings())

    def test_the_bundle_is_complete_and_self_consistent(self):
        z = zipfile.ZipFile(io.BytesIO(dg.bundle_zip(self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, True, self.data)))
        names = set(z.namelist())
        for f in ("metadata.json", "regions.csv", "lines.json", "page_text.txt", "overlay_ocr_frame.png", "overlay_original_frame.png", "SUMMARY.md", "SHA256SUMS.txt"):
            self.assertIn(f, names)
        meta = json.loads(z.read("metadata.json"))
        self.assertEqual(meta["input"]["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertEqual(meta["model"]["checkpoint_sha256"], checkpoints.get("002a").sha256)
        self.assertEqual(z.read("page_text.txt").decode("utf-8"), self.res.text)
        self.assertEqual(hashlib.sha256(z.read("input/page.png")).hexdigest(), self.d.sha256)
        rows = z.read("regions.csv").decode("utf-8-sig").splitlines()
        self.assertEqual(len(rows) - 1, len(self.res.words))
        for r in meta["regions"]:
            self.assertIn(r["crop_file"], names)
        for line in z.read("SHA256SUMS.txt").decode().splitlines():
            h, p = line.split(" *", 1)
            self.assertEqual(hashlib.sha256(z.read(p)).hexdigest(), h)
        self.assertNotIn(str(Path.home()).lower(), z.read("metadata.json").decode("utf-8").lower().replace("\\\\", "\\"))

    def test_the_bundle_without_input_and_with_wrong_input(self):
        z = zipfile.ZipFile(io.BytesIO(dg.bundle_zip(self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, False, self.data)))
        self.assertFalse(any(n.startswith("input/") for n in z.namelist()))
        self.assertFalse(json.loads(z.read("metadata.json"))["input"]["included_in_bundle"])
        with self.assertRaises(ValueError):
            dg.bundle_zip(self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, True, b"other bytes")

    def test_save_bundle_never_overwrites_and_refuses_protected_folders(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        out = dg.save_bundle(tmp / "run1", self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, False)
        self.assertTrue((out / "metadata.json").is_file())
        with self.assertRaises(FileExistsError):
            dg.save_bundle(tmp / "run1", self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, False)
        from pageocr import ProtectedPathError
        for bad in (tmp / "real_docs" / "x", ROOT / "runs" / "diag_x", ROOT / "pageocr" / "diag_x"):
            with self.assertRaises(ProtectedPathError):
                dg.save_bundle(bad, self.d, self.seg, self.res, fullpage.PageSettings(), self.diag, False)


class WarningsOnBadPages(unittest.TestCase):
    def test_ruled_page_regions_are_flagged(self):
        page = sc.make("ruled_through", 32)
        _, d, seg, res = run(page.gray)
        diag = dg.analyse(d, seg, res, fullpage.PageSettings())
        self.assertGreater(diag["regions_with_any_warning"], 0)
        flagged = {k for r in diag["regions"] for k in r["warnings"]}
        self.assertTrue(flagged & {"rule_line_in_crop", "very_wide", "taller_than_line", "ink_at_crop_edge", "overlaps_neighbour"}, flagged)


class CropComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = fx.make_tamil_page(seed=24, lines=3)
        cls.data, cls.d, cls.seg, cls.res = run(cls.page.gray)

    def test_generator_labelled_words_agree_in_both_conditions(self):
        samples = [crop_compare.Sample(id=f"w{i}", box=w.box, reference=w.text, reference_confirmed=True, reference_source="generator")
                   for i, w in enumerate(self.page.words[:12])]
        rep = crop_compare.compare(model(), self.d, self.seg, self.res.margin_px, samples, result=self.res)
        s = rep["summary"]
        self.assertEqual(s["with_confirmed_reference"], 12)
        self.assertEqual(s["crop_integrity"], {"intact": 12})
        self.assertTrue(all(r["pipeline_prediction_matches_page_result"] for r in rep["samples"]))
        self.assertEqual(s["A_manual_words_correct"], 12)

    def test_unconfirmed_references_are_not_scored_and_predictions_are_not_references(self):
        w = self.page.words[0]
        rep = crop_compare.compare(model(), self.d, self.seg, self.res.margin_px,
                                   [crop_compare.Sample(id="s", box=w.box, reference=w.text, reference_confirmed=False)])
        r = rep["samples"][0]
        self.assertFalse(r["reference_confirmed"])
        self.assertNotIn("manual_correct", r)
        self.assertIn("no confirmed reference", rep["summary"]["note"])

    def test_invalid_boxes_are_reported_not_read(self):
        H, W = self.d.gray.shape
        rep = crop_compare.compare(model(), self.d, self.seg, self.res.margin_px,
                                   [crop_compare.Sample(id="out", box=(W - 5, 0, W + 50, 20)), crop_compare.Sample(id="empty", box=(10, 10, 10, 30))])
        self.assertTrue(all("error" in r for r in rep["samples"]))

    def test_integrity_verdicts(self):
        regions = [{"id": "a", "box_original": (0, 0, 50, 20)}, {"id": "b", "box_original": (60, 0, 110, 20)}]
        self.assertEqual(crop_compare.integrity((0, 0, 50, 20), regions)["verdict"].split(":")[0], "intact")
        self.assertEqual(crop_compare.integrity((0, 0, 110, 20), regions)["verdict"].split(":")[0], "split")
        self.assertEqual(crop_compare.integrity((0, 0, 20, 20), regions)["verdict"].split(":")[0], "merged")
        self.assertEqual(crop_compare.integrity((0, 0, 70, 20), [regions[0]])["verdict"].split(":")[0], "clipped")
        self.assertEqual(crop_compare.integrity((200, 0, 250, 20), regions)["verdict"].split(":")[0], "missed")


class RuleRemoval(unittest.TestCase):
    def test_long_horizontal_runs_go_and_crossing_strokes_stay(self):
        ink = np.zeros((60, 400), bool)
        ink[30:32, :] = True                       # a rule
        ink[10:50, 100:103] = True                 # a vertical stroke crossing it
        ink[20:24, 200:230] = True                 # a short horizontal stroke (part of a letter)
        clean, rule = rules.remove_rules(ink, h=20.0)
        self.assertFalse(clean[30, 50] or clean[30, 300])
        self.assertTrue(clean[30:32, 100:103].all())
        self.assertTrue(clean[20:24, 200:230].all())
        self.assertTrue(rule[30, 50])

    def test_text_without_ruling_is_untouched(self):
        page = fx.make_tamil_page(seed=25, lines=3)
        a = fullpage.segment_page(inputs.decode_upload("p.png", fx.png(page.gray)), fullpage.PageSettings())
        b = fullpage.segment_page(inputs.decode_upload("p.png", fx.png(page.gray)), fullpage.PageSettings(exp_rule_removal=True))
        self.assertEqual([w.box for w in a.words], [w.box for w in b.words])

    def test_on_a_ruled_page_it_restores_the_words(self):
        page = sc.make("ruled_through", 33)
        _, d, seg0, res0 = run(page.gray)
        _, d, seg1, res1 = run(page.gray, fullpage.PageSettings(exp_rule_removal=True))
        import editdistance
        e0, e1 = editdistance.eval(res0.text, page.text), editdistance.eval(res1.text, page.text)
        self.assertLess(e1, e0)
        self.assertLessEqual(e1 / len(page.text), 0.05)


class AppDiagnosticMode(unittest.TestCase):
    """app.py through Streamlit's test runner: with the mode off the page is as before; switching it on adds the panel
    and leaves the OCR result exactly as it was."""

    def test_mode_off_by_default_and_on_does_not_change_the_result(self):
        from streamlit.testing.v1 import AppTest
        page = fx.make_tamil_page(seed=26, lines=3)
        data = fx.png(page.gray)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
        at.run()
        self.assertFalse(at.checkbox(key="page_diag").value)
        for k in ("page_exp_rule", "page_exp_clahe", "page_exp_fill"):
            self.assertFalse(at.checkbox(key=k).value)
        at.file_uploader(key=[u.key for u in at.file_uploader if str(u.key).startswith("page_upload")][0]).set_value(("p.png", data, "image/png"))
        at.run()
        at.button(key="run_page").click().run()
        text_off = at.session_state["page"]["result"].text
        self.assertEqual(text_off, page.text)
        self.assertFalse(any("Diagnostics" == s.value for s in at.subheader))
        at.checkbox(key="page_diag").check().run()
        self.assertEqual(at.session_state["page"]["result"].text, text_off)
        self.assertTrue(any("Diagnostics" == s.value for s in at.subheader))
        self.assertEqual(list(at.exception), [])
        self.assertFalse(any("diagnostic view failed" in e.value for e in at.error))


if __name__ == "__main__":
    unittest.main()
