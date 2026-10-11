"""
Tests for pageocr.boxes and pageocr.export (V2): the box schema, JSON/CSV files, validation, cropping, the
preview <-> original coordinate mapping, and the text/JSON export.

    .venv\\Scripts\\python -B -m unittest discover -s tests -p "test_pageocr_*.py"

Only generated arrays and temporary files are used. No model is loaded here.
"""
import json
import random
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pageocr import boxes as bx, export          # noqa: E402

BOXES = [bx.Box("w001", 0, 0, 120, 48, 0, 0), bx.Box("w002", 130, 2, 75, 44, 0, 0),
         bx.Box("தலைப்பு-1", 10, 60, 300, 52, 1, 0), bx.Box("c2,first", 400, 0, 99, 40, 0, 1)]


def gradient_page(height=200, width=500) -> np.ndarray:
    """Every pixel's value depends on its position, so a wrong or shifted crop cannot pass by accident."""
    yy, xx = np.mgrid[0:height, 0:width]
    return ((xx * 7 + yy * 13) % 251).astype(np.uint8)


class BoxSchemaTests(unittest.TestCase):
    def test_fields_are_integers(self):
        b = bx.Box("a", np.int64(3), 4, 5, 6, 0, 1)
        self.assertEqual((b.x, b.y, b.w, b.h, b.line_index, b.column_index), (3, 4, 5, 6, 0, 1))
        for v in (b.x, b.y, b.w, b.h, b.line_index, b.column_index):
            self.assertIs(type(v), int)
        self.assertEqual((b.x1, b.y1), (8, 10))
        self.assertEqual(bx.SCHEMA_VERSION, 1)

    def test_invalid_boxes_rejected(self):
        bad = [dict(x=-1), dict(y=-1), dict(w=0), dict(h=0), dict(w=-5), dict(h=-5), dict(x=1.5), dict(w=10.0),
               dict(y="3"), dict(h=None), dict(x=True), dict(line_index=-1), dict(column_index=-1),
               dict(line_index=0.0), dict(id=""), dict(id="  "), dict(id=" a"), dict(id=7), dict(id=None),
               dict(x=float("nan")), dict(w=float("inf"))]
        for change in bad:
            fields = dict(id="a", x=1, y=1, w=10, h=10, line_index=0, column_index=0)
            fields.update(change)
            with self.assertRaises(bx.BoxError, msg=str(change)):
                bx.Box(**fields)

    def test_box_must_be_inside_the_page(self):
        bx.validate_boxes([bx.Box("edge", 400, 150, 100, 50, 0, 0)], 500, 200)       # touching the border is inside
        for b in (bx.Box("r", 401, 0, 100, 10, 0, 0), bx.Box("b", 0, 151, 10, 50, 0, 0), bx.Box("far", 500, 0, 1, 1, 0, 0),
                  bx.Box("big", 0, 0, 501, 200, 0, 0)):
            with self.assertRaises(bx.BoxError, msg=b.id):
                bx.validate_boxes([b], 500, 200)
        with self.assertRaises(bx.BoxError):
            bx.validate_boxes([BOXES[0], BOXES[0]], 500, 200)                        # duplicate id
        with self.assertRaises(bx.BoxError):
            bx.validate_boxes([{"id": "a", "x": 0, "y": 0, "w": 1, "h": 1}], 500, 200)
        for size in ((0, 200), (500, -1), (500.0, 200), (None, 200)):
            with self.assertRaises(bx.BoxError, msg=str(size)):
                bx.validate_boxes(BOXES, *size)


class BoxFileTests(unittest.TestCase):
    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "boxes.json"
            bx.save_json(f, BOXES, page_width=500, page_height=200, page_sha256="ab" * 32)
            loaded = bx.load_json(f)
            self.assertEqual(loaded.schema_version, bx.SCHEMA_VERSION)
            self.assertEqual(list(loaded.boxes), BOXES)
            self.assertEqual(loaded.page, {"width": 500, "height": 200, "sha256": "ab" * 32})
            raw = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], 1)
            self.assertEqual(list(raw["boxes"][0]), list(bx.FIELDS))
            for k in bx.INT_FIELDS:
                self.assertIs(type(raw["boxes"][0][k]), int)
            with self.assertRaises(FileExistsError):                  # an existing file is not replaced silently
                bx.save_json(f, BOXES[:1])
            bx.save_json(f, BOXES[:1], overwrite=True)
            self.assertEqual(list(bx.load_json(f).boxes), BOXES[:1])
            self.assertIsNone(bx.load_json(f).page)

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "boxes.csv"
            bx.save_csv(f, BOXES)
            loaded = bx.load_csv(f)
            self.assertEqual(loaded.schema_version, bx.SCHEMA_VERSION)
            self.assertEqual(list(loaded.boxes), BOXES)
            lines = f.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "schema_version,id,x,y,w,h,line_index,column_index")
            self.assertEqual(lines[1], "1,w001,0,0,120,48,0,0")
            with self.assertRaises(FileExistsError):
                bx.save_csv(f, BOXES)
        self.assertEqual(list(bx.loads_csv("﻿" + bx.dumps_csv(BOXES).replace("\n", "\r\n")).boxes), BOXES)   # as Excel saves

    def test_json_and_csv_agree(self):
        self.assertEqual(bx.loads_json(bx.dumps_json(BOXES)).boxes, bx.loads_csv(bx.dumps_csv(BOXES)).boxes)

    def test_bad_json_rejected(self):
        good = json.loads(bx.dumps_json(BOXES[:2]))

        def variant(edit):
            doc = json.loads(json.dumps(good))
            edit(doc)
            return json.dumps(doc)

        bad = [
            "not json", "[]", '{"boxes": []}',
            variant(lambda d: d.update(schema_version=2)),
            variant(lambda d: d.update(schema_version="1")),
            variant(lambda d: d.pop("schema_version")),
            variant(lambda d: d.update(boxes={})),
            variant(lambda d: d.update(page=[1, 2])),
            variant(lambda d: d["boxes"][0].update(x=-1)),
            variant(lambda d: d["boxes"][0].update(w=0)),
            variant(lambda d: d["boxes"][0].update(h=1.5)),
            variant(lambda d: d["boxes"][0].update(y="2")),
            variant(lambda d: d["boxes"][0].pop("line_index")),
            variant(lambda d: d["boxes"][0].update(angle=3)),
            variant(lambda d: d["boxes"][1].update(id="w001")),
            variant(lambda d: d["boxes"].append("w003")),
        ]
        for text in bad:
            with self.assertRaises(bx.BoxError, msg=text[:80]):
                bx.loads_json(text)

    def test_bad_csv_rejected(self):
        head = "schema_version,id,x,y,w,h,line_index,column_index\n"
        bad = ["", "id,x,y,w,h\na,0,0,1,1\n", head.replace("line_index", "line") + "1,a,0,0,1,1,0,0\n",
               head + "2,a,0,0,1,1,0,0\n", head + "1,a,0,0,1,1,0\n", head + "1,a,0,0,1,1,0,0,9\n",
               head + "1,a,-1,0,1,1,0,0\n", head + "1,a,0,0,0,1,0,0\n", head + "1,a,0,0,1.0,1,0,0\n",
               head + "1,a,0,0,1,,0,0\n", head + "1,a,0,0,1,1,x,0\n", head + "1,,0,0,1,1,0,0\n",
               head + "1,a,0,0,1,1,0,0\n1,a,5,5,1,1,0,0\n", head + "1,a,1e2,0,1,1,0,0\n"]
        for text in bad:
            with self.assertRaises(bx.BoxError, msg=text[-40:]):
                bx.loads_csv(text)
        self.assertEqual(bx.loads_csv(head).boxes, ())


class CropTests(unittest.TestCase):
    def setUp(self):
        self.page = gradient_page()

    def test_crop_is_grayscale_uint8(self):
        crop = bx.crop_from_page(self.page, bx.Box("a", 30, 40, 120, 48, 0, 0))
        self.assertIsInstance(crop, np.ndarray)
        self.assertEqual((crop.ndim, crop.dtype), (2, np.uint8))

    def test_crop_coordinates(self):
        b = bx.Box("a", 30, 40, 120, 48, 0, 0)
        crop = bx.crop_from_page(self.page, b)
        self.assertEqual(crop.shape, (48, 120))                       # (h, w): no resampling
        np.testing.assert_array_equal(crop, self.page[40:88, 30:150])  # x1, y1 exclusive
        self.assertEqual(int(crop[0, 0]), int(self.page[40, 30]))
        self.assertEqual(int(crop[-1, -1]), int(self.page[87, 149]))
        np.testing.assert_array_equal(bx.crop_from_page(self.page, bx.Box("all", 0, 0, 500, 200, 0, 0)), self.page)
        self.assertEqual(bx.crop_from_page(self.page, bx.Box("px", 499, 199, 1, 1, 0, 0)).tolist(), [[int(self.page[199, 499])]])

    def test_crop_is_a_copy_and_the_page_is_unchanged(self):
        before = self.page.copy()
        crop = bx.crop_from_page(self.page, bx.Box("a", 5, 5, 20, 20, 0, 0))
        crop[:] = 0
        np.testing.assert_array_equal(self.page, before)

    def test_crop_rejects_bad_pages_and_boxes(self):
        b = bx.Box("a", 0, 0, 10, 10, 0, 0)
        for page in (np.dstack([self.page] * 3), self.page.astype(np.float32), self.page > 100, self.page.tolist(), None):
            with self.assertRaises(bx.BoxError):
                bx.crop_from_page(page, b)
        with self.assertRaises(bx.BoxError):
            bx.crop_from_page(self.page, bx.Box("out", 495, 0, 10, 10, 0, 0))


class CoordinateTests(unittest.TestCase):
    def test_scale_record(self):
        s = bx.PreviewScale(original_width=2480, original_height=3508, preview_width=1100, preview_height=1556)
        self.assertAlmostEqual(s.scale_x, 2480 / 1100)
        self.assertAlmostEqual(s.scale_y, 3508 / 1556)
        self.assertEqual(s.to_original(0, 0), (0, 0))
        self.assertEqual(s.to_original(1100, 1556), (2480, 3508))
        self.assertEqual(s.to_original(-5, 99999), (0, 3508))         # clamped to the page
        for bad in ((0, 10, 10, 10), (10, 10, 10, -1), (10.5, 10, 10, 10)):
            with self.assertRaises(bx.BoxError):
                bx.PreviewScale(*bad)

    def test_original_to_preview_to_original_round_trip(self):
        rng = random.Random(0)
        for _ in range(300):
            ow, oh = rng.randint(50, 6000), rng.randint(50, 8000)
            pw = rng.randint(20, ow)
            ph = max(1, round(oh * pw / ow))
            s = bx.PreviewScale(ow, oh, pw, ph)
            for _ in range(20):
                x, y = rng.randint(0, ow - 1), rng.randint(0, oh - 1)
                b = bx.Box("b", x, y, rng.randint(1, ow - x), rng.randint(1, oh - y), 0, 0)
                px0, py0, px1, py1 = s.rect_to_preview(b)
                self.assertTrue(0 <= px0 < px1 <= pw and 0 <= py0 < py1 <= ph)
                self.assertEqual(s.rect_to_original(px0, py0, px1, py1), (b.x, b.y, b.w, b.h))
                self.assertEqual(s.rect_to_original(px1, py1, px0, py0), (b.x, b.y, b.w, b.h))      # corners in any order

    def test_preview_rectangle_maps_into_the_page(self):
        s = bx.PreviewScale(3000, 2000, 750, 500)                     # exactly 4x
        self.assertEqual(s.rect_to_original(10, 20, 110, 70), (40, 80, 400, 200))
        self.assertEqual(s.rect_to_original(700, 450, 900, 900), (2800, 1800, 200, 200))       # clamped at the border
        with self.assertRaises(bx.BoxError):
            s.rect_to_original(10, 10, 10, 50)                        # empty
        x, y, w, h = s.rect_to_original(10.1, 20.1, 110.2, 70.2)
        bx.validate_boxes([bx.Box("drawn", x, y, w, h, 0, 0)], 3000, 2000)

    def test_crop_after_mapping_comes_from_the_original(self):
        page = gradient_page(400, 1000)
        s = bx.PreviewScale(1000, 400, 250, 100)
        x, y, w, h = s.rect_to_original(25, 10, 75, 30)
        self.assertEqual((x, y, w, h), (100, 40, 200, 80))
        np.testing.assert_array_equal(bx.crop_from_page(page, bx.Box("m", x, y, w, h, 0, 0)), page[40:120, 100:300])


@dataclass(frozen=True)
class FakeRecognition:
    text: str
    confidence: float = 0.5
    width_px: int = 100
    timesteps: int = 25
    clipped: bool = False


class ExportTests(unittest.TestCase):
    def setUp(self):
        # supplied out of reading order on purpose
        self.boxes = [bx.Box("d", 600, 0, 50, 40, 0, 1), bx.Box("b", 70, 0, 50, 40, 0, 0), bx.Box("c", 10, 50, 50, 40, 1, 0),
                      bx.Box("a", 10, 0, 50, 40, 0, 0), bx.Box("e", 600, 50, 50, 40, 1, 1), bx.Box("f", 660, 50, 50, 40, 1, 1)]
        self.recs = [FakeRecognition(t) for t in ("நான்கு", "இரண்டு", "மூன்று", "ஒன்று", "ஐந்து", "ஆறு")]

    def test_txt_layout(self):
        # column 0: line 0 = b, a (supplied order inside a line), line 1 = c; column 1: line 0 = d, line 1 = e f
        self.assertEqual(export.to_txt(self.boxes, self.recs), "இரண்டு ஒன்று\nமூன்று\n\nநான்கு\nஐந்து ஆறு\n")

    def test_txt_order_comes_from_box_metadata_not_from_geometry_or_text(self):
        moved = [bx.Box(b.id, 0, 0, 1, 1, b.line_index, b.column_index) for b in self.boxes]
        self.assertEqual(export.to_txt(moved, self.recs), export.to_txt(self.boxes, self.recs))
        swapped = list(self.boxes)
        swapped[1], swapped[3] = swapped[3], swapped[1]
        recs = list(self.recs)
        recs[1], recs[3] = recs[3], recs[1]
        self.assertTrue(export.to_txt(swapped, recs).startswith("ஒன்று இரண்டு\n"))

    def test_txt_empty_predictions(self):
        recs = [FakeRecognition(t) for t in ("", "இரண்டு", "", "", "ஐந்து", "")]
        self.assertEqual(export.to_txt(self.boxes, recs), "இரண்டு\n\nஐந்து\n")
        self.assertEqual(export.to_txt([], []), "")
        with self.assertRaises(ValueError):
            export.to_txt(self.boxes, self.recs[:2])

    def test_json_records(self):
        text = export.to_json(self.boxes, self.recs, "p" * 64, "c" * 64, "pageocr-2.0.0", "page.png", 800, 100,
                              skipped=[(bx.Box("z", 0, 0, 1, 1, 9, 0), "crop is too small")])
        doc = json.loads(text)
        self.assertIn("இரண்டு", text)                                # written as Tamil, not as \\u escapes
        self.assertEqual(doc["statement"], "V2 is an operator-assisted word-region recognition tool. "
                                           "It is not validated full-page OCR.")
        self.assertIn("UNCALIBRATED", doc["confidence_note"])
        self.assertEqual(doc["page"], {"name": "page.png", "sha256": "p" * 64, "width": 800, "height": 100})
        self.assertEqual([r["box_id"] for r in doc["recognitions"]], ["d", "b", "c", "a", "e", "f"])     # supplied order
        first = doc["recognitions"][0]
        self.assertEqual(first, {"box_id": "d", "x": 600, "y": 0, "w": 50, "h": 40, "line_index": 0, "column_index": 1,
                                 "text": "நான்கு", "confidence": 0.5, "clipped": False, "width_px": 100, "timesteps": 25,
                                 "page_sha256": "p" * 64, "checkpoint_sha256": "c" * 64, "engine_version": "pageocr-2.0.0"})
        self.assertEqual(doc["skipped"][0]["box_id"], "z")
        self.assertEqual(doc["skipped"][0]["reason"], "crop is too small")


if __name__ == "__main__":
    unittest.main()
