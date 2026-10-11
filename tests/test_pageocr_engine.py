"""
Tests for pageocr.engine, the page loader's guards, the checkpoint's integrity and the Streamlit page (V2).

    .venv\\Scripts\\python -B -m unittest discover -s tests -p "test_pageocr_*.py"

Every image here is generated in memory. No file under real_docs/ or the R1 review folder is opened, and no accuracy
is measured: the assertions are about shape, order, determinism, metadata and refusals.
"""
import ast
import hashlib
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pageocr                                                    # noqa: E402
from pageocr import boxes as bx, export, imageio as pio          # noqa: E402
from pageocr import engine as eng                                 # noqa: E402
from src import charset as cs                                     # noqa: E402
from src import preprocess as pp                                  # noqa: E402
from src.data.datasets import PAD_W_MULTIPLE                      # noqa: E402

CHECKPOINT = ROOT / "runs" / "experiment_002a" / "best.pth"
FROZEN_SHA256 = "fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1"
NEW_SOURCES = [ROOT / "pageocr" / n for n in ("__init__.py", "engine.py", "boxes.py", "imageio.py", "export.py")] + [ROOT / "app_page.py"]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def synthetic_crop(height: int, width: int, seed: int) -> np.ndarray:
    """A grayscale 'word': noisy paper with a few dark blurred strokes. Not text; only something for the model to chew."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 228.0) + rng.normal(0, 4, (height, width))
    for _ in range(max(2, width // 14)):
        x, y = int(rng.integers(2, max(3, width - 2))), int(rng.integers(height // 4, 3 * height // 4))
        cv2.ellipse(img, (x, y), (int(rng.integers(2, 7)), int(rng.integers(3, max(4, height // 3)))),
                    float(rng.integers(0, 180)), 0, 300, 40.0, 2)
    return np.clip(cv2.GaussianBlur(img, (3, 3), 0), 0, 255).astype(np.uint8)


def synthetic_page(height: int = 300, width: int = 900, seed: int = 0):
    """(page array, boxes): a noisy page with synthetic 'words' pasted at known places, two lines in one column."""
    rng = np.random.default_rng(seed)
    page = np.clip(np.full((height, width), 232.0) + rng.normal(0, 3, (height, width)), 0, 255).astype(np.uint8)
    boxes, n = [], 0
    for line, y in enumerate((40, 170)):
        x = 30
        for w in (150, 90, 220):
            page[y:y + 60, x:x + w] = synthetic_crop(60, w, 100 + n)
            boxes.append(bx.Box(f"w{n:02d}", x, y, w, 60, line, 0))
            x, n = x + w + 35, n + 1
    return page, boxes


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = eng.OcrEngine(CHECKPOINT, device="cpu")
        cls.crops = [synthetic_crop(48, 200, 1), synthetic_crop(60, 90, 2), synthetic_crop(32, 300, 3),
                     synthetic_crop(48, 40, 4), synthetic_crop(96, 500, 5)]

    def test_one_recognition_per_crop(self):
        out = self.engine.recognize_batch(self.crops)
        self.assertEqual(len(out), len(self.crops))
        for r in out:
            self.assertIsInstance(r, eng.Recognition)
            self.assertIsInstance(r.text, str)
            self.assertIsInstance(r.confidence, float)
            self.assertTrue(0.0 <= r.confidence <= 1.0)
            self.assertIsInstance(r.clipped, bool)
        self.assertEqual(self.engine.recognize_batch([]), [])

    def test_cpu_inference(self):
        self.assertEqual(self.engine.device.type, "cpu")
        self.assertEqual(next(self.engine._model.parameters()).device.type, "cpu")
        self.assertFalse(self.engine._model.training)
        self.assertFalse(any(p.requires_grad for p in self.engine._model.parameters()))
        self.assertEqual(len(self.engine.recognize_batch(self.crops[:1])), 1)

    def test_input_order_preserved(self):
        # widths differ per crop, so width_px identifies which crop a result belongs to, whatever the batching did
        expected = [int(np.clip(round(c.shape[1] * pp.TARGET_H / c.shape[0]), pp.MIN_W, pp.MAX_W)) for c in self.crops]
        self.assertEqual(len(set(expected)), len(expected))
        for batch_size in (32, 2, 1):
            out = self.engine.recognize_batch(self.crops, batch_size=batch_size)
            self.assertEqual([r.width_px for r in out], expected)
        forward = self.engine.recognize_batch(self.crops)
        backward = self.engine.recognize_batch(self.crops[::-1])
        self.assertEqual(forward, backward[::-1])

    def test_deterministic_repeat(self):
        a = self.engine.recognize_batch(self.crops)
        b = self.engine.recognize_batch(self.crops)
        self.assertEqual([r.text for r in a], [r.text for r in b])
        self.assertEqual(a, b)

    def test_width_and_timestep_metadata(self):
        for crop, r in zip(self.crops, self.engine.recognize_batch(self.crops)):
            self.assertEqual(r.width_px, pp.resize_to_height(crop).shape[1])
            self.assertEqual(r.timesteps, max(1, r.width_px // 4))
            self.assertFalse(r.clipped)

    def test_clipped_flag(self):
        wide = synthetic_crop(20, 600, 7)                       # natural width at H=48 is 1440 > 1024
        edge = synthetic_crop(48, 1024, 8)                      # exactly the cap: not clipped
        self.assertEqual(eng.natural_width(20, 600), 1440)
        out = self.engine.recognize_batch([wide, edge])
        self.assertTrue(out[0].clipped)
        self.assertEqual(out[0].width_px, pp.MAX_W)
        self.assertEqual(out[0].timesteps, pp.MAX_W // 4)
        self.assertFalse(out[1].clipped)
        self.assertEqual(out[1].width_px, 1024)

    def test_matches_the_existing_single_crop_path(self):
        """A batch of one is the path app.py and tools/evaluate.py use: prepare, pad to PAD_W_MULTIPLE, greedy CTC."""
        crop = self.crops[0]
        tensor, pad_value = pp.prepare(crop)
        width = tensor.shape[2]
        image = torch.full((1, 1, tensor.shape[1], -(-width // PAD_W_MULTIPLE) * PAD_W_MULTIPLE), float(pad_value))
        image[0, :, :, :width] = tensor
        with torch.no_grad():
            logits, lengths = self.engine._model(image, torch.tensor([width]))
        reference = cs.decode(cs.ctc_greedy_collapse(logits.argmax(-1)[: lengths[0], 0].tolist()))
        self.assertEqual(self.engine.recognize_batch([crop])[0].text, reference)

    def test_confidence_definition(self):
        crop = self.crops[4]
        r = self.engine.recognize_batch([crop])[0]
        tensor, pad_value = pp.prepare(crop)
        width = tensor.shape[2]
        image = torch.full((1, 1, 48, -(-width // PAD_W_MULTIPLE) * PAD_W_MULTIPLE), float(pad_value))
        image[0, :, :, :width] = tensor
        with torch.no_grad():
            logits, lengths = self.engine._model(image, torch.tensor([width]))
        prob, best = logits.softmax(-1)[: lengths[0], 0].max(-1)
        keep = best != cs.BLANK
        expected = float(prob[keep].mean()) if bool(keep.any()) else 0.0
        self.assertAlmostEqual(r.confidence, expected, places=6)
        self.assertIn("UNCALIBRATED", eng.__doc__)
        self.assertIn("FLAGGING HEURISTIC", eng.__doc__)

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_cuda_inference(self):
        gpu = eng.OcrEngine(CHECKPOINT, device="cuda")
        out = gpu.recognize_batch(self.crops)
        self.assertEqual(len(out), len(self.crops))
        self.assertEqual([r.width_px for r in out], [r.width_px for r in self.engine.recognize_batch(self.crops)])

    # ---- the input contract
    def assert_rejected(self, crop):
        with self.assertRaises(eng.InvalidCropError):
            self.engine.recognize_batch([crop])
        with self.assertRaises(eng.InvalidCropError):                 # also when it is not the first crop
            self.engine.recognize_batch([self.crops[0], crop])

    def test_rgb_rejected(self):
        self.assert_rejected(np.dstack([self.crops[0]] * 3))

    def test_rgba_and_other_3d_rejected(self):
        self.assert_rejected(np.dstack([self.crops[0]] * 4))
        self.assert_rejected(self.crops[0][:, :, None])

    def test_non_grayscale_inputs_rejected(self):
        gray = self.crops[0]
        self.assert_rejected(gray > 128)                              # boolean mask
        self.assert_rejected(((gray > 128) * 255).astype(np.uint8))   # binarized 0/255
        self.assert_rejected((gray > 128).astype(np.uint8))           # binarized 0/1
        self.assert_rejected(gray.astype(np.float32) / 255.0)         # float image
        self.assert_rejected(gray.astype(np.uint16))
        self.assert_rejected(gray.astype(np.int64))
        self.assert_rejected(gray.ravel())                            # 1-D
        self.assert_rejected(gray[:1, :])                             # too small
        self.assert_rejected(gray.tolist())                           # not an ndarray
        self.assert_rejected(None)
        with self.assertRaises(eng.InvalidCropError):
            self.engine.recognize_batch(gray)                         # a bare array instead of a list

    def test_engine_has_no_streamlit_and_no_image_processing(self):
        src = (ROOT / "pageocr" / "engine.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"(?m)^\s*(import|from)\s+(streamlit|cv2|PIL|skimage)\b")
        for name in ("threshold", "CLAHE", "equalizeHist", "fastNlMeans", "morphologyEx", "erode", "dilate", "GaussianBlur"):
            self.assertNotIn(name, src)
        self.assertIn("pp.prepare(", src)                             # preprocessing is the project's, not a copy


class ProtectedPathTests(unittest.TestCase):
    """The guard is tested as a pure function. Paths below are never opened; most of them do not exist."""

    def test_real_docs_paths_refused(self):
        for p in (ROOT / "real_docs" / "anything.png", ROOT / "real_docs" / "r1_pilot" / "pages" / "p.png",
                  ROOT / "real_docs", Path("D:/copy/REAL_DOCS/a.png")):
            with self.assertRaises(pageocr.ProtectedPathError, msg=str(p)):
                pageocr.assert_path_allowed(p)

    def test_r1_paths_refused(self):
        for p in (Path.home() / "R1_T4b_review" / "freeze" / "x.png", Path.home() / "r1_t4b_review" / "x.json",
                  Path("E:/backup/R1_T4b_review/b4_crops/c.png"), Path("E:/x/r1_pilot/c.png")):
            with self.assertRaises(pageocr.ProtectedPathError, msg=str(p)):
                pageocr.assert_path_allowed(p)

    def test_page_loader_refuses_before_opening(self):
        with self.assertRaises(pageocr.ProtectedPathError):
            pio.load_page(ROOT / "real_docs" / "no_such_file_for_test.png")
        with self.assertRaises(pageocr.ProtectedPathError):
            pio.load_page(Path.home() / "R1_T4b_review" / "no_such_file_for_test.png")

    def test_engine_and_box_loaders_refuse(self):
        with self.assertRaises(pageocr.ProtectedPathError):
            eng.OcrEngine(ROOT / "real_docs" / "no_such_checkpoint_for_test.pth")
        with self.assertRaises(pageocr.ProtectedPathError):
            bx.load_json(ROOT / "real_docs" / "no_such_boxes_for_test.json")
        with self.assertRaises(pageocr.ProtectedPathError):
            bx.load_csv(Path.home() / "R1_T4b_review" / "no_such_boxes_for_test.csv")

    def test_read_only_project_folders_refused_for_writing(self):
        for name in pageocr.READ_ONLY_PROJECT_DIRS:
            with self.assertRaises(pageocr.ProtectedPathError, msg=name):
                pageocr.assert_path_allowed(ROOT / name / "x.json", write=True)
        self.assertEqual(pageocr.assert_path_allowed(CHECKPOINT), CHECKPOINT.resolve())     # reading the model is allowed

    def test_protected_content_match(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.bin").write_bytes(b"frozen page bytes")
            (Path(tmp) / "b.bin").write_bytes(b"other page bytes!")       # same length, different content
            index = pio.protected_size_index([tmp])
            self.assertEqual(pio.protected_match(b"frozen page bytes", index), str(Path(tmp) / "a.bin"))
            self.assertIsNone(pio.protected_match(b"FROZEN page bytes", index))
            self.assertIsNone(pio.protected_match(b"short", index))


class PageLoaderTests(unittest.TestCase):
    def setUp(self):
        self.page, _ = synthetic_page()

    def encode(self, ext, img=None):
        ok, buf = cv2.imencode(ext, self.page if img is None else img)
        self.assertTrue(ok)
        return buf.tobytes()

    def test_png_tiff_jpeg_decode_to_gray_uint8(self):
        for ext, fmt, exact in ((".png", "PNG", True), (".tif", "TIFF", True), (".jpg", "JPEG", False)):
            data = self.encode(ext)
            p = pio.load_page_bytes(data, "page" + ext)
            self.assertEqual((p.gray.ndim, p.gray.dtype, p.dtype), (2, np.uint8, "uint8"), ext)
            self.assertEqual(p.shape, self.page.shape)
            self.assertEqual((p.width, p.height), (self.page.shape[1], self.page.shape[0]))
            self.assertEqual(p.sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual((p.path, p.format), ("page" + ext, fmt))
            self.assertFalse(p.gray.flags.writeable)
            if exact:
                np.testing.assert_array_equal(p.gray, self.page)

    def test_colour_page_becomes_one_gray_array(self):
        p = pio.load_page_bytes(self.encode(".png", cv2.cvtColor(self.page, cv2.COLOR_GRAY2BGR)), "c.png")
        np.testing.assert_array_equal(p.gray, self.page)

    def test_load_from_path_records_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "page.png"
            f.write_bytes(self.encode(".png"))
            p = pio.load_page(f)
            self.assertEqual(p.path, str(f.resolve()))
            self.assertEqual(p.sha256, file_sha256(f))

    def test_rejections(self):
        binar = ((self.page > 128) * 255).astype(np.uint8)
        for data in (b"%PDF-1.7\n...", b"not an image", self.encode(".png", binar), self.encode(".bmp"),
                     self.encode(".png", self.page.astype(np.uint16) * 257)):
            with self.assertRaises(pio.PageError):
                pio.load_page_bytes(data, "x")

    def test_preview_is_derived_from_the_same_array(self):
        preview, scale = pio.make_preview(self.page, 450)
        self.assertEqual(preview.shape, (scale.preview_height, scale.preview_width))
        self.assertEqual((scale.original_width, scale.original_height), (900, 300))
        self.assertEqual((scale.preview_width, scale.preview_height), (450, 150))
        self.assertEqual((scale.scale_x, scale.scale_y), (2.0, 2.0))
        np.testing.assert_array_equal(preview, cv2.resize(self.page, (450, 150), interpolation=cv2.INTER_AREA))
        same, one = pio.make_preview(self.page, 5000)                 # never enlarged
        np.testing.assert_array_equal(same, self.page)
        self.assertEqual((one.scale_x, one.scale_y), (1.0, 1.0))
        overlay = pio.draw_boxes(preview, synthetic_page()[1], scale)
        self.assertEqual(overlay.shape, (150, 450, 3))


class CheckpointIntegrityTests(unittest.TestCase):
    def test_checkpoint_is_the_frozen_002a_file(self):
        self.assertEqual(eng.FROZEN_002A_SHA256, FROZEN_SHA256)
        self.assertEqual(file_sha256(CHECKPOINT), FROZEN_SHA256)

    def test_wrong_checkpoint_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "best.pth"
            fake.write_bytes(b"not the frozen checkpoint")
            with self.assertRaises(eng.CheckpointError):
                eng.OcrEngine(fake)
            with self.assertRaises(eng.CheckpointError):
                eng.OcrEngine(Path(tmp) / "missing.pth")

    def test_loading_and_inference_leave_the_checkpoint_untouched(self):
        before = (file_sha256(CHECKPOINT), os.stat(CHECKPOINT).st_size, os.stat(CHECKPOINT).st_mtime_ns)
        siblings = sorted(p.name for p in CHECKPOINT.parent.iterdir())
        engine = eng.OcrEngine(CHECKPOINT, device="cpu")
        self.assertEqual(engine.checkpoint_sha256, FROZEN_SHA256)
        engine.recognize_batch([synthetic_crop(48, 120, 9)])
        self.assertEqual((file_sha256(CHECKPOINT), os.stat(CHECKPOINT).st_size, os.stat(CHECKPOINT).st_mtime_ns), before)
        self.assertEqual(sorted(p.name for p in CHECKPOINT.parent.iterdir()), siblings)

    def test_new_code_cannot_write_a_checkpoint_or_reach_r1_tools(self):
        for path in NEW_SOURCES:
            src = path.read_text(encoding="utf-8")
            for banned in ("torch.save", ".train(", "optimizer", "backward(", "load_state_dict(strict=False"):
                self.assertNotIn(banned, src, f"{path.name}: {banned}")
            imports = set()
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Import):
                    imports.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add((node.module or "").split(".")[0])
            self.assertFalse(any(n == "tools" or "r1_" in n.lower() for n in imports), path.name)
            self.assertLessEqual(imports, {"pageocr", "src", "torch", "numpy", "PIL", "cv2", "streamlit", "base64", "csv",
                                           "dataclasses", "hashlib", "io", "json", "os", "pathlib", "re", "sys"}, path.name)
        app = (ROOT / "app_page.py").read_text(encoding="utf-8")
        self.assertNotRegex(app, r"\bopen\(|write_bytes|write_text|\.save\(|imwrite")     # the page writes no file


class AppPageTests(unittest.TestCase):
    """The Streamlit page run in-process (streamlit.testing), on a generated page. No server is started."""

    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except Exception as e:                                            # pragma: no cover
            raise unittest.SkipTest(f"streamlit.testing is not available: {e}")
        cls.AppTest = AppTest
        cls.page, cls.boxes = synthetic_page()
        cls.png = cv2.imencode(".png", cls.page)[1].tobytes()

    def start(self):
        at = self.AppTest.from_file(str(ROOT / "app_page.py"), default_timeout=300)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_titles_and_statement(self):
        at = self.start()
        self.assertEqual(at.title[0].value, "Tamil OCR — Operator-Assisted Region OCR")
        self.assertTrue(any("Frozen Experiment 002-A word-level recognizer" in m.value for m in at.markdown))
        self.assertTrue(any(export.STATEMENT in i.value for i in at.info))
        self.assertEqual(len(at.error), 0)

    def test_upload_import_run_and_export(self):
        before = file_sha256(CHECKPOINT)
        at = self.start()
        at.file_uploader(key="page_upload").upload("synthetic_page.png", self.png, "image/png").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        sha = hashlib.sha256(self.png).hexdigest()
        self.assertEqual(at.session_state["page"].sha256, sha)
        self.assertTrue(any("original 900 × 300 px" in c.value for c in at.caption))

        box_file = bx.dumps_json(self.boxes, 900, 300, sha).encode("utf-8")
        at.file_uploader(key=f"box_upload_{sha[:12]}").upload("boxes.json", box_file, "application/json").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual(at.session_state["boxes"], self.boxes)

        next(b for b in at.button if b.label == "Run OCR").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        result = at.session_state["result"]
        self.assertEqual(result["boxes"], self.boxes)
        self.assertEqual(len(result["recognitions"]), len(self.boxes))
        # the page's crops are the original pixels, and its output equals the engine called directly
        for b, crop in zip(self.boxes, result["crops"]):
            np.testing.assert_array_equal(crop, self.page[b.y:b.y + b.h, b.x:b.x + b.w])
        direct = eng.OcrEngine(CHECKPOINT).recognize_batch(result["crops"])      # same default device as the page
        self.assertEqual([r.text for r in direct], [r.text for r in result["recognitions"]])
        shown = at.text_area[0].value
        self.assertEqual(shown, export.to_txt(self.boxes, result["recognitions"]))
        self.assertEqual(file_sha256(CHECKPOINT), before)

    def test_bad_box_file_and_frozen_content_refused(self):
        at = self.start()
        at.file_uploader(key="page_upload").upload("synthetic_page.png", self.png, "image/png").run()
        sha = hashlib.sha256(self.png).hexdigest()
        outside = bx.dumps_json([bx.Box("far", 880, 10, 50, 20, 0, 0)]).encode("utf-8")
        at.file_uploader(key=f"box_upload_{sha[:12]}").upload("boxes.json", outside, "application/json").run()
        self.assertTrue(any("outside the page" in e.value for e in at.error))
        self.assertEqual(at.session_state["boxes"], [])
        other = bx.dumps_json(self.boxes, 900, 300, "0" * 64).encode("utf-8")
        at.file_uploader(key=f"box_upload_{sha[:12]}").upload("boxes.json", other, "application/json").run()
        self.assertTrue(any("different page image" in e.value for e in at.error))
        self.assertEqual(at.session_state["boxes"], [])


if __name__ == "__main__":
    unittest.main()
