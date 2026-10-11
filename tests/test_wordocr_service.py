"""
Tests of the word frontend's logic (wordocr): input validation and decoding, the checkpoint registry, model loading,
recognition, and exports. The model tests use the REAL registered checkpoint and the real pipeline on synthetic word
images (tests/wordocr_fixtures.py); they check wiring, order, determinism and Unicode handling, not accuracy.

    .venv\\Scripts\\python -B -m unittest -v tests/test_wordocr_service.py
"""
import csv
import hashlib
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wordocr_fixtures as fx  # noqa: E402

if not fx.RENDERING:                    # same convention as tests/test_r1_twins.py
    raise unittest.SkipTest(fx.SKIP_REASON)
from pageocr import engine as eng  # noqa: E402
from wordocr import checkpoints, export, inputs, service  # noqa: E402

CUDA = service.cuda_available()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class NoNetwork:
    """Any attempt to open a non-loopback network connection fails the test."""

    def __enter__(self):
        self.attempts = []
        real = socket.socket.connect

        def guarded(sock, addr, *a, **k):
            host = addr[0] if isinstance(addr, tuple) else str(addr)
            if host not in ("127.0.0.1", "::1", "localhost"):
                self.attempts.append(addr)
                raise OSError(f"network access attempted in a test: {addr}")
            return real(sock, addr, *a, **k)
        self.patch = mock.patch.object(socket.socket, "connect", guarded)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


# ---------------------------------------------------------------- inputs

class InputValidationTests(unittest.TestCase):
    def assertRefused(self, name, data, fragment):
        with self.assertRaises(inputs.InputError) as cm:
            inputs.decode_upload(name, data)
        self.assertIn(fragment, str(cm.exception))
        self.assertNotIn("Traceback", str(cm.exception))
        return str(cm.exception)

    def test_a_valid_png_decodes_to_the_grayscale_the_engine_takes(self):
        data = fx.word_png()
        d = inputs.decode_upload("word.png", data)
        self.assertEqual((d.format, d.width, d.height), ("PNG",) + fx.word_gray("வணக்கம்").shape[::-1])
        self.assertEqual(d.gray.dtype, np.uint8)
        self.assertEqual(d.gray.ndim, 2)
        np.testing.assert_array_equal(d.gray, fx.word_gray("வணக்கம்"))
        self.assertEqual(d.sha256, hashlib.sha256(data).hexdigest())
        self.assertFalse(d.gray.flags.writeable)
        eng.validate_crop(d.gray)                               # meets the canonical input contract

    def test_empty_file(self):
        self.assertRefused("w.png", b"", "empty")

    def test_file_too_large(self):
        self.assertRefused("w.png", b"\x89PNG" + b"0" * inputs.MAX_FILE_BYTES, "too large")

    def test_unsupported_and_missing_extensions(self):
        self.assertRefused("w.gif", fx.encode(fx.word_gray("தமிழ்"), "GIF"), "Unsupported file type (.gif)")
        self.assertRefused("word", fx.word_png(), "no extension")
        msg = self.assertRefused("x.![](evil)", fx.word_png(), "Unsupported file type")
        self.assertNotIn("](", msg)                              # nothing from the name is echoed as Markdown

    def test_pdf_is_refused_with_guidance(self):
        self.assertRefused("page.pdf", b"%PDF-1.7\n...", "PDF files are not supported")
        self.assertRefused("page.png", b"%PDF-1.7\n...", "PDF files are not supported")

    def test_misleading_extension(self):
        self.assertRefused("w.png", fx.word_png(fmt="JPEG", quality=95), "ends in .png but the file is a JPEG")
        self.assertRefused("w.jpg", fx.word_png(), "ends in .jpg but the file is a PNG")

    def test_corrupted_and_truncated_bytes(self):
        self.assertRefused("w.png", b"\x89PNG\r\n\x1a\nthis is not an image", "could not be read as an image")
        self.assertRefused("w.jpg", os.urandom(4000), "could not be read as an image")
        png = fx.word_png()
        self.assertRefused("w.png", png[: len(png) // 2], "damaged")
        jpg = fx.word_png(fmt="JPEG", quality=95)
        self.assertRefused("w.jpg", jpg[: len(jpg) // 2], "damaged")

    def test_oversized_dimensions_are_refused_before_decoding(self):
        side = inputs.MAX_SIDE + 1
        self.assertRefused("w.png", fx.encode(Image.new("L", (side, 4), 200)), "too large")
        big = fx.encode(Image.new("L", (7000, 6000), 200))      # 42 Mpx > MAX_PIXELS, a tiny file
        with mock.patch.object(inputs.pio, "load_page_bytes", side_effect=AssertionError("decoded")):
            self.assertRefused("w.png", big, "too large")

    def test_too_small_one_bit_sixteen_bit_and_multi_frame(self):
        self.assertRefused("w.png", fx.encode(Image.new("L", (1, 30), 200)), "too small")
        self.assertRefused("w.png", fx.encode(Image.fromarray(fx.word_gray("தமிழ்")).convert("1")), "1-bit")
        self.assertRefused("w.png", fx.encode(Image.fromarray(fx.word_gray("தமிழ்").astype(np.uint16) * 256)), "more than 8 bits")
        frames = [Image.fromarray(fx.word_gray(w)) for w in ("தமிழ்", "கணினி")]
        b = io.BytesIO()
        frames[0].save(b, "TIFF", save_all=True, append_images=frames[1:])
        self.assertRefused("w.tif", b.getvalue(), "2 images")

    def test_sixteen_bit_colour_gets_the_bit_depth_message(self):         # E-01
        import cv2
        rgb16 = np.dstack([fx.word_gray("தமிழ்").astype(np.uint16) * 257] * 3)
        ok, png = cv2.imencode(".png", rgb16)
        self.assertTrue(ok)
        msg = self.assertRefused("w.png", png.tobytes(), "8 bits")
        self.assertNotIn("damaged", msg)

    def test_two_gray_levels_and_blank_are_refused(self):
        two = np.where(fx.word_gray("தமிழ்") < 128, 0, 255).astype(np.uint8)
        self.assertRefused("w.png", fx.encode(two), "two gray levels")
        self.assertRefused("w.png", fx.encode(np.full((40, 120), 255, np.uint8)), "two gray levels")

    def test_rgb_rgba_jpeg_tiff_webp_all_reach_the_same_kind_of_array(self):
        g = fx.word_gray("கணினி")
        rgb = Image.fromarray(g).convert("RGB")
        for name, data in (("w.jpg", fx.encode(rgb, "JPEG", quality=95)), ("w.tiff", fx.encode(g, "TIFF")),
                           ("w.png", fx.encode(rgb)), ("w.webp", fx.encode(rgb, "WEBP", lossless=True))):
            d = inputs.decode_upload(name, data)
            self.assertEqual((d.gray.ndim, d.gray.dtype, d.gray.shape), (2, np.uint8, g.shape), name)
        lossless = inputs.decode_upload("w.webp", fx.encode(Image.fromarray(g), "WEBP", lossless=True))
        np.testing.assert_array_equal(lossless.gray, inputs.decode_upload("w.png", fx.encode(g)).gray)

    def test_transparency_is_put_on_white(self):
        g = np.pad(fx.word_gray("தமிழ்"), 10, constant_values=0)
        rgba = np.dstack([g, g, g, np.full(g.shape, 255, np.uint8)])
        rgba[:10, :, 3] = rgba[-10:, :, 3] = rgba[:, :10, 3] = rgba[:, -10:, 3] = 0    # a fully transparent (black) border
        for name, fmt in (("w.png", "PNG"), ("w.webp", "WEBP")):
            d = inputs.decode_upload(name, fx.encode(Image.fromarray(rgba, "RGBA"), fmt, **({"lossless": True} if fmt == "WEBP" else {})))
            self.assertTrue(d.transparency_flattened)
            self.assertTrue((d.gray[:10] == 255).all() and (d.gray[:, -10:] == 255).all(), name)   # white paper, not black
            np.testing.assert_array_equal(d.gray[10:-10, 10:-10], g[10:-10, 10:-10])               # opaque pixels unchanged

    def test_exif_rotation_is_applied(self):
        g = fx.word_gray("தமிழ்")
        im = Image.fromarray(g)
        ex = im.getexif()
        ex[0x0112] = 6
        d = inputs.decode_upload("w.jpg", fx.encode(im, "JPEG", exif=ex.tobytes(), quality=95))
        self.assertTrue(d.exif_rotated)
        self.assertEqual((d.width, d.height), (g.shape[0], g.shape[1]))
        self.assertTrue(any("taller than it is wide" in w for w in d.warnings))

    def test_a_very_wide_image_is_flagged_as_squeezed(self):
        wide = np.tile(fx.word_gray("வணக்கம்"), (1, 8))
        d = inputs.decode_upload("w.png", fx.encode(wide))
        self.assertTrue(d.clipped)
        self.assertEqual(d.model_input_width, 1024)
        self.assertTrue(any("squeezed" in w for w in d.warnings))

    def test_safe_stem(self):
        cases = {"வணக்கம்.png": "image", "my word.png": "my_word", r"..\..\evil.png": "evil", "a/b/c.d.png": "c_d",
                 "CON.png": "image_CON", "x" * 200 + ".png": "x" * 60, "": "image", "--.png": "image",
                 "scan (1).jpeg": "scan_1"}
        for name, want in cases.items():
            self.assertEqual(inputs.safe_stem(name), want, name)
            self.assertRegex(inputs.safe_stem(name), r"^[A-Za-z0-9_-]{1,70}$")


# ---------------------------------------------------------------- research-data guard (E-02)

class ResearchDataGuardTests(unittest.TestCase):
    """Only GENERATED images are used. The guard is made to treat their bytes, file hash or pixels as protected, and the
    recognizer is a trap: no real research file and no development page is ever read or given to the model."""

    def setUp(self):
        self.png = fx.word_png("தமிழ்")
        self.sha = hashlib.sha256(self.png).hexdigest()

    def test_the_guard_covers_real_docs_the_r1_folders_and_development_pages(self):
        from pageocr import imageio as pio
        from wordocr import guard
        names = {p.name for p in guard.ROOTS}
        self.assertIn("real_docs", names)
        self.assertIn("development_pages", names)
        self.assertTrue(set(pio.PROTECTED_ROOTS) <= set(guard.ROOTS))

    def test_a_byte_identical_copy_of_protected_data_is_refused(self):
        from wordocr import guard
        with tempfile.TemporaryDirectory() as tmp:
            twin = Path(tmp) / "protected_twin.png"           # a generated file standing in for a protected one
            twin.write_bytes(self.png)
            with mock.patch.object(guard, "_size_index", return_value={len(self.png): [str(twin)]}):
                with self.assertRaises(inputs.InputError) as cm:
                    inputs.decode_upload("w.png", self.png)
        self.assertIn("protected research data", str(cm.exception))

    def test_a_development_page_by_file_hash_is_refused(self):
        from pageocr import devguard
        real = devguard.development_master_hashes()
        with mock.patch.object(devguard, "development_master_hashes", return_value={**real, self.sha: "generated.png"}):
            with self.assertRaises(inputs.InputError) as cm:
                inputs.decode_upload("w.png", self.png)
        self.assertIn("development page", str(cm.exception))

    def test_a_development_page_by_pixels_is_refused(self):
        from pageocr import devguard
        master = sorted(devguard.development_master_hashes())[0]
        with mock.patch.object(devguard, "development_page_for_pixels", return_value=master):
            with self.assertRaises(inputs.InputError) as cm:
                inputs.decode_upload("w.png", self.png)
        self.assertIn("development page", str(cm.exception))

    def test_the_guard_is_checked_again_right_before_recognition(self):
        from pageocr import devguard
        model = service.load_model("002a", "cpu")
        d = inputs.decode_upload("w.png", self.png)               # accepted at upload ...
        real = devguard.development_master_hashes()
        with mock.patch.object(devguard, "development_master_hashes", return_value={**real, self.sha: "generated.png"}), \
                mock.patch.object(model.engine, "recognize_batch", side_effect=AssertionError("the model was called")):
            r = service.recognize_one(model, d)                    # ... but refused before the model is called
        self.assertFalse(r.ok)
        self.assertIn("development page", r.error)

    def test_a_batch_keeps_going_past_a_protected_item(self):
        from pageocr import devguard
        model = service.load_model("002a", "cpu")
        items = [inputs.decode_upload(f"{i}.png", fx.word_png(w)) for i, w in enumerate(("தமிழ்", "கணினி"))]
        real = devguard.development_master_hashes()
        with mock.patch.object(devguard, "development_master_hashes", return_value={**real, items[0].sha256: "generated.png"}):
            out = service.recognize(model, items)
        self.assertEqual([r.ok for r in out], [False, True])
        self.assertEqual(out[1].text, "கணினி")

    def test_the_gate_is_closed_today(self):
        from pageocr import devguard
        self.assertFalse(devguard.status()["ready"])               # so a development page would be refused


# ---------------------------------------------------------------- checkpoints and loading

class CheckpointTests(unittest.TestCase):
    def test_registry_pins_the_recorded_hashes_of_the_files_on_disk(self):
        self.assertEqual(checkpoints.DEFAULT_ID, "002a")
        self.assertEqual(checkpoints.get("002a").sha256, eng.FROZEN_002A_SHA256)
        for c in checkpoints.REGISTRY:
            self.assertTrue(checkpoints.present(c), c.path)
            self.assertEqual(sha(c.file()), c.sha256, c.path)
            self.assertTrue((ROOT / c.evidence).is_file(), c.evidence)
            self.assertIn(c.sha256, (ROOT / c.evidence).read_text(encoding="utf-8"), c.evidence)

    def test_the_engine_default_still_accepts_only_the_frozen_002a_file(self):
        with self.assertRaises(eng.CheckpointError) as cm:
            eng.OcrEngine(checkpoints.get("002b").file(), device="cpu")
        self.assertIn("frozen Experiment 002-A", str(cm.exception))
        e = eng.OcrEngine(checkpoints.get("002a").file(), device="cpu")
        self.assertEqual(e.info()["model"], "Experiment 002-A (CRNNv2, greedy CTC)")

    def test_a_registered_hash_admits_only_that_file(self):
        e = eng.OcrEngine(checkpoints.get("002b").file(), device="cpu", expected_sha256=checkpoints.get("002b").sha256)
        self.assertEqual(e.checkpoint_sha256, checkpoints.get("002b").sha256)
        self.assertEqual(e.info()["model"], "CRNNv2, greedy CTC")
        with self.assertRaises(eng.CheckpointError):
            eng.OcrEngine(checkpoints.get("002a").file(), device="cpu", expected_sha256=checkpoints.get("002b").sha256)

    def test_unknown_id(self):
        with self.assertRaises(service.ModelUnavailableError):
            service.load_model("no-such-model", "cpu")

    def test_missing_file_is_reported_and_nothing_is_substituted(self):
        ghost = checkpoints.CheckpointSpec("ghost", "Ghost", "runs/no_such_dir/best.pth", "0" * 64, "-", "-")
        with mock.patch.dict(checkpoints._BY_ID, {"ghost": ghost}), \
                mock.patch.object(service, "OcrEngine", side_effect=AssertionError("must not load anything")):
            with self.assertRaises(service.ModelUnavailableError) as cm:
                service.load_model("ghost", "cpu")
        self.assertIn("was not found", str(cm.exception))

    def test_a_file_with_another_hash_is_refused_and_nothing_is_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "best.pth"
            fake.write_bytes(b"not a checkpoint")
            spec = checkpoints.CheckpointSpec("fake", "Fake", str(fake), "1" * 64, "-", "-")
            with mock.patch.dict(checkpoints._BY_ID, {"fake": spec}):
                with self.assertRaises(service.ModelUnavailableError) as cm:
                    service.load_model("fake", "cpu")
        self.assertIn("not the registered file", str(cm.exception))
        self.assertIn("no other model was used", str(cm.exception))

    def test_cuda_requested_without_a_gpu_is_an_error_not_a_silent_switch(self):
        with mock.patch.object(service, "cuda_available", return_value=False):
            self.assertEqual(service.resolve_device("auto"), "cpu")
            with self.assertRaises(service.ModelUnavailableError):
                service.resolve_device("cuda")
        with self.assertRaises(service.ModelUnavailableError):
            service.resolve_device("tpu")

    def test_loading_and_inference_never_modify_any_checkpoint(self):
        before = {c.path: (sha(c.file()), os.stat(c.file()).st_mtime_ns) for c in checkpoints.REGISTRY}
        listing = {c.path: sorted(p.name for p in c.file().parent.iterdir()) for c in checkpoints.REGISTRY}
        for cid in checkpoints.ids():
            m = service.load_model(cid, "cpu")
            service.recognize_one(m, inputs.decode_upload("w.png", fx.word_png()))
        self.assertEqual({c.path: (sha(c.file()), os.stat(c.file()).st_mtime_ns) for c in checkpoints.REGISTRY}, before)
        self.assertEqual({c.path: sorted(p.name for p in c.file().parent.iterdir()) for c in checkpoints.REGISTRY}, listing)


# ---------------------------------------------------------------- recognition (real model)

class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = service.load_model("002a", "cpu")
        cls.images = [inputs.decode_upload(f"{i}.png", fx.word_png(w)) for i, w in enumerate(fx.WORDS)]

    def test_the_service_returns_exactly_what_the_canonical_engine_returns(self):
        direct = eng.OcrEngine(checkpoints.get("002a").file(), device="cpu")
        for d in self.images:
            r = service.recognize_one(self.model, d)
            want = direct.recognize_batch([d.gray], batch_size=1)[0]
            self.assertTrue(r.ok)
            self.assertEqual(r.text, want.text)
            self.assertAlmostEqual(r.uncalibrated_score, round(want.confidence, 4), places=4)

    def test_the_rendered_words_are_read_and_their_unicode_is_preserved(self):
        # Smoke check of the wiring on clean synthetic print; NOT a measure of accuracy.
        got = [service.recognize_one(self.model, d).text for d in self.images]
        self.assertEqual(got, list(fx.WORDS))
        for t in got:
            self.assertEqual(t, __import__("unicodedata").normalize("NFC", t))

    def test_output_only_uses_the_model_character_set(self):
        from src import charset as cs
        for d in self.images:
            self.assertTrue(set(service.recognize_one(self.model, d).text) <= set(cs.CHARSET))

    def test_repeated_requests_are_deterministic(self):
        first = [service.recognize_one(self.model, d) for d in self.images]
        for _ in range(3):
            again = [service.recognize_one(self.model, d) for d in self.images]
            self.assertEqual([(r.text, r.uncalibrated_score) for r in again], [(r.text, r.uncalibrated_score) for r in first])

    def test_a_batch_keeps_order_and_matches_single_runs_whatever_the_neighbours(self):
        single = [service.recognize_one(self.model, d).text for d in self.images]
        batch = service.recognize(self.model, list(reversed(self.images)))
        self.assertEqual([r.text for r in batch], list(reversed(single)))
        self.assertEqual([r.index for r in batch], list(range(len(self.images))))

    def test_a_failed_item_stays_in_place_and_the_others_are_read(self):
        bad = service.failed(0, "broken.png", "damaged")
        items = [self.images[0], bad, self.images[1]]
        out = service.recognize(self.model, items)
        self.assertEqual([r.ok for r in out], [True, False, True])
        self.assertEqual([r.name for r in out], ["0.png", "broken.png", "1.png"])
        self.assertEqual([r.text for r in out], [fx.WORDS[0], "", fx.WORDS[1]])

    def test_an_inference_failure_is_contained_and_the_model_keeps_working(self):
        with mock.patch.object(self.model.engine, "recognize_batch", side_effect=RuntimeError("boom")):
            r = service.recognize_one(self.model, self.images[0])
        self.assertFalse(r.ok)
        self.assertIn("Recognition failed (RuntimeError)", r.error)
        self.assertNotIn("boom", r.error)                     # internal detail goes to the log, not the user
        self.assertEqual(service.recognize_one(self.model, self.images[0]).text, fx.WORDS[0])

    def test_an_input_outside_the_contract_is_refused_not_guessed(self):
        bin_img = inputs.DecodedImage(name="b.png", sha256="0" * 64, size_bytes=1, format="PNG", mode="L", width=120,
                                      height=40, exif_rotated=False, transparency_flattened=False,
                                      gray=np.where(np.arange(4800).reshape(40, 120) % 7 == 0, 0, 255).astype(np.uint8))
        r = service.recognize_one(self.model, bin_img)
        self.assertFalse(r.ok)
        self.assertIn("input contract", r.error)

    def test_an_empty_prediction_is_reported_as_empty(self):
        flat = np.full((48, 160), 200, np.uint8)
        flat[20, 10] = 199                                    # three gray levels, no text
        flat[21, 10] = 198
        d = inputs.decode_upload("blank.png", fx.encode(flat))
        r = service.recognize_one(self.model, d)
        self.assertTrue(r.ok)
        if r.text == "":
            self.assertTrue(r.empty)
            self.assertIsNotNone(r.uncalibrated_score)

    @unittest.skipUnless(CUDA, "no CUDA GPU on this machine")
    def test_cuda_gives_the_same_text_as_cpu(self):
        gpu = service.load_model("002a", "cuda")
        self.assertEqual(gpu.device, "cuda")
        self.assertEqual([service.recognize_one(gpu, d).text for d in self.images],
                         [service.recognize_one(self.model, d).text for d in self.images])

    @unittest.skipUnless(CUDA, "no CUDA GPU on this machine")
    def test_repeated_gpu_requests_do_not_grow_memory(self):
        import torch
        gpu = service.load_model("002a", "cuda")
        for _ in range(5):
            service.recognize_one(gpu, self.images[2])
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        texts = {service.recognize_one(gpu, d).text for _ in range(40) for d in self.images}
        torch.cuda.synchronize()
        self.assertLessEqual(torch.cuda.memory_allocated() - base, 1 << 20)
        self.assertEqual(texts, set(fx.WORDS))                 # model state is not corrupted by repetition

    def test_no_network_connection_is_attempted(self):
        with NoNetwork() as nn:
            m = service.load_model("002a", "cpu")
            out = service.recognize(m, self.images)
            export.to_json(out, m.info())
        self.assertEqual(nn.attempts, [])


# ---------------------------------------------------------------- exports

class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = service.load_model("002a", "cpu")
        cls.info = m.info()
        cls.results = service.recognize(m, [inputs.decode_upload("வணக்கம்.png", fx.word_png("வணக்கம்")),
                                            service.failed(0, "bad.jpg", "The file is empty (0 bytes)."),
                                            inputs.decode_upload("two.png", fx.word_png("தமிழ்"))])

    def test_single_txt_is_the_text(self):
        self.assertEqual(export.to_txt(self.results[:1]).decode("utf-8"), "வணக்கம்\n")
        self.assertEqual(export.download_name(self.results[:1], "txt"), "image_ocr.txt")

    def test_batch_txt_names_every_file(self):
        lines = export.to_txt(self.results).decode("utf-8").splitlines()
        self.assertEqual(lines, ["வணக்கம்.png\tவணக்கம்", "bad.jpg\t[not recognized: The file is empty (0 bytes).]", "two.png\tதமிழ்"])
        self.assertEqual(export.download_name(self.results, "csv"), "tamil_ocr_batch_3.csv")

    def test_json_matches_the_results_and_identifies_the_model(self):
        doc = json.loads(export.to_json(self.results, self.info).decode("utf-8"))
        self.assertEqual(doc["schema"], export.SCHEMA)
        self.assertEqual([r["text"] for r in doc["results"]], ["வணக்கம்", None, "தமிழ்"])
        self.assertEqual([r["status"] for r in doc["results"]], ["ok", "error", "ok"])
        self.assertEqual(doc["model"]["checkpoint_sha256"], eng.FROZEN_002A_SHA256)
        self.assertTrue(doc["model"]["sha256_verified"])
        self.assertIn("NOT a probability", doc["score_note"])
        self.assertIn("வணக்கம்", export.to_json(self.results, self.info).decode("utf-8"))   # not \\u-escaped

    def test_csv_round_trips_tamil_with_a_bom(self):
        raw = export.to_csv(self.results)
        self.assertTrue(raw.startswith("﻿".encode("utf-8")))
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual([r["text"] for r in rows], ["வணக்கம்", "", "தமிழ்"])
        self.assertEqual([r["file_name"] for r in rows], ["வணக்கம்.png", "bad.jpg", "two.png"])

    def test_csv_neutralises_formula_cells_but_json_keeps_them(self):
        r = service.ItemResult(index=0, name="=HYPERLINK(1).png", ok=True, text="-1+2")
        row = next(csv.DictReader(io.StringIO(export.to_csv([r]).decode("utf-8-sig"))))
        self.assertEqual((row["file_name"], row["text"]), ("'=HYPERLINK(1).png", "'-1+2"))
        doc = json.loads(export.to_json([r], {}))
        self.assertEqual((doc["results"][0]["file_name"], doc["results"][0]["text"]), ("=HYPERLINK(1).png", "-1+2"))


if __name__ == "__main__":
    unittest.main()
