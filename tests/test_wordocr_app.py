"""
Tests of the Streamlit page (app.py) with Streamlit's own headless test harness (streamlit.testing.v1.AppTest).
The page runs for real: the real registered checkpoint, the real pipeline, synthetic word images
(tests/wordocr_fixtures.py). The bytes behind the download buttons are checked by the browser smoke test
(tests/wordocr_browser_smoke.py), because AppTest does not serve media files.

    .venv\\Scripts\\python -B -m unittest -v tests/test_wordocr_app.py
"""
import re
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import wordocr_fixtures as fx  # noqa: E402

if not fx.RENDERING:                    # same convention as tests/test_r1_twins.py
    raise unittest.SkipTest(fx.SKIP_REASON)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402
from wordocr import checkpoints, inputs  # noqa: E402

APP = str(ROOT / "app.py")
WORD = "வணக்கம்"


def new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def texts(elements) -> list[str]:
    return [e.value for e in elements]


class PageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def assertClean(self, at):
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        for e in at.error:
            self.assertNotIn("Traceback", e.value)

    def upload(self, at, name, data, key="single_upload_0"):
        at.file_uploader(key=key).set_value((name, data, "application/octet-stream"))
        at.run()

    def test_startup_shows_the_model_status_and_the_scope(self):
        at = new_app()
        self.assertClean(at)
        self.assertEqual(at.title[0].value, "Tamil OCR — Word and Full-page Recognizer")
        self.assertTrue(any("Model ready: Experiment 002-A" in s and "SHA-256 verified" in s for s in texts(at.success)))
        self.assertTrue(any("no space character" in c.lower() or "cannot transcribe" in c for c in texts(at.caption)))
        self.assertEqual([t.label for t in at.tabs], ["Single image", "Batch", "Full-page OCR", "About and limits"])
        self.assertTrue(at.button(key="run_single").disabled)           # nothing to run yet
        self.assertEqual(at.selectbox(key="checkpoint").value, checkpoints.DEFAULT_ID)

    def test_upload_run_display_and_no_duplicate_run(self):
        at = new_app()
        self.upload(at, "word.png", fx.word_png(WORD))
        self.assertClean(at)
        self.assertFalse(at.button(key="run_single").disabled)
        self.assertEqual(len(at.get("image")), 2)                      # the grayscale image and the 48 px view
        at.button(key="run_single").click()
        at.run()
        self.assertClean(at)
        self.assertEqual(texts(at.code), [WORD])                       # copyable text
        self.assertTrue(any(f'lang="ta">{WORD}</div>' in m for m in texts(at.markdown)))   # Unicode displayed unaltered
        self.assertTrue(at.button(key="run_single").disabled)          # the same image is not run again
        self.assertEqual([d.proto.label for d in at.get("download_button")], ["Download TXT", "Download JSON"])

    def test_clear_removes_image_and_result(self):
        at = new_app()
        self.upload(at, "word.png", fx.word_png(WORD))
        at.button(key="run_single").click()
        at.run()
        at.button(key="clear_single").click()
        at.run()
        self.assertClean(at)
        self.assertEqual(texts(at.code), [])
        self.assertIsNone(at.file_uploader(key="single_upload_1").value)
        self.assertTrue(at.button(key="run_single").disabled)

    def test_a_new_image_replaces_the_old_result(self):
        at = new_app()
        self.upload(at, "a.png", fx.word_png(WORD))
        at.button(key="run_single").click()
        at.run()
        self.upload(at, "b.png", fx.word_png("தமிழ்"))
        self.assertEqual(texts(at.code), [])                           # the old text is not shown beside the new image
        at.button(key="run_single").click()
        at.run()
        self.assertEqual(texts(at.code), ["தமிழ்"])

    def test_bad_inputs_give_a_message_and_no_run(self):
        cases = [("e.png", b"", "empty"), ("c.png", b"\x89PNG\r\n\x1a\ngarbage", "could not be read"),
                 ("p.pdf", b"%PDF-1.4 x", "PDF files are not supported"), ("x.gif", fx.word_png(fmt="GIF"), "Unsupported"),
                 ("m.jpg", fx.word_png(), "ends in .jpg but the file is a PNG")]
        for name, data, fragment in cases:
            with self.subTest(name=name):
                at = new_app()
                self.upload(at, name, data)
                self.assertClean(at)
                self.assertTrue(any(fragment in e for e in texts(at.error)), texts(at.error))
                self.assertTrue(at.button(key="run_single").disabled)

    def test_a_development_page_is_refused_and_never_reaches_the_model(self):     # E-02
        # GENERATED image; the guard is made to treat its file hash as a development page's. Trap engine.
        import hashlib
        from pageocr import devguard
        from wordocr import service
        png = fx.word_png(WORD)
        real = devguard.development_master_hashes()
        with mock.patch.object(devguard, "development_master_hashes",
                               return_value={**real, hashlib.sha256(png).hexdigest(): "generated.png"}), \
                mock.patch.object(service.OcrEngine, "recognize_batch", side_effect=AssertionError("the model was called")):
            for key, files in (("single_upload_0", ("word.png", png, "image/png")),
                               ("batch_upload_0", [("word.png", png, "image/png"), ("ok.png", fx.word_png("தமிழ்"), "image/png")])):
                at = new_app()
                at.file_uploader(key=key).set_value(files)
                at.run()
                self.assertClean(at)
                if key == "single_upload_0":
                    self.assertTrue(any("development page" in e for e in texts(at.error)), texts(at.error))
                    self.assertTrue(at.button(key="run_single").disabled)
                else:
                    self.assertTrue(any("1 cannot be read" in c for c in texts(at.caption)), texts(at.caption))

    def test_a_hostile_file_name_is_not_rendered_as_markdown(self):
        at = new_app()
        self.upload(at, "![x](http://example.invalid/t.png).png", fx.word_png(WORD))
        self.assertClean(at)
        for m in texts(at.markdown):
            self.assertNotIn("](http://example.invalid", m)

    def test_inference_failure_is_shown_without_internals_and_recovers(self):
        at = new_app()
        self.upload(at, "word.png", fx.word_png(WORD))
        from wordocr import service
        with mock.patch.object(service.OcrEngine, "recognize_batch", side_effect=RuntimeError("secret internal detail")):
            at.button(key="run_single").click()
            at.run()
        self.assertClean(at)
        self.assertTrue(any("Recognition failed (RuntimeError)" in e for e in texts(at.error)))
        self.assertFalse(any("secret internal detail" in e for e in texts(at.error)))
        at.button(key="clear_single").click()
        at.run()
        self.upload(at, "word.png", fx.word_png(WORD), key="single_upload_1")
        at.button(key="run_single").click()
        at.run()
        self.assertEqual(texts(at.code), [WORD])

    def test_batch_with_a_failed_file_keeps_every_row(self):
        at = new_app()
        files = [("one.png", fx.word_png(WORD), "image/png"), ("broken.png", b"\x89PNG\r\n\x1a\nxx", "image/png"),
                 ("two.png", fx.word_png("தமிழ்"), "image/png")]
        at.file_uploader(key="batch_upload_0").set_value(files)
        at.run()
        self.assertFalse(at.button(key="run_batch").disabled)
        at.button(key="run_batch").click()
        at.run()
        self.assertClean(at)
        table = next(m for m in texts(at.markdown) if 'class="ocr-results"' in m)
        rows = re.findall(r'<tr class="st-(\w+)"><td>(\d+)</td><td>([^<]*)</td><td>\w+</td><td class="ta" lang="ta">([^<]*)</td>', table)
        self.assertEqual(rows, [("ok", "1", "one.png", WORD), ("failed", "2", "broken.png", ""), ("ok", "3", "two.png", "தமிழ்")])
        self.assertTrue(any("2 of 3 recognized" in m for m in texts(at.markdown)))
        self.assertEqual([d.proto.label for d in at.get("download_button")], ["Download TXT", "Download JSON", "Download CSV"])
        self.assertTrue(at.button(key="run_batch").disabled)

    def test_batch_limit(self):
        at = new_app()
        files = [(f"{i}.png", fx.word_png(WORD), "image/png") for i in range(inputs.MAX_BATCH_FILES + 1)]
        at.file_uploader(key="batch_upload_0").set_value(files)
        at.run()
        self.assertTrue(any("the limit is" in e for e in texts(at.error)))
        self.assertTrue(at.button(key="run_batch").disabled)

    def test_selecting_another_checkpoint_loads_exactly_that_one(self):
        at = new_app()
        self.upload(at, "word.png", fx.word_png(WORD))
        at.button(key="run_single").click()
        at.run()
        at.selectbox(key="checkpoint").set_value("002b")
        at.run()
        self.assertClean(at)
        self.assertTrue(any("Model ready: Experiment 002-B" in s for s in texts(at.success)))
        self.assertEqual(texts(at.code), [])                           # a result of another model is not kept
        at.button(key="run_single").click()
        at.run()
        self.assertTrue(any("Experiment 002-B" in c for c in texts(at.caption)))

    def test_a_missing_checkpoint_is_reported_and_not_replaced(self):
        st.cache_resource.clear()
        spec = checkpoints.get("002a")
        ghost = checkpoints.CheckpointSpec(spec.id, spec.label, "runs/no_such_dir/best.pth", spec.sha256, spec.training, spec.evidence)
        try:
            with mock.patch.dict(checkpoints._BY_ID, {"002a": ghost}), \
                    mock.patch.object(checkpoints, "REGISTRY", (ghost,) + checkpoints.REGISTRY[1:]):
                at = new_app()
                self.assertClean(at)
                self.assertEqual(texts(at.success), [])
                self.assertTrue(any("Model not available" in e and "was not found" in e for e in texts(at.error)))
                self.upload(at, "word.png", fx.word_png(WORD))
                self.assertTrue(at.button(key="run_single").disabled)
        finally:
            st.cache_resource.clear()

    def test_the_page_makes_no_external_connection(self):
        attempts, real = [], socket.socket.connect

        def guarded(sock, addr, *a, **k):
            if (addr[0] if isinstance(addr, tuple) else str(addr)) not in ("127.0.0.1", "::1", "localhost"):
                attempts.append(addr)
                raise OSError("external connection attempted")
            return real(sock, addr, *a, **k)
        with mock.patch.object(socket.socket, "connect", guarded):
            st.cache_resource.clear()
            at = new_app()
            self.upload(at, "word.png", fx.word_png(WORD))
            at.button(key="run_single").click()
            at.run()
        self.assertEqual(attempts, [])
        self.assertEqual(texts(at.code), [WORD])


if __name__ == "__main__":
    unittest.main()
