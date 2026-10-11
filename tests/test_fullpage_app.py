"""
The frontend (app.py) with the Full-page OCR tab and the word-tab page check, run in-process with streamlit.testing.

    .venv\\Scripts\\python -B -m unittest -v tests/test_fullpage_app.py

Generated images only (tests/fullpage_fixtures.py). The real registered model is used; any attempt to open a
non-loopback network connection fails the test. A real browser run is tests/fullpage_browser_smoke.py.
"""
import re
import socket
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

import streamlit as st                                            # noqa: E402
from streamlit.testing.v1 import AppTest                          # noqa: E402

from pageocr import devguard                                      # noqa: E402
from wordocr import page_export, service                          # noqa: E402

APP = str(ROOT / "app.py")
GUARD_OPEN = devguard.status()["ready"]


def texts(elements) -> list:
    return [e.value for e in elements]


def word_png(text="தமிழ்") -> bytes:
    return fx.png((255 - np.pad(fx.render_word(text), 10).astype(np.float32) * 0.85).astype(np.uint8))


class FullPageAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = fx.make_tamil_page(seed=5, columns=2, heading=True)
        cls.png = fx.png(cls.page.gray)

    def setUp(self):
        st.cache_data.clear()
        self.attempts = []
        real = socket.socket.connect

        def guarded(sock, addr, *a, **k):
            host = addr[0] if isinstance(addr, tuple) else str(addr)
            if host not in ("127.0.0.1", "::1", "localhost"):
                self.attempts.append(addr)
                raise OSError(f"network access attempted in a test: {addr}")
            return real(sock, addr, *a, **k)
        patcher = mock.patch.object(socket.socket, "connect", guarded)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: self.assertEqual(self.attempts, []))     # no external network call in any test

    def app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        self.assertClean(at)
        return at

    def assertClean(self, at):
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        for e in at.error:
            self.assertNotIn("Traceback", e.value)

    def upload(self, at, key, name, data):
        at.file_uploader(key=key).set_value((name, data, "application/octet-stream"))
        at.run()
        self.assertClean(at)

    # ---- startup and word mode

    def test_startup_offers_both_workflows(self):
        at = self.app()
        self.assertEqual([t.label for t in at.tabs], ["Single image", "Batch", "Full-page OCR", "About and limits"])
        self.assertTrue(at.button(key="run_page").disabled)
        self.assertTrue(any(page_export.PAGE_STATEMENT in c for c in texts(at.caption)))
        self.assertTrue(any("tested on generated pages only" in m for m in texts(at.markdown)))

    def test_word_ocr_still_works(self):
        at = self.app()
        self.upload(at, "single_upload_0", "word.png", word_png("கணினி"))
        self.assertFalse(at.button(key="run_single").disabled)
        at.button(key="run_single").click().run()
        self.assertClean(at)
        self.assertEqual(at.session_state["single"]["result"].text, "கணினி")
        self.assertIn("கணினி", texts(at.code))

    def test_a_page_in_the_word_tab_is_refused_and_sent_to_full_page_ocr(self):
        """The reported problem: no word OCR on a page, no one-character 'result', a pointer to Full-page OCR."""
        at = self.app()
        with mock.patch.object(service.OcrEngine, "recognize_batch", side_effect=AssertionError("word OCR ran on a page")):
            self.upload(at, "single_upload_0", "textbook_page.png", self.png)
            self.assertTrue(at.button(key="run_single").disabled)
            self.assertIsNone(at.session_state["single"])
        errors = " ".join(texts(at.error))
        self.assertIn("looks like a page", errors)
        self.assertIn("Full-page OCR", errors)
        self.assertTrue(any("Full-page OCR" in i for i in texts(at.info)))
        self.assertEqual(texts(at.code), [])                           # nothing is shown as recognized text
        self.assertFalse(any('class="ocr-text"' in m for m in texts(at.markdown)))
        # no override box for a page: the only checkboxes are the Full-page tab's own settings (incl. the opt-in
        # diagnostic mode and experimental options added in the diagnostic workspace, all off)
        self.assertEqual(sorted(c.key for c in at.checkbox), ["page_deskew", "page_diag", "page_exp_clahe", "page_exp_fill",
                                                               "page_exp_rule", "page_margin_auto"])
        self.assertFalse(any(c.value for c in at.checkbox if c.key.startswith(("page_diag", "page_exp_"))))

    def test_a_page_in_a_batch_fails_alone(self):
        at = self.app()
        at.file_uploader(key="batch_upload_0").set_value([("w.png", word_png("கல்வி"), "image/png"), ("page.png", self.png, "image/png")])
        at.run()
        at.button(key="run_batch").click().run()
        self.assertClean(at)
        results = at.session_state["batch"]["results"]
        self.assertEqual([(r.ok, r.text) for r in results], [(True, "கல்வி"), (False, "")])
        self.assertIn("Full-page OCR", results[1].error)

    # ---- full-page workflow

    def read_page(self, at):
        self.upload(at, "page_upload_0", "page.png", self.png)
        self.assertRegex(at.button(key="run_page").label, r"Read page \(84 word regions\)")
        self.assertFalse(at.button(key="run_page").disabled)
        self.assertIsNone(at.session_state["page"])                  # segmentation only so far
        at.button(key="run_page").click().run()
        self.assertClean(at)
        return at.session_state["page"]["result"]

    def test_upload_segment_read_and_reconstruct(self):
        at = self.app()
        r = self.read_page(at)
        self.assertEqual(r.counts, {"regions": 84, "ok": 84, "empty": 0, "failed": 0})
        self.assertEqual(r.text, self.page.text)                     # words, spaces, line breaks, blocks
        self.assertIn(self.page.text, texts(at.code))                # copyable
        self.assertTrue(any('class="ocr-page" lang="ta">' in m for m in texts(at.markdown)))
        self.assertTrue(any("Layout found:" in m and "3 block(s)" in m and "84 word region(s)" in m for m in texts(at.markdown)))
        self.assertTrue(any("84 of 84 word regions read" in m for m in texts(at.markdown)))
        self.assertTrue(at.button(key="run_page").disabled)          # no duplicate run
        table = next(m for m in texts(at.markdown) if "ocr-words" in m)
        rows = re.findall(r'<tr class="st-(\w+)"><td>(\d+)</td><td>([^<]*)</td><td>[^<]*</td><td>\w+</td><td class="ta" lang="ta">([^<]*)</td><td>([^<]*)</td>', table)
        self.assertEqual(len(rows), 84)
        import html
        self.assertEqual([(int(n), html.unescape(rid), html.unescape(t)) for _, n, rid, t, _ in rows],
                         [(w.index + 1, w.id, w.text) for w in r.words])       # every row is its own region and text
        self.assertEqual(html.unescape(rows[0][4]), ", ".join(map(str, r.words[0].box)))
        self.assertEqual([d.proto.label for d in at.get("download_button")], ["Download TXT", "Download JSON", "Download CSV"])
        self.assertTrue(any("word regions, numbered in reading" in str(getattr(e, "proto", "")) for e in at.main))   # the overlay caption

    def test_exports_match_what_is_shown(self):
        at = self.app()
        r = self.read_page(at)
        self.assertEqual(page_export.to_txt(r).decode("utf-8"), self.page.text + "\n")
        import json
        doc = json.loads(page_export.to_json(r))
        self.assertEqual((doc["text"], len(doc["words"]), doc["counts"]["ok"]), (r.text, 84, 84))
        self.assertEqual(doc["model"]["checkpoint_id"], "002a")
        self.assertEqual(page_export.download_name(r, "csv"), "page_page_ocr.csv")

    def test_changing_a_setting_clears_the_result_and_allows_a_rerun(self):
        at = self.app()
        self.read_page(at)
        at.slider(key="page_gap").set_value(0.4).run()
        self.assertClean(at)
        self.assertIsNone(at.session_state["page"])
        self.assertFalse(at.button(key="run_page").disabled)
        at.button(key="run_page").click().run()
        self.assertEqual(at.session_state["page"]["result"].settings["gap_floor_h"], 0.4)

    def test_clear_resets_the_page_tab(self):
        at = self.app()
        self.read_page(at)
        at.button(key="clear_page").click().run()
        self.assertClean(at)
        self.assertIsNone(at.session_state["page"])
        self.assertTrue(at.button(key="run_page").disabled)
        self.assertTrue(any("No page yet" in i for i in texts(at.info)))

    def test_bad_and_oversized_files_are_refused_with_a_reason(self):
        import io

        from PIL import Image
        at = self.app()
        self.upload(at, "page_upload_0", "broken.png", b"\x89PNG\r\n\x1a\nnot an image")
        self.assertTrue(any("could not be read as an image" in e for e in texts(at.error)))
        self.assertTrue(at.button(key="run_page").disabled)
        b = io.BytesIO()
        Image.new("L", (13000, 4000), 255).save(b, "PNG", optimize=True)        # beyond the 12,000 px side limit
        at.button(key="clear_page").click().run()
        self.upload(at, "page_upload_1", "huge.png", b.getvalue())
        self.assertTrue(any("too large" in e for e in texts(at.error)))
        at.button(key="clear_page").click().run()
        self.upload(at, "page_upload_2", "doc.pdf", b"%PDF-1.4 x")
        self.assertTrue(any("PDF" in e for e in texts(at.error)))

    def test_a_blank_page_reports_that_nothing_was_found(self):
        rng = np.random.default_rng(0)
        blank = np.clip(235 + rng.normal(0, 3, (700, 500)), 0, 255).astype(np.uint8)
        at = self.app()
        self.upload(at, "page_upload_0", "blank.png", fx.png(blank))
        self.assertTrue(any("No word regions were found" in e for e in texts(at.error)))
        self.assertTrue(at.button(key="run_page").disabled)

    @unittest.skipIf(GUARD_OPEN, "the human-verified set is frozen: the guard is open")
    def test_a_protected_page_is_refused_before_inference(self):
        """A GENERATED page whose hash is made to look like a development page: refused at upload, recognizer never run."""
        import hashlib
        sha = hashlib.sha256(self.png).hexdigest()
        real = devguard.development_master_hashes()
        at = self.app()
        with mock.patch.object(devguard, "development_master_hashes", return_value={**real, sha: "generated.png"}), \
                mock.patch.object(service.OcrEngine, "recognize_batch", side_effect=AssertionError("the recognizer ran on a protected page")):
            self.upload(at, "page_upload_0", "page.png", self.png)
            self.assertTrue(any("development page" in e for e in texts(at.error)))
            self.assertTrue(at.button(key="run_page").disabled)
            self.assertIsNone(at.session_state["page"])

    def test_untrusted_names_are_escaped(self):
        at = self.app()
        self.upload(at, "page_upload_0", "<b>x</b> [l](u).png", fx.png(fx.make_tamil_page(seed=12, lines=2).gray))
        html = " ".join(texts(at.markdown))
        self.assertNotIn("<b>x</b>", html)


if __name__ == "__main__":
    unittest.main()
