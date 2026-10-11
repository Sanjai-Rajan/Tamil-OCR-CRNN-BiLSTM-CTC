"""
Regression tests of the protected-data guard of the clean release (pageocr/protected.py, pageocr/imageio.py,
pageocr/devguard.py, wordocr/guard.py).

The defect they pin: the guard used to find the R1 review folders through Path.home(). A process whose HOME /
USERPROFILE pointed elsewhere no longer saw them, and protected files were accepted. Now protected files are known
by fingerprint (pageocr/protected_manifest.json) and, in addition, by folders named in TAMIL_OCR_PROTECTED_ROOTS;
neither depends on the home folder, and anything that prevents the check makes the guard refuse (fail closed).

    python -B -m unittest -v tests/test_protected_guard.py

No protected file is used. "Protected" files here are generated images, registered as protected for the test either
in a temporary manifest or in a temporary folder. No model is loaded: where recognition could follow, the recognizer
is a trap that fails the test if it is called.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import wordocr_fixtures as fx  # noqa: E402
from pageocr import devguard, layout, pipeline, protected  # noqa: E402
from pageocr import imageio as pio  # noqa: E402
from pageocr_synth_pages import make_page  # noqa: E402
from wordocr import guard, inputs  # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)


def write_manifest(path: Path, protected_files=(), masters=()):
    """A valid manifest that lists the given byte strings as protected, and `masters` (bytes of images) as development masters."""
    files, pixels, master_hashes = {}, {}, []
    for data in list(protected_files) + list(masters):
        files.setdefault(str(len(data)), []).append(hashlib.sha256(data).hexdigest())
    for data in masters:
        h = hashlib.sha256(data).hexdigest()
        master_hashes.append(h)
        pixels[devguard.pixel_digest(pio.load_page_bytes(data, "m.png").gray)] = h
    body = {"schema": protected.SCHEMA, "created": "test", "sources": [], "file_count": sum(len(v) for v in files.values()),
            "files": files, "development_masters": sorted(master_hashes), "development_master_pixels": pixels}
    body["digest"] = protected.body_digest(body)
    path.write_text(json.dumps(body), encoding="ascii")
    return path


def code_names(path: Path) -> str:
    """The identifiers and operators of a source file joined without spaces; comments and string literals left out."""
    import io
    import tokenize
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline):
        if tok.type in (tokenize.NAME, tokenize.OP):
            out.append(tok.string)
    return "".join(out)


class Trap:
    def recognize_batch(self, *a, **k):
        raise AssertionError("the recognizer was called although the guard had to refuse")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="guard_test_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.ordinary = fx.word_png("தமிழ்")
        self.secret = fx.word_png("புத்தகம்")            # a generated image that the test declares protected
        guard._index_for.cache_clear()
        self.addCleanup(guard._index_for.cache_clear)

    def use_manifest(self, path):
        p = mock.patch.object(protected, "MANIFEST", Path(path))
        p.start()
        self.addCleanup(p.stop)

    def redirect_home(self):
        home = self.tmp / "empty_home"
        home.mkdir(exist_ok=True)
        drive, tail = os.path.splitdrive(str(home))
        p = mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home), "HOMEDRIVE": drive, "HOMEPATH": tail})
        p.start()
        self.addCleanup(p.stop)
        self.assertEqual(Path.home().resolve(), home.resolve())
        return home


class ShippedManifestTests(unittest.TestCase):
    """The manifest that comes with the package."""

    def test_it_loads_verifies_and_lists_the_protected_files(self):
        m = protected.manifest()
        self.assertGreater(m["file_count"], 3000)
        self.assertGreaterEqual(len(m["development_masters"]), 16)
        self.assertEqual(len(m["development_master_pixels"]), len(set(m["development_master_pixels"].values())))
        self.assertTrue(protected.describe()["available"])

    def test_it_holds_fingerprints_only(self):
        doc = json.loads(protected.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(doc), {"schema", "created", "note", "sources", "file_count", "files", "development_masters", "development_master_pixels", "digest"})
        text = protected.MANIFEST.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]|Users|\.png|\.jpg|\.pdf|\.json\"", text))        # no path, no file name
        for src in doc["sources"]:
            self.assertEqual(set(src), {"label", "files"})
            self.assertNotRegex(src["label"], r"[\\/:]")

    def test_generated_images_are_not_protected(self):
        self.assertFalse(protected.manifest_match(fx.word_png("தமிழ்")))
        self.assertIsNone(pio.protected_match(fx.word_png("கணினி"), {}))
        d = inputs.decode_upload("word.png", fx.word_png("தமிழ்"))
        self.assertEqual(d.sha256, hashlib.sha256(fx.word_png("தமிழ்")).hexdigest())


class HomeIndependenceTests(Base):
    def test_no_guard_source_asks_for_the_home_folder(self):
        for rel in ("pageocr/__init__.py", "pageocr/protected.py", "pageocr/imageio.py", "pageocr/devguard.py", "pageocr/pipeline.py", "pageocr/engine.py",
                    "wordocr/guard.py", "wordocr/inputs.py", "wordocr/service.py", "wordocr/fullpage.py", "wordocr/checkpoints.py", "app.py", "app_page.py", "app_page_auto.py"):
            names = code_names(ROOT / rel)                     # identifiers only: comments and strings are not code
            for banned in ("Path.home", "expanduser", "gettempdir", "getuser", "getpass"):
                self.assertNotIn(banned, names, f"{rel} uses {banned}")
            if rel not in ("pageocr/protected.py",):           # the one place that reads the environment (TAMIL_OCR_PROTECTED_ROOTS)
                self.assertNotIn("os.environ", names, f"{rel} reads the environment")
                self.assertNotIn("getenv", names, f"{rel} reads the environment")

    def test_a_fingerprinted_file_is_refused_with_the_home_folder_redirected(self):
        self.use_manifest(write_manifest(self.tmp / "m.json", protected_files=[self.secret]))
        for redirected in (False, True):
            if redirected:
                self.redirect_home()
            with self.assertRaises(inputs.InputError) as cm:
                inputs.decode_upload("x.png", self.secret)
            self.assertIn("protected research data", str(cm.exception))
            self.assertEqual(pio.protected_match(self.secret, {}), protected.MANIFEST_HIT)
            inputs.decode_upload("ordinary.png", self.ordinary)                       # an ordinary image is still accepted

    def test_a_file_in_a_configured_folder_is_refused_with_the_home_folder_redirected(self):
        root = self.tmp / "protected_folder" / "deep"
        root.mkdir(parents=True)
        (root / "page.png").write_bytes(self.secret)
        self.redirect_home()
        with mock.patch.dict(os.environ, {protected.ENV_ROOTS: str(self.tmp / "protected_folder")}):
            with self.assertRaises(inputs.InputError) as cm:
                inputs.decode_upload("x.png", self.secret)
            self.assertIn("protected research data", str(cm.exception))
            inputs.decode_upload("ordinary.png", self.ordinary)
        inputs.decode_upload("x.png", self.secret)            # without the folder configured it is an ordinary generated image

    def test_a_development_master_is_refused_by_hash_and_by_pixels_with_the_home_folder_redirected(self):
        master = fx.word_png("வணக்கம்")
        self.use_manifest(write_manifest(self.tmp / "m.json", masters=[master]))
        self.redirect_home()
        sha = hashlib.sha256(master).hexdigest()
        gray = pio.load_page_bytes(master, "m.png").gray
        self.assertTrue(devguard.is_development_page(sha))
        self.assertEqual(devguard.development_page_for_pixels(gray), sha)
        self.assertEqual(devguard.development_page_for_pixels(pio.load_page_bytes(self.ordinary, "o.png").gray), "")
        self.assertFalse(devguard.status()["ready"])
        with self.assertRaises(guard.ProtectedDataError):                   # a re-encoded copy: other bytes, same pixels
            guard.check_image("0" * 64, gray)
        seg = pipeline.segment(layout.OcrImage(gray=make_page(seed=2).gray, page_sha256=sha))
        with self.assertRaises(devguard.DevelopmentBlocked):
            pipeline.recognize(seg, Trap())

    def test_the_whole_check_in_a_child_process_whose_home_is_an_empty_folder(self):
        """As the isolation test ran it: a new process with HOME / USERPROFILE pointing to an empty folder."""
        root = self.tmp / "protected_folder"
        root.mkdir()
        (root / "page.png").write_bytes(self.secret)
        (self.tmp / "ordinary.png").write_bytes(self.ordinary)
        home = self.tmp / "child_home"
        home.mkdir()
        code = ("import sys, pathlib; sys.path.insert(0, sys.argv[1]);"
                "from wordocr import inputs; from pageocr import protected;"
                "tmp = pathlib.Path(sys.argv[2]);"
                "assert pathlib.Path.home().resolve() == (tmp / 'child_home').resolve();"
                "ok = inputs.decode_upload('o.png', (tmp / 'ordinary.png').read_bytes());"
                "\ntry:\n    inputs.decode_upload('p.png', (tmp / 'protected_folder' / 'page.png').read_bytes()); print('ACCEPTED')"
                "\nexcept inputs.InputError as e:\n    print('REFUSED' if 'protected research data' in str(e) else 'OTHER: ' + str(e))"
                "\nprint('MANIFEST', protected.manifest()['file_count'] > 3000)")
        drive, tail = os.path.splitdrive(str(home))
        env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), HOMEDRIVE=drive, HOMEPATH=tail, PYTHONDONTWRITEBYTECODE="1",
                   **{protected.ENV_ROOTS: str(root)})
        r = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(self.tmp)], env=env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(r.stdout.split(), ["REFUSED", "MANIFEST", "True"])


class FailClosedTests(Base):
    def refused_as_unavailable(self):
        with self.assertRaises(inputs.InputError) as cm:
            inputs.decode_upload("ordinary.png", self.ordinary)
        self.assertIn("cannot check this file", str(cm.exception))
        self.assertIn("Nothing was recognized", str(cm.exception))

    def test_a_missing_manifest_refuses_every_upload(self):
        self.use_manifest(self.tmp / "no_such_manifest.json")
        self.assertFalse(protected.describe()["available"])
        with self.assertRaises(protected.GuardUnavailable):
            protected.manifest_match(self.ordinary)
        self.refused_as_unavailable()

    def test_an_altered_manifest_refuses_every_upload(self):
        p = write_manifest(self.tmp / "m.json", protected_files=[self.secret])
        doc = json.loads(p.read_text(encoding="ascii"))
        doc["files"] = {}                                       # someone empties the list but keeps the old digest
        doc["file_count"] = 0
        p.write_text(json.dumps(doc), encoding="ascii")
        self.use_manifest(p)
        with self.assertRaises(protected.GuardUnavailable):
            protected.manifest()
        self.refused_as_unavailable()

    def test_a_damaged_or_foreign_manifest_refuses_every_upload(self):
        for content in ("", "{not json", json.dumps({"schema": "something-else"}), json.dumps([1, 2, 3])):
            p = self.tmp / "m.json"
            p.write_text(content, encoding="ascii")
            self.use_manifest(p)
            protected._cached.cache_clear()
            with self.assertRaises(protected.GuardUnavailable):
                protected.manifest()
            self.refused_as_unavailable()

    def test_a_configured_folder_that_does_not_exist_refuses_every_upload(self):
        for value in (str(self.tmp / "no_such_folder"), "relative/folder", str(self.tmp) + os.pathsep + str(self.tmp / "gone")):
            with mock.patch.dict(os.environ, {protected.ENV_ROOTS: value}):
                with self.assertRaises(protected.GuardUnavailable):
                    protected.configured_roots()
                with self.assertRaises(protected.GuardUnavailable):
                    pio.protected_size_index()
                self.refused_as_unavailable()
        inputs.decode_upload("ordinary.png", self.ordinary)               # with the variable unset again, ordinary input works

    def test_the_recognizer_is_not_reached_when_the_guard_is_unavailable(self):
        d = inputs.decode_upload("ordinary.png", self.ordinary)           # decoded while the guard works
        self.use_manifest(self.tmp / "no_such_manifest.json")
        from wordocr import service
        r = service.recognize_one(mock.Mock(engine=Trap()), d)
        self.assertFalse(r.ok)
        self.assertIn("cannot check this file", r.error)
        seg = pipeline.segment(layout.OcrImage(gray=make_page(seed=3).gray, page_sha256="0" * 64))
        with self.assertRaises(protected.GuardUnavailable):
            pipeline.recognize(seg, Trap())
        with self.assertRaises(guard.GuardUnavailableError):
            guard.check_image("0" * 64, np.full((40, 120), 200, np.uint8))

    def test_there_is_no_switch_that_turns_the_guard_off(self):
        for rel in ("pageocr/protected.py", "wordocr/guard.py", "pageocr/devguard.py"):
            names = code_names(ROOT / rel).upper()
            for banned in ("DISABLE", "SKIP_GUARD", "NO_GUARD", "ALLOW_PROTECTED", "BYPASS", "FORCE"):
                self.assertNotIn(banned, names, rel)
        # the environment can only ADD folders to check; it cannot remove the fingerprints
        with mock.patch.dict(os.environ, {protected.ENV_ROOTS: ""}):
            self.assertEqual(protected.configured_roots(), ())
            self.assertGreater(protected.manifest()["file_count"], 3000)


if __name__ == "__main__":
    unittest.main()
