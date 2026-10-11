"""
Regression test of tools/check_split.py (read-only split audit): every cross-split duplicate group must be printed, even
when many within-split groups exist. Before 2026-10-10 a single cap of 15 hid them. Temporary fixture only.

    .venv\\Scripts\\python -B -m unittest tests.test_check_split
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import check_split                                               # noqa: E402


class CheckSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="check_split_"))
        rows = {"train": [], "val": [], "test": []}
        for g in range(20):                                          # 20 within-split duplicate groups in train
            for k in range(2):
                name = f"train/w{g}_{k}.jpg"
                self.img(name, f"within-{g}")
                rows["train"].append({"image": name, "label": "அ", "orig": f"{g}_{k}.jpg"})
        self.img("train/leak.jpg", "leak")
        rows["train"].append({"image": "train/leak.jpg", "label": "கண்", "orig": "L1.jpg"})
        self.img("test/leak.jpg", "leak")                           # the same bytes in the test split
        rows["test"].append({"image": "test/leak.jpg", "label": "கண்", "orig": "L2.jpg"})
        self.img("val/v.jpg", "unique")
        rows["val"].append({"image": "val/v.jpg", "label": "மலை", "orig": "V.jpg"})
        for s, r in rows.items():
            (self.tmp / f"manifest_{s}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def img(self, name, content):
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode())

    def test_every_cross_split_group_is_printed(self):
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ["check_split.py", "--data", str(self.tmp)]
        try:
            with contextlib.redirect_stdout(out):
                check_split.main()
        finally:
            sys.argv = argv
        text = out.getvalue()
        self.assertIn("20 groups within one split, 1 groups across splits", text)
        self.assertIn("ACROSS SPLITS", text)
        self.assertIn("test row 0 test/leak.jpg", text)
        self.assertEqual(text.count("within split   "), 15)
        self.assertIn("5 more within-split groups not shown", text)


if __name__ == "__main__":
    unittest.main()
