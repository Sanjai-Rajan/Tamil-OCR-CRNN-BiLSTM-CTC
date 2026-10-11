"""
Tests of tools/recompute_eval_metrics.py (read-only metric-reproducibility audit), on temporary fixtures only.

    .venv\\Scripts\\python -B -m unittest tests.test_recompute_eval_metrics
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import recompute_eval_metrics as rem                             # noqa: E402


class RecomputeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="recompute_test_"))
        (self.tmp / "data").mkdir()
        self.rep = self.tmp / "reports"
        self.rep.mkdir()
        labels = [("a.png", "அம்மா", "modern"), ("b.png", "தமிழ்", "old"), ("c.png", "நூல்", "modern"), ("d.png", "கண்", "modern")]
        (self.tmp / "data" / "m.jsonl").write_text("".join(json.dumps({"image": i, "label": l, "style": s}, ensure_ascii=False) + "\n"
                                                           for i, l, s in labels), encoding="utf-8")
        self.rows = []
        for li, (img, lab, style) in enumerate(labels):
            if style != "modern":
                continue
            hyp = lab if li != 2 else "நூ"
            self.rows.append({"manifest_index": li, "image": img, "style": style, "font": None, "ref": lab, "hyp": hyp,
                              "edits": rem.editdistance.eval(lab, hyp), "ref_len": len(lab), "hyp_len": len(hyp), "correct": lab == hyp})
        self.meta = {"_meta": {"checkpoint": "x.pth", "epoch": 1, "manifest": "data/m.jsonl", "style": "modern", "n_scored": len(self.rows)}}
        self.write()
        m = rem.metrics(self.rows)
        (self.rep / "eval_x_summary.json").write_text(json.dumps({"all": {k: m[k] for k in ("n", "cer", "wer", "word_acc", "len_ratio")}}),
                                                      encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rows=None, meta=None):
        body = [meta or self.meta] + (rows if rows is not None else self.rows)
        (self.rep / "eval_x_samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in body), encoding="utf-8")

    def audit(self):
        return rem.audit_file(self.rep / "eval_x_samples.jsonl", self.tmp)

    def test_a_consistent_record_passes_and_the_metrics_are_micro_averaged(self):
        r = self.audit()
        self.assertTrue(r["ok"], r["problems"])
        self.assertTrue(r["summary_match"])
        self.assertEqual(r["provenance"]["rows_checked"], 3)
        m = r["recomputed"]
        self.assertEqual((m["n"], m["edits"], m["ref_chars"]), (3, 2, 5 + 4 + 3))
        self.assertAlmostEqual(m["cer"], 2 / 12)                     # summed edits over summed characters, not a mean of ratios
        self.assertAlmostEqual(m["wer"], 1 / 3)

    def test_a_tampered_prediction_is_detected(self):
        rows = [dict(self.rows[0], hyp="அம்ம")] + self.rows[1:]         # stored edits/correct no longer agree with hyp
        self.write(rows)
        r = self.audit()
        self.assertFalse(r["ok"])
        self.assertTrue(any("disagree with ref and hyp" in p for p in r["problems"]))

    def test_a_label_that_is_not_the_manifest_label_is_detected(self):
        lab = "அப்பா"
        rows = [dict(self.rows[0], ref=lab, hyp=lab, edits=0, ref_len=len(lab), hyp_len=len(lab), correct=True)] + self.rows[1:]
        self.write(rows)
        r = self.audit()
        self.assertFalse(r["ok"])
        self.assertEqual(r["provenance"]["mismatches"], 1)

    def test_a_summary_that_differs_is_detected(self):
        s = json.loads((self.rep / "eval_x_summary.json").read_text(encoding="utf-8"))
        s["all"]["cer"] += 0.001
        (self.rep / "eval_x_summary.json").write_text(json.dumps(s), encoding="utf-8")
        r = self.audit()
        self.assertFalse(r["ok"])
        self.assertFalse(r["summary_match"])

    def test_duplicates_counts_and_missing_manifest_are_detected(self):
        self.write(self.rows + [self.rows[0]])
        r = self.audit()
        self.assertTrue(any("more than once" in p for p in r["problems"]))
        self.assertTrue(any("n_scored" in p for p in r["problems"]))
        self.write(meta={"_meta": dict(self.meta["_meta"], manifest="data/none.jsonl")})
        self.assertTrue(any("not found" in p for p in self.audit()["problems"]))

    def test_the_cli_never_overwrites_its_output(self):
        out = self.tmp / "out.json"
        args = ["--reports-dir", str(self.rep), "--root", str(self.tmp), "--out", str(out)]
        self.assertEqual(rem.main(args), 0)
        before = out.read_bytes()
        with self.assertRaises(FileExistsError):
            rem.main(args)
        self.assertEqual(out.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
