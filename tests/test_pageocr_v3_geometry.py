"""
Tests for pageocr.geometry: IoU, margins, one-to-one matching and the split / merge / missed analysis.

    .venv\\Scripts\\python -B -m unittest discover -s tests -p "test_pageocr_*.py"

Synthetic rectangles only. No image, no annotation and no model is read.
"""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pageocr import geometry as g                                  # noqa: E402
from pageocr.boxes import Box                                      # noqa: E402


class BasicsTests(unittest.TestCase):
    def test_iou_values(self):
        a = (0, 0, 10, 10)
        self.assertEqual(g.iou(a, a), 1.0)
        self.assertEqual(g.iou(a, (10, 0, 20, 10)), 0.0)                    # x2 is exclusive: touching boxes do not overlap
        self.assertAlmostEqual(g.iou(a, (5, 0, 15, 10)), 50 / 150)
        self.assertAlmostEqual(g.iou(a, (0, 0, 5, 10)), 0.5)
        self.assertEqual(g.iou(a, (2, 2, 4, 4)), 4 / 100)

    def test_iou_is_symmetric(self):
        rng = random.Random(3)
        for _ in range(200):
            a = (rng.randrange(50), rng.randrange(50), 60 + rng.randrange(50), 60 + rng.randrange(50))
            b = (rng.randrange(80), rng.randrange(80), 90 + rng.randrange(50), 90 + rng.randrange(50))
            self.assertEqual(g.iou(a, b), g.iou(b, a))
            self.assertTrue(0.0 <= g.iou(a, b) <= 1.0)

    def test_bad_boxes_are_refused(self):
        for bad in ((5, 5, 5, 9), (5, 5, 9, 5), (9, 5, 5, 9), (1.5, 2, 3, 4), (1, 2, 3), "box", (True, 0, 2, 2)):
            with self.assertRaises(g.GeometryError, msg=repr(bad)):
                g.check_box(bad)

    def test_as_xyxy(self):
        self.assertEqual(g.as_xyxy(Box("w", 10, 20, 30, 40, 0, 0)), (10, 20, 40, 60))

    def test_clip_box(self):
        self.assertEqual(g.clip_box((-5, -5, 20, 20), 10, 12), (0, 0, 10, 12))
        self.assertIsNone(g.clip_box((30, 30, 40, 40), 10, 10))


class MarginTests(unittest.TestCase):
    def test_grow_and_shrink(self):
        b = (100, 200, 160, 230)
        self.assertEqual(g.expand_box(b, 0, 1000, 1000)["box"], b)
        self.assertEqual(g.expand_box(b, 4, 1000, 1000)["box"], (96, 196, 164, 234))
        self.assertEqual(g.expand_box(b, -2, 1000, 1000)["box"], (102, 202, 158, 228))
        self.assertEqual(g.expand_box(b, 8, 1000, 1000)["clipped_edges"], [])

    def test_page_boundaries_are_kept(self):
        e = g.expand_box((2, 3, 60, 40), 8, 64, 44)
        self.assertEqual(e["box"], (0, 0, 64, 44))
        self.assertEqual(e["clipped_edges"], ["left", "top", "right", "bottom"])
        self.assertEqual(e["requested"], (-6, -5, 68, 48))
        self.assertTrue(e["valid"])

    def test_too_small_after_shrinking_is_rejected(self):
        e = g.expand_box((10, 10, 17, 40), -4, 100, 100)                    # 7 wide: -4 on both sides leaves nothing
        self.assertFalse(e["valid"])
        self.assertIsNone(e["box"])
        self.assertTrue(g.expand_box((10, 10, 20, 40), -4, 100, 100)["valid"])   # 10 wide: 2 left

    def test_the_six_margins_of_e3(self):
        b = (50, 50, 90, 80)
        got = {k: g.expand_box(b, k, 500, 500)["box"] for k in (-4, -2, 0, 2, 4, 8)}
        self.assertEqual(got[-4], (54, 54, 86, 76)); self.assertEqual(got[8], (42, 42, 98, 88))
        for k, box in got.items():
            self.assertEqual(box[2] - box[0], 40 + 2 * k)

    def test_margin_must_be_whole_pixels(self):
        with self.assertRaises(g.GeometryError):
            g.expand_box((0, 0, 10, 10), 1.5, 100, 100)


class MatchingTests(unittest.TestCase):
    def test_perfect_detection(self):
        ref = [(0, 0, 50, 20), (60, 0, 110, 20), (0, 30, 80, 50)]
        c = g.compare(ref, list(ref))
        self.assertEqual((c.precision, c.recall, c.f1, c.mean_iou_of_matches), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(sorted(i for i, _, _ in c.matches), [0, 1, 2])
        self.assertEqual((c.missed, c.split, c.merged, c.poorly_localised, c.unmatched_detections, c.duplicates), ((), (), (), (), (), ()))

    def test_matching_is_one_to_one_and_takes_the_best_pair(self):
        ref = [(0, 0, 100, 20)]
        det = [(0, 0, 90, 20), (0, 0, 100, 20), (0, 0, 60, 20)]
        m = g.match_one_to_one(ref, det)
        self.assertEqual(m, [(0, 1, 1.0)])

    def test_matching_does_not_depend_on_the_order_of_the_boxes(self):
        rng = random.Random(11)
        ref = [(x, y, x + 40, y + 18) for x in range(0, 400, 55) for y in range(0, 200, 30)]
        det = [(a + rng.randint(-4, 4), b + rng.randint(-3, 3), c + rng.randint(-4, 4), d + rng.randint(-3, 3)) for a, b, c, d in ref]
        base = {(ref[i], det[j]) for i, j, _ in g.match_one_to_one(ref, det)}
        for seed in range(5):
            r2, d2 = ref[:], det[:]
            random.Random(seed).shuffle(r2); random.Random(seed + 9).shuffle(d2)
            self.assertEqual({(r2[i], d2[j]) for i, j, _ in g.match_one_to_one(r2, d2)}, base)
        self.assertEqual(len(base), len(ref))

    def test_merged_words_under_segmentation(self):
        ref = [(0, 0, 40, 20), (46, 0, 90, 20), (200, 0, 240, 20)]
        det = [(0, 0, 90, 20), (200, 0, 240, 20)]                            # the first two words in one box
        c = g.compare(ref, det)
        self.assertEqual(c.merged, ((0, (0, 1)),))
        self.assertEqual(c.counts["reference_in_a_merge (under-segmentation)"], 2)
        self.assertEqual(c.counts["matched"], 1)
        self.assertEqual(c.unmatched_detections, ())                         # the merging box is explained, not "spurious"
        self.assertAlmostEqual(c.recall, 1 / 3)

    def test_split_word_over_segmentation(self):
        ref = [(0, 0, 100, 20)]
        det = [(0, 0, 45, 20), (52, 0, 100, 20)]
        c = g.compare(ref, det)
        self.assertEqual(c.split, ((0, (0, 1)),))
        self.assertEqual(c.counts["reference_split (over-segmentation)"], 1)
        self.assertEqual(c.counts["matched"], 0)

    def test_a_pulli_left_outside_is_a_split_but_the_word_still_matches(self):
        ref = [(0, 0, 100, 30)]
        det = [(0, 8, 100, 30), (40, 0, 46, 5)]                              # the letters, and the dot above them on its own
        c = g.compare(ref, det)
        self.assertEqual(len(c.matches), 1)                                  # IoU 0.73: still the same word
        self.assertEqual(c.split, ((0, (0, 1)),))
        self.assertEqual(c.counts["reference_split (over-segmentation)"], 0)  # counted once, as matched

    def test_missed_and_spurious(self):
        ref = [(0, 0, 40, 20), (100, 100, 140, 120)]
        det = [(0, 0, 40, 20), (300, 300, 340, 320)]
        c = g.compare(ref, det)
        self.assertEqual(c.missed, (1,))
        self.assertEqual(c.unmatched_detections, (1,))
        self.assertEqual((c.precision, c.recall), (0.5, 0.5))

    def test_poorly_localised(self):
        c = g.compare([(0, 0, 100, 20)], [(60, 0, 200, 20)])                 # overlaps, IoU 0.2, neither a split nor a merge
        self.assertEqual(c.poorly_localised, (0,))
        self.assertEqual(c.missed, ())

    def test_duplicates(self):
        c = g.compare([(0, 0, 50, 20)], [(0, 0, 50, 20), (1, 0, 50, 20)])
        self.assertEqual(c.duplicates, ((0, 1),))
        self.assertEqual(c.counts["matched"], 1)

    def test_every_reference_box_is_counted_exactly_once(self):
        rng = random.Random(5)
        for _ in range(60):
            ref = [(x, 0, x + rng.randint(10, 60), 20) for x in range(0, 600, 70)]
            det = [(rng.randint(0, 600), 0, rng.randint(601, 700), 20) if rng.random() < 0.2 else (x + rng.randint(-8, 8), 0, x + rng.randint(15, 70), 20) for x in range(0, 600, rng.choice((35, 70, 140)))]
            k = g.compare(ref, det).counts
            self.assertEqual(k["matched"] + k["reference_in_a_merge (under-segmentation)"] + k["reference_split (over-segmentation)"] + k["reference_missed"] + k["reference_poorly_localised"], k["reference"])

    def test_sparse_reference_does_not_punish_detections_elsewhere(self):
        ref = [(0, 0, 40, 20)]                                               # only one word of the page was verified
        det = [(0, 0, 40, 20), (60, 0, 100, 20), (0, 40, 50, 60), (70, 40, 120, 60)]
        full, sparse = g.compare(ref, det), g.compare(ref, det, sparse_reference=True)
        self.assertEqual(full.precision, 0.25)
        self.assertEqual(sparse.precision, 1.0)
        self.assertEqual(sparse.unjudged_detections, (1, 2, 3))
        self.assertEqual(sparse.recall, full.recall)
        self.assertIn("only detections that overlap a reference box", sparse.precision_scope)

    def test_empty_inputs(self):
        self.assertEqual(g.compare([], []).counts["reference"], 0)
        self.assertEqual(g.compare([(0, 0, 5, 5)], []).missed, (0,))
        self.assertEqual(g.compare([], [(0, 0, 5, 5)]).unmatched_detections, (0,))

    def test_pooling_is_a_micro_average(self):
        a = g.compare([(0, 0, 10, 10)], [(0, 0, 10, 10)])                    # 1 of 1
        b = g.compare([(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)], [(0, 0, 10, 10)])   # 1 of 3
        p = g.pool([a, b])
        self.assertEqual(p["counts"]["reference"], 4)
        self.assertEqual(p["recall"], 0.5)
        self.assertEqual(p["pages"], 2)


if __name__ == "__main__":
    unittest.main()
