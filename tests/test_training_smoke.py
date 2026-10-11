"""
Smoke test of the training pipeline: the real training components on a few generated words, on the CPU.

    python -B -m unittest -v tests/test_training_smoke.py

Uses src.train.build_model / build_optim / TrainStep and src.data.datasets.collate exactly as src/train.py does (model
arguments from configs/experiment_002a.yaml), on 4 words rendered by the project's own renderer. It checks that one
training step runs (CTC loss, backward pass, gradient clipping, optimiser and scheduler step) and that the loss of this
tiny batch falls clearly when the model is fitted to it. No dataset, checkpoint or GPU is needed, and nothing is
written to disk. This is not training, only a check that the training code works end to end.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fullpage_fixtures as fx  # noqa: E402

if not fx.RENDERING:
    raise unittest.SkipTest(fx.SKIP_REASON)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src import charset as cs  # noqa: E402
from src.data.datasets import collate  # noqa: E402
from src.train import TrainStep, build_model, build_optim  # noqa: E402

WORDS = ("தமிழ்", "மொழி", "கல்வி", "நூல்")


def batch_of(words):
    items = []
    for w in words:
        m = fx.render_word(w, size=32)
        img = (238 - (m.astype(np.float32) / 255.0) * 208).astype(np.uint8)        # dark ink on light paper
        items.append((np.pad(img, 8, constant_values=238), w, "synthetic"))
    return collate(items)


class Configs(unittest.TestCase):
    def test_every_config_parses_and_the_frozen_ones_are_complete(self):
        frozen = ("baseline_hw", "experiment_001", "experiment_002a", "experiment_002b")
        for p in sorted((ROOT / "configs").glob("*.yaml")):
            with self.subTest(config=p.name):
                cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
                self.assertIsInstance(cfg, dict)
                if p.stem in frozen:
                    for k in ("experiment", "seed", "model", "train"):
                        self.assertIn(k, cfg)
                    self.assertEqual(cfg["experiment"], p.stem)
                    self.assertEqual(cfg["seed"], 42)


class TrainingStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cfg = yaml.safe_load((ROOT / "configs" / "experiment_002a.yaml").read_text(encoding="utf-8"))
        cls.device = torch.device("cpu")
        cls.model = build_model(cfg["model"], cls.device)
        tc = dict(cfg["train"], lr=0.001, fused_adamw=False)
        cls.steps = 60
        cls.opt, cls.sched = build_optim(cls.model, tc, cls.steps, cls.device)
        cls.scaler = torch.amp.GradScaler("cpu", enabled=False)
        cls.step = TrainStep(cls.model, cls.opt, cls.sched, cls.scaler, cls.device, amp=False, grad_clip=float(tc["grad_clip"]))
        cls.batch = batch_of(WORDS)

    def test_batch_is_built_like_training_batches(self):
        b = self.batch
        self.assertEqual(b["images"].shape[0], len(WORDS))
        self.assertEqual(b["images"].shape[1:3], (1, 48))
        self.assertEqual(b["targets"].tolist(), [i for w in WORDS for i in cs.encode(w)])
        self.assertEqual(b["target_lengths"].tolist(), [len(cs.encode(w)) for w in WORDS])

    def test_loss_is_finite_and_falls_when_fitting_a_tiny_batch(self):
        self.model.train()
        losses = [float(self.step(self.batch)) for _ in range(self.steps)]
        self.assertTrue(all(np.isfinite(losses)), losses[:5])
        first, last = np.mean(losses[:5]), np.mean(losses[-5:])
        self.assertLess(last, 0.5 * first, f"loss {first:.3f} -> {last:.3f}")


if __name__ == "__main__":
    unittest.main()
