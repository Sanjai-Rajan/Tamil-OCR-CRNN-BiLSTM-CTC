"""tools/pretrain_check.py (R1 README §9 rule 11) on fixture repositories, and the PROPOSED Experiment 002-B config
against the frozen configs (read-only). Nothing is written outside a temporary folder; no model is built."""
import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pretrain_check as pc  # noqa: E402

CONFIG = """experiment: fx
seed: 42
train:
  batch_size: 44
sources:
  - name: hw
    type: manifest
    path: data/real/hw/manifest_train.jsonl
  - name: synth
    type: synth
    vocab: data/text_corpus/tamil_correction_vocabulary.json
val:
  - name: v
    path: data/real/hw/manifest_val.jsonl
"""


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="pretrain_fx_"))
        for rel, data in {"configs/fx.yaml": CONFIG.encode(), "data/text_corpus/tamil_correction_vocabulary.json": b'{"a": 3}',
                          "data/real/hw/manifest_train.jsonl": b'{"image": "data/real/hw/img/0.png", "text": "a"}\n',
                          "data/real/hw/manifest_val.jsonl": b'{"image": "data/real/hw/img/1.png", "text": "b"}\n',
                          "data/real/hw/img/0.png": b"image-zero", "data/real/hw/img/1.png": b"image-one!",
                          "assets/fonts/f.ttf": b"font",
                          "real_docs/r1_pilot/metadata/crops.jsonl": b'{"crop_id": "R1-S001_P01_T00", "label": "x"}\n',
                          "real_docs/r1_pilot/crops/R1-S001/R1-S001_P01_T00.png": b"r1-crop-bytes",
                          "real_docs/r1_pilot/diagnostics/synthetic_twins/arm1/crops/R1-S001/R1-S001_P01_T00.png": b"twin-bytes-01",
                          "real_docs/r1_pilot/hashes.sha256": b""}.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self.vocab = sha(b'{"a": 3}')

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def run_check(self, config="configs/fx.yaml"):
        return pc.check(self.repo, config, vocab_sha256=self.vocab)

    def put(self, rel, data):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def fails(self, pattern):
        problems, _ = self.run_check()
        self.assertTrue(any(pattern in p for p in problems), problems)


class Check(Fixture):
    def test_clean_repository_passes(self):
        problems, facts = self.run_check()
        self.assertEqual(problems, [])
        self.assertEqual(facts["vocabulary_sha256"], self.vocab)
        self.assertEqual((facts["r1_workspace_files"], facts["files_under_data"]), (4, 5))
        self.assertEqual(facts["config"]["experiment"], "fx")
        self.assertEqual(facts["config_reads"], ["data/real/hw/manifest_train.jsonl", "data/text_corpus/tamil_correction_vocabulary.json",
                                                 "assets/fonts", "data/real/hw/manifest_val.jsonl"])

    def test_a_vocabulary_changed(self):
        self.put("data/text_corpus/tamil_correction_vocabulary.json", b'{"a": 3, "r1word": 3}')
        self.fails("(a) data/text_corpus/tamil_correction_vocabulary.json")

    def test_a_real_frozen_value_is_the_default(self):
        problems, _ = pc.check(self.repo, "configs/fx.yaml")
        self.assertTrue(any(pc.VOCAB_SHA256 in p for p in problems))
        self.assertEqual(pc.VOCAB_SHA256, "3f19ac2db418d3061162acfd978463b3f3a0c21652e4968c8383fa2e24a2241a")

    def test_b_r1_metadata_copied_under_data(self):
        self.put("data/text_corpus/extra/words.jsonl", b'{"crop_id": "R1-S001_P01_T00", "label": "x"}\n')
        self.fails("is byte-identical to real_docs/r1_pilot/metadata/crops.jsonl")

    def test_b_r1_crop_copied_under_data_with_another_name(self):
        self.put("data/real/hw/img/2.png", b"r1-crop-bytes")
        self.fails("data/real/hw/img/2.png is byte-identical to real_docs/r1_pilot/crops/R1-S001/R1-S001_P01_T00.png")

    def test_b_twin_copied_under_data(self):
        self.put("data/synth_val/images/9.png", b"twin-bytes-01")
        self.fails("is byte-identical to real_docs/r1_pilot/diagnostics/synthetic_twins")

    def test_b_same_size_other_content_is_not_a_match(self):
        self.put("data/real/hw/img/3.png", b"r1-crop-BYTES")
        self.assertEqual(self.run_check()[0], [])

    def test_b_r1_named_file_under_data(self):
        self.put("data/real/hw/img/R1-S001_P01_T00.png", b"other")
        self.fails("(b) R1-named file under data/")

    def test_c_config_references_real_docs(self):
        self.put("configs/other.yaml", b"sources:\n  - path: real_docs/r1_pilot/metadata/crops.jsonl\n")
        self.fails("(c) configs/other.yaml references")

    def test_c_manifest_references_r1(self):
        self.put("data/real/hw/manifest_train.jsonl", b'{"image": "REAL_DOCS/R1_PILOT/crops/R1-S001/x.png", "text": "a"}\n')
        self.fails("(c) data/real/hw/manifest_train.jsonl references")

    def test_c_reference_across_a_read_block_boundary(self):
        self.put("data/real/hw/manifest_val.jsonl", b"x" * ((1 << 20) - 4) + b"synthetic_twins" + b"\n")
        self.fails("(c) data/real/hw/manifest_val.jsonl references")

    def test_config_reads_outside_data(self):
        self.put("configs/fx.yaml", CONFIG.replace("data/real/hw/manifest_val.jsonl", "runs/x/manifest.jsonl").encode())
        self.fails("which is not under data/ or assets/")

    def test_config_reads_a_missing_file(self):
        (self.repo / "data/real/hw/manifest_val.jsonl").unlink()
        self.fails("which does not exist")

    def test_config_uses_another_vocabulary(self):
        self.put("data/text_corpus/other.json", b"{}")
        self.put("configs/fx.yaml", CONFIG.replace("tamil_correction_vocabulary.json", "other.json").encode())
        self.fails("a vocabulary other than the frozen")

    def test_missing_config(self):
        problems, _ = self.run_check("configs/none.yaml")
        self.assertIn("config configs/none.yaml does not exist", problems)

    def test_main_exit_codes(self):
        keep_root, keep_sha = pc.ROOT, pc.VOCAB_SHA256
        pc.ROOT, pc.VOCAB_SHA256 = self.repo, self.vocab
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pc.main(["--config", "configs/fx.yaml"]), 0)
                self.assertEqual(pc.main(["--config", "configs/none.yaml"]), 1)
        finally:
            pc.ROOT, pc.VOCAB_SHA256 = keep_root, keep_sha


class Proposed002B(unittest.TestCase):
    """Read-only: the PROPOSED config is Experiment 001's with one training key changed; the frozen configs are intact."""
    FROZEN = {"baseline_hw": "11f4570724411bbb770f96856bbe2754375dddb52a4e513cf3f6f5b8fdc77724",
              "experiment_001": "311d2d142ea45db39cfe8d5f1120dfb098bdaddd8207e0c4f1f41a8d750b3b3d",
              "experiment_002a": "4588dff5ab90aecada8aa4fe4b8bc723832be11d036163788e8a5ed8af3f0f32"}

    @staticmethod
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(Proposed002B.flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out

    def load(self, name):
        return self.flat(yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8")))

    def diff(self, a, b):
        x, y = self.load(a), self.load(b)
        return {k: (x.get(k), y.get(k)) for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)}

    def test_frozen_configs_unchanged(self):
        for name, digest in self.FROZEN.items():
            self.assertEqual(sha((ROOT / "configs" / f"{name}.yaml").read_bytes()), digest, name)
            run_cfg = ROOT / "runs" / name / "config.yaml"
            if name != "baseline_hw":
                self.assertEqual(sha(run_cfg.read_bytes()), digest, f"runs/{name}/config.yaml")

    def test_one_changed_variable_against_experiment_001(self):
        self.assertEqual(self.diff("experiment_001", "experiment_002b"),
                         {"experiment": ("experiment_001", "experiment_002b"), "train.batch_size": (64, 44)})

    def test_against_experiment_002a(self):
        self.assertEqual(self.diff("experiment_002a", "experiment_002b"),
                         {"experiment": ("experiment_002a", "experiment_002b"), "train.batch_mix": ("paired", None),
                          "train.batch_size": (64, 44)})

    def test_not_a_duplicate_of_a_completed_experiment(self):
        b = self.load("experiment_002b")
        for name in self.FROZEN:
            other = self.load(name)
            self.assertNotEqual({k: v for k, v in b.items() if k != "experiment"}, {k: v for k, v in other.items() if k != "experiment"}, name)

    def test_batch_size_is_the_cap_arithmetic_of_002a(self):
        b = self.load("experiment_002b")
        self.assertEqual(b["train.max_batch_columns"] // b["train.unknown_width_estimate"], b["train.batch_size"])
        self.assertEqual((b["seed"], b["train.epochs"], b["train.epoch_size"]), (42, 40, 151472))

    def test_proposed_config_reads_only_training_data(self):
        cfg = yaml.safe_load((ROOT / "configs" / "experiment_002b.yaml").read_text(encoding="utf-8"))
        for rel in pc.config_paths(cfg):
            self.assertTrue(rel.startswith(("data/", "assets/")), rel)
            self.assertFalse(any(w in rel for w in pc.FORBIDDEN), rel)
        self.assertEqual([v["path"] for v in cfg["val"] if v.get("primary")], ["data/real/iiit_hw/manifest_val.jsonl"])


if __name__ == "__main__":
    unittest.main()
