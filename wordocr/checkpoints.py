"""
The checkpoints the frontend may load. Each is pinned by the SHA-256 recorded at its evaluation:

    baseline, Experiment 001, Experiment 002-A   reports/experiment_002a_final_evaluation.md (section 2)
    Experiment 002-B                             MODEL_PROVENANCE.md (the original evaluation record,
                                                 reports/experiment_002b_evaluation/evaluation_provenance.json, holds
                                                 paths of the owner's computer and is not part of this release)

All four use the same CRNNv2 arguments and the same 101-class character set as src/charset.py (checked again on every
load by pageocr.engine). A file whose bytes do not have the registered hash is refused; no other checkpoint is ever
substituted. last.pth files and the batch-norm recalibration diagnostics (runs/exp001_bn_d0) are not offered.
"""
from dataclasses import dataclass
from pathlib import Path

from pageocr import ROOT


class UnknownCheckpointError(KeyError):
    pass


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    label: str
    path: str            # relative to the project root, forward slashes
    sha256: str
    training: str
    evidence: str

    def file(self, root: Path = ROOT) -> Path:
        return root / Path(self.path)


REGISTRY = (
    CheckpointSpec("002a", "Experiment 002-A (default, frozen)", "runs/experiment_002a/best.pth",
                   "fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1",
                   "IIIT-HW handwritten words and synthetic printed words (50/50), paired-source batches",
                   "reports/experiment_002a_final_evaluation.md"),
    CheckpointSpec("002b", "Experiment 002-B", "runs/experiment_002b/best.pth",
                   "a9db26f91e1ff07d46b9bebe4b8d60154c2ddea11c976f811010b9a2eace1677",
                   "as 002-A, but batch size 44 and without paired-source batches",
                   "docs/MODEL_PROVENANCE.md"),
    CheckpointSpec("001", "Experiment 001", "runs/experiment_001/best.pth",
                   "d095b2e84daea177bb35d5ae386e1670e6ca3c27af685d9491f5ced08e086f2b",
                   "IIIT-HW handwritten words and synthetic printed words (50/50)",
                   "reports/experiment_002a_final_evaluation.md"),
    CheckpointSpec("baseline", "Baseline (handwriting only)", "runs/baseline_hw/best.pth",
                   "0b2655176054793ce6cfc7a41f7a90eefd8f19320c30fa3b6a7b302127b5fe1b",
                   "IIIT-HW handwritten words only",
                   "reports/experiment_002a_final_evaluation.md"),
)
DEFAULT_ID = "002a"
_BY_ID = {c.id: c for c in REGISTRY}


def get(checkpoint_id: str) -> CheckpointSpec:
    try:
        return _BY_ID[checkpoint_id]
    except KeyError:
        raise UnknownCheckpointError(f"unknown checkpoint id {checkpoint_id!r}") from None


def ids() -> list[str]:
    return [c.id for c in REGISTRY]


def present(spec: CheckpointSpec, root: Path = ROOT) -> bool:
    """Whether the file exists. Its hash is checked only when it is loaded (pageocr.engine reads and hashes the bytes)."""
    return spec.file(root).is_file()
