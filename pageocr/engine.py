"""
The frozen Experiment 002-A word recognizer as an importable engine. Inference only; no Streamlit.

    engine = OcrEngine(r"runs\\experiment_002a\\best.pth")
    results = engine.recognize_batch([crop0, crop1, ...])      # one Recognition per crop, in the same order

Everything that touches the pixels or the model is the project's existing code, used as it is:
src/preprocess.py (the one preprocessing path), src/model/crnn_v2.py (the model), src/charset.py (greedy CTC collapse
and decoding), and the batch padding of src/data/datasets.collate (paper-coloured padding, width rounded up to
PAD_W_MULTIPLE, true widths passed to the model).

INPUT CONTRACT, checked on every call by validate_crop():
    a numpy ndarray, 2-D, dtype uint8, grayscale, cut from the ORIGINAL page (see pageocr.boxes.crop_from_page).
RGB, RGBA, any 3-D array, boolean masks, non-uint8 arrays and binarized (two-level) arrays are rejected with an
exception. Nothing in this module binarizes, equalizes, denoises or otherwise alters a crop before src/preprocess.py.

CONFIDENCE IS UNCALIBRATED. Recognition.confidence is the mean, over the frames whose most probable class is not the
CTC blank, of that class's softmax probability. It has not been compared with correctness on any data. It is only a
FLAGGING HEURISTIC for pointing an operator at boxes worth a second look; it is not a probability of being correct
and must not be reported as an accuracy.
"""
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pageocr import ENGINE_VERSION, assert_path_allowed
from src import charset as cs
from src import preprocess as pp
from src.data.datasets import PAD_W_MULTIPLE
from src.model.crnn_v2 import CRNNv2

FROZEN_002A_SHA256 = "fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1"   # runs\experiment_002a\best.pth
DEFAULT_BATCH_SIZE = 32


class CheckpointError(RuntimeError):
    """The checkpoint is missing, is not the frozen Experiment 002-A file, or does not fit the project's code."""


class InvalidCropError(ValueError):
    """A crop does not meet the engine's input contract."""


@dataclass(frozen=True)
class Recognition:
    text: str            # raw greedy-CTC prediction: no lexicon, no language model, no correction
    confidence: float    # UNCALIBRATED flagging heuristic in [0, 1] (see module docstring); 0.0 for an empty prediction
    width_px: int        # width of the model input at height 48, after any clipping, before batch padding
    timesteps: int       # output frames the model used for this crop
    clipped: bool        # True if the natural width at height 48 exceeded the preprocessing cap and was squeezed


def natural_width(height: int, width: int) -> int:
    """Width the crop would have at the model's height, before src.preprocess.resize_to_height clips it."""
    return int(round(width * pp.TARGET_H / height))


def validate_crop(crop) -> None:
    """Raise InvalidCropError unless `crop` is a 2-D uint8 grayscale array that is not binarized."""
    if not isinstance(crop, np.ndarray):
        raise InvalidCropError(f"crop must be a numpy ndarray, got {type(crop).__name__}")
    if crop.ndim != 2:
        raise InvalidCropError(f"crop must be 2-D grayscale, got shape {crop.shape} (RGB/RGBA/3-D arrays are rejected)")
    if crop.dtype != np.uint8:
        raise InvalidCropError(f"crop must be uint8, got {crop.dtype} (boolean masks and float arrays are rejected)")
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        raise InvalidCropError(f"crop is too small: {crop.shape[1]} x {crop.shape[0]} px")
    if np.count_nonzero(np.bincount(crop.ravel(), minlength=256)) == 2:
        raise InvalidCropError("crop has exactly two gray levels: binarized input is rejected, crop the original page")


class OcrEngine:
    def __init__(self, checkpoint_path, device=None, expected_sha256: str = FROZEN_002A_SHA256):
        """`expected_sha256`: the only checkpoint bytes accepted. The default is the frozen Experiment 002-A file, so
        every caller that does not pass it (V2, V3, E1/E2/E3) loads exactly that file and nothing else. The word
        frontend (wordocr.checkpoints) passes the hash recorded at a checkpoint's evaluation."""
        path = assert_path_allowed(checkpoint_path)
        if not path.is_file():
            raise CheckpointError(f"checkpoint not found: {path}")
        with open(path, "rb") as f:                         # read-only; the bytes hashed are the bytes loaded
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != expected_sha256:
            wanted = ("the frozen Experiment 002-A checkpoint" if expected_sha256 == FROZEN_002A_SHA256
                      else f"the registered checkpoint with SHA-256 {expected_sha256}")
            raise CheckpointError(f"{path} is not {wanted} (SHA-256 {sha})")
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=True: tensors and plain containers only, no arbitrary pickled objects (the hash check above
        # already pins the bytes; this is a second barrier). All registered checkpoints load this way.
        ck = torch.load(io.BytesIO(data), map_location=self.device, weights_only=True)
        del data
        if list(ck.get("charset", cs.CHARSET)) != list(cs.CHARSET):
            raise CheckpointError("the checkpoint's character set differs from src/charset.py")
        margs = dict(ck["model_args"])
        margs["channels"] = tuple(margs["channels"])
        self._model = CRNNv2(**margs).to(self.device)
        self._model.load_state_dict(ck["model"])
        self._model.eval()
        for prm in self._model.parameters():
            prm.requires_grad_(False)
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = sha
        self.checkpoint_epoch = ck.get("epoch")
        self.engine_version = ENGINE_VERSION

    def info(self) -> dict:
        return {"engine_version": self.engine_version, "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256, "checkpoint_epoch": self.checkpoint_epoch,
                "device": self.device.type,
                "model": ("Experiment 002-A (CRNNv2, greedy CTC)" if self.checkpoint_sha256 == FROZEN_002A_SHA256
                          else "CRNNv2, greedy CTC"),
                "input_height": pp.TARGET_H, "max_width": pp.MAX_W}

    def recognize_batch(self, crops: list[np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE) -> list[Recognition]:
        """One Recognition per crop, in input order. Every crop is validated before any is processed.

        Crops are grouped by model-input width so little padding is needed. As in training and evaluation, a crop's
        last columns can see the padding beside it, so a prediction is exact for a given list of crops and batch
        size, not guaranteed identical across different groupings.
        """
        if isinstance(crops, np.ndarray) or not isinstance(crops, (list, tuple)):
            raise InvalidCropError("recognize_batch takes a list of 2-D uint8 grayscale arrays")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for i, crop in enumerate(crops):
            try:
                validate_crop(crop)
            except InvalidCropError as e:
                raise InvalidCropError(f"crop {i}: {e}") from None

        prepared = [pp.prepare(crop) for crop in crops]                      # (1 x 48 x W in [-1, 1], padding value)
        order = sorted(range(len(crops)), key=lambda i: prepared[i][0].shape[2])
        out: list[Recognition | None] = [None] * len(crops)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            width = max(prepared[i][0].shape[2] for i in idx)
            width = -(-width // PAD_W_MULTIPLE) * PAD_W_MULTIPLE
            images = torch.empty(len(idx), 1, pp.TARGET_H, width)
            for row, i in enumerate(idx):
                tensor, pad_value = prepared[i]
                images[row].fill_(pad_value)
                images[row, :, :, : tensor.shape[2]] = tensor
            widths = torch.tensor([prepared[i][0].shape[2] for i in idx], dtype=torch.long)
            with torch.no_grad():
                with torch.autocast(self.device.type, enabled=self.device.type == "cuda", dtype=torch.float16):
                    logits, lengths = self._model(images.to(self.device), widths)
                prob, best = logits.float().softmax(-1).max(-1)              # T x B each
            prob, best = prob.cpu(), best.cpu()
            for row, i in enumerate(idx):
                n = int(lengths[row])
                frames = best[:n, row]
                emitted = frames != cs.BLANK
                h, w = crops[i].shape
                out[i] = Recognition(
                    text=cs.decode(cs.ctc_greedy_collapse(frames.tolist())),
                    confidence=float(prob[:n, row][emitted].mean()) if bool(emitted.any()) else 0.0,
                    width_px=int(widths[row]),
                    timesteps=n,
                    clipped=natural_width(h, w) > pp.MAX_W,
                )
        return out
