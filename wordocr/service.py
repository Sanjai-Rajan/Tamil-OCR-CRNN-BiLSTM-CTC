"""
Loading a registered checkpoint and recognizing decoded images. No Streamlit; usable from scripts and tests.

    model = load_model("002a", "auto")                 # LoadedModel, or ModelUnavailableError with a user message
    results = recognize(model, [decoded, ...])         # one ItemResult per image, in input order; never raises for one bad image

The model is pageocr.engine.OcrEngine (input contract, src/preprocess.py, CRNNv2, greedy CTC). Each image is run as a
batch of one, so a word gets exactly the same prediction whether it is uploaded alone or in a batch, whatever the
other images are (in a mixed batch, padding beside a narrow word could change its last frames). Calls into one model
are serialized with a lock, so several browser sessions sharing the cached model never run it concurrently.

Before a word is read, wordocr.pagedetect checks the layout: an image that holds a page (several rows of text) is
refused with an explanation instead of being squeezed to 48 px and read as one or two characters.

Failures are logged with their traceback to the "wordocr" logger (the terminal running the app); the user sees a
short message without internal details.
"""
import logging
import threading
import time
from dataclasses import dataclass, field

import torch

from pageocr.engine import CheckpointError, InvalidCropError, OcrEngine, validate_crop
from wordocr import VERSION, checkpoints, guard, pagedetect
from wordocr.inputs import DecodedImage

log = logging.getLogger("wordocr")


class ModelUnavailableError(RuntimeError):
    """The selected checkpoint could not be loaded. str(error) is a complete, user-facing message."""


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(choice: str) -> str:
    """'auto' -> 'cuda' if a GPU is usable, else 'cpu'. 'cuda' without a GPU is an error, never a silent switch."""
    if choice == "auto":
        return "cuda" if cuda_available() else "cpu"
    if choice == "cuda" and not cuda_available():
        raise ModelUnavailableError("GPU (CUDA) was selected, but no usable CUDA GPU is available. Select CPU or Auto.")
    if choice not in ("cpu", "cuda"):
        raise ModelUnavailableError(f"Unknown device {choice!r}.")
    return choice


@dataclass
class LoadedModel:
    spec: checkpoints.CheckpointSpec
    engine: OcrEngine
    device: str
    load_seconds: float
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def info(self) -> dict:
        i = self.engine.info()
        return {"checkpoint_id": self.spec.id, "checkpoint_label": self.spec.label, "checkpoint_path": self.spec.path,
                "checkpoint_sha256": i["checkpoint_sha256"], "checkpoint_epoch": i["checkpoint_epoch"],
                "sha256_verified": i["checkpoint_sha256"] == self.spec.sha256, "device": i["device"],
                "architecture": "CRNNv2 (CNN + 2-layer BiLSTM), greedy CTC decoding",
                "input_height_px": i["input_height"], "max_input_width_px": i["max_width"],
                "engine_version": i["engine_version"], "wordocr_version": VERSION}


def load_model(checkpoint_id: str, device_choice: str = "auto") -> LoadedModel:
    """Load exactly the selected checkpoint. Any failure raises ModelUnavailableError; nothing else is tried instead."""
    try:
        spec = checkpoints.get(checkpoint_id)
    except checkpoints.UnknownCheckpointError:
        raise ModelUnavailableError(f"Unknown model {checkpoint_id!r}.") from None
    device = resolve_device(device_choice)
    path = spec.file()
    if not path.is_file():
        raise ModelUnavailableError(f"The checkpoint file for {spec.label} was not found at {spec.path}. Restore the file "
                                    "(it is not downloaded automatically) or select another model.")
    t0 = time.perf_counter()
    try:
        engine = OcrEngine(path, device=device, expected_sha256=spec.sha256)
    except CheckpointError:
        log.exception("checkpoint refused: %s", spec.path)
        raise ModelUnavailableError(f"The checkpoint file for {spec.label} ({spec.path}) is not the registered file: its "
                                    "SHA-256 differs from the one recorded at its evaluation, or it does not fit the "
                                    "project's model code. It was not loaded, and no other model was used instead.") from None
    except torch.cuda.OutOfMemoryError:
        log.exception("out of GPU memory while loading %s", spec.path)
        raise ModelUnavailableError("The GPU ran out of memory while loading the model. Select CPU and try again.") from None
    except Exception as e:
        log.exception("could not load %s on %s", spec.path, device)
        raise ModelUnavailableError(f"The model could not be loaded on {device.upper()} ({type(e).__name__}). "
                                    "Details are in the terminal running the app.") from None
    return LoadedModel(spec=spec, engine=engine, device=device, load_seconds=round(time.perf_counter() - t0, 2))


@dataclass(frozen=True)
class ItemResult:
    index: int                       # position in the submitted list (0-based)
    name: str                        # file name as uploaded
    ok: bool
    text: str = ""                   # raw greedy-CTC prediction ('' when not ok, or when the model emitted nothing)
    error: str = ""                  # user-facing reason when not ok
    image: DecodedImage | None = None
    uncalibrated_score: float | None = None     # pageocr.engine "confidence": NOT a probability of being correct
    timesteps: int | None = None
    seconds: float | None = None

    @property
    def empty(self) -> bool:
        return self.ok and self.text == ""


def failed(index: int, name: str, error: str, image: DecodedImage | None = None) -> ItemResult:
    return ItemResult(index=index, name=name, ok=False, error=error, image=image)


def recognize_one(model: LoadedModel, image: DecodedImage, index: int = 0, confirmed_single_word: bool = False) -> ItemResult:
    """`confirmed_single_word`: the user explicitly said the image is one word. It overrides only the weak "multiline"
    layout verdict (a blurred word that splits into two rows), never the "page" verdict."""
    try:                                  # checked at upload too; checked again here, right before the model sees it
        guard.check_image(image.sha256, image.gray)
    except guard.ProtectedDataError as e:
        return failed(index, image.name, str(e), image)
    layout = pagedetect.assess_cached(image.sha256, image.gray)     # a page is never read as one "word"
    if layout.is_page and not (layout.overridable and confirmed_single_word):
        return failed(index, image.name, pagedetect.refusal_message(layout), image)
    try:
        validate_crop(image.gray)
    except InvalidCropError as e:
        return failed(index, image.name, f"The image does not meet the model's input contract: {e}.", image)
    t0 = time.perf_counter()
    try:
        with model.lock, torch.inference_mode():
            rec = model.engine.recognize_batch([image.gray], batch_size=1)[0]
    except torch.cuda.OutOfMemoryError:
        log.exception("out of GPU memory on %s", image.name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return failed(index, image.name, "The GPU ran out of memory for this image. Use a smaller crop or select CPU.", image)
    except Exception as e:
        log.exception("recognition failed for %s", image.name)
        return failed(index, image.name, f"Recognition failed ({type(e).__name__}). Details are in the terminal running the app.", image)
    return ItemResult(index=index, name=image.name, ok=True, text=rec.text, image=image,
                      uncalibrated_score=round(rec.confidence, 4), timesteps=rec.timesteps,
                      seconds=round(time.perf_counter() - t0, 3))


def recognize(model: LoadedModel, items: list) -> list[ItemResult]:
    """`items`: DecodedImage, or ItemResult already failed at upload (kept in place). One result per item, same order."""
    out = []
    for i, item in enumerate(items):
        if isinstance(item, ItemResult):
            out.append(ItemResult(**{**item.__dict__, "index": i}))
        else:
            out.append(recognize_one(model, item, i))
    return out
