# Model checkpoints

**No model weights are distributed with this repository.** The code (architecture, preprocessing, inference, training) is complete. A trained checkpoint is a separate file that you must already have, or produce yourself, under terms that allow you to use it.

## What the application loads

`wordocr/checkpoints.py` is the registry. Each entry names a path relative to the repository root and the SHA-256 recorded when that checkpoint was evaluated:

| ID | Label | Path | SHA-256 |
|---|---|---|---|
| `002a` (default) | Experiment 002-A | `runs/experiment_002a/best.pth` | `fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1` |
| `002b` | Experiment 002-B | `runs/experiment_002b/best.pth` | `a9db26f91e1ff07d46b9bebe4b8d60154c2ddea11c976f811010b9a2eace1677` |
| `001` | Experiment 001 | `runs/experiment_001/best.pth` | `d095b2e84daea177bb35d5ae386e1670e6ca3c27af685d9491f5ced08e086f2b` |
| `baseline` | Baseline (handwriting only) | `runs/baseline_hw/best.pth` | `0b2655176054793ce6cfc7a41f7a90eefd8f19320c30fa3b6a7b302127b5fe1b` |

On every load (`pageocr/engine.py`), the file's SHA-256 is compared with the registry; then the model is built with the project's own `CRNNv2` and the weights are loaded with `torch.load(..., weights_only=True)`. Any mismatch (hash, architecture, character set) refuses the file. **No other file is ever tried instead.** Nothing is downloaded.

## Supplying a checkpoint

1. Obtain an authorised copy of a registered checkpoint from the project owner.
2. Put it at the registered path, for example `runs/experiment_002a/best.pth`.
3. Check its hash:
   - Windows: `certutil -hashfile runs\experiment_002a\best.pth SHA256`
   - Linux/macOS: `sha256sum runs/experiment_002a/best.pth`
4. Start the app; the sidebar shows "Model ready".

`.gitignore` ignores every `*.pth` (and other weight formats), so the file is not committed by accident. Check with `git status` before every commit anyway.

## Using a checkpoint you trained

`src/train.py` writes `runs/<experiment>/best.pth`. Its hash will differ from every registered one, so the application will refuse it until you add a registry entry, with its own ID, path and SHA-256, to `REGISTRY` in `wordocr/checkpoints.py`. It must use the same `CRNNv2` arguments and character set; `pageocr/engine.py` checks both. Do not reuse an existing ID or overwrite a registered file.

## Without a checkpoint

- The application starts. The model status reads "Model not available. The checkpoint file for … was not found at runs/experiment_002a/best.pth. Restore the file (it is not downloaded automatically) or select another model." Recognition buttons stay disabled; segmentation and its overlays still work in the Full-page tab.
- `run_all_tests.py` runs the suites that need no checkpoint and reports the others as NOT RUN, naming the missing file (exit code 2).
