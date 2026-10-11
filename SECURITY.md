# Security

## Reporting

Please report security or privacy problems privately to the repository maintainer (via GitHub's private vulnerability reporting, once the repository is published), not in a public issue. Do not include real document images or personal data in a report.

## What the code does and does not do

- **Local only.** The application and its tests make no network request; the Streamlit server listens on `localhost` with usage statistics off (`.streamlit/config.toml`). No account, API key, token or cloud service is used.
- **No model download.** Checkpoints are never fetched. `wordocr/checkpoints.py` pins each loadable file by SHA-256; anything else is refused, and no other file is substituted.
- **Safe loading in the application.** `pageocr/engine.py` loads weights with `torch.load(..., weights_only=True)` after the hash check.
- **Caution, training and evaluation tools.** `src/train.py` (resuming from its own `last.pth`) and `tools/evaluate.py` call `torch.load(..., weights_only=False)`, which can execute code stored in a checkpoint file. Use them **only with checkpoint files you produced yourself or fully trust.**
- **Protected-data guard.** `wordocr/guard.py` refuses the project's protected research files, recognised by the size and SHA-256 fingerprints in `pageocr/protected_manifest.json` (no names, paths or content). If the manifest is missing or altered, every upload is refused (fail closed). Additional local folders can be protected with `TAMIL_OCR_PROTECTED_ROOTS` (`docs/INSTALLATION.md`).
- **Uploaded images** stay in memory. They are written only if you press a save or download button: the diagnostic bundle can be saved under `diagnostic_runs/`, which is ignored by Git.

## Secrets

The repository was scanned for credential patterns before publication; see `REPOSITORY_AUDIT.md`. If you ever commit a secret by mistake, deleting the file does not help: it stays in the Git history and must be revoked or rotated at its issuer.
