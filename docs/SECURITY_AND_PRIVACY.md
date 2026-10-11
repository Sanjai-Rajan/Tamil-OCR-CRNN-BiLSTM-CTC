> **Note for this repository.** This document was written for the project's private 2026-10-10 review release and is kept unchanged below this note. Differences here: trained weights (`runs/*/best.pth`) are **not included** (see `docs/MODEL_CHECKPOINTS.md`); the release evidence files (`RELEASE_*`, `RELEASE_SHA256SUMS.txt`), `release_tools/`, `release_evidence/`, `GLOBAL_CLEANUP_PLAN.md` and `CHANGELOG_CLEAN_RELEASE.md` are not part of this repository; the other release documents now live in `docs/`. The repository's own record is `EXPORT_MANIFEST.csv` and `REPOSITORY_AUDIT.md`.

# Security and Privacy

## Local operation

- The application processes images on the computer it runs on. It sends nothing anywhere: the inference code imports no network module, and in the audits no connection or name lookup to anything but the local computer was attempted.
- The web interface listens on `localhost` only (`.streamlit/config.toml`). Usage statistics of the interface framework are switched off there.
- No account, API key or token is used or looked for. The package contains no credential; a pattern scan of the original project found none.
- Uploaded images are held in memory for the session. The application writes no upload to disk. Results are saved only when you press a download button.

Not verified: what the browser page of the interface framework loads when it is open in a browser. This was not captured at the network level.

## Checkpoint loading

- Every load reads the file, computes its SHA-256 and compares it with the value registered in `wordocr/checkpoints.py`. A mismatch is an error; no other file is substituted (`pageocr/engine.py`).
- Loading uses `torch.load(..., weights_only=True)`, which accepts tensors and plain containers only and cannot execute code from the file.
- The character set stored in the checkpoint must equal `src/charset.py`.
- Checkpoint paths inside protected folders are refused.

The training and evaluation scripts (`src/train.py --resume`, `tools/evaluate.py`) use the unrestricted loading mode. This was reviewed and deliberately left unchanged, so that the training code here is the code that produced the checkpoints. It is not a defect of the application, but those two scripts must only be given checkpoint files you trust.

## The protected-data guard

The project holds research material on which the recognizer must not be run: the pages and crops of a real-document pilot, and development pages awaiting human verification. Running the model on them could contaminate later evaluation. None of that material is in this package. The guard exists so that such a file is refused if it is ever uploaded.

### How it recognises protected files

1. **Fingerprints.** `pageocr/protected_manifest.json` lists, for each of 3,404 protected files, its size and SHA-256, and for each of the 17 development masters also a digest of its decoded pixels. It contains no file name, no path and no image content. An upload whose bytes match is refused; a development master is also recognised after a lossless re-encoding.
2. **Folders on this computer.** `real_docs/` and `development_pages/` inside the package folder, if present, and every folder named in the environment variable `TAMIL_OCR_PROTECTED_ROOTS`, are indexed and compared byte for byte.
3. **Path names.** A file path containing one of the protected folder names is refused by the page and box loaders.

### What was fixed in this release

In the original project the guard located the pilot's review folders through the user's home folder. A process started with a different home folder no longer saw them, and the files in them were not recognised. The isolation test of 2026-10-10 noticed this.

In this release nothing in the guard depends on the home folder. Verified on the development computer: all 3,571 protected files (3,404 distinct) and all 17 development masters are recognised, both normally and in a process whose home folder points to an empty temporary folder.

### Fail closed

If the guard cannot do its work, it refuses the input instead of accepting it:

| Situation | Behaviour |
|---|---|
| the fingerprint file is missing, unreadable, of unknown format, or its content does not match its own digest | every upload is refused with the reason |
| a folder named in `TAMIL_OCR_PROTECTED_ROOTS` does not exist or is not an absolute path | every upload is refused with the reason |
| the environment variable is unset | normal operation; the fingerprints still apply |

There is no setting that switches the guard off. The environment variable can only add folders.

### Limits of the guard

- It recognises **exact copies** (and, for development masters, exact pixels). A lossy re-encode, a crop, a rescan or any edit of protected material is not recognised. The guard supports a rule; it cannot enforce it against someone determined to break it.
- Files added to the protected folders **after** the fingerprints were made (2026-10-10) are recognised only on a computer where those folders are indexed. After adding protected material, rebuild the fingerprints with `release_tools/build_protected_manifest.py`.
- The fingerprint file is checked against its own digest, which detects damage and casual editing. Someone who can edit the package can also replace the file and its digest together; the package's checksum list (`RELEASE_SHA256SUMS.txt`) is the reference for that.
- Development pages stay refused in this release under all circumstances: the verified set that would open them is not part of the package.

### A decision for the owner

Shipping the fingerprint file means shipping sizes and SHA-256 values of protected files. These reveal nothing about the content, but they do allow anyone who already has a file to confirm that it belongs to the protected set. If that is not acceptable, the alternative is to ship an empty list built for the purpose and rely on configured folders; the guard would then recognise nothing on a reviewer's computer. This choice was made in favour of protection and is listed as an open decision in `RELEASE_AUDIT_SUMMARY.md`.

## Privacy of this package

- No personal directory path appears in the code or documentation of the release.
- No dataset image, document scan or annotation of the project is included. The only images are fonts' glyphs rendered into 32 synthetic test inputs under `release_evidence/comparison_inputs/`.
- The training logs in `runs/` contain timings, losses and error rates, no paths outside the project and no sample text.

## Reporting a problem

Report a suspected security problem to the project owner directly rather than in a public channel.
