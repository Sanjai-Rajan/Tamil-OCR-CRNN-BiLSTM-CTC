# data/

Training and evaluation data are **not** stored in this repository. `DATASETS.md` (repository root) explains each dataset: its source, terms, size and preparation.

Only two things here are tracked by Git:

- this file;
- `manifests/datasets.json`: the size, SHA-256 and line counts of every dataset file the project used. No images, labels or text.

Everything else under `data/` and `external/` is ignored by `.gitignore`, so prepared data cannot be committed by accident.

Check a local copy:

```powershell
python tools/check_datasets.py            # OK / MISSING / DIFFERENT per dataset
python tools/check_datasets.py --quick    # sizes and counts only
```
