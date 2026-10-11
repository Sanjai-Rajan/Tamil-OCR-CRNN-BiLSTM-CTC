"""
Repository readiness check: what Git would publish, scanned for secrets, weights, large files, personal paths and
missing pieces. Prints findings as file:line and pattern name only; it never prints a matched value.

    python tools/check_repository.py            human-readable report
    python tools/check_repository.py --json     the same as JSON

The file set is what `git add -A` would take: tracked files plus untracked files not ignored by .gitignore
(`git ls-files --cached --others --exclude-standard`). Checks:
  secrets      credential and token patterns (cloud, GitHub, Hugging Face, OpenAI, Anthropic, Slack, Google keys,
               private-key blocks, JWTs, assigned passwords/secrets/tokens); credential and .env files
  weights      model-weight files (.pth .pt .ckpt .onnx .safetensors ...)
  size         files over 50 MB (GitHub warns) and over 100 MB (GitHub refuses)
  paths        absolute personal paths (a drive's Users folder, a /home/<name> or /Users/<name> folder)
  layout       nested .git folders, virtual environments, bytecode, caches
  required     files the documented workflows need are present and not ignored
  links        relative links in the top-level Markdown files point to existing files
Exit code: 0 if nothing blocks publication, 1 otherwise (secrets, weights, files over 100 MB, personal paths,
nested repositories or environments, missing required files, broken links).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = {
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
    "huggingface_token": r"\bhf_[A-Za-z0-9]{30,}",
    "openai_key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "slack_token": r"xox[abprs]-[A-Za-z0-9-]{10,}",
    "google_api_key": r"AIza[0-9A-Za-z_-]{35}",
    "private_key_block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "jwt": r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "assigned_secret": r"(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?token|auth[_-]?token|hf_token|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
}
# the user-name segment may not be the documentation placeholder "…" (as in "X:\Users\…")
PERSONAL_PATH = r"(?i)\b[a-z]:[\\/]+users[\\/]+[^\\/\s\"'`<>…]+|/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/"
WEIGHTS = {".pth", ".pt", ".ckpt", ".onnx", ".safetensors", ".bin", ".h5", ".pb", ".tflite", ".gguf", ".pkl", ".pickle", ".joblib"}
CRED_NAMES = {".git-credentials", ".netrc", ".pypirc", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
CRED_SUFFIX = {".pem", ".key", ".p12", ".pfx"}
LAYOUT = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".huggingface", ".cache"}
REQUIRED = ["README.md", "DATASETS.md", "MODEL_CARD.md", "DEPLOYMENT.md", "docs/README_FRONTEND.md", "docs/TRAINING_GUIDE.md",
            "docs/EVALUATION_AND_LIMITATIONS.md", "docs/ARCHITECTURE.md", "docs/INSTALLATION.md", "docs/MODEL_CHECKPOINTS.md",
            "CONTRIBUTING.md", "SECURITY.md", "LICENSE_SELECTION_REQUIRED.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
            "requirements.txt", "requirements-runtime.txt", "requirements-training.txt", "requirements-test.txt", ".gitignore",
            "app.py", "run_all_tests.py", "src/model/crnn_v2.py", "src/train.py", "src/charset.py", "src/preprocess.py",
            "src/data/datasets.py", "src/synth/generator.py", "pageocr/engine.py", "pageocr/pipeline.py", "pageocr/protected_manifest.json",
            "wordocr/checkpoints.py", "wordocr/fullpage.py", "wordocr/diagnostics.py", "wordocr/crop_compare.py",
            "tools/evaluate.py", "tools/import_hf_parquet.py", "tools/synth_export.py", "tools/check_datasets.py",
            "tools/verify_checkpoints.py", "data/manifests/datasets.json", "configs/experiment_002a.yaml",
            "runs/experiment_002a/config.yaml", "assets/fonts/OFL_notosanstamil.txt", "dev_tools/segmentation_experiments.py",
            "dev_tools/compare_manual_crops.py", ".github/workflows/tests.yml"]
TEXT = {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".js", ".cfg", ".ini", ".cff", ""}


def publish_set() -> list:
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        sys.exit("not a Git repository (git ls-files failed)")
    return sorted(p for p in r.stdout.decode("utf-8").split("\0") if p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    files = publish_set()
    f = {k: [] for k in ("secrets", "credential_files", "weights", "over_50MB", "over_100MB", "personal_paths", "layout",
                         "missing_required", "broken_links", "emails")}
    total = 0
    for rel in files:
        p = ROOT / rel
        if not p.is_file():
            continue
        size = p.stat().st_size
        total += size
        parts = Path(rel).parts
        suf = p.suffix.lower()
        if suf in WEIGHTS:
            f["weights"].append(rel)
        if p.name.lower().startswith(".env") or p.name.lower() in CRED_NAMES or suf in CRED_SUFFIX:
            f["credential_files"].append(rel)
        if any(x in LAYOUT for x in parts[:-1]) or suf in (".pyc", ".pyo"):
            f["layout"].append(rel)
        if size > 100_000_000:
            f["over_100MB"].append(f"{rel} ({size // 1_000_000} MB)")
        elif size > 50_000_000:
            f["over_50MB"].append(f"{rel} ({size // 1_000_000} MB)")
        if suf not in TEXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, rx in SECRETS.items():
                if re.search(rx, line):
                    f["secrets"].append(f"{rel}:{i} [{name}]")
            if re.search(PERSONAL_PATH, line):
                f["personal_paths"].append(f"{rel}:{i}")
            if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", line):
                f["emails"].append(f"{rel}:{i}")
    present = set(files)
    f["missing_required"] = [r for r in REQUIRED if r not in present]
    for md in [x for x in files if "/" not in x and x.endswith(".md")]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", (ROOT / md).read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:") or target.startswith("<"):
                continue
            if not (ROOT / target).exists():
                f["broken_links"].append(f"{md} -> {target}")
    blocking = ("secrets", "credential_files", "weights", "over_100MB", "personal_paths", "layout", "missing_required", "broken_links")
    ok = not any(f[k] for k in blocking)
    res = {"files": len(files), "total_MB": round(total / 1e6, 2), "findings": {k: v for k, v in f.items()},
           "counts": {k: len(v) for k, v in f.items()}, "blocking_findings": ok is False, "passed": ok}
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"publishable file set: {len(files)} files, {total / 1e6:.1f} MB")
        for k, v in f.items():
            tag = "BLOCK" if (v and k in blocking) else ("note " if v else "ok   ")
            print(f"{tag} {k:18s} {len(v)}")
            for x in v[:15]:
                print(f"        {x}")
            if len(v) > 15:
                print(f"        ... {len(v) - 15} more")
        print("PASSED: nothing blocks publication" if ok else "NOT READY: blocking findings above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
