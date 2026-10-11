"""Pre-training assertion (R1 README v2.2.1 §9 rule 11, reverse leakage). Read-only: it writes nothing.

    python -B tools\\pretrain_check.py --config configs\\experiment_002b.yaml

Before any training run:
  (a) the synthetic vocabulary file's SHA-256 equals the frozen value;
  (b) no file under data/ has a SHA-256 equal to any R1 annotation or metadata file - extended here to every file of the
      R1 workspace (pages, crops, synthetic twins, annotations, metadata);
  (c) no training configuration or corpus manifest references real_docs/;
and, for the given config: every path it reads lies under data/ or assets/ and exists, and no R1-named file is under
data/. Exit 0 = every assertion holds; 1 = at least one fails. The result is printed as JSON.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
VOCAB_SHA256 = "3f19ac2db418d3061162acfd978463b3f3a0c21652e4968c8383fa2e24a2241a"       # README §9 rule 11 (a)
VOCAB_REL = "data/text_corpus/tamil_correction_vocabulary.json"
FORBIDDEN = ("real_docs", "r1_pilot", "synthetic_twins")
ALLOWED_ROOTS = ("data", "assets")
TEXT_SUFFIXES = (".jsonl", ".json", ".yaml", ".yml", ".csv", ".txt")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def walk(folder):
    for base, _dirs, files in os.walk(folder):
        for name in files:
            yield Path(base) / name


def config_paths(cfg):
    """Every file or folder a training configuration reads: sources (path, vocab, fonts) and validation sets."""
    out = []
    for s in cfg.get("sources", []):
        if s.get("type") == "manifest":
            out.append(s["path"])
        else:
            out += [s.get("vocab")] if s.get("vocab") else []
            out += list(s.get("fonts", ["assets/fonts"]))
    out += [v["path"] for v in cfg.get("val", [])]
    return out


def check(repo, config, vocab_sha256=None):
    """(problems, facts) for the repository `repo` and the training configuration `config` (path relative to repo)."""
    repo = Path(repo)
    data, r1 = repo / "data", repo / "real_docs"
    P, facts = [], {}
    vocab_sha256 = vocab_sha256 or VOCAB_SHA256
    # (a) vocabulary
    vocab = repo / VOCAB_REL
    digest = sha_file(vocab) if vocab.is_file() else None
    facts["vocabulary_sha256"] = digest
    if digest != vocab_sha256:
        P.append(f"(a) {VOCAB_REL}: SHA-256 {digest} is not the frozen {vocab_sha256}")
    # (b) no file under data/ equals an R1 file (same size first, then SHA-256), and no R1-named file under data/
    r1_files = [p for p in walk(r1)] if r1.is_dir() else []
    by_size = {}
    for p in r1_files:
        if p.stat().st_size > 0:
            by_size.setdefault(p.stat().st_size, []).append(p)
    r1_hash, n_data, n_hashed = {}, 0, 0
    for p in walk(data):
        n_data += 1
        low = p.name.lower()
        if low.startswith("r1-") or "r1_pilot" in p.as_posix().lower() or "twin" in low:
            P.append(f"(b) R1-named file under data/: {p.relative_to(repo).as_posix()}")
        same = by_size.get(p.stat().st_size)
        if same:
            n_hashed += 1
            d = sha_file(p)
            for q in same:
                if q not in r1_hash:
                    r1_hash[q] = sha_file(q)
                if r1_hash[q] == d:
                    P.append(f"(b) {p.relative_to(repo).as_posix()} is byte-identical to {q.relative_to(repo).as_posix()}")
    facts.update(r1_workspace_files=len(r1_files), files_under_data=n_data, data_files_hashed_after_size_match=n_hashed)
    # (c) no training configuration or corpus manifest references real_docs/
    scanned = 0
    texts = [p for p in walk(repo / "configs")] + [p for p in walk(data) if p.suffix.lower() in TEXT_SUFFIXES]
    for p in texts:
        scanned += 1
        with open(p, "rb") as f:
            tail = b""
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                chunk = (tail + block).lower()
                hit = [w for w in FORBIDDEN if w.encode() in chunk]
                if hit:
                    P.append(f"(c) {p.relative_to(repo).as_posix()} references {hit}")
                    break
                tail = block[-32:]
    facts["configs_and_manifests_scanned"] = scanned
    # the given configuration
    cpath = repo / config
    if not cpath.is_file():
        P.append(f"config {config} does not exist")
        return P, facts
    cfg = yaml.safe_load(cpath.read_text(encoding="utf-8"))
    facts["config"] = {"file": Path(config).as_posix(), "sha256": sha_file(cpath), "experiment": cfg.get("experiment")}
    read = config_paths(cfg)
    facts["config_reads"] = read
    for rel in read:
        target = (repo / rel).resolve()
        inside = [repo.joinpath(a).resolve() for a in ALLOWED_ROOTS]
        if not any(a == target or a in target.parents for a in inside):
            P.append(f"config reads {rel}, which is not under data/ or assets/")
        elif any(w in Path(rel).as_posix().lower() for w in FORBIDDEN):
            P.append(f"config reads {rel}, an R1 path")
        elif not target.exists():
            P.append(f"config reads {rel}, which does not exist")
    used_vocab = [s.get("vocab") for s in cfg.get("sources", []) if s.get("type") == "synth"]
    if any(v != VOCAB_REL for v in used_vocab):
        P.append(f"a synthetic source uses a vocabulary other than the frozen {VOCAB_REL}: {used_vocab}")
    return P, facts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="training configuration, relative to the repository")
    a = ap.parse_args(argv)
    problems, facts = check(ROOT, a.config)
    print(json.dumps({"schema": "pretrain_r1_leakage_check_v1", "rule": "R1 README v2.2.1 §9 rule 11", "passed": not problems,
                      "problems": problems, "facts": facts}, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
