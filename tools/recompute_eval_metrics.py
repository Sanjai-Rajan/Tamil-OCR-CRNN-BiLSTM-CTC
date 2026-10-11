"""
READ-ONLY reproducibility audit of the stored evaluation outputs: recomputes every metric from the per-sample files,
without loading a model, and checks where each reference label came from.

    .venv\\Scripts\\python -B tools\\recompute_eval_metrics.py                     audit reports/*_samples.jsonl, print a summary
    .venv\\Scripts\\python -B tools\\recompute_eval_metrics.py --out FILE.json     also write the full result (refuses to overwrite)

For every <name>_samples.jsonl written by tools/evaluate.py --samples:
  1. internal consistency: one row per manifest index; the number of rows equals _meta.n_scored; each row's stored
     edits, ref_len, hyp_len and correct agree with its ref and hyp (editdistance, as src/metrics.py);
  2. label provenance: the manifest named in _meta exists; row manifest_index is that manifest's 0-based line; the
     line's image is the row's image; cs.normalize(label) is the row's ref; the style filter of _meta is respected;
  3. metrics: n, CER (sum of edits / sum of reference characters), WER, word accuracy and length ratio recomputed
     exactly as src/metrics.Scoreboard does, and compared with <name>_summary.json["all"] when that file exists.
It reports, it does not judge the model: these are the frozen evaluations re-derived from their own records.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import editdistance                                              # noqa: E402

from src import charset as cs                                    # noqa: E402

TOL = 1e-12


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(rows: list) -> dict:
    edits = sum(editdistance.eval(r["ref"], r["hyp"]) for r in rows)
    chars = sum(len(r["ref"]) for r in rows)
    wrong = sum(r["ref"] != r["hyp"] for r in rows)
    pred = sum(len(r["hyp"]) for r in rows)
    n = len(rows)
    return {"n": n, "edits": edits, "ref_chars": chars, "cer": edits / max(1, chars), "wer": wrong / max(1, n),
            "word_acc": 1 - wrong / max(1, n), "len_ratio": pred / max(1, chars)}


def audit_file(path: Path, root: Path = ROOT) -> dict:
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta, rows = (lines[0].get("_meta"), lines[1:]) if lines and "_meta" in lines[0] else (None, lines)
    problems = []
    if meta is None:
        problems.append("no _meta line")
        meta = {}
    idx = [r.get("manifest_index") for r in rows]
    if len(set(idx)) != len(idx):
        problems.append("a manifest_index occurs more than once")
    if meta.get("n_scored") is not None and meta["n_scored"] != len(rows):
        problems.append(f"_meta.n_scored {meta['n_scored']} != {len(rows)} rows")
    bad_rows = 0
    for r in rows:
        ok = (r.get("edits") == editdistance.eval(r["ref"], r["hyp"]) and r.get("ref_len") == len(r["ref"])
              and r.get("hyp_len") == len(r["hyp"]) and r.get("correct") == (r["ref"] == r["hyp"]))
        bad_rows += not ok
    if bad_rows:
        problems.append(f"{bad_rows} row(s) whose stored edits/lengths/correct disagree with ref and hyp")
    # label provenance
    prov = {"manifest": meta.get("manifest"), "manifest_sha256": None, "rows_checked": 0, "mismatches": 0}
    mpath = (root / meta["manifest"]) if meta.get("manifest") else None
    if mpath is None or not mpath.is_file():
        problems.append(f"manifest {meta.get('manifest')!r} not found; label provenance not checked")
    else:
        prov["manifest_sha256"] = sha_file(mpath)
        mlines = mpath.read_text(encoding="utf-8").splitlines()
        style = meta.get("style")
        for r in rows:
            li = r["manifest_index"]
            if not isinstance(li, int) or li < 0 or li >= len(mlines) or not mlines[li].strip():
                prov["mismatches"] += 1
                continue
            e = json.loads(mlines[li])
            good = (e.get("image") == r.get("image") and cs.normalize(e["label"]) == r["ref"]
                    and (not style or e.get("style") == style))
            prov["mismatches"] += not good
            prov["rows_checked"] += 1
        if prov["mismatches"]:
            problems.append(f"{prov['mismatches']} row(s) whose ref/image/style disagree with the manifest line")
    m = metrics(rows)
    out = {"file": path.name, "checkpoint": meta.get("checkpoint"), "epoch": meta.get("epoch"), "style": meta.get("style"),
           "recomputed": m, "provenance": prov, "summary_file": None, "summary_match": None}
    sp = path.with_name(path.name.replace("_samples.jsonl", "_summary.json"))
    if sp.is_file():
        s = json.loads(sp.read_text(encoding="utf-8"))["all"]
        out["summary_file"] = sp.name
        diffs = {k: (s.get(k), m[k]) for k in ("n", "cer", "wer", "word_acc", "len_ratio")
                 if s.get(k) is None or abs(s[k] - m[k]) > TOL}
        out["summary_match"] = not diffs
        if diffs:
            problems.append(f"recomputed metrics differ from {sp.name}: {diffs}")
    out["problems"] = problems
    out["ok"] = not problems
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recompute stored evaluation metrics (read-only)")
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--root", default=str(ROOT), help="base for the manifest paths recorded in _meta")
    ap.add_argument("--out", default=None, help="write the full result as JSON (never overwrites)")
    a = ap.parse_args(argv)
    files = sorted(Path(a.reports_dir).glob("*_samples.jsonl"))
    results = [audit_file(p, Path(a.root)) for p in files]
    for r in results:
        m = r["recomputed"]
        print(f"{'OK ' if r['ok'] else 'BAD'} {r['file']:52s} n={m['n']:6d} CER={100 * m['cer']:.4f}% WER={100 * m['wer']:.4f}% "
              f"summary={'match' if r['summary_match'] else ('none' if r['summary_match'] is None else 'DIFFERS')} "
              f"labels={r['provenance']['rows_checked'] - r['provenance']['mismatches']}/{r['provenance']['rows_checked']}")
        for p in r["problems"]:
            print("      ", p)
    doc = {"tool": "tools/recompute_eval_metrics.py", "files": len(results), "all_ok": all(r["ok"] for r in results), "results": results}
    if a.out:
        with open(a.out, "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{sum(r['ok'] for r in results)} of {len(results)} files consistent")
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
