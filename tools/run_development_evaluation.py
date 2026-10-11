"""
The one entry point for everything that follows the human review of the development pages.

    .venv\\Scripts\\python -B tools\\run_development_evaluation.py             run everything (refused until the verified set is frozen)
    .venv\\Scripts\\python -B tools\\run_development_evaluation.py --check     say exactly what is missing; start nothing

In order:
    1  integrity of the verified set (files match SHA256SUMS, one record per queued region, protocol lock, checkpoint)
    2  E1  oracle boxes                 tools/development_e1.py
    3  E2  automatic segmentation       tools/development_e2.py   (segmentation, box matching, text through found boxes)
    4  E3  box perturbation             tools/development_e3.py
    5  aggregate, per-page and per-condition metrics; failure cases   (inside the three reports)
    6  the decision logic               tools/development_decision.py
    7  a summary report                 reports/development_evaluation_summary_<date>.md

THE GUARD CANNOT BE BYPASSED. On the real development pages this program starts nothing unless every condition of
pageocr.devguard holds: human_verified01 exists, its manifest exists, 252 of 252 regions are in it, it is frozen, the
protocol lock still matches, and the Experiment 002-A checkpoint has its frozen SHA-256. Otherwise it exits with code
3 and lists what is missing. There is no option to skip a condition.

A stage that fails is recorded with its error and the later stages still run; the summary says what is missing.
Each stage writes into its own folder under development_pages/experiments/ and never overwrites a finished run.
No training. The recognizer, the verified set and the AI annotations are not changed.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import development_decision as dec
import development_e1 as e1
import development_e2 as e2
import development_e3 as e3
import development_eval_core as core
from pageocr import devguard


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Post-verification evaluation of the development pages: E1, E2, E3, decision, reports")
    ap.add_argument("--verified-dir", default=str(devguard.VERIFIED))
    ap.add_argument("--pages-dir", default=str(core.DEV / "original"))
    ap.add_argument("--out-dir", default=None, help="fixture runs only: a folder outside the project")
    ap.add_argument("--device", default=None)
    ap.add_argument("--check", action="store_true", help="list what is missing; start nothing")
    a = ap.parse_args(argv)
    if a.check:
        st = devguard.status()
        for c in st["conditions"]:
            print(("  ok       " if c["ok"] else "  MISSING  ") + c["condition"] + ("" if c["ok"] else "\n             " + c["missing"]))
        print("\n" + devguard.explain(st))
        return 0 if st["ready"] else core.EXIT_NOT_READY
    table = core.page_table(Path(a.pages_dir))
    mode = core.resolve_mode(Path(a.verified_dir), Path(a.pages_dir), a.out_dir, table)
    guard = core.guard(mode)                                   # real mode: exits here with the list of what is missing
    records, manifest_sha = core.load_verified(Path(a.verified_dir))
    base = core.EXPERIMENTS if mode == "real" else Path(a.out_dir)
    args = ["--verified-dir", a.verified_dir, "--pages-dir", a.pages_dir] + (["--out-dir", a.out_dir] if a.out_dir else []) + (["--device", a.device] if a.device else [])
    started, stages = core.now(), []
    print(f"verified set: {len(records)} records, manifest {manifest_sha[:16]}; mode {mode}")
    for name, mod in (("E1 oracle boxes", e1), ("E2 automatic segmentation", e2), ("E3 box perturbation", e3)):
        try:
            code = mod.main(args)
            stages.append({"stage": name, "ok": code == 0, "error": "" if code == 0 else f"exit code {code}"})
        except SystemExit as ex:
            stages.append({"stage": name, "ok": False, "error": f"stopped: {ex.code}"})
        except Exception as ex:                                 # a failed stage must not hide the others
            stages.append({"stage": name, "ok": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-2000:]})
            print(f"{name} FAILED: {type(ex).__name__}: {ex}")
    d1 = core.run_dir(base, "E1_oracle_boxes", manifest_sha, e1.CONFIG)
    d2 = core.run_dir(base, "E2_automatic_segmentation", manifest_sha, e2.CONFIG)
    d3 = core.run_dir(base, "E3_box_perturbation", manifest_sha, e3.CONFIG)
    rules, rules_sha, rules_src = dec.load_rules(dec.RULES_PATH if mode == "real" else Path(a.out_dir) / "DECISION_RULES.json")
    decision = dec.decide(_load(d1 / "metrics.json"), _load(d2 / "metrics.json"), _load(d2 / "box_metrics.json"), _load(d3 / "metrics.json"), rules)
    finished = core.now()
    summary = {"mode": mode, "stamp": core.FIXTURE_STAMP if mode != "real" else "development data, tuned-on; not a held-out test set", "started": started, "finished": finished,
               "verified_manifest_sha256": manifest_sha, "verified_records": len(records), "guard": guard, "stages": stages,
               "run_folders": {"E1": str(d1), "E2": str(d2), "E3": str(d3)}, "decision_rules": {"source": rules_src, "sha256": rules_sha, "rules": rules}, "decision": decision,
               "checkpoint_sha256_after": core.sha_file(devguard.CHECKPOINT)}
    out = Path(base) / "summary" / f"{manifest_sha[:12]}_{finished[:19].replace(':', '')}"
    core.write_new(out / "summary.json", core.dump(summary))
    L = []
    A = L.append
    A(f"# Development pages — evaluation summary — {finished[:10]}\n")
    if mode != "real":
        A(f"**{core.FIXTURE_STAMP}**\n")
    A("Development data, tuned-on: not a held-out test set, not R1 data. The reference is the human-verified subset: one reviewer, who saw the AI annotation, on part of each page.\n")
    A("## 1. Stages\n\n| Stage | Result | Run folder |\n|---|---|---|")
    for s, d in zip(stages, (d1, d2, d3)):
        A(f"| {s['stage']} | {'done' if s['ok'] else 'FAILED: ' + s['error']} | `{d.as_posix()}` |")
    n = decision["numbers"]
    A("\n## 2. The three results side by side\n\n| | Value | Measured on |\n|---|---|---|")
    A(f"| E1 CER, verified boxes | {core.fmt_pct(n['E1_cer'])} | {n['E1_scope']}, {n['E1_regions']} regions |")
    A(f"| E2 CER, found boxes | {core.fmt_pct(n['E2_cer'])} | {n['E2_scope']} |")
    A(f"| E2 minus E1 | {'n/a' if n['E2_cer_excess_over_E1'] is None else format(100 * n['E2_cer_excess_over_E1'], '+.2f') + ' pp'} | |")
    A(f"| E2 box recall on verified boxes | {core.fmt_pct(n['E2_box_recall'])} | all usable verified boxes |")
    if n["E3"]:
        A(f"| E3 CER spread over margins | {100 * n['E3']['spread']:.2f} pp | common set; lowest at {n['E3']['best_margin_px']:+d} px |")
    A("\n## 3. Decision\n")
    g = decision["grades"]
    A(f"| | Grade |\n|---|---|\n| E1 | {g['E1']} |\n| E2 | {g['E2']} |\n| E3 box sensitivity | {g['E3_sensitivity']} |\n")
    A(f"**Case {decision['case']}.** {decision['statement']}\n")
    A(decision["caution"] + "\n")
    A(f"The thresholds behind 'good', 'poor' and 'strong' come from `{rules_src}` (SHA-256 `{rules_sha}`). {rules.get('status', '')}\n")
    A("## 4. Where the details are\n\nEach run folder holds `REPORT.md`, `metrics.json`, `predictions.jsonl` and `run_record.json` (configuration, hashes of the checkpoint, the verified set, the pages and the code).\n")
    text = "\n".join(L) + "\n"
    core.write_new(out / "SUMMARY.md", text)
    if mode == "real":
        rp = core.REPORTS / f"development_evaluation_summary_{finished[:10]}.md"
        if not rp.exists():
            core.write_new(rp, text)
    print(f"\nDECISION: case {decision['case']} | E1 {g['E1']} | E2 {g['E2']} | E3 sensitivity {g['E3_sensitivity']}")
    print(f"summary: {out}")
    return 0 if all(s["ok"] for s in stages) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except core.NotReady as e:
        if not isinstance(e.code, int):
            print(e.code)
        sys.exit(core.EXIT_NOT_READY)
