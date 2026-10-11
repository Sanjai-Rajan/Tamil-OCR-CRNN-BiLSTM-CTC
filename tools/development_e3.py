"""
E3, the box-perturbation experiment: how much does the recognizer's reading depend on the exact box?

Every verified box is given a margin of k pixels on all four edges, for k = -4, -2, 0, +2, +4, +8 (the values fixed by
the annotation guide; k = 0 is the tight box, negative k cuts into the word). The frozen Experiment 002-A recognizer
reads each crop of the ORIGINAL grayscale page. A box that would leave the page is kept inside it and the clipped
edges are recorded; a box that shrinks below 2 x 2 pixels is rejected and counted.

    .venv\\Scripts\\python -B tools\\development_e3.py               the real run (refused until the verified set is frozen)
    .venv\\Scripts\\python -B tools\\development_e3.py --check       say whether the real run may start
    ... --verified-dir D --pages-dir P --out-dir O                   a fixture run on generated pages (tests)

The comparison between margins is made on the regions that could be read at EVERY margin (the common set), so that
each margin is judged on the same words. No training, no segmentation, no change to the checkpoint or the verified set.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import development_e1 as e1
import development_eval_core as core
from pageocr import devguard

MARGINS = (-4, -2, 0, 2, 4, 8)
CONFIG = {
    "experiment": "E3 box perturbation",
    "margins_px": list(MARGINS),
    "margin_rule": "the same number of pixels on all four edges of the verified tight box; negative values shrink it; kept inside the page; "
                   "rejected if smaller than 2 x 2 pixels",
    "recognizer": e1.CONFIG["recognizer"], "checkpoint_sha256": devguard.FROZEN_002A_SHA256, "crop_source": e1.CONFIG["crop_source"],
    "preprocessing": e1.CONFIG["preprocessing"], "batch_size": 32, "primary_rule": core.PRIMARY_RULE, "reference": e1.CONFIG["reference"],
    "comparison_set": "primary regions scored at every margin",
    "uncertainty": e1.CONFIG["uncertainty"], "implementation_version": core.IMPLEMENTATION_VERSION, "estimand": e1.CONFIG["estimand"],
}


def report(mode, record, per_k, common, per_page, n_common, out_dir, acc=None):
    L = []
    A = L.append
    A(f"# Development pages — E3 box perturbation — {record['finished'][:10]}\n")
    if mode != "real":
        A(f"**{core.FIXTURE_STAMP}**\n")
    A("**What this measures:** how the frozen Experiment 002-A recognizer's reading of a word changes when its human-verified box is made larger or smaller by a few pixels. "
      "It says how exact a word box has to be. Development data, tuned-on: not a held-out test set.\n")
    A(f"## 1. Result on the common set ({n_common} primary regions read at every margin)\n")
    A("| Margin (px) | CER | WER | Word accuracy | Mean confidence | Width-clipped | CER change vs 0 px | Confidence change vs 0 px |\n|---|---|---|---|---|---|---|---|")
    z = common.get(0)
    for k in MARGINS:
        m = common.get(k)
        if not m:
            continue
        dc = "" if not z else f"{100 * (m['cer'] - z['cer']):+.2f} pp"
        df = "" if not z or m["mean_confidence"] is None or z["mean_confidence"] is None else f"{m['mean_confidence'] - z['mean_confidence']:+.3f}"
        A(f"| {k:+d} | {core.fmt_pct(m['cer'])} | {core.fmt_pct(m['wer'])} | {core.fmt_pct(m['word_acc'])} | {'n/a' if m['mean_confidence'] is None else format(m['mean_confidence'], '.3f')} | {core.fmt_pct(m['width_clipped_rate'])} | {dc} | {df} |")
    if common:
        best = min(common, key=lambda k: (common[k]["cer"], abs(k)))
        worst = max(common, key=lambda k: (common[k]["cer"], abs(k)))
        A(f"\nLowest CER at {best:+d} px ({core.fmt_pct(common[best]['cer'])}); highest at {worst:+d} px ({core.fmt_pct(common[worst]['cer'])}); spread {100 * (common[worst]['cer'] - common[best]['cer']):.2f} percentage points.\n")
    A("Confidence is uncalibrated: a flagging heuristic, not a probability of being correct.\n")
    if common:
        A("CER of the common set with one source document left out, per margin (the range over the sources; recomputed from summed edits, never averaged):\n")
        A("| Margin (px) | Sources | CER | Range with one source left out |\n|---|---|---|---|")
        for k in MARGINS:
            m = common.get(k)
            if m:
                lo = m.get("loso") or {}
                A(f"| {k:+d} | {m['n_clusters']} | {core.fmt_pct(m['cer'])} | "
                  + (f"{core.fmt_pct(lo['cer_min'])} to {core.fmt_pct(lo['cer_max'])}" if "cer_min" in lo else lo.get("refused", "")) + " |")
        if 0 in common:
            A(f"\nAt 0 px: {core.uncertainty_sentence(common[0])}\n")
    A("## 2. Every margin on the regions it could read\n")
    A("| Margin (px) | Scored | Rejected (too small) | Crop refused | Clipped by the page edge | Failure rate (rejected, refused or empty) | CER | WER |\n|---|---|---|---|---|---|---|---|")
    for k in MARGINS:
        s = per_k[k]
        A(f"| {k:+d} | {s['scored']} | {s['rejected_too_small']} | {s['crop_refused']} | {s['clipped_by_page_edge']} | {core.fmt_pct(s['failure_rate'])} | {core.fmt_pct(s['metrics'].get('cer'))} | {core.fmt_pct(s['metrics'].get('wer'))} |")
    A("\n## 3. By page (common set, CER)\n")
    A("| Page | Regions | " + " | ".join(f"{k:+d} px" for k in MARGINS) + " |\n|---|---|" + "---|" * len(MARGINS))
    for page, d in sorted(per_page.items()):
        A(f"| {page} | {d['n']} | " + " | ".join(core.fmt_pct(d['cer'].get(k)) for k in MARGINS) + " |")
    A("\n## 4. What was run\n")
    A("| | |\n|---|---|")
    A(f"| Checkpoint | `runs/experiment_002a/best.pth`, SHA-256 `{record['checkpoint_sha256_before']}`, unchanged after the run: {record['checkpoint_sha256_before'] == record['checkpoint_sha256_after']} |")
    A(f"| Verified set | `SHA256SUMS.txt` hash `{record['verified_manifest_sha256']}` |")
    A(f"| Margins | {', '.join(f'{k:+d}' for k in MARGINS)} px on all four edges |")
    A(f"| Run folder | `{out_dir.as_posix()}` |")
    A(f"| Configuration hash | `{record['config_sha256']}` |\n")
    A("## 5. Limits\n")
    A("- The margins are in pixels of pages scanned at different resolutions, so the same number of pixels is a different share of a letter on different pages. Section 3 shows the pages apart.\n"
      "- A uniform margin on all four edges is one kind of box error. A box that is shifted, or wrong on one side only, is not tested here.\n"
      "- The reference comes from one reviewer who saw the AI annotation.\n- Development pages: no figure here is a held-out test result.\n")
    if acc is not None:
        A("## 6. Sample accounting\n")
        A(f"Where every verified record went, at each margin. The common set of section 1 is the {acc['common_set_size']} records whose disposition is "
          "one of the three scored ones at every margin (listed in `accounting.json`).\n")
        A(core.accounting_section(acc))
    A("## 7. Provenance and sample design\n")
    A(core.provenance_section(record))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="E3 box-perturbation experiment on the development pages")
    ap.add_argument("--verified-dir", default=str(devguard.VERIFIED))
    ap.add_argument("--pages-dir", default=str(core.DEV / "original"))
    ap.add_argument("--out-dir", default=None, help="fixture runs only: a folder outside the project")
    ap.add_argument("--device", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        st = devguard.status()
        print(devguard.explain(st))
        return 0 if st["ready"] else core.EXIT_NOT_READY
    table = core.page_table(Path(a.pages_dir))
    mode = core.resolve_mode(Path(a.verified_dir), Path(a.pages_dir), a.out_dir, table)
    guard = core.guard(mode)
    records, manifest_sha = core.load_verified(Path(a.verified_dir))
    records = core.attach_sources(records, core.source_table(Path(a.pages_dir), {r["page_id"] for r in records}))   # refuses before any model is loaded
    out = core.run_dir(core.EXPERIMENTS if mode == "real" else a.out_dir, "E3_box_perturbation", manifest_sha, CONFIG)
    if out.exists():
        print(f"This run exists already and is not repeated or overwritten: {out}")
        return 0
    pages = core.load_pages(table, {r["page_id"] for r in records})
    started, ck_before = core.now(), core.sha_file(devguard.CHECKPOINT)
    engine = core.load_engine(a.device)
    rows_by_k, per_k = {}, {}
    for k in MARGINS:
        rows = e1.run_setting(engine, records, pages, {"kind": "pixels", "value": k}, CONFIG["batch_size"])
        for r in rows:
            r["margin_px"] = k
        rows_by_k[k] = rows
        prim = [r for r in rows if "primary" in r["groups"]]
        scored = [r for r in prim if r["scored"]]
        too_small = sum((not r["scored"]) and "smaller than 2 x 2" in r["reason"] for r in prim)
        refused = sum((not r["scored"]) and "smaller than 2 x 2" not in r["reason"] for r in prim)
        empty = sum(r["hyp"] == "" for r in scored)
        per_k[k] = {"primary_regions": len(prim), "scored": len(scored), "rejected_too_small": too_small, "crop_refused": refused,
                    "clipped_by_page_edge": sum(bool(r.get("clipped_edges")) for r in prim),
                    "failure_rate": (too_small + refused + empty) / max(1, len(prim)), "metrics": core.tally(scored) if scored else {}}
    ok_ids = set.intersection(*[{r["verification_id"] for r in rows_by_k[k] if r["scored"] and "primary" in r["groups"]} for k in MARGINS]) if records else set()
    common, per_page, common_by_source = {}, {}, {}
    for k in MARGINS:
        sel = [r for r in rows_by_k[k] if r["verification_id"] in ok_ids]
        if sel:
            common[k] = core.tally(sel)
            common[k].update(core.uncertainty(sel))
            common_by_source[k] = core.score(sel)
            for r in sel:
                d = per_page.setdefault(r["page_id"], {"rows": {}})
                d["rows"].setdefault(k, []).append(r)
    per_page = {p: {"n": len(d["rows"][0]) if 0 in d["rows"] else 0, "cer": {k: core.tally(v)["cer"] for k, v in d["rows"].items()}} for p, d in per_page.items()}
    acc = core.accounting(CONFIG["experiment"], records, {f"{k:+d}px": rows_by_k[k] for k in MARGINS}, pages,
                          {"common_set_size": len(ok_ids), "common_set": sorted(ok_ids)})
    for k in MARGINS:
        core.check_denominator(acc["settings"][f"{k:+d}px"], {"n": per_k[k]["scored"]})
    prov = core.provenance(mode, records, Path(a.pages_dir), Path(a.verified_dir), a.out_dir)
    record = {"experiment": CONFIG["experiment"], "mode": mode, "stamp": core.FIXTURE_STAMP if mode != "real" else "development data, tuned-on; not a held-out test set",
              "started": started, "finished": core.now(), "config": CONFIG, "config_sha256": core.sha_json(CONFIG), "verified_manifest_sha256": manifest_sha,
              "verified_records": len(records), "common_set": len(ok_ids), "page_sha256": {k: v.sha256 for k, v in sorted(pages.items())},
              "checkpoint_sha256_before": ck_before, "checkpoint_sha256_after": core.sha_file(devguard.CHECKPOINT), "guard": guard,
              "code_sha256": core.code_hashes(), "environment": core.environment(engine), **prov}
    assert record["checkpoint_sha256_before"] == record["checkpoint_sha256_after"] == devguard.FROZEN_002A_SHA256
    core.write_new(out / "run_record.json", core.dump(record))
    core.write_new(out / "accounting.json", core.dump(acc))
    core.write_new(out / "predictions.jsonl", core.lines(r for k in MARGINS for r in rows_by_k[k]))
    core.write_new(out / "metrics.json", core.dump({"common_set": {str(k): v for k, v in common.items()}, "per_margin": {str(k): v for k, v in per_k.items()},
                                                    "per_page_common_set": {p: {"n": d["n"], "cer": {str(k): v for k, v in d["cer"].items()}} for p, d in per_page.items()},
                                                    "design": core.design_block(prov),
                                                    "by_source_common_set": {str(k): core.by_source(v, prov["source_mapping"]) for k, v in common_by_source.items()}}))
    text = report(mode, record, per_k, common, per_page, len(ok_ids), out, acc)
    core.write_new(out / "REPORT.md", text)
    if mode == "real":
        rp = core.REPORTS / f"development_E3_box_perturbation_{record['finished'][:10]}.md"
        if not rp.exists():
            core.write_new(rp, text)
    print(f"E3 done ({mode}): common set {len(ok_ids)} regions; CER by margin: " + ", ".join(f"{k:+d}px {core.fmt_pct(common[k]['cer'])}" for k in MARGINS if k in common))
    print(f"run folder: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except core.NotReady as e:
        if not isinstance(e.code, int):
            print(e.code)
        sys.exit(core.EXIT_NOT_READY)
