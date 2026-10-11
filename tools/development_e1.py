"""
E1, the oracle-box experiment: how well does the frozen recognizer read a word when it is given the right box?

    verified human word box -> crop of the ORIGINAL grayscale page -> src/preprocess.py -> frozen Experiment 002-A
    -> greedy CTC -> CER, WER, word accuracy, length ratio, confidence, clipping, failure cases

    .venv\\Scripts\\python -B tools\\development_e1.py                      the real run (refused until the verified set is frozen)
    .venv\\Scripts\\python -B tools\\development_e1.py --check              say whether the real run may start; load nothing
    ... --verified-dir D --pages-dir P --out-dir O                         a fixture run on generated pages (tests)

The configuration below is fixed before any result exists and is hashed into the run folder's name.
No training, no segmentation, no change to the checkpoint or to the verified set.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import development_eval_core as core
from pageocr import devguard

# ---- the experiment, fixed in advance ---------------------------------------------------------------------------------------
CONFIG = {
    "experiment": "E1 oracle boxes",
    "recognizer": "Experiment 002-A, frozen (runs/experiment_002a/best.pth), greedy CTC, no lexicon, no language model",
    "checkpoint_sha256": devguard.FROZEN_002A_SHA256,
    "crop_source": "original grayscale page (pageocr.imageio.load_page), no binarization, no enhancement",
    "preprocessing": "src/preprocess.py as it is (height 48, width cap 1024, paper-coloured batch padding)",
    # The annotation guide (section 3) stores tight boxes and makes the margin a crop-time parameter, with about 10 % of
    # the box height on every side as the default for normal recognition runs. That default is the primary setting here.
    "padding_primary": {"kind": "height_fraction", "value": 0.10},
    "padding_secondary": {"kind": "pixels", "value": 0},
    "batch_size": 32,
    "primary_rule": core.PRIMARY_RULE,
    "reference": "human-verified transcription, src.charset.normalize",
    # Stage 5B: the implementation version is part of the configuration, hence of the run folder name (core.run_dir).
    "implementation_version": core.IMPLEMENTATION_VERSION,
    "estimand": "D-01 Option 2: finite-population descriptive; per-page and per-source tables; leave-one-source-out range; no headline interval",
    "uncertainty": {"loso": {"unit": core.CLUSTER_UNIT},
                    "source_cluster_interval": {"min_clusters": core.MIN_CLUSTERS_FOR_CI, "min_clusters_status": core.MIN_CLUSTERS_STATUS,
                                                "seed": core.BOOTSTRAP_SEED, "replicates": core.BOOTSTRAP_REPS, "label": "indicative only"},
                    "region_bootstrap_diagnostic": {"seed": core.BOOTSTRAP_SEED, "replicates": core.BOOTSTRAP_REPS, "unit": "region", "label": "diagnostic only"}},
}


def run_setting(engine, records, pages, rule, batch_size):
    """Rows for one padding rule: one per verified record, scored or with the reason it could not be."""
    rows, todo = [], []
    for r in records:
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        row = {"verification_id": r["verification_id"], "region_id": r["region_id"], "page_id": r["page_id"], "source_id": r["source_id"],
               "verified_box": list(box), "groups": core.groups_of(r), "final_state": r["final_state"], "usable": bool(r["usable_as_verified_reference"])}
        if not row["usable"]:
            rows.append({**row, "scored": False, "reason": r.get("not_usable_because") or "not usable as verified reference"})
            continue
        c = core.make_crop(pages[r["page_id"]], box, core.pad_for(rule, box))
        row.update(box_used=list(c["box_used"]) if c["box_used"] else None, pad_px=c["pad_px"], clipped_edges=c["clipped_edges"], ref=core.reference_text(r))
        if not c["valid"]:
            rows.append({**row, "scored": False, "reason": c["reason"]})
            continue
        todo.append((len(rows), c["crop"]))
        rows.append({**row, "scored": True, "reason": ""})
    recs = core.recognise(engine, [c for _, c in todo], batch_size)
    for (i, _), rec in zip(todo, recs):
        hyp = core.cs.normalize(rec.text)
        rows[i].update(hyp=hyp, correct=hyp == rows[i]["ref"], edits=core.editdistance.eval(rows[i]["ref"], hyp), confidence=rec.confidence,
                       width_clipped=rec.clipped, model_width_px=rec.width_px, timesteps=rec.timesteps)
    return rows


def report(mode, record, metrics, rows, secondary, out_dir, acc=None):
    L, A = [], None
    A = L.append
    p = metrics.get("primary")
    A(f"# Development pages — E1 oracle boxes — {record['finished'][:10]}\n")
    if mode != "real":
        A(f"**{core.FIXTURE_STAMP}**\n")
    A("**What this measures:** how well the frozen Experiment 002-A recognizer reads a word of the development pages when it is given the human-verified box. "
      "No segmentation is involved. Development data, tuned-on: not a held-out test set, not R1 data.\n")
    A(f"**Reference:** the human-verified subset, {record['verified_records']} regions reviewed by one person who saw the AI annotation while judging. "
      "It is not an independent double reading.\n")
    A("## 1. Result\n")
    if p:
        A(f"On the primary set: **{core.uncertainty_sentence(p)}**\n")
        A(f"WER {core.fmt_pct(p['wer'])}, word accuracy {core.fmt_pct(p['word_acc'])}, length ratio {p['len_ratio']:.3f}.\n")
    else:
        A("No region is in the primary set.\n")
    A(f"**Estimand:** {record['estimand']}\n")
    A(core.PRIMARY_RULE + "\n")
    A("The queue is not a random sample. Part B is the seeded stratified sample of regions that were not flagged as doubtful; part A holds the regions chosen because they were doubtful. "
      "Part B is the figure that speaks for ordinary regions.\n")
    A(core.METRICS_HEADER)
    for g in ("primary", "primary|part:B", "primary|part:A", "all_usable"):
        if g in metrics:
            A(core.metrics_row(g, metrics[g]))
    A("\nConfidence is uncalibrated: a flagging heuristic, not a probability of being correct.\n")
    A("### By source document (primary set)\n")
    A("The source documents, not the regions, are the independent units: regions of one book share its type, paper and scan.\n")
    A(core.source_section(metrics, record["source_mapping"]))
    A("## 2. Coverage\n")
    A("| | Regions |\n|---|---|")
    n = len(rows)
    A(f"| In the verified set | {n} |")
    A(f"| Usable as verified reference | {sum(r['usable'] for r in rows)} |")
    A(f"| Scored | {sum(r['scored'] for r in rows)} |")
    A(f"| Not usable (ambiguous, unreadable, or unit split / merge) | {sum(not r['usable'] for r in rows)} |")
    A(f"| Usable but not scorable (crop refused) | {sum(r['usable'] and not r['scored'] for r in rows)} |")
    A(f"| Crop margin cut by the page edge | {sum(bool(r.get('clipped_edges')) for r in rows)} |\n")
    for title, prefix in (("3. By page", "page:"), ("4. By page condition", "condition:"), ("5. By script (all usable regions)", "script:"),
                          ("6. By region type", "region_type:"), ("7. By other properties", ("final_state:", "ai_flagged_uncertain:", "fragment:", "legibility:", "unit:", "encodable:"))):
        keys = [g for g in metrics if g.startswith(prefix)]
        if keys:
            A(f"## {title}\n")
            if prefix == "condition:":
                A("A page can carry several tags; a region counts under each tag of its page.\n")
            A(core.METRICS_HEADER)
            for g in keys:
                A(core.metrics_row(g, metrics[g]))
            A("")
    A("## 8. Tight crops (secondary setting)\n")
    A("The same regions cut with no margin at all. This is also the 0-pixel point of the box-perturbation experiment E3.\n")
    A(core.METRICS_HEADER)
    for g in ("primary", "primary|part:B", "primary|part:A"):
        if g in secondary:
            A(core.metrics_row(g, secondary[g]))
    wrong = sorted((r for r in rows if r["scored"] and not r["correct"] and "primary" in r["groups"]), key=lambda r: (-r["edits"], r["verification_id"]))
    A(f"\n## 9. Failure cases\n\n{len(wrong)} primary regions were read wrongly. The 25 with the most character edits (all are in `failures.jsonl`):\n")
    A("| Region | Reference | Prediction | Edits | Confidence |\n|---|---|---|---|---|")
    for r in wrong[:25]:
        A(f"| `{r['region_id']}` | {r['ref']} | {r['hyp'] or '(empty)'} | {r['edits']} | {r['confidence']:.3f} |")
    A("\n## 10. What was run\n")
    A("| | |\n|---|---|")
    A(f"| Checkpoint | `runs/experiment_002a/best.pth`, SHA-256 `{record['checkpoint_sha256_before']}`, unchanged after the run: {record['checkpoint_sha256_before'] == record['checkpoint_sha256_after']} |")
    A(f"| Verified set | `SHA256SUMS.txt` hash `{record['verified_manifest_sha256']}` |")
    A(f"| Crop margin | primary: {CONFIG['padding_primary']['value'] * 100:.0f} % of the box height on every side, kept inside the page; secondary: none |")
    A(f"| Device | {record['environment'].get('engine', {}).get('device')} |")
    A(f"| Run folder | `{out_dir.as_posix()}` |")
    A(f"| Configuration hash | `{record['config_sha256']}` |\n")
    A("## 11. Limits\n")
    A("- One reviewer, who saw the AI annotation. Agreement between AI text and human text may be higher than an independent reading would give, and E1 inherits that.\n"
      "- The primary set mixes doubtful regions (part A) with a sample of ordinary ones (part B); read the two rows separately.\n"
      "- These pages were used to build and tune the page tools. No figure here is a held-out test result.\n"
      "- No lexicon or language model is used; the prediction is the raw greedy output.\n")
    if acc is not None:
        A("## 12. Sample accounting\n")
        A("Where every verified record went, in each crop setting. No record is left out of this table.\n")
        A(core.accounting_section(acc))
    A("## 13. Provenance and sample design\n")
    A(core.provenance_section(record))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="E1 oracle-box experiment on the development pages")
    ap.add_argument("--verified-dir", default=str(devguard.VERIFIED))
    ap.add_argument("--pages-dir", default=str(core.DEV / "original"))
    ap.add_argument("--out-dir", default=None, help="fixture runs only: a folder outside the project")
    ap.add_argument("--device", default=None, help="cpu or cuda; default as the existing tools: cuda if available")
    ap.add_argument("--check", action="store_true", help="report whether the real run may start; load no model")
    a = ap.parse_args(argv)
    if a.check:
        st = devguard.status()
        print(devguard.explain(st))
        return 0 if st["ready"] else core.EXIT_NOT_READY
    table = core.page_table(Path(a.pages_dir))
    mode = core.resolve_mode(Path(a.verified_dir), Path(a.pages_dir), a.out_dir, table)
    guard = core.guard(mode)                                    # real mode: exits here, before any model is loaded, unless everything holds
    records, manifest_sha = core.load_verified(Path(a.verified_dir))
    records = core.attach_sources(records, core.source_table(Path(a.pages_dir), {r["page_id"] for r in records}))   # refuses before any model is loaded
    out = core.run_dir(core.EXPERIMENTS if mode == "real" else a.out_dir, "E1_oracle_boxes", manifest_sha, CONFIG)
    if out.exists():
        print(f"This run exists already and is not repeated or overwritten: {out}")
        return 0
    pages = core.load_pages(table, {r["page_id"] for r in records})
    for r in records:
        if r.get("master_sha256") and r["master_sha256"] != pages[r["page_id"]].sha256:
            raise core.NotReady(f"Refused: {r['region_id']} was verified on a different image of {r['page_id']}.")
    started, ck_before = core.now(), core.sha_file(devguard.CHECKPOINT)
    engine = core.load_engine(a.device)
    rows = run_setting(engine, records, pages, CONFIG["padding_primary"], CONFIG["batch_size"])
    rows0 = run_setting(engine, records, pages, CONFIG["padding_secondary"], CONFIG["batch_size"])
    metrics, secondary = core.score([r for r in rows if r["scored"]]), core.score([r for r in rows0 if r["scored"]])
    acc = core.accounting(CONFIG["experiment"], records, {"primary_padding": rows, "tight": rows0}, pages)
    core.check_denominator(acc["settings"]["primary_padding"], metrics.get("primary"))
    core.check_denominator(acc["settings"]["tight"], secondary.get("primary"))
    prov = core.provenance(mode, records, Path(a.pages_dir), Path(a.verified_dir), a.out_dir)
    record = {"experiment": CONFIG["experiment"], "mode": mode, "stamp": core.FIXTURE_STAMP if mode != "real" else "development data, tuned-on; not a held-out test set",
              "started": started, "finished": core.now(), "config": CONFIG, "config_sha256": core.sha_json(CONFIG), "verified_manifest_sha256": manifest_sha,
              "verified_records": len(records), "page_sha256": {k: v.sha256 for k, v in sorted(pages.items())},
              "checkpoint_sha256_before": ck_before, "checkpoint_sha256_after": core.sha_file(devguard.CHECKPOINT), "guard": guard,
              "code_sha256": core.code_hashes(), "environment": core.environment(engine), **prov}
    assert record["checkpoint_sha256_before"] == record["checkpoint_sha256_after"] == devguard.FROZEN_002A_SHA256
    core.write_new(out / "run_record.json", core.dump(record))
    core.write_new(out / "accounting.json", core.dump(acc))
    core.write_new(out / "predictions.jsonl", core.lines(rows))
    core.write_new(out / "predictions_tight.jsonl", core.lines(rows0))
    core.write_new(out / "metrics.json", core.dump({"primary_padding": metrics, "tight": secondary, "design": core.design_block(prov),
                                                    "by_source": {"primary_padding": core.by_source(metrics, prov["source_mapping"]),
                                                                  "tight": core.by_source(secondary, prov["source_mapping"])}}))
    core.write_new(out / "failures.jsonl", core.lines(sorted((r for r in rows if r["scored"] and not r["correct"]), key=lambda r: (-r["edits"], r["verification_id"]))))
    text = report(mode, record, metrics, rows, secondary, out, acc)
    core.write_new(out / "REPORT.md", text)
    if mode == "real":
        rp = core.REPORTS / f"development_E1_oracle_boxes_{record['finished'][:10]}.md"
        if not rp.exists():
            core.write_new(rp, text)
    p = metrics.get("primary", {})
    print(f"E1 done ({mode}): {sum(r['scored'] for r in rows)} regions scored; primary CER {core.fmt_pct(p.get('cer'))}, word accuracy {core.fmt_pct(p.get('word_acc'))}")
    print(f"run folder: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except core.NotReady as e:
        if not isinstance(e.code, int):
            print(e.code)
        sys.exit(core.EXIT_NOT_READY)
