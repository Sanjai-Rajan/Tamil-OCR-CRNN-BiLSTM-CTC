"""
E2, the automatic-segmentation experiment: what happens when the word boxes are found by the program, not given?

    page -> pageocr.pipeline.segment (classical layout analysis, lines, words, reading order)
         -> detected boxes compared with the human-verified boxes (pageocr.geometry)
         -> the detected words that lie on verified regions are read by the frozen Experiment 002-A recognizer
         -> the same text metrics as E1, so that E2 and E1 differ only in where the boxes come from

    .venv\\Scripts\\python -B tools\\development_e2.py               the real run (refused until the verified set is frozen)
    .venv\\Scripts\\python -B tools\\development_e2.py --check       say whether the real run may start
    ... --verified-dir D --pages-dir P --out-dir O                   a fixture run on generated pages (tests)

THE REFERENCE IS SPARSE. Only some words of each page were verified. A detected box that touches no verified box is
not judged: nobody checked that place. So recall-type figures (matched, split, merged, missed) are over the verified
boxes, and precision is over the detections that touch a verified box. Nothing here is a whole-page detection score.

END-TO-END TEXT. A verified word counts as read only if exactly one detection matches it (IoU at least 0.5); its
prediction is then that detection's. A verified word that was missed, split, merged or badly localised gets an empty
prediction, so a segmentation failure costs the whole word, as it would in use.

The segmentation parameters are the defaults of pageocr.layout.SegParams, fixed before this run and never tuned on
the development pages. No training; the recognizer and the verified set are not changed.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import development_e1 as e1
import development_eval_core as core
from pageocr import devguard, geometry, pipeline
from pageocr.layout import SegParams

CONFIG = {
    "experiment": "E2 automatic segmentation",
    "segmentation": {"pipeline_version": pipeline.PIPELINE_VERSION, "parameters": {k: getattr(SegParams(), k) for k in SegParams.__dataclass_fields__}},
    "matching": {"iou_threshold": geometry.IOU_MATCH, "part_of": geometry.PART_OF, "sparse_reference": True,
                 "reference_boxes": "every verified box usable as reference, all scripts"},
    "recognizer": e1.CONFIG["recognizer"], "checkpoint_sha256": devguard.FROZEN_002A_SHA256,
    "crop_source": "the segmentation's OCR image: original grayscale, rotated by the measured skew when it is at least the deskew minimum; never the layout image",
    "padding": e1.CONFIG["padding_primary"], "preprocessing": e1.CONFIG["preprocessing"], "batch_size": 32,
    "primary_rule": core.PRIMARY_RULE, "reference": e1.CONFIG["reference"], "uncertainty": e1.CONFIG["uncertainty"],
    "implementation_version": core.IMPLEMENTATION_VERSION, "estimand": e1.CONFIG["estimand"],
    "end_to_end_rule": "a verified word is read only through a one-to-one matched detection; otherwise its prediction is empty",
}


def page_run(engine, page, recs, batch_size):
    """One page: segmentation, comparison with the verified boxes, recognition of the matched detections."""
    seg = pipeline.segment(page)
    det = [tuple(w.box_original) for w in seg.words]
    usable = [r for r in recs if r["usable_as_verified_reference"]]
    ref = [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in usable]
    cmp_ = geometry.compare(ref, det, sparse_reference=True)
    matched = {i: (j, v) for i, j, v in cmp_.matches}
    status = {}
    for i in range(len(ref)):
        status[i] = "matched" if i in matched else None
    for j, group in cmp_.merged:
        for i in group:
            status[i] = status[i] or "merged"
    for i, _ in cmp_.split:
        status[i] = status[i] or "split"
    for i in cmp_.missed:
        status[i] = status[i] or "missed"
    for i in cmp_.poorly_localised:
        status[i] = status[i] or "poorly_localised"
    # a split whose smaller part is no bigger than a mark: a pulli, superscript or punctuation mark left outside its word
    mark_side = seg.params.mark_side_h * max(seg.h, 1.0)
    mark_splits = sum(1 for _, parts in cmp_.split if min(max(det[j][2] - det[j][0], det[j][3] - det[j][1]) for j in parts) <= mark_side)
    devguard.assert_ocr_allowed(seg.ocr.page_sha256)
    todo, crops = [], []
    for i, (j, _) in sorted(matched.items()):
        w = seg.words[j]
        pad = core.pad_for(CONFIG["padding"], w.box)
        c = pipeline.ocr_crops(seg.ocr, [w], pad_px=pad)[0]
        try:
            from pageocr.engine import validate_crop
            validate_crop(c)
            todo.append(i); crops.append(c)
        except Exception:
            status[i] = "matched_but_crop_refused"
    out = {i: rec for i, rec in zip(todo, core.recognise(engine, crops, batch_size))}
    rows = []
    for i, r in enumerate(usable):
        row = {"verification_id": r["verification_id"], "region_id": r["region_id"], "page_id": r["page_id"], "source_id": r["source_id"], "verified_box": list(ref[i]),
               "groups": core.groups_of(r),
               "segmentation_status": status[i], "ref": core.reference_text(r), "scored": True,
               "matched_detection": seg.words[matched[i][0]].id if i in matched else None, "iou": round(matched[i][1], 4) if i in matched else None}
        rec = out.get(i)
        hyp = core.cs.normalize(rec.text) if rec is not None else ""
        row.update(hyp=hyp, correct=hyp == row["ref"], edits=core.editdistance.eval(row["ref"], hyp), confidence=rec.confidence if rec is not None else None,
                   width_clipped=bool(rec.clipped) if rec is not None else False)
        rows.append(row)
    diag = pipeline.diagnostics(seg)
    return seg, cmp_, rows, {"box_counts": cmp_.counts, "precision": cmp_.precision, "recall": cmp_.recall, "f1": cmp_.f1, "mean_iou_of_matches": cmp_.mean_iou_of_matches,
                             "mark_splits": mark_splits, "flags": list(seg.flags), "skew": seg.skew, "blocks": len(seg.blocks),
                             "lines": sum(len(l) for l in seg.lines), "words_detected": len(seg.words), "text_height_px": seg.h, "seconds": round(seg.seconds, 2)}, diag


def report(mode, record, pooled, per_page, metrics, rows, out_dir, acc=None):
    L = []
    A = L.append
    c = pooled["counts"]
    A(f"# Development pages — E2 automatic segmentation — {record['finished'][:10]}\n")
    if mode != "real":
        A(f"**{core.FIXTURE_STAMP}**\n")
    A("**What this measures:** the word boxes found by the classical segmentation pipeline, compared with the human-verified boxes, and how the frozen Experiment 002-A recognizer reads "
      "the words through those found boxes. Development data, tuned-on: not a held-out test set.\n")
    A("**The reference is sparse.** Only the verified words are judged. Detections elsewhere on the page are not counted as right or wrong.\n")
    A("## 1. Boxes\n")
    A("| | Count | Share of verified boxes |\n|---|---|---|")
    n = max(1, c["reference"])
    for label, key in (("Verified boxes", "reference"), ("Matched one-to-one (IoU at least 0.5)", "matched"), ("Inside a merged detection (under-segmentation)", "reference_in_a_merge (under-segmentation)"),
                       ("Split into several detections (over-segmentation)", "reference_split (over-segmentation)"), ("Missed: nothing detected there", "reference_missed"),
                       ("Detected but badly localised", "reference_poorly_localised")):
        A(f"| {label} | {c[key]} | {100 * c[key] / n:.1f} % |")
    A(f"\nRecall {core.fmt_pct(pooled['recall'])}; precision over the judged detections {core.fmt_pct(pooled['precision'])}; F1 {core.fmt_pct(pooled['f1'])}; "
      f"mean IoU of the matches {pooled['mean_iou_of_matches']:.3f}.\n")
    A(f"Detections judged: {c['judged_detections']} of {c['detected']} (the rest touch no verified box). Duplicate detections: {c['duplicate_pairs']}. "
      f"Splits where a part is only a small mark (a pulli, superscript or punctuation mark left outside its word): {pooled['mark_splits']}.\n")
    A("## 2. Text read through the found boxes\n")
    A(core.METRICS_HEADER)
    for g in ("primary", "primary|part:B", "primary|part:A", "all_usable"):
        if g in metrics:
            A(core.metrics_row(g, metrics[g]))
    A("\nA verified word that was not matched one-to-one has an empty prediction. Compare these rows with the same rows of E1 to see what the found boxes cost.\n")
    if metrics.get("primary"):
        A(f"On the primary set: {core.uncertainty_sentence(metrics['primary'])}\n")
    A("### By source document (primary set)\n")
    A(core.source_section(metrics, record["source_mapping"]))
    A("## 3. By page\n")
    A("| Page | Verified | Matched | Merged | Split | Missed | Badly localised | Recall | Skew (deg) | Blocks | Lines | Words found | Flags |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, d in sorted(per_page.items()):
        k = d["box_counts"]
        A(f"| {p} | {k['reference']} | {k['matched']} | {k['reference_in_a_merge (under-segmentation)']} | {k['reference_split (over-segmentation)']} | {k['reference_missed']} | "
          f"{k['reference_poorly_localised']} | {core.fmt_pct(d['recall'])} | {d['skew']['angle_deg']:+.2f} | {d['blocks']} | {d['lines']} | {d['words_detected']} | {'; '.join(d['flags']) or '-'} |")
    for title, prefix in (("4. Text by page", "page:"), ("5. Text by page condition", "condition:")):
        keys = [g for g in metrics if g.startswith(prefix)]
        if keys:
            A(f"\n## {title}\n\n" + core.METRICS_HEADER)
            for g in keys:
                A(core.metrics_row(g, metrics[g]))
    bad = [r for r in rows if "primary" in r["groups"] and r["segmentation_status"] != "matched"]
    A(f"\n## 6. Verified words the segmentation did not deliver ({len(bad)} primary regions)\n")
    A("| Region | What happened | Reference |\n|---|---|---|")
    for r in bad[:40]:
        A(f"| `{r['region_id']}` | {r['segmentation_status'].replace('_', ' ')} | {r['ref']} |")
    A("\n## 7. What was run\n")
    A("| | |\n|---|---|")
    A(f"| Segmentation | `{pipeline.PIPELINE_VERSION}`, default parameters, fixed before the run, not tuned on these pages |")
    A(f"| Checkpoint | `runs/experiment_002a/best.pth`, SHA-256 `{record['checkpoint_sha256_before']}`, unchanged after the run: {record['checkpoint_sha256_before'] == record['checkpoint_sha256_after']} |")
    A(f"| Verified set | `SHA256SUMS.txt` hash `{record['verified_manifest_sha256']}` |")
    A(f"| Run folder | `{out_dir.as_posix()}` |")
    A(f"| Configuration hash | `{record['config_sha256']}` |\n")
    A("## 8. Limits\n")
    A("- The verified boxes cover part of each page, chosen partly for difficulty. Recall here is recall on those words, not on the page.\n"
      "- A word set solid with its neighbours is one unit by the annotation policy; where the pipeline and the policy differ on what a unit is, that shows as a split or a merge.\n"
      "- Reading order is not scored here: the verified set does not cover whole lines.\n- The pipeline was developed on generated pages only. Development pages: no figure here is a held-out test result.\n")
    if acc is not None:
        A("## 9. Sample accounting\n")
        A("Where every verified record went. A verified word the segmentation did not deliver is scored with an empty prediction; "
          "its segmentation outcome is the `detail` of its row in `accounting.json`.\n")
        A(core.accounting_section(acc))
    A("## 10. Provenance and sample design\n")
    A(core.provenance_section(record))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="E2 automatic-segmentation experiment on the development pages")
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
    out = core.run_dir(core.EXPERIMENTS if mode == "real" else a.out_dir, "E2_automatic_segmentation", manifest_sha, CONFIG)
    if out.exists():
        print(f"This run exists already and is not repeated or overwritten: {out}")
        return 0
    pages = core.load_pages(table, {r["page_id"] for r in records})
    started, ck_before = core.now(), core.sha_file(devguard.CHECKPOINT)
    engine = core.load_engine(a.device)
    rows, per_page, comps, diags = [], {}, [], {}
    for pid in sorted(pages):
        seg, cmp_, prow, info, diag = page_run(engine, pages[pid], [r for r in records if r["page_id"] == pid], CONFIG["batch_size"])
        rows += prow; per_page[pid] = info; comps.append(cmp_); diags[pid] = diag
    pooled = geometry.pool(comps)
    pooled["mark_splits"] = sum(d["mark_splits"] for d in per_page.values())
    metrics = core.score(rows)
    acc = core.accounting(CONFIG["experiment"], records, {"end_to_end": rows}, pages)
    core.check_denominator(acc["settings"]["end_to_end"], metrics.get("primary"))
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
    core.write_new(out / "box_metrics.json", core.dump({"pooled": pooled, "per_page": per_page, "precision_scope": comps[0].precision_scope if comps else ""}))
    core.write_new(out / "metrics.json", core.dump({"end_to_end": metrics, "design": core.design_block(prov),
                                                    "by_source": {"end_to_end": core.by_source(metrics, prov["source_mapping"])}}))
    for pid, d in diags.items():
        core.write_new(out / "segmentation" / f"{pid}.json", core.dump(d))
    text = report(mode, record, pooled, per_page, metrics, rows, out, acc)
    core.write_new(out / "REPORT.md", text)
    if mode == "real":
        rp = core.REPORTS / f"development_E2_automatic_segmentation_{record['finished'][:10]}.md"
        if not rp.exists():
            core.write_new(rp, text)
    p = metrics.get("primary", {})
    print(f"E2 done ({mode}): box recall {core.fmt_pct(pooled['recall'])}; end-to-end primary CER {core.fmt_pct(p.get('cer'))}, word accuracy {core.fmt_pct(p.get('word_acc'))}")
    print(f"run folder: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except core.NotReady as e:
        if not isinstance(e.code, int):
            print(e.code)
        sys.exit(core.EXIT_NOT_READY)
