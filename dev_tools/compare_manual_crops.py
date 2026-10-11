"""
Manual-versus-pipeline crop comparison for ONE non-sensitive page, from the command line (the same code as the
"Manual crop comparison" panel of the app's diagnostic mode).

    <python> -B dev_tools/compare_manual_crops.py --page <image> --samples <samples.json> --out <new folder> [--rule-removal]

samples.json (write it by hand; see dev_tools/manual_samples_TEMPLATE.json):
    [{"id": "S01", "box": [x1, y1, x2, y2], "reference": "<Tamil text you verified>", "reference_confirmed": true, "note": ""}, ...]
Boxes are in the pixels of the image file as it is (after EXIF rotation), tight around ONE word. A reference counts
only with "reference_confirmed": true, and must be your own reading of the image, never the model's output.

The page goes through the app's own decoder and guard (protected research images are refused), the production
segmentation and recognizer (registered Experiment 002-A checkpoint). Writes into a NEW folder: comparison.json, a
diagnostic bundle of the run, and both crops of every sample. Nothing is scored without confirmed references.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wordocr import checkpoints, crop_compare, fullpage, inputs, service  # noqa: E402
from wordocr import diagnostics as dg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule-removal", action="store_true", help="also run with the experimental rule removal (both runs are kept)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; choose a new folder (results are never overwritten)")
    data = Path(a.page).read_bytes()
    try:
        d = inputs.decode_upload(Path(a.page).name, data)
    except inputs.InputError as e:
        sys.exit(f"refused: {e}")
    raw = json.loads(Path(a.samples).read_text(encoding="utf-8"))
    samples = [crop_compare.Sample(id=str(s["id"]), box=tuple(int(v) for v in s["box"]), reference=str(s.get("reference", "")),
                                   reference_confirmed=bool(s.get("reference_confirmed")), note=str(s.get("note", ""))) for s in raw]
    model = service.load_model(checkpoints.DEFAULT_ID, a.device)
    runs = {"default": fullpage.PageSettings()}
    if a.rule_removal:
        runs["rule_removal"] = fullpage.PageSettings(exp_rule_removal=True)
    report = {"page": {"name": d.name, "sha256": d.sha256, "width": d.width, "height": d.height}, "runs": {}}
    for name, settings in runs.items():
        seg = fullpage.segment_page(d, settings)
        res = fullpage.recognize_page(model, d, seg, settings)
        rep = crop_compare.compare(model, d, seg, res.margin_px, samples, result=res)
        report["runs"][name] = {"settings": settings.as_dict(), "summary": rep["summary"], "samples": rep["samples"], "page_text": res.text}
        diag = dg.analyse(d, seg, res, settings)
        dg.save_bundle(out / f"bundle_{name}", d, seg, res, settings, diag, include_input=False)
        crops_dir = out / f"sample_crops_{name}"
        crops_dir.mkdir(parents=True)
        from pageocr import pipeline
        pcs = pipeline.ocr_crops(seg.ocr, seg.words, res.margin_px)
        idx = {w.id: i for i, w in enumerate(seg.words)}
        H, W = d.gray.shape
        m = res.margin_px
        for s, row in zip(samples, rep["samples"]):
            if "error" in row:
                continue
            x1, y1, x2, y2 = s.box
            cv2.imwrite(str(crops_dir / f"{s.id}_A_manual.png"), d.gray[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)])
            if row["pipeline_region"]:
                cv2.imwrite(str(crops_dir / f"{s.id}_B_pipeline_{row['pipeline_region']}.png"), pcs[idx[row["pipeline_region"]]])
    report["model"] = model.info()
    (out / "comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, r in report["runs"].items():
        print(name, json.dumps(r["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
