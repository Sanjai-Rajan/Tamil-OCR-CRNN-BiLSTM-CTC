"""
Score a checkpoint on any manifest (use this ONCE on test splits, at the end).

  .venv\\Scripts\\python tools\\evaluate.py --ckpt runs\\baseline_hw\\best.pth ^
      --manifest data\\real\\iiit_hw\\manifest_test.jsonl --out runs\\baseline_hw\\test_iiit

Prints micro CER, WER (word accuracy = 1 - WER), per length bucket; writes all errors to <out>_errors.tsv.
--samples additionally writes <out>_samples.jsonl (every scored sample with its 0-based manifest line index),
the input of tools/compare_eval.py. Without it, the output is exactly as before.
"""
import argparse
import json
import sys
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import charset as cs  # noqa: E402
from src.data.datasets import ManifestDataset, collate  # noqa: E402
from src.metrics import Scoreboard  # noqa: E402
from src.model.crnn_v2 import CRNNv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--style", choices=["modern", "old"], default=None,
                    help="filter to entries whose manifest 'style' field matches (e.g. data/synth_val); omit for all")
    ap.add_argument("--samples", action="store_true",
                    help="also write <out>_samples.jsonl: one row per scored sample with its manifest line index "
                         "(for paired comparison with tools/compare_eval.py)")
    a = ap.parse_args()
    if a.samples and not a.out:
        sys.exit("--samples needs --out")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if list(ck.get("charset", cs.CHARSET)) != list(cs.CHARSET):
        sys.exit("checkpoint charset differs from src/charset.py")
    margs = dict(ck["model_args"]); margs["channels"] = tuple(margs["channels"])
    model = CRNNv2(**margs).to(dev).eval()
    model.load_state_dict(ck["model"])

    ds = ManifestDataset(a.manifest, where={"style": a.style} if a.style else None)
    rows = manifest_rows(ds, a.manifest, a.style) if a.samples else None
    data = IndexedDataset(ds) if a.samples else ds
    dl = DataLoader(data, batch_size=a.batch_size, num_workers=a.workers,
                    collate_fn=indexed_collate if a.samples else collate)
    board, errors, samples = Scoreboard(), [], []
    with torch.no_grad():
        for b in dl:
            if b is None:
                continue
            with torch.autocast(dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                logits, lengths = model(b["images"].to(dev), b["widths"])
            best = logits.float().argmax(-1).cpu()
            for i, ref in enumerate(b["texts"]):
                hyp = cs.decode(cs.ctc_greedy_collapse(best[: lengths[i], i].tolist()))
                board.add(ref, hyp, b["sources"][i])
                if hyp != ref:
                    errors.append((ref, hyp))
                if rows is not None:
                    r = rows[b["positions"][i]]
                    assert r["label"] == ref, f"sample/manifest mismatch at manifest line {r['manifest_index']}"
                    samples.append({"manifest_index": r["manifest_index"], "image": r["image"],
                                    "style": r["style"], "font": r["font"], "ref": ref, "hyp": hyp,
                                    "edits": editdistance.eval(ref, hyp), "ref_len": len(ref),
                                    "hyp_len": len(hyp), "correct": hyp == ref})
    s = board.summary()
    print(f"checkpoint {a.ckpt} (epoch {ck.get('epoch')})   set {a.manifest}   n={s['all']['n']}")
    print(f"  CER {s['all']['cer']*100:.2f}%   WER {s['all']['wer']*100:.2f}%   word accuracy {s['all']['word_acc']*100:.2f}%"
          f"   length ratio {s['all']['len_ratio']:.2f}")
    for k in sorted(k for k in s if k.startswith("len:")):
        v = s[k]
        print(f"  {k:10} n={v['n']:6d}  CER {v['cer']*100:6.2f}%  word acc {v['word_acc']*100:6.2f}%")
    if ds.dropped:
        print(f"  ({ds.dropped} labels skipped: characters outside the charset)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{a.out}_errors.tsv", "w", encoding="utf-8") as f:
            f.write("reference\tprediction\n")
            f.writelines(f"{r}\t{h}\n" for r, h in errors)
        with open(f"{a.out}_summary.json", "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        print(f"  wrote {a.out}_errors.tsv and {a.out}_summary.json")
    if rows is not None:
        # manifest rows the loader skipped (unreadable image) are listed so the pairing can never silently shrink
        missing = len(rows) - len(samples)
        meta = {"_meta": {"checkpoint": a.ckpt, "epoch": ck.get("epoch"), "manifest": a.manifest,
                          "style": a.style, "n_manifest_rows_kept": len(rows), "n_scored": len(samples),
                          "n_unreadable_skipped": missing}}
        with open(f"{a.out}_samples.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for s_ in sorted(samples, key=lambda x: x["manifest_index"]):
                f.write(json.dumps(s_, ensure_ascii=False) + "\n")
        print(f"  wrote {a.out}_samples.jsonl ({len(samples)} samples, {missing} unreadable skipped)")


def manifest_rows(ds, manifest, style):
    """Re-read the manifest with ManifestDataset's own filters, keeping the 0-based line index of every kept
    entry. Cross-checked against ds.items so the two can never disagree."""
    where = {"style": style} if style else None
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for li, line in enumerate(f):
            if not line.strip():
                continue
            e = json.loads(line)
            if where and any(e.get(k) != v for k, v in where.items()):
                continue
            label = cs.normalize(e["label"])
            if not cs.is_encodable(label):
                continue
            rows.append({"manifest_index": li, "image": e["image"], "label": label,
                         "style": e.get("style"), "font": e.get("font")})
    assert len(rows) == len(ds.items), "manifest re-read disagrees with ManifestDataset"
    for r, (path, label) in zip(rows, ds.items):
        p = Path(r["image"])
        assert label == r["label"] and Path(path) == (p if p.is_absolute() else ds.root / p), \
            f"manifest re-read disagrees with ManifestDataset at line {r['manifest_index']}"
    return rows


class IndexedDataset(torch.utils.data.Dataset):
    """ManifestDataset items plus their position, so each prediction maps back to its manifest line."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return (*self.ds[idx], idx)


def indexed_collate(batch):
    """collate() skips samples with an empty label (unreadable image); keep positions for exactly the kept ones."""
    out = collate([item[:3] for item in batch])
    if out is not None:
        out["positions"] = [item[3] for item in batch if item[1]]
        assert len(out["positions"]) == len(out["texts"])
    return out


if __name__ == "__main__":
    main()
