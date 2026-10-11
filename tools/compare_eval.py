"""
Paired control-vs-treatment comparison for Experiment 001, exactly as pre-registered in
reports/experiment_001_preregistration_addendum.md (sections 2 and 3). Read-only: it reads two sets of
<out>_samples.jsonl files written by tools/evaluate.py --samples and writes <out>.json and <out>.md.

  .venv\\Scripts\\python tools\\compare_eval.py ^
      --iiit-test      reports\\eval_test_baseline_rescore_samples.jsonl         reports\\eval_test_exp001_samples.jsonl ^
      --synth-combined reports\\eval_synth_baseline_rescore_samples.jsonl        reports\\eval_synth_exp001_samples.jsonl ^
      --synth-modern   reports\\eval_synth_baseline_modern_rescore_samples.jsonl reports\\eval_synth_modern_exp001_samples.jsonl ^
      --synth-old      reports\\eval_synth_baseline_old_rescore_samples.jsonl    reports\\eval_synth_old_exp001_samples.jsonl ^
      --out reports\\exp001_comparison

Per set, subset (all / covered / uncovered) and length bucket: n, CER, WER, word accuracy, length ratio for
both arms; delta = treatment - control in percentage points with a paired percentile-bootstrap 95% CI
(10,000 resamples, numpy default_rng(20260923), micro-averaged CER inside every resample); McNemar's exact test.
Both arms must be scored with the same evaluate.py command (manifest, --style, --batch-size): predictions depend
very slightly on which images share a batch (padding width), so only identical batching pairs exactly.
Then the design-review section-8 outcome, applied mechanically. The thresholds are constants on purpose: they
were fixed before any treatment result existed and are not a command-line option.
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import charset as cs  # noqa: E402
from src.metrics import BUCKETS, bucket_name  # noqa: E402

DELTA_CER_PP = 0.10          # pre-registered, IIIT test, absolute percentage points
DELTA_WER_PP = 0.50
BOOT_SEED = 20260923
BOOT_RESAMPLES = 10_000
SETS = [("iiit_test", "IIIT test"), ("synth_combined", "Synthetic combined"),
        ("synth_modern", "Synthetic modern"), ("synth_old", "Synthetic old")]
BUCKET_NAMES = [bucket_name(lo) for lo, _ in BUCKETS]


def load_samples(path):
    rows, meta = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
            else:
                rows.append(r)
    return meta, rows


def pair(ctrl_path, treat_path):
    mc, rc = load_samples(ctrl_path)
    mt, rt = load_samples(treat_path)
    for m, p in ((mc, ctrl_path), (mt, treat_path)):
        if m.get("n_unreadable_skipped"):
            sys.exit(f"{p}: {m['n_unreadable_skipped']} samples were skipped as unreadable; pairing refused")
    key = lambda r: (r["manifest_index"], r["image"], r["ref"])     # noqa: E731
    kc, kt = [key(r) for r in rc], [key(r) for r in rt]
    if kc != kt:
        sys.exit(f"control and treatment samples differ ({len(kc)} vs {len(kt)} rows, "
                 f"{len(set(kc) ^ set(kt))} unmatched keys); pairing refused")
    return mc, mt, rc, rt


def covered_charset(train_manifest):
    chars = set()
    with open(train_manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                label = cs.normalize(json.loads(line)["label"])
                if cs.is_encodable(label):
                    chars.update(label)
    return chars


def arm_metrics(edits, wrong, ref_len, hyp_len):
    ch = ref_len.sum()
    return {"cer": edits.sum() / max(1, ch), "wer": wrong.mean() if len(wrong) else 0.0,
            "word_acc": 1 - wrong.mean() if len(wrong) else 0.0, "len_ratio": hyp_len.sum() / max(1, ch)}


def bootstrap(ec, et, wc, wt, ch):
    """Paired percentile bootstrap of (delta CER, delta WER); same indices for both arms."""
    n = len(ch)
    rng = np.random.default_rng(BOOT_SEED)
    dcer, dwer = np.empty(BOOT_RESAMPLES), np.empty(BOOT_RESAMPLES)
    m = max(1, 8_000_000 // n)                        # resamples per chunk (bounded memory)
    for s in range(0, BOOT_RESAMPLES, m):
        k = min(m, BOOT_RESAMPLES - s)
        idx = rng.integers(0, n, size=(k, n))
        dcer[s:s + k] = (et[idx].sum(1) - ec[idx].sum(1)) / np.maximum(1, ch[idx].sum(1))
        dwer[s:s + k] = (wt[idx].sum(1) - wc[idx].sum(1)) / n
    lo, hi = np.percentile(dcer, [2.5, 97.5]), np.percentile(dwer, [2.5, 97.5])
    return (float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def analyse(rc, rt):
    ec = np.array([r["edits"] for r in rc], float); et = np.array([r["edits"] for r in rt], float)
    wc = np.array([not r["correct"] for r in rc], float); wt = np.array([not r["correct"] for r in rt], float)
    ch = np.array([r["ref_len"] for r in rc], float)
    hc = np.array([r["hyp_len"] for r in rc], float); ht = np.array([r["hyp_len"] for r in rt], float)
    out = {"n": len(rc)}
    if not len(rc):
        return out
    c, t = arm_metrics(ec, wc, ch, hc), arm_metrics(et, wt, ch, ht)
    (cer_lo, cer_hi), (wer_lo, wer_hi) = bootstrap(ec, et, wc, wt, ch)
    b = int(((wc == 0) & (wt == 1)).sum()); cc = int(((wc == 1) & (wt == 0)).sum())
    out.update({"control": c, "treatment": t,
                "delta_pp": {"cer": 100 * (t["cer"] - c["cer"]), "wer": 100 * (t["wer"] - c["wer"]),
                             "word_acc": 100 * (t["word_acc"] - c["word_acc"]),
                             "len_ratio": t["len_ratio"] - c["len_ratio"]},
                "ci95_pp": {"cer": [100 * cer_lo, 100 * cer_hi], "wer": [100 * wer_lo, 100 * wer_hi]},
                "mcnemar": {"b_ctrl_right_treat_wrong": b, "c_ctrl_wrong_treat_right": cc,
                            "p_exact_two_sided": mcnemar_exact(b, cc)}})
    return out


def analyse_set(rc, rt, charset):
    res = {}
    for subset in ("all", "covered", "uncovered"):
        keep = [i for i, r in enumerate(rc)
                if subset == "all" or (all(ch in charset for ch in r["ref"]) == (subset == "covered"))]
        sc, st = [rc[i] for i in keep], [rt[i] for i in keep]
        res[subset] = {"overall": analyse(sc, st)}
        for bn in BUCKET_NAMES:
            idx = [i for i, r in enumerate(sc) if bucket_name(r["ref_len"]) == bn]
            res[subset][f"len:{bn}"] = analyse([sc[i] for i in idx], [st[i] for i in idx])
    return res


def decide(results):
    """Design review section 8 / addendum section 3. Run validity (step 1) is checked from the run record."""
    need = ("iiit_test", "synth_modern", "synth_old")
    if any(k not in results for k in need):
        return {"outcome": "NOT DECIDED (iiit_test, synth_modern and synth_old are all required)"}
    t = results["iiit_test"]["all"]["overall"]
    hw_ok = t["ci95_pp"]["cer"][1] <= DELTA_CER_PP and t["ci95_pp"]["wer"][1] <= DELTA_WER_PP
    below = {s: results[s]["covered"]["overall"]["ci95_pp"]["cer"][1] < 0 for s in ("synth_modern", "synth_old")}
    S = sum(below.values())
    if S == 2:
        outcome = "H-001 SUPPORTED" if hw_ok else "SYNTHETIC GAIN WITH HANDWRITING COST"
    elif S == 1:
        outcome = "STYLE-ASYMMETRIC"
    else:
        outcome = "NO SYNTHETIC GAIN"
    unc = any(results[s]["uncovered"]["overall"].get("ci95_pp", {}).get("cer", [0, 0])[1] < 0
              for s in ("synth_modern", "synth_old"))
    return {"outcome": outcome, "S": S, "covered_cer_ci_upper_below_0": below, "hw_ok": hw_ok,
            "iiit_cer_ci_upper_pp": t["ci95_pp"]["cer"][1], "iiit_wer_ci_upper_pp": t["ci95_pp"]["wer"][1],
            "delta_cer_pp": DELTA_CER_PP, "delta_wer_pp": DELTA_WER_PP,
            "flag_gain_only_on_uncovered": S == 0 and unc,
            "run_validity": "must be confirmed from the run record (addendum section 3, step 1)"}


def fmt(x, pct=True):
    return f"{100 * x:.4f}%" if pct else f"{x:.4f}"


def markdown(results, decision, metas):
    L = ["# Experiment 001 — paired comparison", "",
         f"Δ = treatment − control (percentage points). 95% CI: paired percentile bootstrap, "
         f"{BOOT_RESAMPLES:,} resamples, seed {BOOT_SEED}. δ_CER = +{DELTA_CER_PP} pp, δ_WER = +{DELTA_WER_PP} pp.", ""]
    for key, title in SETS:
        if key not in results:
            continue
        mc, mt = metas[key]
        L += [f"## {title}", "", f"control `{mc.get('checkpoint')}` (epoch {mc.get('epoch')}), "
              f"treatment `{mt.get('checkpoint')}` (epoch {mt.get('epoch')}), manifest `{mc.get('manifest')}`"
              f"{', style ' + mc['style'] if mc.get('style') else ''}", "",
              "| Subset | Bucket | n | Control CER | Treat. CER | ΔCER pp [95% CI] | Control WER | Treat. WER | "
              "ΔWER pp [95% CI] | Len ratio C→T | McNemar b/c, p |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for subset in ("all", "covered", "uncovered"):
            for part, r in results[key][subset].items():
                if not r.get("n"):
                    L.append(f"| {subset} | {part.replace('len:', '')} | 0 | | | | | | | | |")
                    continue
                c, t, d, ci, mn = r["control"], r["treatment"], r["delta_pp"], r["ci95_pp"], r["mcnemar"]
                L.append(f"| {subset} | {part.replace('len:', '')} | {r['n']:,} | {fmt(c['cer'])} | {fmt(t['cer'])} | "
                         f"{d['cer']:+.4f} [{ci['cer'][0]:+.4f}, {ci['cer'][1]:+.4f}] | {fmt(c['wer'])} | {fmt(t['wer'])} | "
                         f"{d['wer']:+.4f} [{ci['wer'][0]:+.4f}, {ci['wer'][1]:+.4f}] | "
                         f"{c['len_ratio']:.4f}→{t['len_ratio']:.4f} | {mn['b_ctrl_right_treat_wrong']}/"
                         f"{mn['c_ctrl_wrong_treat_right']}, {mn['p_exact_two_sided']:.4g} |")
        L.append("")
    L += ["## Outcome (design review §8, addendum §3)", ""]
    for k, v in decision.items():
        L.append(f"- **{k}**: {v}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    for key, _ in SETS:
        ap.add_argument("--" + key.replace("_", "-"), nargs=2, metavar=("CONTROL", "TREATMENT"))
    ap.add_argument("--train-manifest", default="data/real/iiit_hw/manifest_train.jsonl",
                    help="defines the covered character set")
    ap.add_argument("--out", required=True, help="writes <out>.json and <out>.md")
    a = ap.parse_args()

    charset = covered_charset(ROOT / a.train_manifest)
    print(f"covered character set: {len(charset)} characters from {a.train_manifest}")
    results, metas = {}, {}
    for key, title in SETS:
        paths = getattr(a, key)
        if not paths:
            continue
        mc, mt, rc, rt = pair(*paths)
        metas[key] = (mc, mt)
        results[key] = analyse_set(rc, rt, charset)
        o = results[key]["all"]["overall"]
        print(f"{title:20} n={o['n']:6,}  ΔCER {o['delta_pp']['cer']:+.4f} pp "
              f"[{o['ci95_pp']['cer'][0]:+.4f}, {o['ci95_pp']['cer'][1]:+.4f}]  ΔWER {o['delta_pp']['wer']:+.4f} pp "
              f"[{o['ci95_pp']['wer'][0]:+.4f}, {o['ci95_pp']['wer'][1]:+.4f}]", flush=True)
    decision = decide(results)
    print("outcome:", decision["outcome"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{a.out}.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "decision": decision,
                   "inputs": {k: getattr(a, k) for k, _ in SETS if getattr(a, k)},
                   "covered_charset_size": len(charset)}, f, ensure_ascii=False, indent=1)
    with open(f"{a.out}.md", "w", encoding="utf-8") as f:
        f.write(markdown(results, decision, metas))
    print(f"wrote {a.out}.json and {a.out}.md")


if __name__ == "__main__":
    main()
