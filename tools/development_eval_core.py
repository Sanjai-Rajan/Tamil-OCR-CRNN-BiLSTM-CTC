"""
Shared core of the development-page experiments E1, E2 and E3: the guard, loading, cropping, recognition and scoring.

THE GUARD. An experiment runs in one of two modes, decided from its inputs and never by a switch:
    real     the inputs are the project's development pages or its human-verified set. Then every condition of
             pageocr.devguard must hold (verified set exists, complete, frozen; protocol lock matches; frozen 002-A
             checkpoint), or the run exits before a model is loaded and says exactly what is missing.
    fixture  the inputs are generated pages and a generated "verified" folder, outside the project. Results go outside
             the project and are stamped as fixture output; they are tests of the plumbing, never evidence.
A fixture input that turns out to be a development page (by SHA-256) switches the run to real mode.

WHAT IS SCORED. Reference = the human-verified transcription, normalised by src.charset.normalize. Prediction = the
frozen Experiment 002-A recognizer's greedy CTC output (pageocr.engine.OcrEngine, which hashes the checkpoint and
loads it read-only) on a crop of the ORIGINAL grayscale page. Metrics are those of src.metrics: micro CER, WER, word
accuracy and length ratio. As in the R1 scoring, text the recognizer cannot emit is scored, not dropped.

Nothing here trains, changes a checkpoint, or writes into an annotation folder.
"""
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import editdistance                                              # noqa: E402

from pageocr import devguard, geometry                           # noqa: E402
from src import charset as cs                                    # noqa: E402

CORE_VERSION = "development-eval-1.2.0"
# Recorded in every run record (CH-07) and part of every experiment's CONFIG, so it is part of the run folder's name: a
# run made by another implementation version lands in another folder and never meets an earlier finished run.
# 1.1.0 = Stage 5A of the 2026-10-09 implementation blueprint: provenance and sample accounting.
# 1.2.0 = Stage 5B: source_id (CH-03) and the source-aware uncertainty statement of owner decision D-01, Option 2 (CH-01).
#         Thresholds, the primary rule and the point estimates are unchanged.
IMPLEMENTATION_VERSION = CORE_VERSION + " (stage 5B: CH-03 source_id, CH-01 source-aware uncertainty; includes stage 5A: CH-07, CH-08)"
DEV = ROOT / "development_pages"
EXPERIMENTS = DEV / "experiments"
REPORTS = ROOT / "reports"
BOOTSTRAP_SEED, BOOTSTRAP_REPS = 20261008, 2000
EXIT_NOT_READY = 3
FIXTURE_STAMP = "FIXTURE RUN on generated pages: a test of the code, not evidence about the recognizer or the development pages."


class NotReady(SystemExit):
    """The guard refused the run. Exit code 3."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha_file(p) -> str:
    return devguard.sha_file(p)


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def inside(path: Path, parent: Path) -> bool:
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents


# ---- inputs and the guard --------------------------------------------------------------------------------------------------
def page_table(pages_dir: Path) -> dict:
    """page_id -> {"path", "sha256", "condition_tags"}. Real pages come from the manifest; fixture pages from pages.json."""
    pages_dir = Path(pages_dir).resolve()
    if pages_dir == (DEV / "original").resolve():
        with open(DEV / "manifests" / "development_pages_manifest.csv", encoding="utf-8", newline="") as f:
            return {r["page_id"]: {"path": pages_dir / r["filename"], "sha256": r["file_hash"], "condition_tags": r["condition_tags"]} for r in csv.DictReader(f)}
    spec = json.loads((pages_dir / "pages.json").read_text(encoding="utf-8"))
    return {k: {"path": pages_dir / v["filename"], "sha256": v["sha256"], "condition_tags": v.get("condition_tags", "")} for k, v in spec.items()}


def resolve_mode(verified_dir: Path, pages_dir: Path, out_dir, table: dict) -> str:
    verified_dir, pages_dir = Path(verified_dir).resolve(), Path(pages_dir).resolve()
    masters = devguard.development_master_hashes()
    real = (verified_dir == devguard.VERIFIED.resolve() or inside(verified_dir, DEV) or inside(pages_dir, DEV)
            or any(p["sha256"] in masters or (p["path"].is_file() and sha_file(p["path"]) in masters) for p in table.values()))
    if real:
        if verified_dir != devguard.VERIFIED.resolve() or pages_dir != (DEV / "original").resolve():
            raise NotReady("Refused: development pages may only be evaluated together with the project's own verified set "
                           f"({devguard.VERIFIED}) and its own masters ({DEV / 'original'}). Mixed inputs are not allowed.")
        return "real"
    if out_dir is None or inside(Path(out_dir), ROOT):
        raise NotReady("Refused: a fixture run must write to a folder outside the project (give --out-dir).")
    return "fixture"


def guard(mode: str) -> dict:
    """Real mode: every devguard condition, or exit with the list of what is missing. Returns the status for the record."""
    if mode != "real":
        return {"ready": None, "mode": "fixture", "conditions": []}
    st = devguard.status()
    if not st["ready"]:
        print(devguard.explain(st))
        raise NotReady(EXIT_NOT_READY)
    return st


def load_verified(verified_dir: Path) -> tuple:
    """(records, manifest hash). The folder must match its own SHA256SUMS.txt; the hash of that file identifies the set."""
    verified_dir = Path(verified_dir)
    sums_path = verified_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise NotReady(f"Refused: {sums_path} is missing; the verified set is not frozen.")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split("  ", 1)
            if not (verified_dir / name).is_file() or sha_file(verified_dir / name) != h:
                raise NotReady(f"Refused: {name} of the verified set does not match SHA256SUMS.txt.")
    recs = [json.loads(l) for l in (verified_dir / "verified_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len({r["region_id"] for r in recs}) != len(recs):
        raise NotReady("Refused: the verified set holds a region twice.")
    return recs, sha_file(sums_path)


def load_pages(table: dict, wanted: set) -> dict:
    """page_id -> pageocr.imageio.Page (original grayscale, decoded once), each checked against its recorded SHA-256."""
    from pageocr import imageio
    out = {}
    for pid in sorted(wanted):
        if pid not in table:
            raise NotReady(f"Refused: page {pid} of the verified set is not in the page table.")
        page = imageio.load_page(table[pid]["path"])
        if page.sha256 != table[pid]["sha256"]:
            raise NotReady(f"Refused: the image of {pid} does not match its recorded SHA-256.")
        out[pid] = page
    return out


def load_engine(device=None):
    """The frozen Experiment 002-A recognizer. OcrEngine hashes the file and refuses any other checkpoint."""
    from pageocr.engine import FROZEN_002A_SHA256, OcrEngine
    before = sha_file(devguard.CHECKPOINT)
    if before != FROZEN_002A_SHA256 or FROZEN_002A_SHA256 != devguard.FROZEN_002A_SHA256:
        raise NotReady(f"Refused: {devguard.CHECKPOINT} is not the frozen Experiment 002-A checkpoint.")
    return OcrEngine(devguard.CHECKPOINT, device=device)


TOO_SMALL_REASON = "the box is smaller than 2 x 2 pixels after shrinking"


# ---- cropping and recognition ------------------------------------------------------------------------------------------------
def pad_for(rule: dict, box: tuple) -> int:
    """Margin in pixels for a box under a padding rule: {"kind": "pixels", "value": k} or {"kind": "height_fraction", "value": f}."""
    if rule["kind"] == "pixels":
        return int(rule["value"])
    if rule["kind"] == "height_fraction":
        return int(round(rule["value"] * (box[3] - box[1])))
    raise ValueError(f"unknown padding rule {rule!r}")


def make_crop(page, box: tuple, pad_px: int) -> dict:
    """The crop of a verified box from the original page, with a margin. {"crop" or None, "box_used", "clipped_edges", "valid", "reason"}."""
    from pageocr.engine import InvalidCropError, validate_crop
    e = geometry.expand_box(box, pad_px, page.width, page.height)
    out = {"box_used": e["box"], "clipped_edges": e["clipped_edges"], "valid": e["valid"], "pad_px": pad_px, "crop": None, "reason": ""}
    if not e["valid"]:
        out["reason"] = TOO_SMALL_REASON
        return out
    x1, y1, x2, y2 = e["box"]
    crop = page.gray[y1:y2, x1:x2].copy()
    try:
        validate_crop(crop)
    except InvalidCropError as ex:
        out.update(valid=False, reason=str(ex))
        return out
    out["crop"] = crop
    return out


def recognise(engine, crops: list, batch_size: int = 32) -> list:
    return engine.recognize_batch(crops, batch_size=batch_size) if crops else []


# ---- which regions are scored ----------------------------------------------------------------------------------------------
PRIMARY_RULE = ("primary = regions the reviewer confirmed or corrected (usable_as_verified_reference), script 'tamil'. "
                "Ambiguous and unreadable regions, regions needing a unit split or merge, and non-Tamil regions are outside the "
                "primary figure and are reported separately or counted as coverage.")


def groups_of(r: dict) -> list:
    """The groups a verified record's score is added to. Per-page and per-condition groups are over the primary set."""
    usable, tamil = bool(r["usable_as_verified_reference"]), r.get("script") == "tamil"
    g = []
    if not usable:
        return g
    g.append("all_usable")
    g.append(f"script:{r.get('script')}")
    if tamil:
        part = r.get("queue_part") or "?"
        g += ["primary", f"primary|part:{part}", f"page:{r['page_id']}", f"region_type:{r.get('region_type')}", f"final_state:{r['final_state']}",
              f"ai_flagged_uncertain:{(r.get('ai_flags') or {}).get('uncertainty') == 'uncertain'}", f"fragment:{bool(r.get('fragment'))}",
              f"legibility:{r.get('legibility')}", f"unit:{r.get('word_unit_type')}", f"encodable:{r.get('encoding_status') == 'encodable'}"]
        g += [f"condition:{t}" for t in (r.get("page_condition_tags") or "").split()]
        if "source_id" in r:                                     # set by attach_sources() in every run; score() refuses rows without it
            g.append(f"source:{r['source_id']}")
    return g


def reference_text(r: dict) -> str:
    return cs.normalize(r["normalized_text"])


# ---- scoring ---------------------------------------------------------------------------------------------------------------
def tally(rows: list) -> dict:
    """Metrics of a list of scored rows ({"ref", "hyp", "confidence", "width_clipped"}). Definitions as in src/metrics.py."""
    n = len(rows)
    edits = sum(editdistance.eval(r["ref"], r["hyp"]) for r in rows)
    chars = sum(len(r["ref"]) for r in rows)
    wrong = sum(r["ref"] != r["hyp"] for r in rows)
    pred = sum(len(r["hyp"]) for r in rows)
    conf = [r["confidence"] for r in rows if r.get("confidence") is not None]
    return {"n": n, "ref_chars": chars, "edits": edits, "cer": edits / max(1, chars), "wer": wrong / max(1, n), "word_acc": 1 - wrong / max(1, n),
            "len_ratio": pred / max(1, chars), "mean_confidence": sum(conf) / len(conf) if conf else None,
            "width_clipped_rate": sum(bool(r.get("width_clipped")) for r in rows) / max(1, n), "empty_prediction_rate": sum(r["hyp"] == "" for r in rows) / max(1, n)}


def bootstrap_cer(rows: list, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED):
    """95 % percentile interval of the micro CER, resampling regions. Deterministic for a given list."""
    if len(rows) < 2:
        return None
    e = [editdistance.eval(r["ref"], r["hyp"]) for r in rows]
    c = [len(r["ref"]) for r in rows]
    rng, n, vals = random.Random(seed), len(rows), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(e[i] for i in idx) / max(1, sum(c[i] for i in idx)))
    vals.sort()
    return [vals[int(0.025 * reps)], vals[min(reps - 1, int(0.975 * reps))]]


def score(rows: list) -> dict:
    """group -> metrics, for rows carrying "groups". The primary groups and all_usable also get the uncertainty statement of
    D-01 Option 2 (uncertainty()): every one of their rows must carry a source_id."""
    by = {}
    for r in rows:
        for g in r["groups"]:
            by.setdefault(g, []).append(r)
    out = {}
    for g in sorted(by):
        out[g] = tally(by[g])
        if g.startswith("primary") or g == "all_usable":
            out[g].update(uncertainty(by[g]))
    return out


# ---- sources (CH-03) and the uncertainty statement (CH-01) ----------------------------------------------------------------
DECISIONS_DIR = "reports/cowork_implementation_blueprint_2026-10-09"
OWNER_DECISIONS = {
    "D-01": "Option 2: finite-population descriptive over the verified regions of the 16 development pages; per-page and per-source "
            "tables; leave-one-source-out range; no headline population-level confidence interval "
            f"({DECISIONS_DIR}/OWNER_DECISION_RECORD_2026-10-09_D01_D04.md)",
    "D-04": "documented fallback: source_id derived from source_title; manifest and protocol lock unchanged "
            f"({DECISIONS_DIR}/OWNER_DECISION_RECORD_2026-10-09_D04_IMPLEMENTATION.md)",
    "D-05": "NOT DECIDED by the owner; MIN_CLUSTERS_FOR_CI = 20 used provisionally, as the register's deferral rule says",
}
SOURCE_ID_RULE = ("source_id = the first 8 hexadecimal digits of the SHA-256 of the page's source_title (UTF-8, outer spaces removed), "
                  "taken from the page manifest; a page without a title, or two titles with one id, stops the run (owner decision D-04, fallback)")
CLUSTER_UNIT = "source_id"
MIN_CLUSTERS_FOR_CI = 20
MIN_CLUSTERS_STATUS = "provisional: owner decision D-05 not taken; the decision register's deferral default"
REGION_DIAGNOSTIC_LABEL = ("DIAGNOSTIC ONLY, not a confidence interval: regions resampled as if independent, which they are not (they are "
                           "nested in pages and source documents); it describes nothing beyond these regions")
CLUSTER_INTERVAL_LABEL = ("indicative only: source documents resampled with replacement; it describes variation between these source documents "
                          "and is not a population interval (D-01 Option 2: no headline interval)")


class SourceMappingError(NotReady):
    """A page has no source title, or two different titles give one source_id. The run is refused; no default is assigned."""


def source_id_of(title: str) -> str:
    return hashlib.sha256(title.strip().encode("utf-8")).hexdigest()[:8]


def source_table(pages_dir: Path, pages=None) -> dict:
    """page_id -> {"source_id", "source_title"} for the given pages (default: every page of the table). Refuses a page with no
    source title and a source_id shared by two different titles."""
    titles = page_sources(pages_dir)
    wanted = sorted(titles) if pages is None else sorted(pages)
    missing = [p for p in wanted if not titles.get(p)]
    if missing:
        raise SourceMappingError(f"Refused: no source title for page(s) {', '.join(missing)}; a source is never assigned by default.")
    seen, out = {}, {}
    for p in wanted:
        sid = source_id_of(titles[p])
        if seen.setdefault(sid, titles[p]) != titles[p]:
            raise SourceMappingError(f"Refused: source_id {sid} is produced by two different titles ({seen[sid]!r}, {titles[p]!r}).")
        out[p] = {"source_id": sid, "source_title": titles[p]}
    return out


def attach_sources(records: list, table: dict) -> list:
    """Copies of the verified records, each with the source_id of its page. The records on disk are not touched."""
    out = []
    for r in records:
        if r["page_id"] not in table:
            raise SourceMappingError(f"Refused: page {r['page_id']} of {r['verification_id']} has no source mapping.")
        out.append(dict(r, source_id=table[r["page_id"]]["source_id"]))
    return out


def _require_sources(rows: list) -> list:
    bad = [r.get("verification_id") for r in rows if not r.get("source_id")]
    if bad:
        raise AssertionError(f"scored rows without a source_id: {bad[:5]}; a source is never assigned by default")
    return sorted({r["source_id"] for r in rows})


def loso(rows: list) -> dict:
    """Leave one source out: the metric recomputed with each source document removed in turn. Each value re-sums the edits and
    reference characters of the remaining rows (tally()); per-source CERs are never averaged."""
    srcs = _require_sources(rows)
    if len(srcs) < 2:
        return {"cluster_unit": CLUSTER_UNIT, "n_clusters": len(srcs), "refused": "leave-one-source-out needs at least two sources"}
    per = {}
    for s in srcs:
        t = tally([r for r in rows if r["source_id"] != s])
        per[s] = {"n": t["n"], "edits": t["edits"], "ref_chars": t["ref_chars"], "cer": t["cer"], "wer": t["wer"]}
    cers = [v["cer"] for v in per.values()]
    return {"cluster_unit": CLUSTER_UNIT, "n_clusters": len(srcs), "per_source_excluded": per, "cer_min": min(cers), "cer_max": max(cers),
            "method": "metric recomputed from summed edits and reference characters with one source document removed; not an average of per-source CERs"}


def cluster_bootstrap(rows: list, cluster_key: str = CLUSTER_UNIT, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """95 % percentile interval of the micro CER, resampling whole clusters with replacement. Deterministic for a given list."""
    clusters = sorted({r[cluster_key] for r in rows})
    e = {c: 0 for c in clusters}
    ch = {c: 0 for c in clusters}
    for r in rows:
        e[r[cluster_key]] += editdistance.eval(r["ref"], r["hyp"])
        ch[r[cluster_key]] += len(r["ref"])
    rng, k, vals = random.Random(seed), len(clusters), []
    for _ in range(reps):
        pick = [clusters[rng.randrange(k)] for _ in range(k)]
        vals.append(sum(e[c] for c in pick) / max(1, sum(ch[c] for c in pick)))
    vals.sort()
    return {"interval": [vals[int(0.025 * reps)], vals[min(reps - 1, int(0.975 * reps))]], "n_clusters": k, "cluster_unit": cluster_key,
            "replicates": reps, "seed": seed}


def uncertainty(rows: list, min_clusters: int = MIN_CLUSTERS_FOR_CI) -> dict:
    """The uncertainty statement of a group under D-01 Option 2: LOSO range; a source-cluster interval only from min_clusters
    sources up, and then labelled indicative only; otherwise a recorded refusal. The region bootstrap is kept as a labelled diagnostic."""
    srcs = _require_sources(rows)
    out = {"cluster_unit": CLUSTER_UNIT, "n_clusters": len(srcs), "loso": loso(rows),
           "cer_95ci_region_bootstrap_diagnostic": {"interval": bootstrap_cer(rows), "indicative_only": True, "unit": "region", "label": REGION_DIAGNOSTIC_LABEL}}
    if len(srcs) >= min_clusters:
        cb = cluster_bootstrap(rows)
        out["source_cluster_interval"] = {"cer_95ci_indicative_only": cb["interval"], "indicative_only": True, "n_clusters": cb["n_clusters"],
                                          "replicates": cb["replicates"], "seed": cb["seed"], "min_clusters_for_ci": min_clusters, "label": CLUSTER_INTERVAL_LABEL}
    else:
        out["source_cluster_interval"] = {"refused": f"{len(srcs)} source documents, fewer than MIN_CLUSTERS_FOR_CI = {min_clusters} "
                                                     f"({MIN_CLUSTERS_STATUS}); no confidence interval is emitted. The leave-one-source-out range is reported instead.",
                                          "n_clusters": len(srcs), "min_clusters_for_ci": min_clusters}
    return out


def design_block(prov: dict) -> dict:
    """The design fields metrics.json carries beside the numbers, so the numbers are never read without them."""
    return {k: prov[k] for k in ("estimand", "cluster_unit", "n_clusters", "n_pages", "n_sources", "selection_design", "min_clusters_for_ci",
                                 "min_clusters_for_ci_status", "implementation_version")}


def by_source(metrics: dict, table: dict) -> dict:
    """source_id -> title, pages and the metrics of its 'source:<id>' group (primary set)."""
    out = {}
    for p, s in sorted(table.items()):
        d = out.setdefault(s["source_id"], {"source_title": s["source_title"], "pages": []})
        d["pages"].append(p)
    for sid, d in out.items():
        d["metrics"] = metrics.get(f"source:{sid}")
    return out


def uncertainty_sentence(m: dict) -> str:
    """The headline under D-01 Option 2: the point estimate, the number of sources, the LOSO range. No interval."""
    if not m:
        return "No region is in this set."
    lo = m.get("loso") or {}
    s = f"CER {fmt_pct(m['cer'])} over {m['n']} regions from {m.get('n_clusters', 'n/a')} source documents"
    if "cer_min" in lo:
        s += f"; removing any single source document moves it to between {fmt_pct(lo['cer_min'])} and {fmt_pct(lo['cer_max'])}"
    s += (". This describes these verified regions of these pages (a purposive convenience sample); it is not an estimate for Tamil print in general, "
          "and no confidence interval is given.")
    sci = m.get("source_cluster_interval") or {}
    if "refused" in sci:
        s += f" Source-cluster interval: refused ({sci['n_clusters']} sources, minimum {sci['min_clusters_for_ci']}, {MIN_CLUSTERS_STATUS})."
    return s


def source_section(metrics: dict, table: dict) -> str:
    """REPORT.md table: one row per source document of the primary set, with its pages, and the LOSO value without it."""
    lo = (metrics.get("primary") or {}).get("loso") or {}
    L = ["| source_id | Source document | Pages | Regions | CER | WER | CER without this source |", "|---|---|---|---|---|---|---|"]
    for sid, d in by_source(metrics, table).items():
        m = d["metrics"] or {}
        wo = (lo.get("per_source_excluded") or {}).get(sid)
        L.append(f"| `{sid}` | {d['source_title']} | {', '.join(d['pages'])} | {m.get('n', 0)} | {fmt_pct(m.get('cer'))} | {fmt_pct(m.get('wer'))} | "
                 f"{fmt_pct(wo['cer']) if wo else 'n/a'} |")
    return "\n".join(L) + "\n\n" + SOURCE_ID_RULE + ".\n"


# ---- provenance of a run (CH-07) ---------------------------------------------------------------------------------------------
SELECTION_DESIGN = "purposive convenience sample; no page-level frame"
ESTIMAND = ("D-01 Option 2 (owner decision 2026-10-09): finite-population descriptive. Micro-CER (sum of edits / sum of reference "
            "characters) over the primary set of the verified regions of these development pages, treated as a finite set, not a sample; "
            "reported with per-page and per-source tables and a leave-one-source-out range; no headline population-level confidence interval.")


def page_sources(pages_dir: Path) -> dict:
    """page_id -> source title, or None where the page table records none. source_table() turns these into source_ids."""
    pages_dir = Path(pages_dir).resolve()
    if pages_dir == (DEV / "original").resolve():
        with open(DEV / "manifests" / "development_pages_manifest.csv", encoding="utf-8", newline="") as f:
            return {r["page_id"]: (r.get("source_title") or "").strip() or None for r in csv.DictReader(f)}
    spec = json.loads((pages_dir / "pages.json").read_text(encoding="utf-8"))
    return {k: (v.get("source_title") or "").strip() or None for k, v in spec.items()}


def provenance(mode: str, records: list, pages_dir: Path, verified_dir: Path, out_dir) -> dict:
    """The CH-07 fields of run_record.json. A value that cannot be established is null, with the reason in provenance_notes;
    nothing is filled in by guess. A page without a source title stops the run (source_table)."""
    notes = {}
    used = sorted({r["page_id"] for r in records})
    sources = source_table(pages_dir, used)
    n_sources = len({s["source_id"] for s in sources.values()}) if used else None
    queue_sha = None
    try:
        inputs = json.loads((Path(verified_dir) / "integrity.json").read_text(encoding="utf-8")).get("inputs_sha256") or {}
        hits = sorted({h for k, h in inputs.items() if k.replace("\\", "/").endswith("queue.json")})
        queue_sha = hits[0] if len(hits) == 1 else None
        if len(hits) != 1:
            notes["queue_sha256"] = f"the verified set's integrity.json records {len(hits)} queue hashes; exactly one is required"
    except Exception as ex:
        notes["queue_sha256"] = f"the verified set's integrity.json could not be read ({type(ex).__name__})"
    out = {"queue_sha256": queue_sha, "queue_sha256_source": "inputs_sha256 of the verified set's integrity.json (the queue the set was frozen from)"}
    if mode == "real":
        out["queue_sha256_on_disk"] = sha_file(devguard.QUEUE) if devguard.QUEUE.is_file() else None
    design = devguard.QUEUE.parent / "queue_design.json"
    queue_design_sha = sha_file(design) if mode == "real" and design.is_file() else None
    if queue_design_sha is not None:                             # a stale sidecar is refused, never recorded (follow-up of Stage 2)
        try:
            recorded = json.loads(design.read_text(encoding="utf-8"))["derived_from"]["queue_sha256"]
        except Exception as ex:
            raise NotReady(f"Refused: {design} cannot be read as a queue design ({type(ex).__name__}).")
        actual = {h for h in (queue_sha, out.get("queue_sha256_on_disk")) if h}
        if not actual or any(recorded != h for h in actual):
            raise NotReady(f"Refused: {design.name} was derived from queue {recorded}, but the queue of this run is "
                           f"{', '.join(sorted(actual)) or 'unknown'}. A design record of another queue is never attached to a run.")
    if queue_design_sha is None:
        notes["queue_design_sha256"] = ("queue_design.json does not exist (CH-02, blueprint Stage 2, not implemented)" if mode == "real"
                                        else "fixture run: no queue design sidecar")
    import development_decision as dec                                # here, not at the top: development_decision imports this module
    rules_path = dec.RULES_PATH
    if mode != "real" and out_dir is not None and (Path(out_dir) / "DECISION_RULES.json").is_file():
        rules_path = Path(out_dir) / "DECISION_RULES.json"
    rules, rules_sha, rules_src = dec.load_rules(rules_path)
    approval = rules.get("approval") if isinstance(rules.get("approval"), dict) else None
    if approval is None:
        notes["decision_rules_approved"] = ("the rules carry no machine-readable approval block (CH-06 not implemented); status text: "
                                            + str(rules.get("status", "")))
    out.update({"estimand": ESTIMAND, "cluster_unit": CLUSTER_UNIT, "n_clusters": n_sources, "n_pages": len(used), "n_sources": n_sources,
                "source_mapping": sources, "source_id_rule": SOURCE_ID_RULE, "min_clusters_for_ci": MIN_CLUSTERS_FOR_CI,
                "min_clusters_for_ci_status": MIN_CLUSTERS_STATUS, "owner_decisions": OWNER_DECISIONS,
                "selection_design": SELECTION_DESIGN, "queue_design_sha256": queue_design_sha, "decision_rules_sha256": rules_sha,
                "decision_rules_source": rules_src, "decision_rules_approved": bool(approval and approval.get("approved") is True),
                "implementation_version": IMPLEMENTATION_VERSION, "provenance_notes": notes})
    return out


def provenance_section(record: dict) -> str:
    """REPORT.md text of the CH-07 fields. The selection design is printed verbatim."""
    na = lambda v: "n/a" if v is None else v
    L = ["| | |", "|---|---|",
         f"| Selection design | {record['selection_design']} |",
         f"| Estimand | {record['estimand']} |",
         f"| Cluster unit | {record['cluster_unit']} |",
         f"| Pages / sources / clusters | {record['n_pages']} / {na(record['n_sources'])} / {na(record['n_clusters'])} |",
         f"| Queue SHA-256 | `{record['queue_sha256']}` |",
         f"| Queue design SHA-256 | `{record['queue_design_sha256']}` |",
         f"| Decision rules SHA-256 | `{record['decision_rules_sha256']}` ({record['decision_rules_source']}) |",
         f"| Decision rules approved by the owner | {record['decision_rules_approved']} |",
         f"| Minimum source documents for an interval | {record['min_clusters_for_ci']} ({record['min_clusters_for_ci_status']}) |",
         f"| Owner decisions applied | " + "; ".join(f"{k}: {v}" for k, v in record["owner_decisions"].items()) + " |",
         f"| Implementation version | {record['implementation_version']} |"]
    if record.get("provenance_notes"):
        L.append("\nNot established, and why:\n")
        L += [f"- `{k}`: {v}" for k, v in record["provenance_notes"].items()]
    return "\n".join(L) + "\n"


# ---- sample accounting (CH-08) -------------------------------------------------------------------------------------------------
DISPOSITIONS = ("scored", "ineligible_not_usable", "ineligible_non_tamil", "ineligible_unreadable", "ineligible_ambiguous", "ineligible_unit_dispute",
                "excluded_invalid_geometry", "excluded_crop_refused", "excluded_too_small", "scored_empty_prediction", "scored_non_encodable")
SCORED_DISPOSITIONS = ("scored", "scored_empty_prediction", "scored_non_encodable")
DENOMINATOR_RULE = ("Every verified record has exactly one disposition. The primary figure's denominator is the three scored dispositions; a "
                    "non-encodable reference stays in it (scored_non_encodable), as does a region read as nothing (scored_empty_prediction). "
                    "A record that is both is counted as scored_non_encodable. Usable regions of another script are ineligible for the "
                    "primary figure but are still scored in the all-usable and per-script groups.")


def eligibility(r: dict):
    """The ineligible disposition of a verified record, or None if it is in the primary set. Agrees with groups_of()."""
    if not r["usable_as_verified_reference"]:
        if r["final_state"] == "UNREADABLE":
            return "ineligible_unreadable"
        if r["final_state"] == "AMBIGUOUS":
            return "ineligible_ambiguous"
        if (r.get("not_usable_because") or "").startswith("unit correction"):
            return "ineligible_unit_dispute"
        return "ineligible_not_usable"
    if r.get("script") != "tamil":
        return "ineligible_non_tamil"
    return None


def disposition(r: dict, row, page=None) -> tuple:
    """(disposition, detail) of one verified record, given its result row in one setting (None if the setting made none)."""
    d = eligibility(r)
    if d:
        return d, r.get("not_usable_because") or ""
    if row is None:
        raise AssertionError(f"{r['verification_id']} is in the primary set but has no result row")
    if not row["scored"]:
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        off_page = page is not None and (x2 <= 0 or y2 <= 0 or x1 >= page.width or y1 >= page.height)
        if x2 <= x1 or y2 <= y1 or off_page:
            return "excluded_invalid_geometry", row.get("reason", "")
        if row.get("reason") == TOO_SMALL_REASON:
            return "excluded_too_small", row["reason"]
        return "excluded_crop_refused", row.get("reason", "")
    if r.get("encoding_status") == "non_encodable":
        return "scored_non_encodable", row.get("segmentation_status") or ""
    if row["hyp"] == "":
        return "scored_empty_prediction", row.get("segmentation_status") or ""
    return "scored", row.get("segmentation_status") or ""


def ledger(records: list, rows: list, pages: dict = None) -> dict:
    """One setting: a row per verified record with its disposition, the counts of every disposition, and the invariant
    len(records) == sum of all counts, asserted. No record may be absent."""
    by_id = {}
    for x in rows:
        if x["verification_id"] in by_id:
            raise AssertionError(f"{x['verification_id']} has two result rows in one setting")
        by_id[x["verification_id"]] = x
    stray = set(by_id) - {r["verification_id"] for r in records}
    if stray:
        raise AssertionError(f"result rows for records not in the verified set: {sorted(stray)[:5]}")
    counts, out = {d: 0 for d in DISPOSITIONS}, []
    for r in records:
        row = by_id.get(r["verification_id"])
        d, detail = disposition(r, row, (pages or {}).get(r["page_id"]))
        if d not in DISPOSITIONS:
            raise AssertionError(f"disposition {d!r} is not in the closed list")
        counts[d] += 1
        out.append({"verification_id": r["verification_id"], "region_id": r["region_id"], "page_id": r["page_id"], "source_id": r.get("source_id"),
                    "final_state": r["final_state"],
                    "script": r.get("script"), "encoding_status": r.get("encoding_status"), "queue_part": r.get("queue_part"), "disposition": d,
                    "in_primary_denominator": d in SCORED_DISPOSITIONS, "scored_outside_primary": bool(d == "ineligible_non_tamil" and row and row.get("scored")),
                    "empty_prediction": bool(row and row.get("scored") and row.get("hyp") == ""), "detail": detail})
    n = len(records)
    if len({r["verification_id"] for r in records}) != n or len(out) != n or sum(counts.values()) != n:
        raise AssertionError(f"accounting invariant broken: {n} verified records, {len(out)} ledger rows, {sum(counts.values())} counted")
    return {"verified_records": n, "counts": counts, "primary_denominator": sum(counts[d] for d in SCORED_DISPOSITIONS), "rows": out}


def accounting(experiment: str, records: list, settings: dict, pages: dict = None, extra: dict = None) -> dict:
    """accounting.json of a run: one ledger per setting (E1: two crop settings; E2: one; E3: one per margin)."""
    return {"schema": "accounting-1", "experiment": experiment, "implementation_version": IMPLEMENTATION_VERSION, "verified_records": len(records),
            "dispositions": list(DISPOSITIONS), "denominator_rule": DENOMINATOR_RULE, "primary_rule": PRIMARY_RULE,
            "settings": {name: ledger(records, rows, pages) for name, rows in settings.items()}, **(extra or {})}


def check_denominator(led: dict, primary_metrics) -> None:
    """The ledger's scored count must equal the n of the primary metrics computed from the same rows."""
    n = (primary_metrics or {}).get("n", 0)
    if led["primary_denominator"] != n:
        raise AssertionError(f"accounting says {led['primary_denominator']} primary regions were scored, the metrics say {n}")


def accounting_section(acc: dict) -> str:
    """REPORT.md table: every disposition (zeros included) for every setting, and the total, which equals the verified records."""
    names = list(acc["settings"])
    L = ["| Disposition | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for d in DISPOSITIONS:
        L.append(f"| {d} | " + " | ".join(str(acc["settings"][s]["counts"][d]) for s in names) + " |")
    L.append("| **Primary denominator (the three scored dispositions)** | " + " | ".join(str(acc["settings"][s]["primary_denominator"]) for s in names) + " |")
    L.append("| **Total = verified records** | " + " | ".join(str(sum(acc["settings"][s]["counts"].values())) for s in names) + " |")
    return "\n".join(L) + "\n\n" + DENOMINATOR_RULE + " Per-record dispositions are in `accounting.json`.\n"


# ---- output ----------------------------------------------------------------------------------------------------------------
def run_dir(base, name: str, manifest_sha: str, config: dict) -> Path:
    """Deterministic folder of a run: <base>/<name>/<verified-set hash 12>_<config hash 12>."""
    return Path(base) / name / f"{manifest_sha[:12]}_{sha_json(config)[:12]}"


def write_new(path: Path, text: str) -> None:
    """Create a file; never overwrite one. A finished run is a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def lines(objs) -> str:
    return "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs)


def code_hashes() -> dict:
    names = ["tools/development_eval_core.py", "tools/development_e1.py", "tools/development_e2.py", "tools/development_e3.py", "pageocr/engine.py", "pageocr/geometry.py",
             "pageocr/pipeline.py", "pageocr/layout.py", "pageocr/devguard.py", "src/preprocess.py", "src/charset.py", "src/metrics.py", "src/model/crnn_v2.py"]
    return {n: sha_file(ROOT / n) for n in names if (ROOT / n).is_file()}


def environment(engine=None) -> dict:
    import numpy, torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__, "core_version": CORE_VERSION}
    if engine is not None:
        info["engine"] = engine.info()
    return info


def fmt_pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.2f} %"


def metrics_row(name: str, m: dict) -> str:
    lo = m.get("loso") or {}
    rng = f"{fmt_pct(lo['cer_min'])} to {fmt_pct(lo['cer_max'])} ({lo['n_clusters']} sources)" if "cer_min" in lo else ""
    return (f"| {name} | {m['n']} | {fmt_pct(m['cer'])} | {rng} | {fmt_pct(m['wer'])} | {fmt_pct(m['word_acc'])} | "
            f"{m['len_ratio']:.3f} | {'n/a' if m['mean_confidence'] is None else format(m['mean_confidence'], '.3f')} | {fmt_pct(m['width_clipped_rate'])} |")


# No interval column: under D-01 Option 2 the CER is a description of these regions; the LOSO range shows its dependence on single sources.
METRICS_HEADER = ("| Group | Regions | CER | CER with one source document left out (range) | WER | Word accuracy | Length ratio | Mean confidence (uncalibrated) | Width-clipped |\n"
                  "|---|---|---|---|---|---|---|---|---|")
