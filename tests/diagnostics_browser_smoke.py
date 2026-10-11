"""
End-to-end test of the diagnostic panel of Full-page OCR: the REAL Streamlit server (app.py) and the real model, driven
in headless Microsoft Edge (tests/diagnostics_browser_smoke.js):

    Full-page OCR: experimental options and diagnostic mode are off -> page read with diagnostics off (no panel) ->
    diagnostic mode on (OCR text unchanged) -> metrics, both overlays, sortable region table, warning filter,
    single-region inspector, line table -> diagnostic bundle (ZIP) downloaded and checked here -> manual-crop
    comparison form filled with one generator-labelled box -> comparison JSON downloaded and checked here ->
    rule removal switched on and off again from the interface

    .venv\\Scripts\\python -B tests\\diagnostics_browser_smoke.py [OUT_DIR]

The page is a GENERATED ruled synthetic page (dev_tools/synthetic_conditions.py, condition ruled_below, seed 7000, a seed
used by no experiment). The reference typed into the comparison form is the GENERATOR's text of that word, never a model
output. The "Save bundle to diagnostic_runs/" button is checked for presence but NOT clicked (it writes inside the
project; save_bundle is covered by tests/test_diagnostics.py). The server listens on localhost:8793 only. Screenshots and
downloads go to OUT_DIR (default: a new temporary folder). Prints a JSON summary; exit 0 only if every check holds;
exit 3 if Edge or Node is missing (not a pass). Nothing is written inside the project.
"""
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "dev_tools"))
import fullpage_fixtures as fx  # noqa: E402

PORT = 8793
BASE = f"http://localhost:{PORT}"
CONDITION, SEED, SAMPLE_WORD = "ruled_below", 7000, 2


def reachable(host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, PORT), timeout=1.5):
            return True
    except OSError:
        return False


def check_bundle(path: Path, page_sha: str, ui: dict) -> dict:
    """Open the downloaded ZIP and check it against itself and against what the browser showed."""
    z = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
    names = set(z.namelist())
    meta = json.loads(z.read("metadata.json"))
    sums = {ln.split(" *", 1)[1]: ln.split(" ", 1)[0] for ln in z.read("SHA256SUMS.txt").decode("ascii").splitlines() if ln.strip()}
    crops = sorted(n for n in names if n.startswith("crops/"))
    inputs_ = [n for n in names if n.startswith("input/")]
    regions_csv = z.read("regions.csv").decode("utf-8").lstrip("﻿").strip().splitlines()
    c = {
        "bundle_has_required_files": {"metadata.json", "regions.csv", "lines.json", "page_text.txt", "overlay_ocr_frame.png",
                                      "overlay_original_frame.png", "SUMMARY.md", "SHA256SUMS.txt"} <= names,
        "bundle_checksums_hold": set(sums) == names - {"SHA256SUMS.txt"} and all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in sums.items()),
        "bundle_one_crop_per_region": len(crops) == len(meta["regions"]) == ui.get("metric_regions") and len(regions_csv) == len(crops) + 1,
        "bundle_page_text_equals_ui": z.read("page_text.txt").decode("utf-8").strip() == (ui.get("page_text_after_diag") or "").strip(),
        "bundle_input_is_the_uploaded_page": len(inputs_) == 1 and hashlib.sha256(z.read(inputs_[0])).hexdigest() == page_sha == meta["input"]["sha256"],
        "bundle_records_official_checkpoint": meta["model"].get("checkpoint_sha256") == "fb29131ed71fe0c89229497d0171b1ff2de4e1e581b4d8c1e09d6fbbaf62d5f1",
    }
    return {"checks": c, "observed": {"files": len(names), "crops": len(crops), "regions_with_any_warning": meta.get("regions_with_any_warning"),
                                      "warning_counts": meta.get("warning_counts")}}


def check_comparison(path: Path, sample: dict) -> dict:
    rep = json.loads(path.read_text(encoding="utf-8"))
    s = rep["samples"]
    one = s[0] if len(s) == 1 else {}
    c = {
        "comparison_one_sample": len(s) == 1,
        "comparison_box_as_entered": list(one.get("box") or []) == list(sample["box"]),
        "comparison_reference_as_entered": one.get("reference") == sample["text"],
        "comparison_reference_confirmed_and_scored": one.get("reference_confirmed") is True and rep["summary"].get("with_confirmed_reference") == 1
                                                     and rep["summary"].get("A_manual_words_correct") is not None,
        "comparison_both_conditions_read": "prediction_manual" in one and "prediction_pipeline" in one and not one.get("error_manual"),
    }
    return {"checks": c, "observed": {"summary": rep["summary"], "A": one.get("prediction_manual"), "B": one.get("prediction_pipeline"),
                                      "pipeline_region": one.get("pipeline_region"), "reference": sample["text"]}}


def main():
    if not fx.RENDERING:
        sys.exit(fx.SKIP_REASON)
    if shutil.which("node") is None:
        print(json.dumps({"skipped": "Node.js not found", "passed": False}))
        sys.exit(3)
    import synthetic_conditions as sc
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="diag_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="diag_smoke_files_"))
    page = sc.make(CONDITION, SEED)
    data = fx.png(page.gray)
    text, box = page.words[SAMPLE_WORD]
    sample = {"id": "S01", "box": [int(v) for v in box], "text": text}
    files = {"page": str(work / f"{CONDITION}_{SEED}.png"), "page_words": len(page.words), "sample": sample}
    Path(files["page"]).write_bytes(data)
    spec = work / "files.json"
    spec.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")

    if reachable():
        sys.exit(f"port {PORT} is already in use; stop that program first")
    log = open(out_dir / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "app.py", "--server.port", str(PORT), "--server.headless", "true"],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    res = {"out_dir": str(out_dir), "page": {"condition": CONDITION, "seed": SEED, "size": list(page.gray.shape[::-1]), "words": len(page.words),
                                             "sha256": hashlib.sha256(data).hexdigest()}, "checks": {}}
    try:
        t0 = time.time()
        while time.time() - t0 < 90:
            try:
                if urlopen(f"{BASE}/_stcore/health", timeout=2).read().decode().strip() == "ok":
                    break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError("the server stopped; see server.log")
                time.sleep(0.5)
        res["server_start_seconds"] = round(time.time() - t0, 1)
        res["checks"]["server_health_ok"] = urlopen(f"{BASE}/_stcore/health", timeout=5).read().decode().strip() == "ok"
        r = subprocess.run(["node", str(HERE / "diagnostics_browser_smoke.js"), BASE, str(spec), str(out_dir)],
                           capture_output=True, text=True, encoding="utf-8", timeout=900)
        try:
            browser = json.loads(r.stdout[r.stdout.index("{"):])
        except ValueError:
            browser = {"error": f"no JSON from the browser driver (exit {r.returncode}): {r.stderr[-400:]}"}
        res["browser"] = browser
        res["checks"].update({f"browser.{k}": v for k, v in browser.get("checks", {}).items()})
        res["checks"]["browser_ran_to_the_end"] = "error" not in browser and "skipped" not in browser and bool(browser.get("checks"))
        dl = out_dir / "downloads"
        z = sorted(dl.glob("diagnostic_*.zip")) if dl.is_dir() else []
        if z:
            b = check_bundle(z[0], res["page"]["sha256"], browser.get("observed", {}))
            res["bundle"] = b["observed"]
            res["checks"].update(b["checks"])
        else:
            res["checks"]["bundle_downloaded"] = False
        cj = sorted(dl.glob("crop_comparison_*.json")) if dl.is_dir() else []
        if cj:
            c = check_comparison(cj[0], sample)
            res["comparison"] = c["observed"]
            res["checks"].update(c["checks"])
        else:
            res["checks"]["comparison_downloaded"] = False
        res["checks"]["nothing_saved_inside_project"] = not (ROOT / "diagnostic_runs").exists()
    finally:
        if os.name == "nt":                      # the .venv launcher starts a child interpreter: stop the whole tree
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)], capture_output=True)
        else:
            server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        t1 = time.time()
        while reachable() and time.time() - t1 < 20:
            time.sleep(0.5)
        res["checks"]["server_stopped"] = server.poll() is not None and not reachable()
        shutil.rmtree(work, ignore_errors=True)
    res["passed"] = all(v is True for v in res["checks"].values()) and bool(res["checks"])
    (out_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=True, indent=1))
    skipped = "skipped" in res.get("browser", {})
    sys.exit(0 if res["passed"] else (3 if skipped else 1))


if __name__ == "__main__":
    main()
