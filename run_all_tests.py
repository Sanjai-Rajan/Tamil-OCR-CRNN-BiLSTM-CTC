"""
Run every test suite of this release, one after another, and print one summary table.

    python -B run_all_tests.py                    all suites
    python -B run_all_tests.py --list             show the suites and commands, run nothing
    python -B run_all_tests.py --only guard word  run the named suites only
    python -B run_all_tests.py --no-browser       leave out the three browser suites (they need Microsoft Edge and Node.js)
    python -B run_all_tests.py --log out.txt      also write the full output to a file

Use the Python of the environment described in docs/INSTALLATION.md (the one this script is started with runs the tests;
if a .venv folder exists next to this file, that one is used).

Trained weights are not distributed with this repository. If the registered default checkpoint (wordocr/checkpoints.py)
is missing, the test modules that load it (NEEDS_CHECKPOINT below) and the three browser suites are reported as NOT RUN,
naming the file, and every other module runs as usual. With the checkpoint in place, everything runs.

All inputs are generated in memory by the test fixtures. Nothing is downloaded, no dataset is needed, and every suite
writes only to temporary folders. The three browser suites start their own local Streamlit server on ports 8765, 8781
and 8793 (127.0.0.1) and stop it again; they do not touch a server you started yourself on another port.

Nothing here edits a test, skips a failing test, or changes an assertion. A suite that cannot be started is reported
as NOT RUN, never as passed.

Exit code: 0 if every suite ran and passed, 1 if any failed, 2 if any could not be run.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
UNIT = ["-B", "-m", "unittest"]


def python() -> str:
    return str(VENV) if VENV.is_file() else sys.executable


def files(pattern: str) -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob(pattern))


def browser_tools() -> str:
    """'' if Edge and Node.js are available, else what is missing."""
    missing = []
    if not shutil.which("node"):
        missing.append("Node.js")
    edge = [Path(os.environ.get(v, "")) / "Microsoft/Edge/Application/msedge.exe" for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    if not any(p.is_file() for p in edge) and not shutil.which("msedge"):
        missing.append("Microsoft Edge")
    return ", ".join(missing)


def suites(browser: bool = True) -> list[tuple[str, str, list[str] | None, str]]:
    """(name, description, command or None, reason if None) in run order."""
    py = python()
    out = [
        ("guard", "protected-data guard: fingerprints, redirected home folder, fail closed", [py] + UNIT + ["-v", "tests/test_protected_guard.py"], ""),
        ("pageocr", "recognizer engine, boxes, region app, segmentation", [py] + UNIT + ["discover", "-v", "-s", "tests", "-p", "test_pageocr_*.py"], ""),
        ("word", "word OCR service and app.py (Word OCR)", [py] + UNIT + ["-v"] + files("test_wordocr_*.py"), ""),
        ("fullpage", "full-page OCR service and app.py (Full-page OCR)", [py] + UNIT + ["-v"] + files("test_fullpage_*.py"), ""),
        ("diagnostics", "diagnostic mode, evidence export, manual-vs-pipeline crops, experimental options, angle-aware lines (workspace additions)",
         [py] + UNIT + ["-v", "tests/test_diagnostics.py", "tests/test_line_geometry.py", "tests/test_slant_lines.py"], ""),
        ("training_code", "model, loss, metrics, synthetic generator (minimal smoke test); split audit; metric recomputation; pre-training check",
         [py] + UNIT + ["-v", "tests/test_minimal.py", "tests/test_check_split.py", "tests/test_recompute_eval_metrics.py", "tests/test_pretrain_check.py"], ""),
    ]
    if browser:
        why = browser_tools()
        out += [
            ("word_browser", "word OCR: real Streamlit server + model in headless Edge (port 8765)", None if why else [py, "-B", "tests/wordocr_browser_smoke.py"], why and f"missing: {why}"),
            ("fullpage_browser", "full-page OCR: real Streamlit server + model in headless Edge (port 8781)", None if why else [py, "-B", "tests/fullpage_browser_smoke.py"], why and f"missing: {why}"),
            ("diag_browser", "diagnostic panel of full-page OCR: real Streamlit server + model in headless Edge (port 8793; workspace addition)",
             None if why else [py, "-B", "tests/diagnostics_browser_smoke.py"], why and f"missing: {why}"),
        ]
    return without_checkpoint(out) if not checkpoint_present() else out


# --- repository export: trained weights are not distributed --------------------------------------------------------------
# The test modules below load the registered default checkpoint (measured: run without it, each has tests that error or
# fail because the file is missing). When the file is absent they are reported as NOT RUN, naming the file; every other
# module still runs, unchanged. With the checkpoint in place, all suites run exactly as listed above. No test is edited.
NEEDS_CHECKPOINT = ("test_diagnostics.py", "test_fullpage_app.py", "test_fullpage_ocr.py", "test_line_geometry.py", "test_pageocr_engine.py",
                    "test_pageocr_v3_segmentation.py", "test_wordocr_app.py", "test_wordocr_service.py")
BROWSER_SUITES = ("word_browser", "fullpage_browser", "diag_browser")


def checkpoint_present() -> bool:
    sys.path.insert(0, str(ROOT))
    from wordocr import checkpoints
    return checkpoints.get(checkpoints.DEFAULT_ID).file().is_file()


def without_checkpoint(out: list) -> list:
    from wordocr import checkpoints
    path = checkpoints.get(checkpoints.DEFAULT_ID).path
    why = f"needs the registered checkpoint {path}, which is not distributed (docs/MODEL_CHECKPOINTS.md)"
    res = []
    for name, desc, cmd, reason in out:
        if cmd is None:
            res.append((name, desc, cmd, reason))
            continue
        if name in BROWSER_SUITES:
            res.append((name, desc, None, why))
            continue
        mods = files("test_pageocr_*.py") if "discover" in cmd else [c for c in cmd if c.startswith("tests/test_")]
        keep = [m for m in mods if Path(m).name not in NEEDS_CHECKPOINT]
        skipped = [Path(m).name for m in mods if Path(m).name in NEEDS_CHECKPOINT]
        if keep:
            res.append((name, desc + (f" [without {', '.join(skipped)}]" if skipped else ""), [cmd[0]] + UNIT + ["-v"] + keep, ""))
        if skipped:
            res.append((name + "*", f"{name}: {', '.join(skipped)}", None, why))
    return res


def counts(text: str) -> str:
    """A one-line count taken from the suite's own output (never computed by this script)."""
    m = re.findall(r"Ran (\d+) tests? in [\d.]+s\s+(OK.*|FAILED.*)", text)
    if m:
        return "; ".join(f"Ran {n}, {res.strip()}" for n, res in m)
    m = re.search(r'\n\s*"passed":\s*(true|false)\s*\n\}\s*$', text)
    if m:
        return f"passed={m.group(1)} (JSON summary)"
    tail = [ln for ln in text.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    return tail[0][:120]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--log")
    a = ap.parse_args()
    todo = [s for s in suites(not a.no_browser) if not a.only or s[0] in a.only]
    if a.list:
        for name, desc, cmd, why in todo:
            print(f"{name:18s} {desc}\n{'':18s} {' '.join(cmd) if cmd else 'NOT RUNNABLE: ' + why}")
        return 0
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    log, rows = [], []
    for name, desc, cmd, why in todo:
        if cmd is None:
            rows.append((name, "NOT RUN", "-", why))
            continue
        print(f"--- {name}: {' '.join(cmd)}", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = r.stdout + r.stderr
        log.append(f"===== {name} ({desc})\n$ {' '.join(cmd)}\nexit {r.returncode}\n{text}\n")
        rows.append((name, "PASS" if r.returncode == 0 else "FAIL", r.returncode, f"{counts(text)}  [{time.time() - t:.0f}s]"))
        print(f"    exit {r.returncode}: {rows[-1][3]}", flush=True)
    print("\nSUITE              RESULT   EXIT  COUNTS")
    for name, res, code, c in rows:
        print(f"{name:18s} {res:8s} {str(code):5s} {c}")
    if a.log:
        Path(a.log).write_text("".join(log) + "\n", encoding="utf-8")
    if any(r[1] == "FAIL" for r in rows):
        return 1
    return 2 if any(r[1] == "NOT RUN" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
