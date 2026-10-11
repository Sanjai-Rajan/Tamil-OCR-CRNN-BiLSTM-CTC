"""
End-to-end test of Full-page OCR: the REAL Streamlit server (app.py) and the real model, driven in headless Microsoft
Edge (tests/fullpage_browser_smoke.js):

    a page uploaded to the word tab is refused (the reported problem) -> a word is still read ->
    Full-page OCR: upload page -> validate -> segment -> word boundaries shown -> recognize the word crops ->
    page text rebuilt -> word table inspected -> TXT / JSON / CSV downloaded and read back from disk -> reset

    .venv\\Scripts\\python -B tests\\fullpage_browser_smoke.py [OUT_DIR]

The page is a GENERATED two-column Tamil page with a heading (tests/fullpage_fixtures.py), written to a temporary
folder. The server listens on localhost:8781 only. Screenshots and downloads go to OUT_DIR (default: a new temporary
folder). Prints a JSON summary; exit 0 only if every check holds; exit 3 if Edge or Node is missing (not a pass).
Nothing is written inside the project.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import fullpage_fixtures as fx  # noqa: E402

PORT = 8781
BASE = f"http://localhost:{PORT}"


def reachable(host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, PORT), timeout=1.5):
            return True
    except OSError:
        return False


def main():
    if not fx.RENDERING:
        sys.exit(fx.SKIP_REASON)
    if shutil.which("node") is None:
        print(json.dumps({"skipped": "Node.js not found", "passed": False}))
        sys.exit(3)
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="fullpage_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fullpage_smoke_files_"))
    page = fx.make_tamil_page(seed=5, columns=2, heading=True)
    word_text = "புத்தகம்"
    import numpy as np
    word = (255 - np.pad(fx.render_word(word_text), 10).astype(np.float32) * 0.85).astype(np.uint8)
    files = {"page": str(work / "tamil_page.png"), "page_text": page.text, "page_words": len(page.words),
             "word": str(work / "word.png"), "word_text": word_text}
    Path(files["page"]).write_bytes(fx.png(page.gray))
    Path(files["word"]).write_bytes(fx.png(word))
    spec = work / "files.json"
    spec.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")

    if reachable():
        sys.exit(f"port {PORT} is already in use; stop that program first")
    log = open(out_dir / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "app.py", "--server.port", str(PORT)], cwd=ROOT,
                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    res = {"out_dir": str(out_dir), "page": {"size": list(page.gray.shape[::-1]), "words": len(page.words), "blocks": 3}, "checks": {}}
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
        r = subprocess.run(["node", str(HERE / "fullpage_browser_smoke.js"), BASE, str(spec), str(out_dir)],
                           capture_output=True, text=True, encoding="utf-8", timeout=900)
        try:
            browser = json.loads(r.stdout[r.stdout.index("{"):])
        except ValueError:
            browser = {"error": f"no JSON from the browser driver (exit {r.returncode}): {r.stderr[-400:]}"}
        res["browser"] = browser
        res["checks"].update({f"browser.{k}": v for k, v in browser.get("checks", {}).items()})
        res["checks"]["browser_ran_to_the_end"] = "error" not in browser and "skipped" not in browser and bool(browser.get("checks"))
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
