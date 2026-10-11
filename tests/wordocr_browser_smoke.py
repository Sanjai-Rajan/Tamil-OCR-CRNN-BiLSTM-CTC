"""
End-to-end smoke test of the word frontend: the REAL Streamlit server (app.py) and the real model, driven in headless
Microsoft Edge (tests/wordocr_browser_smoke.js): upload -> Run OCR -> displayed text -> real browser downloads
(TXT, JSON, CSV) read back from disk -> a corrupted file -> a batch with one failing file.

    .venv\\Scripts\\python -B tests\\wordocr_browser_smoke.py [OUT_DIR]

Inputs are synthetic words rendered by the project's renderer (tests/wordocr_fixtures.py), written to a temporary
folder. The server listens on localhost:8765 only; the test also checks that it is NOT reachable on the machine's other
addresses. Screenshots and downloads go to OUT_DIR (default: a new temporary folder). Prints a JSON summary; exit 0
only if every check holds. Nothing is written inside the project.
"""
import json
import os
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
import wordocr_fixtures as fx  # noqa: E402

PORT = 8765
BASE = f"http://localhost:{PORT}"


def other_addresses() -> list[str]:
    try:
        return sorted({a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)} - {"127.0.0.1"})
    except OSError:
        return []


def reachable(host: str) -> bool:
    try:
        with socket.create_connection((host, PORT), timeout=1.5):
            return True
    except OSError:
        return False


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="wordocr_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wordocr_smoke_files_"))
    files = {"word": str(work / "vanakkam.png"), "word_text": "வணக்கம்", "corrupt": str(work / "corrupt.png"),
             "batch": [str(work / "b1_tamil.png"), str(work / "b2_broken.jpg"), str(work / "b3_kanini.jpg")],
             "batch_texts": ["தமிழ்", None, "கணினி"]}
    Path(files["word"]).write_bytes(fx.word_png("வணக்கம்"))
    Path(files["corrupt"]).write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(200))
    Path(files["batch"][0]).write_bytes(fx.word_png("தமிழ்"))
    Path(files["batch"][1]).write_bytes(b"\xff\xd8\xff\xe0 truncated")
    Path(files["batch"][2]).write_bytes(fx.word_png("கணினி", fmt="JPEG", quality=95))
    spec = work / "files.json"
    spec.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")

    if reachable("127.0.0.1"):
        sys.exit(f"port {PORT} is already in use; stop that program first")
    log = open(out_dir / "server.log", "w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "app.py", "--server.port", str(PORT)], cwd=ROOT,
                              stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    res = {"out_dir": str(out_dir), "checks": {}}
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
        page = urlopen(BASE, timeout=10).read().decode("utf-8", "replace")
        res["checks"]["serves_the_streamlit_page"] = "<div id=\"root\">" in page or "id=\"root\"" in page
        others = other_addresses()
        res["other_addresses_tried"] = others
        res["checks"]["not_reachable_on_other_addresses"] = not any(reachable(a) for a in others)
        r = subprocess.run(["node", str(HERE / "wordocr_browser_smoke.js"), BASE, str(spec), str(out_dir)],
                           capture_output=True, text=True, encoding="utf-8", timeout=600)
        browser = json.loads(r.stdout.strip().splitlines()[-1])
        res["browser"] = browser
        res["checks"].update({f"browser.{k}": v for k, v in browser.get("checks", {}).items()})
        if "error" in browser or "skipped" in browser:
            res["checks"]["browser_ran"] = False
    finally:
        # the .venv launcher starts a child interpreter: stop the whole process tree, then wait for the port to close
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)], capture_output=True)
        else:
            server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        t1 = time.time()
        while reachable("127.0.0.1") and time.time() - t1 < 20:
            time.sleep(0.5)
        res["checks"]["server_stopped"] = server.poll() is not None and not reachable("127.0.0.1")
    res["passed"] = all(res["checks"].values()) and bool(res["checks"])
    (out_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=True, indent=1))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
