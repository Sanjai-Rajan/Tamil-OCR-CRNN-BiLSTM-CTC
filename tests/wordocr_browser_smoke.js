// Browser smoke test of the word frontend (app.py) in headless Microsoft Edge, driven over the DevTools protocol.
// Started by tests/wordocr_browser_smoke.py, which runs the real Streamlit server on localhost and prepares the files.
//
//     node tests/wordocr_browser_smoke.js BASE FILES_JSON OUT_DIR
//
// FILES_JSON: {"word": path, "word_text": "...", "corrupt": path, "batch": [paths], "batch_texts": [...]}.
// Real clicks and real file inputs; downloads are saved by the browser into OUT_DIR/downloads and read back.
// Prints one JSON object (the observations) as its last line. Exit 3 = Edge not found.
// Name resolution is refused for everything but 127.0.0.1 and localhost; every request the page makes is recorded.
"use strict";
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const [BASE, FILES_JSON, OUT] = process.argv.slice(2);
const F = JSON.parse(fs.readFileSync(FILES_JSON, "utf-8"));
const EDGE = ["C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"].find(p => fs.existsSync(p));
if (!EDGE) { console.log(JSON.stringify({ skipped: "Microsoft Edge not found" })); process.exit(3); }
const PORT = 9420;
const DL = path.join(OUT, "downloads");
fs.mkdirSync(DL, { recursive: true });
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "wordocr_smoke_edge_"));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const out = { checks: {}, requests: [], external_requests: [], console_errors: [], downloads: {} };
let edge = null;

async function main() {
  edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, "--window-size=1400,1300",
    "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--disable-background-networking",
    "--disable-component-update", "--disable-default-apps", "--disable-domain-reliability", "--disable-client-side-phishing-detection",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1, EXCLUDE localhost", "about:blank"], { stdio: "ignore" });
  let target = null;
  for (let i = 0; i < 80 && !target; i++) {
    await sleep(250);
    try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find(t => t.type === "page"); } catch (e) { /* not up */ }
  }
  if (!target) throw new Error("the browser did not start");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let seq = 0; const pending = new Map(); const listeners = [];
  ws.onmessage = m => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { const p = pending.get(d.id); pending.delete(d.id); d.error ? p.rej(new Error(JSON.stringify(d.error))) : p.res(d.result); }
    else if (d.method) listeners.forEach(f => f(d));
  };
  const cdp = (method, params = {}) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
  const js = async expr => (await cdp("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result.value;
  const waitFor = async (expr, ms = 60000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await js(expr)) return true; await sleep(250); } return false; };
  const bodyText = () => js("document.body.innerText");
  const clickButton = label => js(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(label)} && !b.disabled && b.offsetParent !== null); if (!b) return false; b.click(); return true; })()`);
  const buttonEnabled = label => `(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(label)} && b.offsetParent !== null); return !!b && !b.disabled; })()`;
  // a real mouse click at the centre of the smallest visible element whose text is exactly `label`
  const mouseClick = async label => {
    const r = await js(`(() => { const els = [...document.querySelectorAll('[role=tab], button, p, span, div')].filter(e => e.innerText && e.innerText.trim() === ${JSON.stringify(label)} && e.offsetParent !== null);
      if (!els.length) return null; els.sort((a, b) => a.getBoundingClientRect().width * a.getBoundingClientRect().height - b.getBoundingClientRect().width * b.getBoundingClientRect().height);
      const b = els[0]; b.scrollIntoView({block: "center"}); const q = b.getBoundingClientRect(); return {x: q.left + q.width / 2, y: q.top + q.height / 2}; })()`);
    if (!r) return false;
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await cdp("Input.dispatchMouseEvent", { type, x: r.x, y: r.y, button: "left", clickCount: 1 });
    return true;
  };
  const shot = async name => { const r = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }); fs.writeFileSync(path.join(OUT, name), Buffer.from(r.data, "base64")); };
  const setFiles = async (index, files) => {
    const doc = await cdp("DOM.getDocument", { depth: -1, pierce: true });
    const q = await cdp("DOM.querySelectorAll", { nodeId: doc.root.nodeId, selector: "input[type=file]" });
    if (q.nodeIds.length <= index) return false;
    await cdp("DOM.setFileInputFiles", { nodeId: q.nodeIds[index], files });
    return true;
  };
  let done = new Map();
  const download = async (label, name) => {
    const before = new Set(fs.readdirSync(DL));
    if (!await clickButton(label)) return null;
    const t = Date.now();
    while (Date.now() - t < 20000) {
      const now = fs.readdirSync(DL).filter(f => !before.has(f) && !f.endsWith(".crdownload") && !f.endsWith(".tmp"));
      if (now.length) { await sleep(300); const f = now[0]; out.downloads[name] = f; return fs.readFileSync(path.join(DL, f)); }
      await sleep(200);
    }
    return null;
  };

  listeners.push(ev => {
    if (ev.method === "Network.requestWillBeSent") {
      const u = new URL(ev.params.request.url);
      out.requests.push(u.protocol + "//" + u.host + u.pathname.split("/").slice(0, 3).join("/"));
      if (!["127.0.0.1", "localhost", "[::1]"].includes(u.hostname) && !["data:", "blob:"].includes(u.protocol)) out.external_requests.push(ev.params.request.url);
    }
    if (ev.method === "Runtime.exceptionThrown") out.console_errors.push(ev.params.exceptionDetails.text);
  });
  await cdp("Network.enable"); await cdp("Runtime.enable"); await cdp("Page.enable"); await cdp("DOM.enable");
  await cdp("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: DL });
  await cdp("Page.navigate", { url: BASE });

  // ---- startup
  out.checks.page_loads_and_model_ready = await waitFor("document.body.innerText.includes('Model ready')", 120000);
  const t0 = await bodyText();
  out.checks.scope_statement_shown = t0.includes("cannot transcribe") && t0.includes("one cropped Tamil word");
  out.checks.run_disabled_without_image = !(await js(buttonEnabled("Run OCR")));
  await shot("01_start.png");

  // ---- single image: upload -> run -> result -> downloads
  out.checks.single_file_set = await setFiles(0, [F.word]);
  out.checks.run_enabled_after_upload = await waitFor(buttonEnabled("Run OCR"), 30000);
  out.checks.preview_shown = await waitFor("[...document.querySelectorAll('img')].some(i => i.naturalWidth > 0 && (i.alt||'').length >= 0 && i.closest('[data-testid=stImage]'))", 15000);
  await clickButton("Run OCR");
  out.checks.result_appears = await waitFor("!!document.querySelector('.ocr-text')", 60000);
  const shown = await js("document.querySelector('.ocr-text') && document.querySelector('.ocr-text').innerText");
  out.single_text = shown;
  out.checks.result_is_the_expected_tamil_text = shown === F.word_text;
  out.checks.result_text_selectable = await js("getComputedStyle(document.querySelector('.ocr-text')).userSelect !== 'none'");
  out.checks.run_disabled_after_result = !(await js(buttonEnabled("Run OCR")));
  await shot("02_single_result.png");
  const txt = await download("Download TXT", "single_txt");
  out.checks.txt_download_equals_displayed_text = !!txt && txt.toString("utf-8") === shown + "\n";
  const jsn = await download("Download JSON", "single_json");
  let doc = null; try { doc = JSON.parse(jsn.toString("utf-8")); } catch (e) { /* checked below */ }
  out.checks.json_download_matches = !!doc && doc.results.length === 1 && doc.results[0].text === shown && doc.model.sha256_verified === true;
  out.checks.download_names_are_safe = Object.values(out.downloads).every(n => /^[A-Za-z0-9_.-]+$/.test(n));

  // ---- clear, then a corrupted file
  await clickButton("Clear");
  out.checks.clear_removes_result = await waitFor("!document.querySelector('.ocr-text')", 15000);
  await setFiles(0, [F.corrupt]);
  out.checks.corrupt_file_message = await waitFor("document.body.innerText.includes('could not be read as an image')", 30000);
  const tc = await bodyText();
  out.checks.no_traceback_in_page = !/Traceback|File "/.test(tc);
  out.checks.run_disabled_for_corrupt = !(await js(buttonEnabled("Run OCR")));
  await shot("03_corrupt.png");

  // ---- batch with one failing file
  out.checks.batch_tab_opened = await mouseClick("Batch") && await waitFor(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.trim() === 'Run OCR on all'); return !!b && b.offsetParent !== null; })()`, 15000);
  out.checks.batch_files_set = await setFiles(1, F.batch);
  out.checks.batch_run_enabled = await waitFor(buttonEnabled("Run OCR on all"), 30000);
  await clickButton("Run OCR on all");
  const n = F.batch.length, okN = F.batch_texts.filter(t => t !== null).length;
  out.checks.batch_summary = await waitFor(`document.body.innerText.includes('${okN} of ${n} recognized')`, 120000);
  await waitFor(`document.querySelectorAll('table.ocr-results tr').length === ${n + 1}`, 30000);
  out.batch_table = await js("[...document.querySelectorAll('table.ocr-results tr')].slice(1).map(tr => [...tr.children].map(td => td.innerText))");
  out.checks.batch_table_rows_match = Array.isArray(out.batch_table) && out.batch_table.length === n &&
    out.batch_table.every((r, i) => r[2] === (F.batch_texts[i] === null ? "failed" : "ok") && r[3] === (F.batch_texts[i] || ""));
  await shot("04_batch.png");
  const csvb = await download("Download CSV", "batch_csv");
  if (csvb) {
    const s = csvb.toString("utf-8");
    out.checks.csv_has_bom = s.charCodeAt(0) === 0xfeff;
    const rows = s.replace(/^\ufeff/, "").trim().split(/\r?\n/).slice(1).map(l => l.split(","));
    out.batch_csv_rows = rows.map(r => [r[1], r[2], r[3]]);
    out.checks.csv_rows_match_expected = rows.length === n && rows.every((r, i) => F.batch_texts[i] === null ? r[2] === "error" : r[3] === F.batch_texts[i]);
  } else { out.checks.csv_has_bom = false; out.checks.csv_rows_match_expected = false; }
  const bj = await download("Download JSON", "batch_json");
  let bdoc = null; try { bdoc = JSON.parse(bj.toString("utf-8")); } catch (e) { /* below */ }
  out.checks.batch_json_matches = !!bdoc && bdoc.results.map(r => r.text).join("|") === F.batch_texts.map(t => t === null ? null : t).join("|");

  out.checks.no_external_requests = out.external_requests.length === 0;
  out.checks.no_page_exceptions = out.console_errors.length === 0;
  out.requests = [...new Set(out.requests)].sort();
  ws.close();
}

main().catch(e => { out.error = String(e && e.stack || e); }).finally(() => {
  try { edge && edge.kill(); } catch (e) { /* gone */ }
  setTimeout(() => { try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* locked */ } console.log(JSON.stringify(out)); process.exit(0); }, 1500);
});
