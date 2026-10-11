// End-to-end browser test of Full-page OCR and of the word-tab page check, in headless Microsoft Edge over the DevTools
// protocol. Started by tests/fullpage_browser_smoke.py against the REAL Streamlit server (app.py) and the real model.
//
//   node fullpage_browser_smoke.js BASE FIXTURES_JSON OUT_DIR
//
// FIXTURES_JSON: {"page": path, "page_text": str, "page_words": n, "word": path, "word_text": str}. Inputs are GENERATED
// images only. Name resolution is refused for everything but 127.0.0.1 and localhost; every request is recorded; the
// browser profile is thrown away. Prints one JSON object; exit 0 only if every check holds.
"use strict";
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const [BASE, FIXTURES, OUT] = process.argv.slice(2);
const F = JSON.parse(fs.readFileSync(FIXTURES, "utf-8"));
const EDGE = ["C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"].find(p => fs.existsSync(p));
const PORT = 9441;
const DL = path.join(OUT, "downloads");
fs.mkdirSync(DL, { recursive: true });
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "fullpage_smoke_edge_"));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const out = { base: BASE, checks: {}, observed: {}, external_requests: [], console_errors: [], passed: false };
let edge = null;

async function main() {
  if (!EDGE) { out.skipped = "Microsoft Edge not found"; return; }
  edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, "--window-size=1500,1400",
    "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--disable-background-networking",
    "--disable-component-update", "--disable-default-apps", "--disable-domain-reliability", "--disable-client-side-phishing-detection",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1 , EXCLUDE localhost", "about:blank"], { stdio: "ignore" });
  let target = null;
  for (let i = 0; i < 80 && !target; i++) {
    await sleep(250);
    try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find(t => t.type === "page"); } catch (e) { /* not up yet */ }
  }
  if (!target) throw new Error("the browser did not start");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let seq = 0; const pending = new Map();
  const cdp = (method, params = {}) => new Promise((res, rej) => {
    const id = ++seq; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params }));
    setTimeout(() => { if (pending.has(id)) { pending.delete(id); rej(new Error("no answer to " + method)); } }, 60000);
  });
  const local = u => /^(about|data|blob):/.test(u) || /^(https?|wss?):\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/.test(u);
  ws.onmessage = m => {
    const d = JSON.parse(m.data);
    if (d.id) { const p = pending.get(d.id); pending.delete(d.id); if (p) d.error ? p.rej(new Error(JSON.stringify(d.error))) : p.res(d.result); return; }
    if (d.method === "Network.requestWillBeSent" && !local(d.params.request.url)) out.external_requests.push(d.params.request.url);
    if (d.method === "Network.webSocketCreated" && !local(d.params.url)) out.external_requests.push(d.params.url);
    if (d.method === "Runtime.exceptionThrown") out.console_errors.push(JSON.stringify(d.params.exceptionDetails).slice(0, 300));
  };
  await cdp("Page.enable"); await cdp("Runtime.enable"); await cdp("Network.enable"); await cdp("DOM.enable");
  await cdp("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: DL, eventsEnabled: true });
  const js = async expr => { const r = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 400)); return r.result.value; };
  const until = async (expr, what, ms = 120000) => { for (let t = 0; t < ms; t += 300) { if (await js(expr)) return true; await sleep(300); } throw new Error("timed out waiting for " + what); };
  const shot = async name => { const r = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }); fs.writeFileSync(path.join(OUT, name + ".png"), Buffer.from(r.data, "base64")); };
  const visibleButton = label => `[...document.querySelectorAll('button')].find(b => b.innerText.trim().startsWith(${JSON.stringify(label)}) && b.offsetParent !== null)`;
  const clickButton = async label => { await until(`!!${visibleButton(label)} && !${visibleButton(label)}.disabled`, `button '${label}' enabled`); await js(`${visibleButton(label)}.click(), true`); };
  const tab = async label => { await js(`[...document.querySelectorAll('[role=tab]')].find(b => b.innerText.trim() === ${JSON.stringify(label)}).click(), true`); await sleep(600); };
  const setFiles = async (index, files) => {
    const doc = await cdp("DOM.getDocument", { depth: -1, pierce: true });
    const q = await cdp("DOM.querySelectorAll", { nodeId: doc.root.nodeId, selector: "input[type=file]" });
    if (q.nodeIds.length <= index) throw new Error(`file input ${index} not found (have ${q.nodeIds.length})`);
    await cdp("DOM.setFileInputFiles", { nodeId: q.nodeIds[index], files });
  };
  const waitDownload = async (pattern, ms = 30000) => {
    for (let t = 0; t < ms; t += 300) {
      const f = fs.readdirSync(DL).filter(n => pattern.test(n) && !n.endsWith(".crdownload"));
      if (f.length) { await sleep(400); return path.join(DL, f[0]); }
      await sleep(300);
    }
    throw new Error("no download matching " + pattern);
  };
  const visibleText = sel => js(`(() => { const e = [...document.querySelectorAll(${JSON.stringify(sel)})].find(e => e.offsetParent !== null); return e ? e.innerText : null; })()`);
  const body = () => js("document.body.innerText");

  // 1. the app starts, the model is ready, both workflows are offered
  await cdp("Page.navigate", { url: BASE + "/" });
  await until("document.body && document.body.innerText.includes('Model ready')", "model ready", 240000);
  out.observed.title = await js("document.title");
  out.observed.tabs = await js("[...document.querySelectorAll('[role=tab]')].map(t => t.innerText.trim())");
  out.checks.both_workflows_offered = out.observed.tabs.includes("Single image") && out.observed.tabs.includes("Full-page OCR");
  await shot("1_home");

  // 2. THE REPORTED PROBLEM: a whole page uploaded to the word tab is refused, not read as one character
  await setFiles(0, [F.page]);
  await until("document.body.innerText.includes('looks like a page')", "the page refusal in the word tab");
  const t2 = await body();
  out.checks.word_tab_refuses_page = t2.includes("looks like a page") && t2.includes("Full-page OCR");
  out.checks.word_tab_run_disabled_for_page = await js(`${visibleButton("Run OCR")}.disabled`);
  out.checks.word_tab_shows_no_result_for_page = (await visibleText(".ocr-text")) === null;
  await shot("2_word_tab_page_refused");

  // 3. a normal word is still read in the word tab
  await clickButton("Clear");
  await until("document.body.innerText.includes('No image yet')", "word tab cleared");
  await setFiles(0, [F.word]);
  await until("document.body.innerText.includes('What the model receives')", "word preview");
  await clickButton("Run OCR");
  await until("[...document.querySelectorAll('.ocr-text')].some(e => e.offsetParent !== null)", "word result");
  out.observed.word_text = (await visibleText(".ocr-text")).trim();
  out.checks.word_still_read = out.observed.word_text === F.word_text;

  // 4. Full-page OCR: upload -> validate -> segment -> word boundaries shown
  await tab("Full-page OCR");
  await setFiles(2, [F.page]);
  await until("document.body.innerText.includes('Layout found:')", "the layout summary");
  const t4 = await body();
  out.observed.layout_line = (t4.match(/Layout found:[^\n]*/) || [""])[0];
  out.observed.read_button = await js(`${visibleButton("Read page")}.innerText.trim()`);
  out.checks.segmentation_counts_shown = out.observed.read_button === `Read page (${F.page_words} word regions)`;
  out.checks.overlay_shown_before_reading = await until("[...document.querySelectorAll('img')].filter(i => i.offsetParent !== null && i.complete && i.naturalWidth > 300).length >= 2", "the page and the overlay images", 30000).catch(() => false);
  out.checks.no_text_before_reading = (await visibleText(".ocr-page")) === null;
  await shot("3_page_segmented");

  // 5. recognize the word crops and rebuild the page text
  await clickButton("Read page");
  await until("[...document.querySelectorAll('.ocr-page')].some(e => e.offsetParent !== null)", "the page text", 240000);
  out.observed.page_text = await visibleText(".ocr-page");
  out.checks.page_text_matches_reference = out.observed.page_text.trim() === F.page_text.trim();
  out.checks.page_text_has_lines_and_spaces = out.observed.page_text.includes("\n") && out.observed.page_text.includes(" ");
  const t5 = await body();
  out.observed.read_line = (t5.match(/\d+ of \d+ word regions read[^\n]*/) || [""])[0];
  out.checks.all_regions_read = out.observed.read_line.startsWith(`${F.page_words} of ${F.page_words} word regions read`);
  out.checks.read_disabled_after_result = await js(`${visibleButton("Read page")}.disabled`);
  await shot("4_page_read");

  // 6. the word regions can be inspected: table rows with number, id, text, box
  await js(`[...document.querySelectorAll('summary')].find(s => s.innerText.includes('Word regions (')).click(), true`);
  await until("!!document.querySelector('table.ocr-words') && document.querySelector('table.ocr-words').offsetParent !== null", "the word table");
  const rows = await js("[...document.querySelectorAll('table.ocr-words tr')].slice(1).map(tr => [...tr.cells].map(c => c.innerText.trim()))");
  out.observed.word_rows = rows.length;
  out.observed.first_rows = rows.slice(0, 3);
  out.checks.word_table_complete = rows.length === F.page_words && rows.every((r, i) => r[0] === String(i + 1) && r[3] === "ok" && /^\d+, \d+, \d+, \d+$/.test(r[5]));
  const tableText = rows.map(r => r[4]);
  out.checks.table_words_are_the_page_words = tableText.join(" ") === F.page_text.split(/\s+/).join(" ");
  await shot("5_word_table");

  // 7. downloads: TXT, JSON, CSV read back from disk
  await clickButton("Download TXT");
  const txt = fs.readFileSync(await waitDownload(/_page_ocr\.txt$/), "utf-8");
  await clickButton("Download JSON");
  const doc = JSON.parse(fs.readFileSync(await waitDownload(/_page_ocr\.json$/), "utf-8"));
  await clickButton("Download CSV");
  const csv = fs.readFileSync(await waitDownload(/_page_ocr\.csv$/), "utf-8");
  out.observed.downloads = fs.readdirSync(DL);
  out.checks.txt_equals_page_text = txt === F.page_text + "\n";
  out.checks.json_matches = doc.text === F.page_text && doc.words.length === F.page_words && doc.words.every((w, i) => w.sequence === i + 1 && w.text === tableText[i] && w.box.length === 4) &&
    doc.model.sha256_verified === true && typeof doc.page.sha256 === "string" && doc.segmentation.word_regions === F.page_words;
  const csvLines = csv.replace(/^\ufeff/, "").trim().split(/\r\n/);
  out.checks.csv_matches = csv.charCodeAt(0) === 0xfeff && csvLines.length === F.page_words + 1 && csvLines[0].startsWith("file_name,sequence,region_id") &&
    csvLines.slice(1).every((l, i) => l.split(",")[1] === String(i + 1) && l.includes(tableText[i]));
  out.observed.json_model = { checkpoint_id: doc.model.checkpoint_id, device: doc.model.device, margin: doc.crop_margin_px, pipeline: doc.segmentation.pipeline_version };

  // 8. reset
  await clickButton("Clear page");
  await until("document.body.innerText.includes('No page yet')", "page tab cleared");
  out.checks.clear_resets_page = (await visibleText(".ocr-page")) === null;

  out.checks.no_external_requests = out.external_requests.length === 0;
  out.checks.no_page_exceptions = out.console_errors.length === 0;
  out.passed = Object.values(out.checks).every(v => v === true);
}

main().catch(e => { out.error = String(e && e.stack || e); }).finally(async () => {
  try { edge && edge.kill(); } catch (e) { /* ignore */ }
  await sleep(800);
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { out.profile_cleanup_error = String(e); }
  console.log(JSON.stringify(out, null, 1));
  process.exit(out.passed ? 0 : (out.skipped ? 3 : 1));
});
