// End-to-end browser test of the diagnostic panel of Full-page OCR, in headless Microsoft Edge over the DevTools
// protocol. Started by tests/diagnostics_browser_smoke.py against the REAL Streamlit server (app.py) and the real model.
//
//   node diagnostics_browser_smoke.js BASE FIXTURES_JSON OUT_DIR
//
// FIXTURES_JSON: {"page": path, "page_words": n, "sample": {"id", "box": [x1,y1,x2,y2], "text"}}. The page is GENERATED;
// sample.text is the generator's label of that word. Name resolution is refused for everything but 127.0.0.1 and
// localhost; every request is recorded; the browser profile is thrown away. Never clicks "Save bundle" (it writes inside
// the project). Prints one JSON object; exit 0 only if every check holds.
"use strict";
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const [BASE, FIXTURES, OUT] = process.argv.slice(2);
const F = JSON.parse(fs.readFileSync(FIXTURES, "utf-8"));
const EDGE = ["C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe"].find(p => fs.existsSync(p));
const PORT = 9443;
const DL = path.join(OUT, "downloads");
fs.mkdirSync(DL, { recursive: true });
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "diag_smoke_edge_"));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const out = { base: BASE, checks: {}, observed: {}, steps: [], external_requests: [], console_errors: [], passed: false };
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
  const shot = async name => { const r = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }); fs.writeFileSync(path.join(OUT, name + ".png"), Buffer.from(r.data, "base64")); out.observed.screenshots = (out.observed.screenshots || []).concat(name + ".png"); };
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
  const mouse = async (x, y, clicks = 1) => {
    await cdp("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
    for (let c = 1; c <= clicks; c++) {
      await cdp("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", buttons: 1, clickCount: c });
      await sleep(60);
      await cdp("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", buttons: 0, clickCount: c });
      await sleep(60);
    }
  };
  const key = async (k, code, vk, text) => {
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", key: k, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk, ...(text ? { text } : {}) });
    await cdp("Input.dispatchKeyEvent", { type: "keyUp", key: k, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
  };
  const checkbox = label => `[...document.querySelectorAll('label')].find(l => l.innerText.trim().startsWith(${JSON.stringify(label)}))`;
  const isChecked = label => js(`(() => { const l = ${checkbox(label)}; const i = l && l.querySelector('input[type=checkbox]'); return i ? i.checked : null; })()`);
  const toggle = async label => {                 // a real mouse click on the box (a synthetic label click can be ignored)
    const r = await js(`(() => { const l = ${checkbox(label)}; l.scrollIntoView({block: "center"}); const b = l.getBoundingClientRect(); return [b.left, b.top, b.height]; })()`);
    await sleep(300); await mouse(r[0] + 9, r[1] + r[2] / 2); await sleep(600);
  };
  const openExpander = async label => {          // a real mouse click on the expander header; returns whether it is open afterwards
    const sel = `[...document.querySelectorAll('summary')].find(s => s.innerText.includes(${JSON.stringify(label)}))`;
    const r = await js(`(() => { const s = ${sel}; if (!s) return null; const d = s.closest('details'); if (d && d.open) return "open"; s.scrollIntoView({block: "center"}); const b = s.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })()`);
    if (r === null) return false;
    if (r !== "open") { await sleep(300); await mouse(r[0] + 40, r[1] + r[3] / 2); await sleep(800); }
    return js(`(() => { const s = ${sel}; const d = s && s.closest('details'); return !!(d && d.open); })()`);
  };
  // the data grids (st.dataframe / st.data_editor) draw on a canvas; their accessibility table carries the cell text
  const grids = () => js(`[...document.querySelectorAll('[data-testid=stDataFrame]')].filter(g => g.offsetParent !== null).length`);
  const gridInfo = (i, scroll = false) => js(`(() => { const g = [...document.querySelectorAll('[data-testid=stDataFrame]')].filter(g => g.offsetParent !== null)[${i}]; if (!g) return null;
      const t = g.querySelector('[role=grid]') || g.querySelector('table');
      const heads = [...g.querySelectorAll('[role=columnheader]')].map(h => h.innerText.trim());
      const rows = [...g.querySelectorAll('[role=row]')].slice(1).map(r => [...r.querySelectorAll('[role=gridcell]')].map(c => c.innerText.trim()));
      if (${scroll}) g.scrollIntoView({block: "center"}); const c = g.querySelector('canvas'); const b = (c || g).getBoundingClientRect();
      return { heads, rows, rowcount: t && t.getAttribute('aria-rowcount'), rect: [b.left, b.top, b.width, b.height] }; })()`);
  const step = s => out.steps.push(s);

  // 1. the app starts; Full-page OCR: every experimental option and diagnostic mode are off
  await cdp("Page.navigate", { url: BASE + "/" });
  await until("document.body && document.body.innerText.includes('Model ready')", "model ready", 240000);
  out.observed.tabs = await js("[...document.querySelectorAll('[role=tab]')].map(t => t.innerText.trim())");
  out.checks.both_workflows_offered = out.observed.tabs.includes("Single image") && out.observed.tabs.includes("Full-page OCR");
  await tab("Full-page OCR");
  await openExpander("Segmentation settings");
  await sleep(500);
  const opts = ["Remove ruling lines", "Local contrast equalisation", "Fill ruling in crops", "Diagnostic mode"];
  out.observed.option_states = {};
  for (const o of opts) out.observed.option_states[o] = await isChecked(o);
  out.checks.experimental_and_diagnostic_off_by_default = opts.every(o => out.observed.option_states[o] === false);
  step("defaults checked");

  // 2. a page is read with diagnostics off: no diagnostic panel
  await setFiles(2, [F.page]);
  await until("document.body.innerText.includes('Layout found:')", "the layout summary");
  out.observed.layout_line = ((await body()).match(/Layout found:[^\n]*/) || [""])[0];
  out.observed.read_button_default = await js(`${visibleButton("Read page")}.innerText.trim()`);
  await clickButton("Read page");
  await until("[...document.querySelectorAll('.ocr-page')].some(e => e.offsetParent !== null)", "the page text", 240000);
  const textOff = await visibleText(".ocr-page");
  out.observed.read_line = ((await body()).match(/\d+ of \d+ word regions read[^\n]*/) || [""])[0];
  out.checks.no_panel_when_diagnostics_off = !(await body()).includes("Inspect one region");
  await shot("1_page_read_diagnostics_off");
  step("page read, diagnostics off");

  // 3. diagnostic mode on: the panel appears, the OCR result is unchanged, no diagnostic error
  await toggle("Diagnostic mode");
  await until("document.body.innerText.includes('Inspect one region')", "the diagnostic panel", 120000);
  await sleep(1500);
  const textOn = await visibleText(".ocr-page");
  out.observed.page_text_after_diag = textOn;
  out.checks.ocr_text_unchanged_by_diagnostics = textOn === textOff;
  const tb = await body();
  out.checks.no_diagnostic_error = !tb.includes("The diagnostic view failed") && !tb.includes("Traceback");
  out.observed.metrics = await js("[...document.querySelectorAll('[data-testid=stMetric]')].map(m => m.innerText.replace(/\\n+/g, ' | ').trim())");
  const metric = name => { const m = out.observed.metrics.find(s => s.startsWith(name + " |")); return m ? Number(m.split("|")[1].trim()) : null; };
  out.observed.metric_regions = metric("Regions");
  out.observed.metric_with_warning = metric("With a warning");
  const nRegions = Number((out.observed.read_line.match(/^\d+ of (\d+)/) || [0, NaN])[1]);
  out.checks.metric_regions_equals_regions_of_the_run = out.observed.metric_regions === nRegions && nRegions > 0;
  out.checks.warnings_section_shown = tb.includes("What each warning means");
  await shot("2_diagnostics_panel");
  step("panel shown");

  // 4. both overlays are rendered images; line boxes: angle-aware by default, horizontal bands one click away
  const bigImages = () => js("[...document.querySelectorAll('img')].filter(i => i.offsetParent !== null && i.complete && i.naturalWidth > 300).length");
  const overlaySrc = () => js("(() => { const i = [...document.querySelectorAll('img')].filter(i => i.offsetParent !== null && i.naturalWidth > 300); return i.length ? i[i.length - 1].src : null; })()");
  const radioChecked = label => js(`(() => { const l = ${checkbox(label)}; const i = l && l.querySelector('input[type=radio]'); return i ? i.checked : null; })()`);
  const imgs1 = await bigImages();
  out.observed.line_mode_default = { angle: await radioChecked("Angle-aware line boxes"), horizontal: await radioChecked("Horizontal line bands") };
  out.checks.angle_aware_line_boxes_default_in_diagnostic_mode = out.observed.line_mode_default.angle === true && out.observed.line_mode_default.horizontal === false;
  out.checks.overlay_original_frame_shown = tb.includes("Boxes in the coordinates of the uploaded image") && tb.includes("angle-aware line boxes") && imgs1 >= 3;
  out.observed.line_mode_caption = ((tb.match(/Global deskew:[^\n]*/) || [""])[0]).slice(0, 220);
  out.checks.angle_caption_states_global_and_local = /Global deskew: measured [+-]?\d/.test(tb) && /\d+ with a local angle/.test(tb) && tb.includes("Display only");
  await tab("Crop frame, lines and regions");
  await sleep(800);
  const angleSrc = await overlaySrc();
  out.checks.overlay_crop_frame_shown = (await body()).includes("angle-aware line boxes") && (await bigImages()) >= 3;
  await shot("3_overlay_crop_frame_angle_lines");
  await toggle("Horizontal line bands");
  await until("document.body.innerText.includes('green = line bands')", "the horizontal-band overlay", 30000).catch(() => false);
  await sleep(800);
  const horizSrc = await overlaySrc();
  out.checks.horizontal_bands_still_available = (await radioChecked("Horizontal line bands")) === true && (await body()).includes("green = line bands") && !!horizSrc && horizSrc !== angleSrc;
  await shot("3b_overlay_crop_frame_horizontal_bands");
  await toggle("Angle-aware line boxes");
  await until("!document.body.innerText.includes('green = line bands')", "the angle-aware overlay again", 30000).catch(() => false);
  out.checks.switch_back_to_angle_aware = (await radioChecked("Angle-aware line boxes")) === true;
  await tab("Original page, region numbers");
  step("overlays");

  // 5. region table: all regions, the expected columns, sortable by clicking a column header
  const g0 = await gridInfo(0);
  out.observed.region_table = { heads: g0 && g0.heads.slice(0, 12), rowcount: g0 && g0.rowcount, first_rows: g0 && g0.rows.slice(0, 3) };
  out.checks.region_table_has_columns = !!g0 && ["number", "id", "text", "warnings", "box_original"].every(h => g0.heads.includes(h));
  out.checks.region_table_has_all_regions = !!g0 && Number(g0.rowcount) === out.observed.metric_regions + 1;
  const firstNumber = async () => { const g = await gridInfo(0); return g && g.rows.length ? g.rows[0][0] : null; };
  const before = await firstNumber();
  // the "number" column is the first column: click its header (top band of the canvas) up to two times
  const [gx, gy] = (await gridInfo(0, true)).rect;
  await sleep(500);
  const sortSeq = [before];
  for (let k = 0; k < 2; k++) { await mouse(gx + 25, gy + 17); await sleep(1200); sortSeq.push(await firstNumber()); }
  out.observed.sort_first_number_sequence = sortSeq;
  out.checks.region_table_sortable = sortSeq.includes(String(out.observed.metric_regions)) && before === "1";
  await shot("4_region_table_sorted");
  // the warning filter
  await toggle("Only regions with a warning");
  await sleep(1200);
  const gF = await gridInfo(0);
  out.observed.rows_with_filter = gF && Number(gF.rowcount) - 1;
  out.checks.warning_filter_works = !!gF && Number(gF.rowcount) - 1 === out.observed.metric_with_warning;
  await toggle("Only regions with a warning");
  await sleep(1000);
  step("region table");

  // 6. inspect one region: step the number up to 3; the crop, its context and its record are shown
  const up = `(() => { const c = [...document.querySelectorAll('[data-testid=stNumberInput]')].find(c => c.innerText.includes('Region number')); return c && c.querySelector('[data-testid=stNumberInputStepUp]'); })()`;
  for (let k = 0; k < 2; k++) { await js(`${up}.click(), true`); await sleep(1200); }
  await until("document.body.innerText.includes('#3 ')", "the inspector of region 3", 30000);
  const ti = await body();
  out.observed.inspector_caption = (ti.match(/#3 [^\n]*/) || [""])[0];
  out.checks.inspector_shows_crop_context_record = /#3 \S+: the crop the model read \(\d+ x \d+ px/.test(ti) && ti.includes("red = tight box, blue = crop with margin")
    && ti.includes("crop_box_ocr") && ti.includes("uncalibrated_score");
  await shot("5_region_inspector");
  step("inspector");

  // 7. line table
  const gBefore = await grids();
  out.observed.lines_expander_open = await openExpander("Lines (");
  await sleep(1200);
  const gl = await js(`(() => { const s = [...document.querySelectorAll('summary')].find(s => s.innerText.includes("Lines (")); const d = s && s.closest('details'); const g = d && d.querySelector('[data-testid=stDataFrame]'); if (!g) return null; g.scrollIntoView({block: "center"}); const t = g.querySelector('[role=grid]') || g.querySelector('table'); return { visible: g.offsetParent !== null, rowcount: t && t.getAttribute('aria-rowcount'), heads: [...g.querySelectorAll('[role=columnheader]')].map(h => h.innerText.trim()) }; })()`);
  out.observed.line_table = gl;
  out.observed.visible_grids_before_after = [gBefore, await grids()];
  out.checks.line_table_shown = !!gl && gl.visible && Number(gl.rowcount) === metric("Lines") + 1;
  const gg = await js(`(() => { const s = [...document.querySelectorAll('summary')].find(s => s.innerText.includes("Lines (")); const d = s && s.closest('details'); const g = d && d.querySelectorAll('[data-testid=stDataFrame]')[1]; if (!g) return null; const t = g.querySelector('[role=grid]') || g.querySelector('table'); return { rowcount: t && t.getAttribute('aria-rowcount'), heads: [...g.querySelectorAll('[role=columnheader]')].map(h => h.innerText.trim()) }; })()`);
  out.observed.line_geometry_table = gg && { rowcount: gg.rowcount, heads: gg.heads.slice(0, 8) };
  out.checks.line_geometry_table_shown = !!gg && ["id", "angle_original_deg", "source", "reliable"].every(h => gg.heads.includes(h)) && Number(gg.rowcount) === metric("Lines") + 1;
  await shot("5b_line_table");
  step("lines");

  // 8. evidence bundle: downloaded (checked by the Python harness); the save button is present but not clicked
  out.checks.bundle_include_input_on_by_default = (await isChecked("Include the input image in the bundle")) === true;
  await clickButton("Download diagnostic bundle (ZIP)");
  out.observed.bundle_file = path.basename(await waitDownload(/^diagnostic_.*\.zip$/));
  out.checks.save_bundle_button_present = await js(`!!${visibleButton("Save bundle to diagnostic_runs/")}`);
  step("bundle");

  // 9. manual crop comparison: fill the one template row with the generator's box and text, tick confirmed, compare
  const gE = (await grids()) - 1;
  const ed = await gridInfo(gE);
  out.observed.editor_heads = ed && ed.heads;
  out.checks.comparison_form_shown = !!ed && ["id", "x1", "y1", "x2", "y2", "reference", "confirmed"].every(h => ed.heads.includes(h));
  const [ex, ey] = (await gridInfo(gE, true)).rect;
  await sleep(500);
  // A first synthetic click only focuses the canvas; a following double-click on the x1 cell opens its editor. Each value
  // is written by selecting the editor's text and inserting the new value; Enter commits (and moves one row down);
  // ArrowUp and ArrowRight go to the next column, where Enter opens the editor again. Space toggles "confirmed".
  const editorOpen = () => js(`(() => { const e = document.activeElement; return !!e && (e.tagName === "INPUT" || e.tagName === "TEXTAREA") && (e.classList.contains("gdg-input") || !!e.closest("#portal")); })()`);
  const writeCell = async v => {
    await js(`(() => { const e = document.activeElement; if (e && e.select) e.select(); return true; })()`);
    await cdp("Input.insertText", { text: v }); await sleep(200);
    await key("Enter", "Enter", 13); await sleep(600);
    await key("ArrowUp", "ArrowUp", 38); await sleep(250);
  };
  await mouse(ex + 33 + 40, ey + 35 + 17);           // the id cell of the first row (right of the 33 px row-selection column)
  await sleep(400);
  await mouse(ex + 33 + 122 + 60, ey + 35 + 17, 2);  // the x1 cell
  await sleep(700);
  const vals = [String(F.sample.box[0]), String(F.sample.box[1]), String(F.sample.box[2]), String(F.sample.box[3]), F.sample.text];
  out.observed.editor_opened = [];
  for (let k = 0; k < vals.length; k++) {
    if (k > 0) { await key("ArrowRight", "ArrowRight", 39); await sleep(250); await key("Enter", "Enter", 13); await sleep(400); }
    out.observed.editor_opened.push(await editorOpen());
    await writeCell(vals[k]);
  }
  await key("ArrowRight", "ArrowRight", 39); await sleep(250);
  await key(" ", "Space", 32, " "); await sleep(1500);
  const filled = await gridInfo(gE, true);
  out.observed.editor_rows = filled && filled.rows.slice(0, 2);
  out.checks.comparison_form_filled_as_typed = !!filled && filled.rows.length >= 1 && filled.rows[0].slice(0, 6).join("|") === ["S01", ...vals].join("|");
  await shot("6_comparison_form_filled");
  await clickButton("Compare A and B");
  await until(`!!${visibleButton("Download comparison (JSON)")}`, "the comparison result", 120000);
  await shot("7_comparison_result");
  await clickButton("Download comparison (JSON)");
  out.observed.comparison_file = path.basename(await waitDownload(/^crop_comparison_.*\.json$/));
  step("comparison");

  // 10. rule removal switched on from the interface re-segments the page; switched off, the default segmentation returns
  await toggle("Remove ruling lines");
  await until(`!!${visibleButton("Read page")} && ${visibleButton("Read page")}.innerText.trim() !== ${JSON.stringify(out.observed.read_button_default)}`, "re-segmentation with rule removal", 60000).catch(() => false);
  out.observed.read_button_rule_removal = await js(`(${visibleButton("Read page")} || {innerText: ''}).innerText.trim()`);
  out.observed.rule_removal_checked = await isChecked("Remove ruling lines");
  await shot("8_rule_removal_on");
  await toggle("Remove ruling lines");
  await until(`!!${visibleButton("Read page")} && ${visibleButton("Read page")}.innerText.trim() === ${JSON.stringify(out.observed.read_button_default)}`, "default segmentation again", 60000).catch(() => false);
  out.observed.read_button_after_off = await js(`(${visibleButton("Read page")} || {innerText: ''}).innerText.trim()`);
  out.checks.rule_removal_toggle_from_ui = out.observed.rule_removal_checked === true && out.observed.read_button_after_off === out.observed.read_button_default;
  out.observed.rule_removal_changed_region_count = out.observed.read_button_rule_removal !== out.observed.read_button_default;
  step("rule removal toggle");

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
