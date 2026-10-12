const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
async function api(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const post = (url, data) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const PHASES = ["LOCALIZING", "REPRODUCING", "PATCHING", "TESTING", "SELF_CORRECTING", "COMPLETE"];

let cur = null, sock = null, lastSeq = -1, seen = new Set(), tab = "diff", state = null, fileSel = null;

/* ---------- sessions list ---------- */
async function loadSessions() {
  const list = await api("/api/sessions");
  $("#session-empty").hidden = list.length > 0;
  $("#session-list").innerHTML = list.map(s => `
    <li data-id="${s.id}" class="st-${s.status} ${s.id === cur ? "sel" : ""}">
      <span class="s-title"><span class="dot"></span>${esc(s.title)}</span>
      <span class="s-meta">${esc(s.status)} · ${esc(s.phase)} · ${s.telemetry.steps}/${s.max_steps || 15} steps</span>
    </li>`).join("");
  $$("#session-list li").forEach(li => li.onclick = () => select(li.dataset.id));
  return list;
}

function select(id) {
  if (sock) sock.close();
  cur = id; lastSeq = -1; seen = new Set(); fileSel = null; state = null;
  $("#activity").innerHTML = ""; $("#term-out").textContent = "";
  $("#diff-view").innerHTML = '<div class="empty">No changes yet.</div>';
  $("#file-tree").innerHTML = '<div class="empty">No workspace loaded</div>';
  $("#file-view").innerHTML = '<div class="empty">Pick a file to inspect it.</div>';
  $("#repomap").innerHTML = '<div class="empty">The map appears after localisation.</div>';
  renderPipeline(); connect(); loadSessions();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const id = cur;
  sock = new WebSocket(`${proto}://${location.host}/ws/sessions/${id}?after=${lastSeq}`);
  sock.onmessage = m => {
    if (id !== cur) return;
    const d = JSON.parse(m.data);
    (d.events || []).forEach(handleEvent);
    if (d.diff !== undefined) renderDiff(d.diff);
    if (d.state) { state = d.state; renderTelemetry(); renderPipeline(); }
    if ((d.events || []).length) loadSessions();
    refreshTab();
  };
  sock.onclose = () => { if (id === cur && state && !["resolved", "unresolved", "failed"].includes(state.status)) setTimeout(() => id === cur && connect(), 800); };
}

/* ---------- events ---------- */
const fmtTime = ts => new Date(ts * 1000).toLocaleTimeString();
function card(cls, head, body = "", when = "") {
  const el = document.createElement("div");
  el.className = `card ${cls}`;
  el.innerHTML = `<div class="card-h"><span>${head}</span><span class="when">${when}</span></div>${body}`;
  const box = $("#activity");
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  box.appendChild(el);
  if (stick) box.scrollTop = box.scrollHeight;
}
function term(html) { const t = $("#term-out"); t.insertAdjacentHTML("beforeend", html); t.scrollTop = t.scrollHeight; }

function handleEvent(e) {
  lastSeq = Math.max(lastSeq, e.seq);
  const when = fmtTime(e.ts);
  switch (e.type) {
    case "phase_change":
      seen.add(e.to);
      card("phase", `Phase: ${esc(e.to)}`, `<div class="sub">${e.prev && e.prev !== "QUEUED" ? `${esc(e.prev)} took ${(e.elapsed_ms / 1000).toFixed(2)} s` : "pipeline started"}</div>`, when); break;
    case "log": card("log", esc(e.message), "", when); break;
    case "repomap": card("log", "AST repo map built", `<div class="sub">${e.stats.files} files, ${e.stats.symbols} symbols. Top: ${e.top.map(t => `${esc(t.path)} (${t.score})`).join(", ") || "none"}</div>`, when); break;
    case "llm": card("llm", `LLM call · ${esc(e.model)}`, `<div class="sub">${e.prompt_tokens} prompt + ${e.completion_tokens} completion tokens · ${e.latency_ms} ms${e.tool_calls.length ? " · tools: " + esc(e.tool_calls.join(", ")) : ""}</div>${e.content ? `<pre>${esc(e.content)}</pre>` : ""}`, when); break;
    case "tool": {
      const bad = /^(AST_LINTER_ERROR|ERROR|SANDBOX_BLOCKED)/.test(e.result);
      card(bad ? "bad" : "tool", `Step ${e.step}: ${esc(e.name)}`, `<div class="sub">${esc(JSON.stringify(e.args)).slice(0, 220)}</div><pre>${esc(e.result)}</pre>`, when); break;
    }
    case "terminal_start": term(`<span class="t-cmd">$ ${esc(e.cmd)}</span>\n`); break;
    case "terminal_chunk": term(esc(e.text)); break;
    case "terminal_end": term(`<span class="${e.rc === 0 ? "t-rc-ok" : "t-rc-bad"}">[exit ${e.rc} · ${e.duration_ms} ms]</span>\n\n`);
      card(e.rc === 0 ? "ok" : "bad", `Command finished with exit code ${e.rc}`, `<div class="sub">${e.duration_ms} ms</div>`, when); break;
    case "result": card(e.resolved ? "ok" : "bad", esc(e.message), "", when); break;
    case "error": card("bad", "Pipeline failed", `<pre>${esc(e.message)}</pre>`, when); break;
  }
}

/* ---------- pipeline chips + telemetry ---------- */
function renderPipeline() {
  const active = state?.phase;
  $("#pipeline").innerHTML = PHASES.map(p => {
    const cls = active === p ? "active" : seen.has(p) ? "done" : "";
    return `<span class="chip ${cls}">${p.replace("_", " ").toLowerCase()}</span>`;
  }).join("") + (active === "FAILED" ? '<span class="chip fail">failed</span>' : "");
  if (active === "COMPLETE") $$(".chip.active").forEach(c => c.className = "chip done");
}
function renderTelemetry() {
  const t = state.telemetry;
  $("#t-step").textContent = `Step: ${t.steps}/${state.max_steps || 15}`;
  $("#t-tokens").textContent = `Tokens: ${t.total_tokens} (${t.prompt_tokens} in / ${t.completion_tokens} out)`;
  $("#t-latency").textContent = `LLM latency: ${t.llm_latency_ms} ms (${t.llm_calls} calls)`;
  $("#t-tests").textContent = `Test runs: ${t.tests_run}`;
  $("#t-elapsed").textContent = `Elapsed: ${(t.elapsed_ms / 1000).toFixed(1)} s`;
  $("#t-engine").textContent = `Engine: ${state.engine}`;
  const st = $("#t-status"); st.textContent = state.status; st.className = `status ${state.status}`;
}

/* ---------- right-hand tabs ---------- */
function renderDiff(diff) {
  if (!diff.trim()) { $("#diff-view").innerHTML = '<div class="empty">No changes yet.</div>'; return; }
  $("#diff-view").innerHTML = diff.split("\n").map(l => {
    let c = "ctx", g = " ";
    if (l.startsWith("@@")) { c = "hunk"; g = "@"; }
    else if (/^(diff |index |--- |\+\+\+ |new file|deleted file)/.test(l)) { c = "meta"; g = ""; }
    else if (l.startsWith("+")) { c = "add"; g = "+"; }
    else if (l.startsWith("-")) { c = "del"; g = "−"; }
    return `<div class="dl ${c}"><span class="g">${g}</span><code>${esc(c === "add" || c === "del" ? l.slice(1) : l) || " "}</code></div>`;
  }).join("");
}
async function refreshTab() {
  if (!cur || !state) return;
  try {
    if (tab === "files" && state.has_workspace) await loadFiles();
    if (tab === "repomap" && state.has_repomap) await loadRepomap();
  } catch (e) { /* workspace may not exist yet */ }
}
let filesKey = "", mapKey = "";
async function loadFiles() {
  const key = `${cur}:${state.phase}:${state.telemetry.steps}`;
  if (key === filesKey) return; filesKey = key;
  const tree = await api(`/api/sessions/${cur}/files`);
  const html = nodes => nodes.map(n => n.type === "dir"
    ? `<details open><summary>${esc(n.name)}/</summary>${html(n.children)}</details>`
    : `<div class="f ${n.path === fileSel ? "sel" : ""}" data-path="${esc(n.path)}">${esc(n.name)}</div>`).join("");
  $("#file-tree").innerHTML = html(tree);
  $$("#file-tree .f").forEach(f => f.onclick = () => openFile(f.dataset.path));
  if (fileSel) openFile(fileSel, true);
}
async function openFile(path, keep) {
  fileSel = path;
  if (!keep) $$("#file-tree .f").forEach(f => f.classList.toggle("sel", f.dataset.path === path));
  const d = await api(`/api/sessions/${cur}/file?path=${encodeURIComponent(path)}`);
  $("#file-view").innerHTML = d.content.split("\n").map((l, i) => `<div class="ln"><span class="n">${i + 1}</span><code>${esc(l) || " "}</code></div>`).join("");
}
async function loadRepomap() {
  if (mapKey === cur) return; mapKey = cur;
  const rm = await api(`/api/sessions/${cur}/repomap`);
  const max = Math.max(1, ...rm.files.map(f => f.score));
  $("#repomap").innerHTML = (rm.query_keywords.length ? `<div class="sub" style="margin-bottom:8px;color:var(--muted)">Issue keywords: ${rm.query_keywords.map(esc).join(", ")}</div>` : "") +
    rm.files.map(f => `<details class="rm-file ${f.is_test ? "rm-test" : ""}" ${f.score > 0 && !f.is_test ? "open" : ""}>
      <summary>${esc(f.path)}<span class="rm-bar"><i style="width:${Math.round(100 * f.score / max)}%"></i></span><span class="rm-score">${f.score}</span></summary>
      ${f.functions.map(fn => `<div class="rm-sym"><span class="ln">L${fn.lineno}–${fn.end_lineno}</span><span>${esc(fn.signature)}</span></div>`).join("")}
      ${f.classes.map(c => `<div class="rm-sym rm-class"><span class="ln">L${c.lineno}–${c.end_lineno}</span><span>class ${esc(c.name)}</span></div>` +
        c.methods.map(m => `<div class="rm-sym" style="padding-left:44px"><span class="ln">L${m.lineno}–${m.end_lineno}</span><span>${esc(m.signature)}</span></div>`).join("")).join("")}
    </details>`).join("") +
    rm.errors.map(e => `<div class="err">${esc(e.path)}: ${esc(e.error)}</div>`).join("");
}
$$(".tab").forEach(b => b.onclick = () => {
  tab = b.dataset.tab;
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".pane").forEach(p => p.classList.toggle("active", p.id === `pane-${tab}`));
  filesKey = mapKey = ""; refreshTab();
});
$("#wrap").onchange = e => { $("#diff-view").classList.toggle("wrap", e.target.checked); $("#file-view").classList.toggle("wrap", e.target.checked); };
$("#term-toggle").onclick = () => {
  const t = $("#terminal"); t.classList.toggle("min");
  $("#term-toggle").textContent = t.classList.contains("min") ? "restore" : "minimise";
};

/* ---------- dialogs ---------- */
$$("[data-close]").forEach(b => b.onclick = () => b.closest("dialog").close());
const syncSource = () => { const v = $("#f-source").value; ["template", "github", "local"].forEach(k => $(`#row-${k}`).hidden = k !== v); };
$("#f-source").onchange = syncSource;
$("#btn-new").onclick = async () => {
  const tpls = await api("/api/templates");
  $("#f-template").innerHTML = tpls.map(t => `<option value="${t.id}">${esc(t.id)}: ${esc(t.title.split(": ")[1] || t.title)}</option>`).join("");
  $("#issue-err").textContent = ""; syncSource(); $("#dlg-issue").showModal();
};
$("#issue-go").onclick = async () => {
  const src = $("#f-source").value;
  const body = { title: $("#f-title").value, description: $("#f-desc").value, max_steps: $("#f-steps").value ? +$("#f-steps").value : null };
  if (src === "template") body.template = $("#f-template").value;
  if (src === "github") body.repo_url = $("#f-url").value.trim();
  if (src === "local") body.workspace_path = $("#f-path").value.trim();
  try { const s = await post("/api/issues", body); $("#dlg-issue").close(); await loadSessions(); select(s.id); }
  catch (e) { $("#issue-err").textContent = e.message; }
};
$("#btn-bench").onclick = async () => {
  const list = await post("/api/benchmarks/run", {}); await loadSessions(); select(list[0].id);
};
async function loadConfig() {
  const c = await api("/api/config");
  const b = $("#engine-badge"); b.textContent = `engine: ${c.engine}`; b.classList.toggle("offline", !c.has_key);
  return c;
}
$("#btn-settings").onclick = async () => {
  const c = await loadConfig();
  $("#s-provider").innerHTML = Object.keys(c.providers).map(p => `<option ${p === c.provider ? "selected" : ""}>${p}</option>`).join("");
  $("#s-model").value = c.model; $("#s-key").value = ""; $("#settings-err").textContent = ""; $("#dlg-settings").showModal();
};
$("#s-provider").onchange = async e => { const c = await api("/api/config"); $("#s-model").value = c.providers[e.target.value]; };
async function saveSettings(clear) {
  const body = { provider: $("#s-provider").value, model: $("#s-model").value };
  if (clear) body.api_key = ""; else if ($("#s-key").value) body.api_key = $("#s-key").value;
  try { await post("/api/config", body); await loadConfig(); $("#dlg-settings").close(); }
  catch (e) { $("#settings-err").textContent = e.message; }
}
$("#s-save").onclick = () => saveSettings(false);
$("#s-clear").onclick = () => saveSettings(true);

/* ---------- boot ---------- */
(async () => {
  await loadConfig();
  const list = await loadSessions();
  if (list.length) select(list[0].id);
  setInterval(async () => { const l = await loadSessions(); if (state && cur) { const s = l.find(x => x.id === cur); if (s && ["queued", "preparing", "running"].includes(s.status)) renderTelemetry(); } }, 2000);
})();
