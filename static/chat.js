/* "Ask AI" chat for a report. History lives in this tab (sessionStorage), so a
   refresh keeps the conversation; closing the tab clears it. */
(function () {
  const root = document.getElementById("chat");
  if (!root) return;
  const log = document.getElementById("chat-log");
  const empty = document.getElementById("chat-empty");
  const form = document.getElementById("chat-form");
  const text = document.getElementById("chat-text");
  const send = document.getElementById("chat-send");
  const clear = document.getElementById("chat-clear");
  const provider = document.getElementById("ask-provider");
  const KEY = "chat:" + root.dataset.job;
  let history = [];
  try { history = JSON.parse(sessionStorage.getItem(KEY) || "[]"); } catch (e) { history = []; }

  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v ?? "");

  // Small, safe formatter: escape everything, then allow **bold** and "- " bullets.
  function format(t) {
    const lines = esc(t).split(/\r?\n/);
    let html = "", list = false;
    for (const raw of lines) {
      const line = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
      const m = /^\s*[-•*]\s+(.*)$/.exec(line);
      if (m) { if (!list) { html += "<ul>"; list = true; } html += `<li>${m[1]}</li>`; continue; }
      if (list) { html += "</ul>"; list = false; }
      html += line.trim() ? `<p>${line}</p>` : "";
    }
    return html + (list ? "</ul>" : "");
  }

  function stepsHTML(steps) {
    if (!steps || !steps.length) return "";
    const items = steps.map((s) => {
      const head = `<p class="step-why">${esc(s.why || "Query")}</p>`;
      if (s.error) return `<li>${head}<p class="step-err">Query error: ${esc(s.error)}</p></li>`;
      const r = s.result;
      const rows = r.rows.slice(0, 12).map((row) => `<tr>${row.map((v) => `<td${typeof v === "number" ? ' class="num"' : ""}>${esc(num(v))}</td>`).join("")}</tr>`).join("");
      const more = r.result_rows > Math.min(12, r.rows.length) ? `<p class="quiet small">Showing ${Math.min(12, r.rows.length)} of ${r.result_rows.toLocaleString()} result rows.</p>` : "";
      return `<li>${head}<p class="quiet small">${esc(r.sheet)}: ${r.matched_rows.toLocaleString()} matching rows</p>
        <div class="table-wrap"><table><thead><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>${more}</li>`;
    }).join("");
    return `<details class="qsteps"><summary>Numbers used (${steps.length} ${steps.length === 1 ? "query" : "queries"})</summary><ol>${items}</ol></details>`;
  }

  function add(msg, opts = {}) {
    empty.hidden = true;
    const el = document.createElement("div");
    el.className = `msg ${msg.role}${msg.error ? " error" : ""}`;
    if (msg.role === "user") el.innerHTML = `<div class="bubble">${esc(msg.text).replace(/\n/g, "<br>")}</div>`;
    else el.innerHTML = `<div class="bubble">${msg.error ? `<p>${esc(msg.text)}</p>` : format(msg.text)}
      ${msg.model ? `<p class="msg-meta">${esc(msg.model)}</p>` : ""}${stepsHTML(msg.steps)}</div>`;
    log.appendChild(el);
    if (opts.scroll !== false) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    clear.hidden = false;
    return el;
  }

  function save() {
    try { sessionStorage.setItem(KEY, JSON.stringify(history.slice(-30))); } catch (e) { /* storage full or blocked */ }
  }

  history.forEach((m) => add(m, { scroll: false }));

  async function ask(question) {
    question = question.trim();
    if (!question || send.disabled) return;
    const prior = history.filter((m) => !m.error).map((m) => ({ role: m.role, text: m.text }));
    const userMsg = { role: "user", text: question };
    history.push(userMsg); add(userMsg); save();
    text.value = ""; send.disabled = true; text.disabled = true;
    const thinking = add({ role: "assistant", text: "Working on it… checking the numbers. This can take up to a minute." });
    thinking.classList.add("thinking");
    let reply;
    try {
      const res = await fetch(root.dataset.url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history: prior, provider: provider ? provider.value : null }),
      });
      const data = await res.json().catch(() => ({ error: `The server returned an error (${res.status}).` }));
      reply = res.ok && data.answer
        ? { role: "assistant", text: data.answer, steps: data.steps, model: data.model }
        : { role: "assistant", error: true, text: data.error || `The server returned an error (${res.status}).` };
    } catch (e) {
      reply = { role: "assistant", error: true, text: "Couldn't reach the app. Check your connection; if the app was asleep, try again in a minute." };
    }
    thinking.remove();
    history.push(reply); add(reply); save();
    send.disabled = false; text.disabled = false; text.focus();
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); ask(text.value); });
  text.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(text.value); }
  });
  root.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
  clear.addEventListener("click", () => {
    history = []; save();
    log.querySelectorAll(".msg").forEach((m) => m.remove());
    empty.hidden = false; clear.hidden = true;
  });
})();
