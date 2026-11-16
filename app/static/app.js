"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (sec) => `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);

let appConfig = null;
let pollTimer = null;
let charts = [];
let current = null; // { status, result }

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

/* ------------------------------------------------------------------ home */

async function initHome() {
  appConfig = await api("/api/config");
  const sel = $("#whisper-model");
  sel.innerHTML = appConfig.whisper_models
    .map((m) => `<option value="${m}" ${m === appConfig.whisper_default ? "selected" : ""}>${m}</option>`)
    .join("");

  if (!appConfig.openai_available) {
    $("#engine-openai").disabled = true;
    $("#engine-openai").parentElement.classList.add("disabled");
    $("#openai-note").textContent = "OPENAI_API_KEY 未設定";
  }
  if (!appConfig.anthropic_available) {
    $("#use-ai").disabled = true;
    $("#use-ai").parentElement.classList.add("disabled");
    $("#ai-note").textContent = "ANTHROPIC_API_KEY 未設定";
  } else {
    $("#use-ai").checked = true;
  }
  const syncOptions = () => {
    const local = document.querySelector('input[name="engine"]:checked').value === "local";
    $("#whisper-model-row").style.display = local ? "" : "none";
    $("#context-row").style.display = $("#use-ai").checked ? "" : "none";
  };
  document.querySelectorAll('input[name="engine"]').forEach((el) => el.addEventListener("change", syncOptions));
  $("#use-ai").addEventListener("change", syncOptions);
  syncOptions();

  const dz = $("#dropzone");
  const fileInput = $("#file");
  const setFile = (f) => {
    if (!f) return;
    const dt = new DataTransfer();
    dt.items.add(f);
    fileInput.files = dt.files;
    $("#file-name").textContent = `${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`;
    $("#submit").disabled = false;
  };
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); setFile(e.dataTransfer.files[0]); });

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = fileInput.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("engine", document.querySelector('input[name="engine"]:checked').value);
    fd.append("whisper_model", $("#whisper-model").value);
    fd.append("use_vision", $("#use-vision").checked);
    fd.append("use_ai", $("#use-ai").checked);
    fd.append("context", $("#context").value);
    const btn = $("#submit");
    btn.disabled = true;
    btn.textContent = "アップロード中…";
    try {
      const { id } = await api("/api/jobs", { method: "POST", body: fd });
      location.hash = `#/job/${id}`;
    } catch (err) {
      alert(`開始できませんでした: ${err.message}`);
    } finally {
      btn.textContent = "分析を開始";
      btn.disabled = false;
    }
  });
}

async function renderHistory() {
  const jobs = await api("/api/jobs");
  const el = $("#history");
  if (!jobs.length) { el.innerHTML = '<p class="muted">まだありません</p>'; return; }
  const label = { done: "完了", running: "解析中", queued: "待機中", error: "エラー" };
  el.innerHTML = jobs.map((j) => `
    <a class="history-item" href="#/job/${j.id}">
      <span class="name">${esc(j.filename)}</span>
      <span class="badge ${j.state}">${label[j.state] || j.state}</span>
      <span class="date">${new Date(j.created_at * 1000).toLocaleString("ja-JP")}</span>
    </a>`).join("");
}

/* ------------------------------------------------------------------ job */

async function openJob(id) {
  clearTimeout(pollTimer);
  $("#report").hidden = true;
  $("#job-error").hidden = true;
  $("#progress").hidden = true;
  let loadedResult = false;

  const tick = async () => {
    let data;
    try { data = await api(`/api/jobs/${id}`); } catch (e) {
      $("#job-error").hidden = false;
      $("#job-error").textContent = `読み込めませんでした: ${e.message}`;
      return;
    }
    const st = data.status;
    $("#job-title").textContent = st.filename;
    if (st.state === "error") {
      $("#progress").hidden = true;
      $("#job-error").hidden = false;
      $("#job-error").innerHTML = `<strong>解析に失敗しました</strong><br>${esc(st.error)}`;
      return;
    }
    const running = st.state === "running" || st.state === "queued";
    if (running && !loadedResult) {
      $("#progress").hidden = false;
      $("#progress-step").textContent = `${st.step} (${Math.round(st.progress * 100)}%)`;
      $("#progress-bar").style.width = `${st.progress * 100}%`;
    }
    if (data.result && !loadedResult) {
      loadedResult = true;
      $("#progress").hidden = true;
      current = data;
      renderReport(id, data.status, data.result);
    } else if (running && loadedResult) {
      // AI 講評の再実行中
      $("#ai-body").innerHTML = `<p class="muted">${esc(st.step)}…</p>`;
    }
    if (running) pollTimer = setTimeout(tick, 1500);
  };
  await tick();
}

function seek(t) {
  const v = $("#video");
  v.currentTime = Math.max(0, t - 0.5);
  v.play();
  v.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderReport(id, status, r) {
  $("#report").hidden = false;
  const v = $("#video");
  if (!v.src.endsWith(`/api/jobs/${id}/video`)) v.src = `/api/jobs/${id}/video`;

  renderKpis(r);
  renderChecks(r);
  renderAi(id, status, r);
  renderCharts(r);
  renderTranscript(r);

  $("#export-md").onclick = () => exportMarkdown(status, r);
  $("#delete-job").onclick = async () => {
    if (!confirm("この分析結果と動画を削除します。元に戻せません。よろしいですか？")) return;
    await api(`/api/jobs/${id}`, { method: "DELETE" });
    location.hash = "#/";
  };
}

function renderKpis(r) {
  const s = r.speech, vis = r.vision;
  const minutes = Math.max(s.audio.speaking_time / 60, 0.1);
  const kpis = [];
  if (r.review) kpis.push(["AI 総合スコア", r.review.overall_score, "/100", ""]);
  kpis.push(["話すスピード", s.rate.chars_per_minute, "字/分", "目安 300 前後"]);
  kpis.push(["フィラー", s.fillers.total, "回", `${(s.fillers.total / minutes).toFixed(1)} 回/分`]);
  kpis.push(["2 秒以上の沈黙", s.audio.long_pauses.length, "回", ""]);
  if (vis && vis.available) {
    kpis.push(["目線 (カメラ方向)", Math.round(vis.eye_contact_ratio * 100), "%", ""]);
    kpis.push(["笑顔の割合", Math.round(vis.smile_ratio * 100), "%", ""]);
  }
  $("#kpis").innerHTML = kpis.map(([label, value, unit, sub]) => `
    <div class="kpi"><div class="label">${label}</div>
      <div class="value">${value}<small>${unit}</small></div>
      ${sub ? `<div class="sub">${sub}</div>` : ""}</div>`).join("");
}

function renderChecks(r) {
  const items = [...r.speech.evaluation, ...((r.vision && r.vision.evaluation) || [])];
  const icon = { good: "◎", warn: "△", bad: "×" };
  let html = items.map((it) => `
    <div class="check-item ${it.level}">
      <span class="icon">${icon[it.level]}</span>
      <div class="head">${esc(it.title)}<span>${esc(it.value)}</span></div>
      <div class="advice">${esc(it.advice)}</div>
    </div>`).join("");
  if (r.vision && !r.vision.available) {
    html += `<p class="muted small">表情・視線: ${esc(r.vision.reason)}</p>`;
  }
  $("#checks").innerHTML = html;
}

function renderAi(id, status, r) {
  const body = $("#ai-body");
  const rv = r.review;
  if (!rv) {
    const reason = r.review_error
      ? `<p class="card error">AI 講評に失敗しました: ${esc(r.review_error)}</p>`
      : `<p class="muted">AI 講評は実行されていません。</p>`;
    const canRun = appConfig && appConfig.anthropic_available;
    body.innerHTML = `${reason}
      <div class="ai-placeholder">
        ${canRun ? '<button id="run-review" class="primary">AI 講評を実行</button>' : '<span class="muted small">.env に ANTHROPIC_API_KEY を設定してサーバーを再起動すると実行できます。</span>'}
      </div>`;
    const btn = $("#run-review");
    if (btn) btn.onclick = async () => {
      btn.disabled = true;
      try {
        const fd = new FormData();
        fd.append("context", status.options.context || "");
        await api(`/api/jobs/${id}/review`, { method: "POST", body: fd });
        location.hash = `#/job/${id}?t=${Date.now()}`;
      } catch (e) { alert(e.message); btn.disabled = false; }
    };
    return;
  }
  const list = (arr, tag = "ul") => `<${tag}>${(arr || []).map((x) => `<li>${esc(x)}</li>`).join("")}</${tag}>`;
  const stars = (n) => "★".repeat(n) + "☆".repeat(Math.max(0, 5 - n));
  body.innerHTML = `
    <div class="ai-score"><span class="num">${rv.overall_score}</span><span>${esc(rv.overall_summary)}</span></div>
    <div class="ai-cols">
      <div><h3>良かった点</h3>${list(rv.strengths)}</div>
      <div><h3>次回までに取り組むこと</h3>${list(rv.action_items, "ol")}</div>
    </div>
    <div class="ai-cols">
      <div><h3>話し方について</h3><p>${esc(rv.delivery_feedback)}</p></div>
      <div><h3>表情・態度について</h3><p>${esc(rv.nonverbal_feedback)}</p></div>
    </div>
    <h3>質問ごとの講評</h3>
    ${(rv.qa || []).map((q, i) => `
      <details class="qa" ${i === 0 ? "open" : ""}>
        <summary>
          <span class="time-link" data-t="${q.question_time}">${fmtTime(q.question_time)}</span>
          <span class="q">${esc(q.question)}</span>
          <span class="stars">${stars(q.score)}</span>
        </summary>
        <div class="qa-body">
          <p><strong>回答の要約:</strong> ${esc(q.answer_summary)}</p>
          <div class="ai-cols">
            <div><h3>良い点</h3>${list(q.good_points)}</div>
            <div><h3>改善点</h3>${list(q.improvements)}</div>
          </div>
          <h3>回答例</h3>
          <div class="better">${esc(q.better_answer)}</div>
        </div>
      </details>`).join("")}
    <p class="muted small">講評モデル: ${esc(rv.model)} / AI の講評は参考意見です。</p>`;
  body.querySelectorAll(".time-link").forEach((el) => el.addEventListener("click", (e) => {
    e.preventDefault();
    seek(Number(el.dataset.t));
  }));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderCharts(r) {
  charts.forEach((c) => c.destroy());
  charts = [];
  if (!window.Chart) return;
  const text = cssVar("--muted"), grid = cssVar("--border");
  const base = {
    responsive: true, maintainAspectRatio: false, animation: false, spanGaps: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: text, boxWidth: 12 } } },
    onClick: (_e, els, chart) => { if (els.length) seek(chart.data.labels[els[0].index]); },
  };
  const xAxis = { type: "linear", ticks: { color: text, callback: (v) => fmtTime(v) }, grid: { color: grid } };

  const tl = r.speech.audio.timeline;
  charts.push(new Chart($("#chart-voice"), {
    type: "line",
    data: {
      labels: tl.map((p) => p.t),
      datasets: [
        { label: "声の大きさ (dB)", data: tl.map((p) => ({ x: p.t, y: p.volume_db })), borderColor: cssVar("--accent"), pointRadius: 0, borderWidth: 1.5, yAxisID: "y" },
        { label: "声の高さ (Hz)", data: tl.map((p) => ({ x: p.t, y: p.pitch_hz })), borderColor: cssVar("--warn"), pointRadius: 0, borderWidth: 1.5, yAxisID: "y1" },
      ],
    },
    options: { ...base, scales: {
      x: xAxis,
      y: { position: "left", ticks: { color: text }, grid: { color: grid }, title: { display: true, text: "dB", color: text } },
      y1: { position: "right", ticks: { color: text }, grid: { display: false }, title: { display: true, text: "Hz", color: text } },
    } },
  }));

  const vis = r.vision;
  $("#vision-chart-section").hidden = !(vis && vis.available);
  if (vis && vis.available) {
    const vt = vis.timeline;
    charts.push(new Chart($("#chart-vision"), {
      type: "line",
      data: {
        labels: vt.map((p) => p.t),
        datasets: [
          { label: "カメラ方向を見ている割合", data: vt.map((p) => ({ x: p.t, y: p.eye_contact })), borderColor: cssVar("--accent"), pointRadius: 0, borderWidth: 1.5, stepped: true },
          { label: "笑顔度", data: vt.map((p) => ({ x: p.t, y: p.smile })), borderColor: cssVar("--good"), pointRadius: 0, borderWidth: 1.5 },
        ],
      },
      options: { ...base, scales: { x: xAxis, y: { min: 0, max: 1, ticks: { color: text }, grid: { color: grid } } } },
    }));
  }
}

function renderTranscript(r) {
  const segs = r.transcript.segments;
  const occ = r.speech.fillers.occurrences;
  const pauses = [...r.speech.audio.long_pauses];
  const html = [];
  segs.forEach((seg, i) => {
    while (pauses.length && pauses[0].start <= seg.start) {
      const p = pauses.shift();
      html.push(`<div><span class="pause-chip">${fmtTime(p.start)} 沈黙 ${p.duration.toFixed(1)} 秒</span></div>`);
    }
    const marks = occ.filter((o) => o.segment === i).sort((a, b) => a.start - b.start);
    let text = "", pos = 0;
    for (const m of marks) {
      text += esc(seg.text.slice(pos, m.start)) + `<mark class="filler">${esc(seg.text.slice(m.start, m.end))}</mark>`;
      pos = m.end;
    }
    text += esc(seg.text.slice(pos));
    html.push(`<div class="seg" data-start="${seg.start}" data-end="${seg.end}">
      <span class="time-link" data-t="${seg.start}">${fmtTime(seg.start)}</span><span>${text}</span></div>`);
  });
  const el = $("#transcript");
  el.innerHTML = html.join("") || '<p class="muted">発話を検出できませんでした。</p>';
  el.querySelectorAll(".time-link").forEach((a) => a.addEventListener("click", () => seek(Number(a.dataset.t) + 0.5)));

  const video = $("#video");
  video.ontimeupdate = () => {
    const t = video.currentTime;
    el.querySelectorAll(".seg").forEach((s) => {
      s.classList.toggle("active", t >= Number(s.dataset.start) && t < Number(s.dataset.end));
    });
  };
}

function exportMarkdown(status, r) {
  const s = r.speech, vis = r.vision, rv = r.review;
  const lines = [`# 面接振り返りレポート: ${status.filename}`, "", `分析日時: ${new Date(status.created_at * 1000).toLocaleString("ja-JP")}`, ""];
  lines.push("## 計測結果", "");
  for (const it of [...s.evaluation, ...((vis && vis.evaluation) || [])]) {
    lines.push(`- **${it.title}** (${it.value}): ${it.advice}`);
  }
  if (rv) {
    lines.push("", `## AI 講評 (総合 ${rv.overall_score}/100)`, "", rv.overall_summary, "", "### 良かった点");
    rv.strengths.forEach((x) => lines.push(`- ${x}`));
    lines.push("", "### 次回までに取り組むこと");
    rv.action_items.forEach((x, i) => lines.push(`${i + 1}. ${x}`));
    lines.push("", "### 話し方について", rv.delivery_feedback, "", "### 表情・態度について", rv.nonverbal_feedback, "");
    for (const q of rv.qa) {
      lines.push(`### [${fmtTime(q.question_time)}] ${q.question} (${q.score}/5)`, "", `**回答の要約:** ${q.answer_summary}`, "", "**良い点**");
      q.good_points.forEach((x) => lines.push(`- ${x}`));
      lines.push("", "**改善点**");
      q.improvements.forEach((x) => lines.push(`- ${x}`));
      lines.push("", "**回答例**", "", ...q.better_answer.split("\n").map((l) => `> ${l}`), "");
    }
  }
  lines.push("## 文字起こし", "");
  r.transcript.segments.forEach((seg) => lines.push(`- [${fmtTime(seg.start)}] ${seg.text}`));
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${status.filename.replace(/\.[^.]+$/, "")}_振り返り.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ------------------------------------------------------------------ router */

async function route() {
  const m = location.hash.match(/^#\/job\/([0-9a-f]{32})/);
  $("#view-home").hidden = !!m;
  $("#view-job").hidden = !m;
  if (m) {
    if (!appConfig) appConfig = await api("/api/config");
    openJob(m[1]);
  } else {
    clearTimeout(pollTimer);
    $("#video").pause();
    renderHistory();
  }
}

initHome();
window.addEventListener("hashchange", route);
route();
