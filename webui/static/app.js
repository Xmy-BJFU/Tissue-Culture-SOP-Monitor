const $ = (s) => document.querySelector(s);
const state = { filter: "all", lastReport: null };

function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const m = String(Math.floor(sec / 60)).padStart(2, "0");
  const s = String(sec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

async function api(url, opt) {
  const res = await fetch(url, opt);
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
  return data;
}

function renderSOP(sop) {
  const done = sop.filter((s) => s.state === "done").length;
  $("#sopProgress").textContent = `${done}/${sop.length}`;
  $("#sopList").innerHTML = sop
    .map(
      (s, i) => `
    <li class="${s.state}">
      <i class="dot"></i>
      <div><b>${String(i + 1).padStart(2, "0")} ${s.title}</b><small>${s.hint}</small></div>
    </li>`,
    )
    .join("");
}

function renderTimers(soaks) {
  const names = ["酒精", "无菌水", "次氯酸钠"];
  $("#timers").innerHTML = names
    .map((n) => {
      const t = soaks[n] || { elapsed: 0, target: 1, progress: 0, remain: 0, running: false, done: false };
      const verdict = t.verdict === "ok" ? "合格" : t.verdict === "early" ? "偏早" : t.verdict === "late" ? "超时" : "";
      const label = t.running ? "计时中" : t.done ? verdict || "完成" : "待命";
      const cls =
        t.verdict === "ok" ? "ok" : t.verdict === "early" || t.verdict === "late" ? "warn" : t.running ? "on" : "";
      const win =
        t.window || t.lo ? `窗 ${t.window || `${Math.round(t.lo)}–${Math.round(t.hi)}s`}` : `目标 ${fmtTime(t.target)}`;
      const shift = t.running || t.done ? ` 倒出 ${t.dump_hold || 0}/${t.dump_need || 8}帧` : "";
      return `<div class="timer ${cls}">
        <div>${n}</div><div>${fmtTime(t.elapsed)} / ${fmtTime(t.target)} · ${label}</div>
        <div class="sub">${win}${shift}</div>
        <div class="bar"><i style="width:${Math.round(Math.min(1, t.progress || 0) * 100)}%"></i></div>
      </div>`;
    })
    .join("");
}

function renderActions(flags, trace) {
  if (trace && trace.length) {
    $("#actions").innerHTML = trace
      .map((g) => {
        const steps = (g.steps || [])
          .map((s) => {
            const cls = s.ok ? "ok" : s.warn ? "warn" : "";
            return `<li class="${cls}"><span class="k">${s.label}</span><span>${s.value}</span></li>`;
          })
          .join("");
        return `<div class="act ${g.state || ""}">
          <div style="width:100%">
            <div class="act-h"><b>${g.title}</b><span class="st">${g.summary || ""}</span></div>
            <ul class="trace-steps">${steps}</ul>
          </div>
        </div>`;
      })
      .join("");
    return;
  }
  const items = [
    [
      "灭菌",
      flags.step1_ok ? "完成" : flags.tweezers_ok || flags.knife_ok ? "进行中" : "未检出",
      flags.step1_ok,
      false,
    ],
    [
      "倾倒",
      flags.live_pour || (flags.last_pour ? `最近 ${flags.last_pour}` : "待命"),
      Boolean(flags.live_pour),
      false,
    ],
    ["倒出", flags.live_dump ? "进行中" : `累计 ${flags.dump_n || 0} 次`, Boolean(flags.live_dump), false],
    [
      "摇晃",
      flags.live_shake || (flags.shake_scored ? "计分已检出" : flags.shake_unscored ? "不计分已检出" : "未检出"),
      flags.shake_scored,
      flags.shake_unscored && !flags.shake_scored,
    ],
    ["切割", flags.cutting_ok ? "已检出" : "未检出", flags.cutting_ok, false],
    [
      "斜插",
      flags.insert_ok ? (flags.insert_angle_ok ? "发生且角度合格" : "已发生") : "等待鳞茎/培养基",
      flags.insert_ok,
      false,
    ],
  ];
  $("#actions").innerHTML = items
    .map(
      ([n, st, on, warn]) =>
        `<div class="act ${on ? "on" : ""} ${warn ? "warn" : ""}"><span>${n}</span><span class="st">${st}</span></div>`,
    )
    .join("");
}

function renderPPE(ppe) {
  $("#ppe").innerHTML = ["手套", "口罩", "帽子"]
    .map((k) => `<div class="item ${ppe[k] ? "on" : ""}">${k}<div>${ppe[k] ? "已检出" : "未检出"}</div></div>`)
    .join("");
}

function renderObjs(objs, holding) {
  $("#holdChip").textContent = holding.length ? `持 ${holding.length}` : "未持物";
  $("#holding").textContent = holding.length ? holding.join("、") : "—";
  $("#objBody").innerHTML = (objs || [])
    .slice(0, 12)
    .map((o) => `<tr><td>${o.tid}</td><td>${o.name}</td><td>${o.tilt}°</td><td>${o.holding ? "持" : "—"}</td></tr>`)
    .join("");
}

function renderScore(score) {
  $("#totalScore").textContent = (score.total ?? 0).toFixed(1);
  $("#dims").innerHTML = (score.dims || [])
    .map((d) => `<div class="dim"><div>${d.name}</div><b>${d.score}</b> / ${d.weight}<small>${d.detail}</small></div>`)
    .join("");
}

function renderLog(events) {
  const f = state.filter;
  const rows = (events || []).filter((e) => f === "all" || e.kind === f);
  $("#log").innerHTML = rows
    .slice(0, 40)
    .map((e) => `<li class="${e.level || ""}"><span class="t">${e.clock}</span>[${e.kind}] ${e.message}</li>`)
    .join("");
}

function fillSettings(s) {
  const f = $("#setForm");
  f.student.value = s.student || "";
  f.source.value = s.source || "0";
  f.hand_mode.value = s.hand_mode || "glove";
  f.device.value = s.device || "0";
  f.imgsz.value = s.imgsz || 640;
  f.conf.value = s.conf || 0.25;
  const t = s.thresholds || {};
  f.pour_tilt_deg.value = t.pour_tilt_deg ?? 60;
  f.pour_upright_deg.value = t.pour_upright_deg ?? 14;
  f.cut_span_px.value = t.cut_span_px ?? 55;
  f.cut_hold_frames.value = t.cut_hold_frames ?? 12;
  f.shake_min_range.value = t.shake_min_range ?? 12;
}

function applyState(s) {
  $("#sessionId").textContent = s.session_id || "—";
  $("#elapsed").textContent = fmtTime(s.elapsed);
  $("#fps").textContent = (s.fps ?? 0).toFixed(1);
  $("#frameId").textContent = s.frame_id || 0;
  const pill = $("#livePill");
  pill.textContent = s.running ? (s.paused ? "PAUSE" : "LIVE") : s.loading ? "LOADING" : "STANDBY";
  pill.className = "live-pill" + (s.running ? (s.paused ? " pause" : " on") : "");
  $("#handModeTag").textContent = s.hand_mode === "bare" ? "裸手·整图" : "手套·框内";
  const btnHm = $("#btnHandMode");
  if (btnHm) btnHm.textContent = s.hand_mode === "bare" ? "手关键点：裸手（整图）" : "手关键点：手套（框内）";
  const sel = $("#setForm")?.hand_mode;
  if (sel && sel.value !== s.hand_mode) sel.value = s.hand_mode || "glove";
  $("#sourceTag").textContent = `src ${s.source || "—"}`;
  $("#rinseCount").textContent = s.rinse_count || 0;
  renderSOP(s.sop || []);
  renderTimers(s.soaks || {});
  renderPPE(s.ppe || {});
  renderActions(s.flags || {}, s.action_trace || []);
  renderObjs(s.objects || [], s.holding || []);
  renderScore(s.score || { total: 0, dims: [] });
  renderLog(s.events || []);
  $("#btnPause").textContent = s.paused ? "继续" : "暂停";
}

async function tick() {
  try {
    const s = await api("/api/state");
    applyState(s);
    if (!state._inited) {
      fillSettings(s);
      state._inited = true;
    }
  } catch (err) {
    $("#livePill").textContent = "OFFLINE";
  }
}

async function control(action, extra = {}) {
  try {
    await api("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    await tick();
  } catch (err) {
    alert(err.message);
  }
}

$("#btnLoad").onclick = () => control("load");
$("#btnStart").onclick = () => {
  const f = $("#setForm");
  control("start", { source: f.source.value || "0", student: f.student.value, hand_mode: f.hand_mode.value });
};
$("#btnPause").onclick = async () => {
  const s = await api("/api/state");
  control(s.paused ? "resume" : "pause");
};
$("#btnStop").onclick = () => control("stop");
$("#btnSnap").onclick = () => control("snapshot");
$("#btnReset").onclick = () => control("reset");
$("#btnHandMode").onclick = async () => {
  const s = await api("/api/state");
  const next = s.hand_mode === "bare" ? "glove" : "bare";
  control("hand_mode", { hand_mode: next });
};
$("#btnSettings").onclick = () => {
  $("#mask").classList.add("on");
  $("#drawer").classList.add("on");
};
$("#btnCloseSet").onclick = () => {
  $("#mask").classList.remove("on");
  $("#drawer").classList.remove("on");
};
$("#mask").onclick = $("#btnCloseSet").onclick;

$("#setForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = e.target;
  await api("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      student: f.student.value,
      source: f.source.value,
      hand_mode: f.hand_mode.value,
      device: f.device.value,
      imgsz: Number(f.imgsz.value),
      conf: Number(f.conf.value),
      thresholds: {
        pour_tilt_deg: Number(f.pour_tilt_deg.value),
        pour_upright_deg: Number(f.pour_upright_deg.value),
        cut_span_px: Number(f.cut_span_px.value),
        cut_hold_frames: Number(f.cut_hold_frames.value),
        shake_min_range: Number(f.shake_min_range.value),
      },
    }),
  });
  alert("设置已保存");
};

$("#videoFile").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const data = await api("/api/upload", { method: "POST", body: fd });
  $("#setForm").source.value = data.path;
};

document.querySelectorAll(".filters .mini").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".filters .mini").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on");
    state.filter = btn.dataset.f;
    tick();
  };
});

document.querySelectorAll("#overlayToggles input").forEach((el) => {
  el.onchange = () => {
    const overlay = {};
    document.querySelectorAll("#overlayToggles input").forEach((i) => (overlay[i.dataset.ov] = i.checked));
    api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ overlay }),
    });
  };
});

$("#btnReport").onclick = async () => {
  const data = await api("/api/report");
  state.lastReport = data;
  $("#reportJson").textContent = JSON.stringify(data, null, 2);
  $("#reportMask").classList.add("on");
};
$("#btnCloseReport").onclick = () => $("#reportMask").classList.remove("on");
$("#btnDlJson").onclick = () => {
  const blob = new Blob([$("#reportJson").textContent], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `实训报告_${state.lastReport?.session_id || "session"}.json`;
  a.click();
};
$("#btnDlTxt").onclick = async () => {
  await navigator.clipboard.writeText($("#reportJson").textContent);
  alert("已复制报告 JSON");
};

setInterval(tick, 400);
tick();
