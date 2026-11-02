/* 小蟹学习助手 · 演示站脚本
   纯静态：数据来自 data/demo.json（由 tools/export_demo_json.py 用真实算法在合成课程上算出）。
   没有构建步骤、没有外部依赖、没有网络请求（除了读取那一个 JSON）。 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const hms = (sec) => {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};
const pct = (v) => Math.round(Math.max(0, Math.min(1, v)) * 100);

/* 重讲文本里带 LaTeX 行内公式。没装渲染器，就把反斜杠宏换成 Unicode 字符，
   比让反斜杠裸露干净，也不假装排版已经好了。
   这里刻意不用正则：反斜杠穿过 shell / python / 正则三层转义太容易写坏，
   改成 split/join 反而不会出错。 */
const BS = String.fromCharCode(92);          // 一个反斜杠
const MACROS = [
  ["mu", "μ"], ["sigma", "σ"], ["lambda", "λ"], ["theta", "θ"],
  ["alpha", "α"], ["beta", "β"], ["pi", "π"], ["rho", "ρ"],
  ["times", "×"], ["cdot", "·"], ["leq", "≤"], ["le", "≤"], ["geq", "≥"], ["ge", "≥"],
  ["neq", "≠"], ["approx", "≈"], ["to", "→"], ["sum", "Σ"], ["infty", "∞"], ["sqrt", "√"],
];

const TOKENS = [
  ["operatorname{Var}", "Var"], ["operatorname{E}", "E"], ["operatorname{Cov}", "Cov"],
  ["operatorname{Corr}", "Corr"], ["quad", " "], ["qquad", "  "], ["left", ""], ["right", ""],
];

function prettyMath(text) {
  let out = esc(text);
  for (const d of ["(", ")", "[", "]"]) out = out.split(BS + d).join("");   // 去定界符
  for (const [k, v] of TOKENS) out = out.split(BS + k).join(v);             // 复合记号
  for (const [k, v] of MACROS) out = out.split(BS + k).join(v);             // 宏 → 符号
  for (const [k, v] of [["^{-1}", "⁻¹"], ["^{-2}", "⁻²"], ["^{2}", "²"], ["^{3}", "³"],
                        ["^2", "²"], ["^3", "³"]]) out = out.split(k).join(v);

  // **粗体**：按标记切段、奇数段加粗，别为这一件事引正则
  const parts = out.split("**");
  if (parts.length > 1) {
    out = parts.map((p, i) => (i % 2 ? "<strong>" + p + "</strong>" : p)).join("");
  }
  const NL = String.fromCharCode(10);                        // 换行符，同样避开转义
  return out.split(NL + NL)
    .map((para) => `<p>${para.split(NL).join("<br>")}</p>`)
    .join("");
}

/* ---------- 动效基础设施 ----------
   全部集中在这里：进场观察、数字滚动、重播。原则是"能关"——
   系统设了"减少动效"就一步到位，不做任何过渡。 */
const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
// 能跑动效才给 html 挂 js-motion；挂不上就保持"静态完整可见"
if (!REDUCE && "IntersectionObserver" in window) {
  document.documentElement.classList.add("js-motion");
}

let _io = null;
function observeReveals(root = document) {
  const els = $$("[data-reveal]", root).filter((el) => !el.classList.contains("in"));
  if (!els.length) return;
  if (REDUCE || !("IntersectionObserver" in window)) {
    els.forEach((el) => { el.classList.add("in"); el.dispatchEvent(new Event("revealed")); });
    return;
  }
  _io = _io || new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      e.target.dispatchEvent(new Event("revealed"));
      _io.unobserve(e.target);
    }
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  els.forEach((el) => _io.observe(el));
}

/* 数字从 0 滚到目标值；非数字（比如 08:05）原样显示 */
function countUp(el, target, dur = 900) {
  const to = Number(target);
  if (!isFinite(to)) { el.textContent = String(target); return; }
  if (REDUCE) { el.textContent = String(to); return; }
  const t0 = performance.now();
  const done = () => { el.textContent = String(to); };
  const step = (now) => {
    const p = Math.max(0, Math.min(1, (now - t0) / dur));
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = String(Math.round(to * e));
    if (p < 1) requestAnimationFrame(step); else done();
  };
  requestAnimationFrame(step);
  // 兜底：浏览器节流或标签页切走时帧循环可能不回来，数字不能停在半路
  setTimeout(done, dur + 600);
}

let DATA = null;

async function boot() {
  const res = await fetch("data/demo.json", { cache: "no-store" });
  DATA = await res.json();
  renderVideos();
  renderKpis();
  renderTimelineMeta();
  initChart();
  renderKnots();
  renderReword(0);
  renderDiag();
  renderQuiz();
  renderAbout();
  observeReveals(document);
  initScrollSpy();
  initToTop();
}

/* ---------- 视频 ----------
   没有 webdemo/video/*.mp4 时整块隐藏：宁可没有，也不给空播放器。
   服务器上行只有约 1.6 Mbps，所以体积、时长、「可能要缓冲」全都写在卡片上，
   并且给一个"下载完再看"的出口——不装作是秒开的 CDN。 */
function renderVideos() {
  const list = DATA.videos || (DATA.video ? [DATA.video] : []);
  const wrap = $("#videos");
  if (!list.length) { wrap.hidden = true; return; }
  wrap.hidden = false;

  const total = list.reduce((a, v) => a + (v.size_mb || 0), 0);
  wrap.innerHTML = list.map((v, i) => {
    const meta = [v.duration_text && `时长 ${v.duration_text}`, v.size_mb && `${v.size_mb} MB`]
      .filter(Boolean).join(" · ");
    return `
    <section class="card video-card" id="v-${esc(v.id || i)}">
      <div class="card-head">
        <div>
          <h2>${esc(v.title || "视频")}</h2>
          <p class="sub">${esc(v.note || "")}</p>
        </div>
        <span class="chip">${esc(v.tag || "视频")}${meta ? " · " + esc(meta) : ""}</span>
      </div>
      <div class="player">
        <video id="el-${esc(v.id || i)}" controls preload="metadata" playsinline
          ${v.poster ? `poster="${esc(v.poster)}"` : ""}></video>
        <button class="play-overlay" type="button" data-el="el-${esc(v.id || i)}">
          <span class="po-icon">▶</span>
          <span class="po-text">播放${v.duration_text ? "（" + esc(v.duration_text) + "）" : ""}</span>
        </button>
      </div>
      <p class="player-note">
        这台演示机上行只有约 1.6 Mbps，这段 ${esc(v.size_mb || "?")} MB：点开会边下边播，
        开头可能要等十几秒，拖动进度条也会短暂缓冲。想一口气看完，先下下来更稳。
        <a class="dl" href="${esc(v.src)}" download>下载这段（${esc(v.size_mb || "?")} MB）</a>
      </p>
    </section>`;
  }).join("") + `
    <p class="videos-foot">两支片子合计约 ${total.toFixed(1)} MB。视频是软件在本地跑的真实画面，
      不含任何登录凭证；画面里的网课页面来自 B 站公开课程。</p>`;

  list.forEach((v, i) => {
    const id = `el-${v.id || i}`;
    const el = document.getElementById(id);
    const card = el.closest(".video-card");
    const overlay = card.querySelector(".play-overlay");
    const note = card.querySelector(".player-note");
    el.src = v.src;
    overlay.onclick = () => {
      overlay.hidden = true;
      el.play().catch(() => { overlay.hidden = false; });
    };
    el.addEventListener("play", () => { overlay.hidden = true; });
    el.addEventListener("error", () => {
      overlay.hidden = false;
      note.insertAdjacentHTML("afterbegin",
        '<b style="color:#B42318">这段视频没加载出来</b>——可能文件还没同步到服务器。<br>');
    });
  });
}

/* ---------- KPI ---------- */
function renderKpis() {
  const s = DATA.session || {}, st = DATA.stats || {}, d = DATA.diagnosis || {};
  const items = [
    { k: "这一节时长", v: s.duration_text || hms(st.duration || 0), u: "", num: false },
    { k: "分析窗口", v: st.n_windows ?? DATA.series.length, u: "× 60 秒", num: true },
    { k: "回看点", v: d.n_knots ?? DATA.knots.length, u: `其中强 ${d.n_strong ?? 0}`, num: true },
    { k: "跨节根因", v: (d.root_causes || []).length, u: "缺的前置概念", num: true },
  ];
  $("#kpis").innerHTML = items.map(({ k, v, u, num }, i) => `
    <div class="kpi">
      <div class="k">${esc(k)}</div>
      <div class="v">${num
        ? `<span class="num" data-count="${esc(v)}">${esc(v)}</span>`
        : esc(v)}<small>${esc(u)}</small></div>
    </div>`).join("");

  // 首屏可见就立刻滚；不能滚就保持 HTML 里的真值（上面已经写好）
  if (!document.documentElement.classList.contains("js-motion")) return;
  $$("#kpis .num").forEach((el, i) => {
    setTimeout(() => countUp(el, el.dataset.count, 900), 160 + i * 90);
  });
}

function renderTimelineMeta() {
  const s = DATA.session || {};
  const n = (DATA.course?.sessions || []).length;
  $("#timeline-sub").textContent =
    `${s.title || ""} · ${s.duration_text || ""} · 这门课共 ${n} 节，每节都跑过同一套分析`;
  $("#knot-count").textContent = ` ${DATA.knots.length} 条`;
}

/* ---------- 时间轴 ---------- */
/* 时间轴：可以按 progress（0→1）画，用来演"曲线往前推、卡点被点亮"。
   progress=1 就是最终态，这时才开放 hover 交互。 */
function drawChart(progress = 1) {
  const cv = $("#chart");
  const box = cv.parentElement;
  const cssW = box.clientWidth || 900;
  const cssH = 300;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  cv.style.height = cssH + "px";
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  const pad = { l: 42, r: 14, t: 18, b: 30 };
  const W = cssW - pad.l - pad.r;
  const H = cssH - pad.t - pad.b;
  const series = DATA.series;
  const t0 = series[0]?.t ?? 0;
  const t1 = series[series.length - 1]?.e ?? 1;
  const X = (t) => pad.l + ((t - t0) / Math.max(1e-6, t1 - t0)) * W;
  const Y = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * H;
  const scanX = pad.l + W * Math.max(0, Math.min(1, progress));
  const alpha = Math.min(1, progress * 4);           // 网格先出来
  g.globalAlpha = alpha;

  // 网格 + y 轴
  g.font = "11px ui-monospace, Consolas, monospace";
  g.textBaseline = "middle";
  for (let v = 0; v <= 1.0001; v += 0.25) {
    const y = Y(v);
    g.strokeStyle = v === 0 ? "#E2D9CE" : "#F2EEE8";
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + W, y); g.stroke();
    g.fillStyle = "#8C949C";
    g.fillText(v.toFixed(2), 8, y);
  }

  // x 轴刻度
  g.textBaseline = "top";
  const stepT = t1 - t0 > 480 ? 120 : 60;
  for (let t = Math.ceil(t0 / stepT) * stepT; t <= t1; t += stepT) {
    const x = X(t);
    g.strokeStyle = "#F7F4F0";
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + H); g.stroke();
    g.fillStyle = "#8C949C";
    g.textAlign = "center";
    g.fillText(hms(t), x, pad.t + H + 8);
    g.textAlign = "left";
  }

  // 曲线之后的东西都裁到扫描位置：视觉上就是"还没读到的地方不画"
  g.save();
  g.beginPath();
  g.rect(0, 0, scanX, cssH);
  g.clip();

  // 回看点色带（宽度是它真实的跨度）
  for (const k of DATA.knots) {
    const x0 = X(k.t_start);
    const x1 = X(k.t_end > k.t_start ? k.t_end : k.t_start + 60);
    const lit = Math.min(1, Math.max(0, (scanX - x0) / 60));   // 扫到时渐亮
    if (lit <= 0) continue;
    g.globalAlpha = alpha * lit;
    g.fillStyle = "rgba(234,88,12,.13)";
    g.fillRect(x0, pad.t, Math.max(3, x1 - x0), H);
    g.fillStyle = "rgba(234,88,12,.85)";
    g.fillRect(x0, pad.t, 2, H);
  }
  g.globalAlpha = alpha;

  const line = (key, color, dash, width) => {
    g.beginPath();
    series.forEach((w, i) => {
      const x = X((w.t + w.e) / 2), y = Y(w[key]);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.setLineDash(dash || []);
    g.strokeStyle = color;
    g.lineWidth = width;
    g.lineJoin = "round";
    g.lineCap = "round";
    g.stroke();
    g.setLineDash([]);
  };
  line("B", "#8C949C", [4, 4], 1.6);
  line("L", "#F2621B", [], 2.4);

  // 回看点标记点：被扫到才出现
  for (const k of DATA.knots) {
    const x = X(k.t_start + 30);
    if (x > scanX) continue;
    const w = series.find((w) => w.idx === nearestIdx(k.t_start));
    const y = Y(w ? w.L : 0.8);
    const pop = Math.min(1, Math.max(0, (scanX - x) / 40));
    g.beginPath(); g.arc(x, y, 4 * (0.6 + 0.4 * pop), 0, Math.PI * 2);
    g.fillStyle = "#EA580C"; g.fill();
    g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke();
  }
  g.restore();

  // 扫描线（只在动画中显示）
  if (progress < 1) {
    const grad = g.createLinearGradient(scanX - 26, 0, scanX, 0);
    grad.addColorStop(0, "rgba(242,98,27,0)");
    grad.addColorStop(1, "rgba(242,98,27,.16)");
    g.globalAlpha = 1;
    g.fillStyle = grad;
    g.fillRect(scanX - 26, pad.t, 26, H);
    g.strokeStyle = "rgba(242,98,27,.55)";
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(scanX, pad.t - 4); g.lineTo(scanX, pad.t + H); g.stroke();
    g.beginPath(); g.arc(scanX, pad.t - 4, 2.5, 0, Math.PI * 2);
    g.fillStyle = "#F2621B"; g.fill();
  }
  g.globalAlpha = 1;

  // ---- 交互：只在最终态开放 ----
  const tip = $("#tip");
  const onMove = (ev) => {
    if (progress < 1) { tip.hidden = true; return; }
    const r = cv.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const t = t0 + ((x - pad.l) / W) * (t1 - t0);
    let best = null, bd = 1e9;
    for (const w of series) {
      const d = Math.abs((w.t + w.e) / 2 - t);
      if (d < bd) { bd = d; best = w; }
    }
    if (!best || x < pad.l - 4 || x > pad.l + W + 4) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.innerHTML =
      `<div class="tt">${hms(best.t)} – ${hms(best.e)}</div>` +
      `<div class="tl">认知负荷 L ${best.L.toFixed(2)} · 行为信号 B ${best.B.toFixed(2)}</div>` +
      `<div class="tl">${best.text ? esc(best.text.slice(0, 90)) + (best.text.length > 90 ? "…" : "") : "（无转写）"}</div>`;
    const tw = 300, left = Math.min(Math.max(8, x - tw / 2), cssW - tw - 8);
    tip.style.left = left + "px";
    tip.style.top = "8px";
  };
  cv.onmousemove = onMove;
  cv.onmouseleave = () => { tip.hidden = true; };

  cv.onclick = (ev) => {
    if (progress < 1) return;
    const r = cv.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const t = t0 + ((x - pad.l) / W) * (t1 - t0);
    const hit = DATA.knots.find((k) => t >= k.t_start - 2 && t <= k.t_start + 62);
    if (!hit) return;
    const el = document.getElementById(`knot-${hit.id}`);
    if (!el) return;
    el.classList.add("flash");
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    setTimeout(() => el.classList.remove("flash"), 1600);
  };
}

/* 播放一次生长动画：进视口自动放，也能点"重播"再看 */
let _chartAnim = null;
let _chartGuard = null;
let _chartDone = false;
function playChart() {
  if (REDUCE) { drawChart(1); _chartDone = true; return; }
  cancelAnimationFrame(_chartAnim);
  clearTimeout(_chartGuard);
  _chartDone = false;
  const DUR = 2400;
  const start = performance.now();
  const finish = () => {
    cancelAnimationFrame(_chartAnim);           // 关键：先停掉帧循环，否则它会把画面改回中途
    _chartAnim = null;
    clearTimeout(_chartGuard);
    drawChart(1);
    _chartDone = true;
  };
  const step = (now) => {
    if (_chartDone) return;                     // 兜底已收尾就不再画中间态
    const p = Math.max(0, Math.min(1, (now - start) / DUR));
    drawChart(1 - Math.pow(1 - p, 3));          // easeOutCubic
    if (p < 1) _chartAnim = requestAnimationFrame(step);
    else finish();
  };
  _chartAnim = requestAnimationFrame(step);
  // 兜底：帧循环被节流/中断时（后台标签页、省电模式）也必须落到完整曲线，
  // 不能把动画中途当成最终画面留下来。
  _chartGuard = setTimeout(finish, DUR + 1200);
}

function nearestIdx(t) {
  let best = 0, bd = 1e9;
  for (const w of DATA.series) {
    const d = Math.abs((w.t + w.e) / 2 - (t + 30));
    if (d < bd) { bd = d; best = w.idx; }
  }
  return best;
}

/* ---------- 回看点 ---------- */
function renderKnots() {
  $("#knot-list").innerHTML = DATA.knots.map((k, i) => `
    <article class="knot" id="knot-${k.id}" data-reveal>
      <div class="knot-top">
        <span class="time">${esc(k.t || hms(k.t_start))}</span>
        <span class="pill ${esc(k.grade || "medium")}">${esc(k.grade_cn || "")}级</span>
        <span class="pill kind">${esc(k.level_cn || "回看点")}</span>
        <span class="score">分数 ${k.score.toFixed(2)}
          <span class="bar"><i data-w="${pct(k.score)}" style="width:${pct(k.score)}%"></i></span>
        </span>
      </div>
      <h3>为什么标记这里</h3>
      <div class="ev">${(k.evidence || "").split("；").filter(Boolean)
        .map((e) => `<span>${esc(e.trim())}</span>`).join("")}</div>
      <blockquote>${esc(k.excerpt || "（这一段没有可用摘录）")}</blockquote>
      ${k.terms?.length ? `<div class="terms">${k.terms.slice(0, 8)
        .map((t) => `<i>${esc(t)}</i>`).join("")}</div>` : ""}
      <button class="btn" data-open="${i}">看这段怎么换讲法 <span class="arw">→</span></button>
    </article>`).join("");

  // 卡片进视口时，分数条才从 0 长到真实分数（先把真值存下，清零仅限动效开启时）
  if (document.documentElement.classList.contains("js-motion")) {
    $$("#knot-list .knot").forEach((a) => {
      const bar = a.querySelector(".bar i");
      if (!bar) return;
      bar.style.width = "0";
      a.addEventListener("revealed", () => { bar.style.width = bar.dataset.w + "%"; });
    });
  }
  observeReveals($("#knot-list"));

  $$("#knot-list .btn").forEach((b) => {
    b.onclick = () => {
      const i = Number(b.dataset.open);
      renderReword(i);
      $("#reword").scrollIntoView({ behavior: "smooth", block: "start" });
    };
  });
}

/* ---------- 重讲 ---------- */
function renderReword(i) {
  const list = DATA.rewords || [];
  if (!list.length) {
    $("#reword-src").textContent = "（这次导出没有重讲内容）";
    $("#reword-dst").textContent = "";
    return;
  }
  const item = list[Math.min(i, list.length - 1)];
  const knot = DATA.knots.find((k) => k.id === item.knot_id) || DATA.knots[0];

  $("#reword-tabs").innerHTML = list.map((r, j) => {
    const k = DATA.knots.find((x) => x.id === r.knot_id);
    return `<button class="tab" role="tab" aria-selected="${j === i}" data-i="${j}">
      ${esc(k?.t || "")} · ${esc(r.strategy)}</button>`;
  }).join("");
  $$("#reword-tabs .tab").forEach((t) => {
    t.onclick = () => renderReword(Number(t.dataset.i));
  });

  $("#reword-src").textContent = knot?.excerpt || "（无摘录）";
  $("#reword-src-meta").textContent =
    `${knot?.t || ""} 起的那一段 · ${knot?.grade_cn || ""}${knot?.level_cn || ""} · 分数 ${(knot?.score ?? 0).toFixed(2)}`;
  const srcTerms = $("#reword-src-terms");
  if (srcTerms) {
    srcTerms.innerHTML = `<p class="pane-label">这一段频繁出现的术语</p>` +
      `<div class="terms">${(knot?.terms || []).map((t) => `<i>${esc(t)}</i>`).join("")}</div>` +
      `<p class="pane-label" style="margin-top:14px">为什么被标成回看点</p>` +
      `<div class="ev">${(knot?.evidence || "").split("；").filter(Boolean)
        .map((e) => `<span>${esc(e.trim())}</span>`).join("")}</div>`;
  }
  const src = DATA.meta?.reword_source === "rules" || item.degraded
    ? "离线规则版" : "模型生成";
  $("#reword-tag").textContent = src;
  $("#reword-tag").title = src === "离线规则版"
    ? "这次导出没接模型：文案由 review.py 的规则模板生成，展示的是结构与取材逻辑，不是模型措辞"
    : "由配置的模型生成";
  $("#reword-dst").innerHTML = prettyMath(item.text);
  $$("#reword-dst p").forEach((p, i) => {
    p.style.animationDelay = (i * 90) + "ms";
  });
}

/* ---------- 诊断 ---------- */
function renderDiag() {
  const d = DATA.diagnosis || {};

  // 位置条：把回看点画在整节课的时间轴上（比按 5 分钟分桶更直观，
  // 分桶在只有一两节课时全是等高的柱子，看不出信息）
  const dur = DATA.session?.duration || DATA.series[DATA.series.length - 1]?.e || 1;
  const strip = $("#dist-strip");
  strip.setAttribute("data-reveal", "soft");
  strip.innerHTML =
    `<div class="inner"><div class="rail"></div>` +
    [0, 0.5, 1].map((f, i) => `<span class="mark" style="left:${f * 100}%">${hms(f * dur)}</span>`).join("") +
    DATA.knots.map((k, i) => {
      const p = Math.max(0, Math.min(1, k.t_start / dur)) * 100;
      const hot = k.grade === "strong" ? " hot" : "";
      return `<button class="tick${hot}" style="left:${p}%;animation-delay:${i * 90}ms" data-k="${k.id}"
        title="${esc(k.t || "")} · 分数 ${k.score} · ${esc(k.grade_cn || "")}${esc(k.level_cn || "")}">
        <b>${esc(k.t || "")}</b><i></i></button>`;
    }).join("") + `</div>`;
  $$("#dist-strip .tick").forEach((b) => {
    b.onclick = () => {
      const el = document.getElementById(`knot-${b.dataset.k}`);
      if (!el) return;
      el.classList.add("flash");
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      setTimeout(() => el.classList.remove("flash"), 1600);
    };
  });
  const dist = d.distribution || [];
  $("#dist-hint").textContent = DATA.knots.length
    ? `共 ${DATA.knots.length} 个回看点，占全片 ${pct(DATA.knots.reduce((a, k) => a + Math.max(0, k.t_end - k.t_start), 0) / dur)}% 的时长；`
      + `最长的一段 ${hms(Math.max(...DATA.knots.map((k) => Math.max(0, k.t_end - k.t_start))))}。`
    : "这一节没有回看点。";

  // 根因
  const rc = (d.root_causes || []).slice(0, 6);
  $("#root-tbl").innerHTML = rc.length
    ? `<thead><tr><th>概念</th><th class="num">出现节数</th><th class="num">回看次数</th><th class="num">均分</th><th class="num">掌握度</th></tr></thead>
       <tbody>${rc.map((r) => `<tr>
         <td><b>${esc(r.name)}</b></td>
         <td class="num">${r.sessions}</td>
         <td class="num">${r.knots}</td>
         <td class="num">${Number(r.avg_score).toFixed(2)}</td>
         <td class="num ${Number(r.mastery) === 0 ? "zero" : ""}">${pct(r.mastery)}%</td>
       </tr>`).join("")}</tbody>`
    : `<tbody><tr><td class="num">这门课"我的节"之间还没有重复出现的概念。</td></tr></tbody>`;

  // 复习顺序
  const plan = d.plan || [];
  $("#plan-list").innerHTML = plan.length ? plan.map((p, i) => `
    <li>
      <div class="pl-top">
        <span class="time">${esc(p.t || "")}</span>
        <b>${i + 1}. ${esc((p.terms || [])[0] || "这一段的术语群")}</b>
        <span class="pill ${esc(p.grade || p.level || "medium")}">分数 ${Number(p.score || 0).toFixed(2)}</span>
      </div>
      <div class="pl-ev">${esc(p.evidence || "")}</div>
    </li>`).join("") : `<li>暂无建议。</li>`;

  // 跨节关联
  observeReveals($("#diag"));

  const rel = d.related || [];
  $("#related-list").innerHTML = rel.length ? rel.slice(0, 12).map((r) => `
    <div><b>${esc(r.name)}</b><small>${esc(r.session || "")} ${esc(r.t || "")} · 分数 ${Number(r.score || 0).toFixed(2)}</small></div>`
  ).join("") : "<div>没有跨节关联。</div>";
}

/* ---------- 小测 ---------- */
function renderQuiz() {
  const items = DATA.quiz || [];
  const total = items.length;
  let answered = 0, correct = 0;
  const score = $("#quiz-score");
  const paint = () => { score.textContent = `${correct} / ${total}`; };
  paint();

  if (!total) {
    $("#quiz-list").innerHTML = `<p class="sub">这次导出没有生成题目（模型与规则都没能给出可核对的出处）。</p>`;
    return;
  }

  $("#quiz-list").innerHTML = items.map((q, i) => `
    <div class="q" id="q-${q.id}">
      <h3>${i + 1}. ${esc(q.prompt)}</h3>
      <ul class="opts">
        ${q.choices.map((c, j) => `
          <li><button class="opt" data-q="${q.id}" data-j="${j}" data-c="${q.correct}">${esc(c)}</button></li>`).join("")}
      </ul>
      <p class="why" id="why-${q.id}">出自原文：<code>${esc(q.evidence || "—")}</code></p>
    </div>`).join("");

  $$("#quiz-list .opt").forEach((btn) => {
    btn.onclick = () => {
      const qid = btn.dataset.q;
      const group = $$(`#quiz-list .opt[data-q="${qid}"]`);
      if (group.some((b) => b.disabled)) return;
      const right = Number(btn.dataset.j) === Number(btn.dataset.c);
      group.forEach((b) => {
        b.disabled = true;
        if (Number(b.dataset.j) === Number(b.dataset.c)) b.classList.add("right");
        else if (b === btn) b.classList.add("wrong");
      });
      $(`#why-${qid}`).classList.add("show");
      const qCard = btn.closest(".q");
      qCard.classList.add(right ? "correct" : "shake");
      setTimeout(() => qCard.classList.remove("correct", "shake"), 800);
      answered++; if (right) correct++;
      paint();
      score.classList.remove("pop");
      void score.offsetWidth;              // 强制重排，动画才能重播
      score.classList.add("pop");
    };
  });
}

/* ---------- 关于 ---------- */
function renderAbout() {
  const m = DATA.meta || {};
  $("#about-gen").textContent = `快照生成于 ${m.generated_at || "—"}（${m.data_source || "合成数据"}）`;
  const a = $("#repo-link");
  if (m.repo) a.href = m.repo;
  $("#foot-meta").textContent = `${m.app || "小蟹学习助手"} v${m.version || "0.1.0"} · ${m.data_source || ""}`;
}

/* ---------- 时间轴：进视口自动播一次，可手动重播 ---------- */
let _chartStarted = false;
function initChart() {
  const replay = $("#replay");
  if (replay) {
    replay.onclick = () => {
      _chartStarted = true;
      replay.classList.add("spin");
      setTimeout(() => replay.classList.remove("spin"), 520);
      playChart();
    };
  }
  if (REDUCE || !("IntersectionObserver" in window)) { playChart(); _chartStarted = true; return; }
  drawChart(1);                                   // 先给最终态：动画失败也不会是空白
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !_chartStarted) { _chartStarted = true; playChart(); io.disconnect(); }
    }
  }, { threshold: 0.28 });
  io.observe($("#timeline"));
}

/* ---------- 导航高亮当前区块 ---------- */
function initScrollSpy() {
  const links = new Map();
  $$(".nav a").forEach((a) => links.set((a.getAttribute("href") || "").slice(1), a));
  if (!links.size || !("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const a = links.get(e.target.id);
      if (!a || !e.isIntersecting) continue;
      links.forEach((l) => l.classList.remove("active"));
      a.classList.add("active");
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
}

/* ---------- 回到顶部 ---------- */
function initToTop() {
  const b = $("#totop");
  if (!b) return;
  const check = () => b.classList.toggle("show", window.scrollY > 720);
  window.addEventListener("scroll", check, { passive: true });
  check();
  b.onclick = () => window.scrollTo({ top: 0, behavior: REDUCE ? "auto" : "smooth" });
}

/* 宽度变化时重画（色带和刻度是按像素算的，不重画会错位） */
let rt = null;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (!DATA) return;
    cancelAnimationFrame(_chartAnim);
    _chartDone = true;
    drawChart(1);
  }, 150);
});

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="padding:14px 24px;background:#FEF2F2;color:#7F1D1D;font-size:14px">
       演示数据加载失败：${esc(e.message)}。请确认 webdemo/data/demo.json 存在。
     </div>`);
});