/* Results tables and charts (numbers from the paper), method widget, theme, copy buttons. */
(() => {
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";

  // ---------- theme ----------
  const root = document.documentElement;
  const toggle = $("theme-toggle");
  const stored = (() => { try { return localStorage.getItem("cast-theme"); } catch { return null; } })();
  if (stored) root.dataset.theme = stored;
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  const syncIcon = () => { toggle.querySelector("use").setAttribute("href", isDark() ? "#i-sun" : "#i-moon"); };
  syncIcon();
  toggle.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("cast-theme", root.dataset.theme); } catch {}
    syncIcon();
    window.dispatchEvent(new Event("themechange"));
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    syncIcon(); window.dispatchEvent(new Event("themechange"));
  });

  // ---------- copy ----------
  document.querySelectorAll(".code .copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.parentElement.querySelector("pre").innerText;
      try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
      catch { btn.textContent = "Select and copy"; }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });

  // ---------- math ----------
  if (window.renderMathInElement) {
    renderMathInElement(document.body, { delimiters: [{ left: "$$", right: "$$", display: true }], throwOnError: false });
  }

  // ---------- tables ----------
  const v = (x, sd, bold) => {
    const s = sd ? `<span class="sd">±${sd}</span>` : "";
    return `<td>${bold ? `<b>${x}</b>` : x}${s}</td>`;
  };

  // [system, ours, [mAP, sd], [mAPc, sd], [FAc, sd], [DER, sd], [few mAP, sd]]
  const MAIN = {
    causal: [
      ["CAST-ASD ×1 (Mimi)", true, ["90.56", ".10"], ["71.51", ".41"], ["8.61", ".18"], ["27.59", ".23"], ["96.99", ".02"]],
      ["LoCoNet ×3 (VGGish)", false, ["89.15", ".13"], ["69.63", ".81"], ["11.14", ".04"], ["31.57", ".37"], ["96.71", ".05"]],
      ["Per-face backbone (Mimi)", false, ["88.58", ".16"], ["69.95", ".43"], ["12.72", ".34"], ["32.17", ".46"], ["96.17", ".05"]],
      ["TalkNet (MFCC)", false, ["87.15", ".51"], ["68.48", ".96"], ["13.84", ".76"], ["33.92", ".98"], ["95.27", ".09"]],
      ["Light-ASD (MFCC)", false, ["85.10", ".22"], ["65.78", ".42"], ["16.73", ".35"], ["38.16", ".21"], ["94.22", ".08"]],
    ],
    offline: [
      ["CAST-ASD ×1 (Mimi)", false, ["91.70", ".11"], ["74.63", ".36"], ["7.20", ".02"], ["25.68", ".16"], ["97.51", ".06"]],
      ["CAST-ASD ×3 (Mimi)", false, ["92.08", ".05"], ["75.64", ".26"], ["6.71", ".08"], ["25.08", ".10"], ["97.65", ".02"]],
      ["LoCoNet ×3 (VGGish)", false, ["91.82", ".34"], ["75.35", "1.1"], ["7.92", ".54"], ["27.11", ".41"], ["98.10", ".06"]],
      ["TalkNet (MFCC)", false, ["88.78", ".21"], ["70.64", ".40"], ["11.65", ".28"], ["30.80", ".21"], ["96.20", ".06"]],
      ["Light-ASD (MFCC)", false, ["88.64", ".28"], ["71.03", ".81"], ["12.18", ".55"], ["32.13", ".71"], ["95.96", ".13"]],
    ],
  };
  $("t-main").tBodies[0].innerHTML =
    `<tr class="group"><td colspan="6">Causal: zero lookahead, verified by perturbation</td></tr>` +
    MAIN.causal.map((r) => `<tr class="${r[1] ? "ours" : ""}"><td>${r[0]}</td>${r.slice(2).map((c) => v(c[0], c[1], r[1])).join("")}</tr>`).join("") +
    `<tr class="group"><td colspan="6">Offline reference: reads the whole clip</td></tr>` +
    MAIN.offline.map((r) => `<tr><td>${r[0]}</td>${r.slice(2).map((c) => v(c[0], c[1], false)).join("")}</tr>`).join("");

  // Table 3: [audio, mixer, s, blocks, params(M), mAP, sd, FAc, sd, boldMap, boldFa]
  const OP = [
    ["Mimi", "CAST", null, 1, 1.57, "90.56", ".10", "8.61", ".18"],
    ["Mimi", "CAST", null, 3, 4.72, "90.65", ".14", "8.44", ".17", true, true],
    ["Mimi", "LoCoNet", 3, 1, 5.18, "89.60", ".05", "10.77", ".29"],
    ["Mimi", "LoCoNet", 3, 3, 15.55, "89.91", ".24", "10.26", ".42"],
    ["Mimi", "LoCoNet", 4, 3, 25.18, "90.14", ".17", "9.68", ".31"],
    ["Mimi", "LoCoNet", 5, 3, 37.57, "90.37", ".15", "9.42", ".18"],
    ["VGGish", "CAST", null, 1, 1.57, "89.45", ".38", "9.88", ".60"],
    ["VGGish", "CAST", null, 3, 4.72, "89.74", ".45", "9.46", ".62"],
    ["VGGish", "LoCoNet", 3, 1, 5.18, "88.92", ".09", "11.52", ".14"],
    ["VGGish", "LoCoNet", 3, 3, 15.55, "89.15", ".13", "11.14", ".04"],
  ];
  $("t-op").tBodies[0].innerHTML = OP.map((r, i) => {
    const first = i === 0 || OP[i - 1][0] !== r[0];
    const ours = r[1] === "CAST";
    return `<tr class="${ours ? "ours" : ""}"${first && i ? ' style="border-top:2px solid var(--border)"' : ""}>` +
      `<td>${first ? `<b>${r[0]}</b>` : ""}</td><td class="l">${ours ? "<b>CAST</b>" : "LoCoNet"}</td>` +
      `<td>${r[2] ?? "—"}</td><td>${r[3]}</td><td>${r[4].toFixed(2)}M</td>` +
      v(r[5], r[6], r[9]) + v(r[7], r[8], r[10]) + `</tr>`;
  }).join("");

  // Table 4
  const OCC = {
    share: [5.4, 9.6, 14.5, 69.6],
    series: [
      { name: "Per-face backbone (Mimi)", color: "var(--faint)", fa: [8.25, 9.97, 17.17, 12.45], map: [96.74, 93.66, 83.99, 86.87] },
      { name: "LoCoNet ×3 (VGGish)", color: "#f08c00", fa: [6.94, 7.02, 13.33, 11.39], map: [97.01, 94.13, 85.59, 87.28] },
      { name: "CAST-ASD ×1 (Mimi)", color: "var(--accent)", fa: [6.72, 7.83, 12.28, 8.15], map: [97.46, 94.58, 86.32, 89.23], ours: true },
      { name: "CAST-ASD ×3 (Mimi)", color: "#13a36b", fa: [6.30, 7.15, 11.56, 8.14], map: [97.35, 94.50, 86.79, 89.32], ours: true },
    ],
  };
  const best = (key, col, lower) => {
    const vals = OCC.series.map((s) => s[key][col]);
    return lower ? Math.min(...vals) : Math.max(...vals);
  };
  const occRows = (key, lower) => OCC.series.map((s) =>
    `<tr class="${s.ours ? "ours" : ""}"><td>${s.name}</td>${s[key].map((x, c) =>
      `<td>${x === best(key, c, lower) ? `<b>${x.toFixed(2)}</b>` : x.toFixed(2)}</td>`).join("")}</tr>`).join("");
  $("t-occ").tBodies[0].innerHTML =
    `<tr><td>Proportion of frames (%)</td>${OCC.share.map((x) => `<td>${x.toFixed(1)}</td>`).join("")}</tr>` +
    `<tr class="group"><td colspan="5">Competing false alarms FA<sub>c</sub> ↓</td></tr>` + occRows("fa", true) +
    `<tr class="group"><td colspan="5">Average precision mAP ↑</td></tr>` + occRows("map", false);

  // Table 5
  const COST = [
    ["CAST-ASD ×1 (Mimi)", 90.56, 91.70, "1.14", ".11", true],
    ["CAST-ASD ×3 (Mimi)", 90.65, 92.08, "1.42", ".24"],
    ["TalkNet (MFCC)", 87.15, 88.78, "1.63", ".89"],
    ["LoCoNet ×3 (VGGish)", 89.15, 91.82, "2.67", ".59"],
    ["Light-ASD (MFCC)", 85.10, 88.64, "3.54", ".56"],
  ];
  $("t-cost").tBodies[0].innerHTML = COST.map((r) =>
    `<tr class="${r[5] ? "ours" : ""}"><td>${r[0]}</td><td>${r[1].toFixed(2)}</td><td>${r[2].toFixed(2)}</td>${v(r[3], r[4], r[5])}</tr>`).join("");

  // Table 6
  $("t-ava").tBodies[0].innerHTML = [
    `<tr class="ours"><td>CAST-ASD ×1 (Mimi)</td>${v("91.07", ".15", true)}${v("94.2", ".17")}</tr>`,
    `<tr><td>LoCoNet ×3 (VGGish)</td>${v("90.19", ".06")}<td>95.2<sup>†</sup></td></tr>`,
    `<tr><td>TalkNet (MFCC)</td>${v("89.11", ".18")}<td>92.3<sup>†</sup></td></tr>`,
  ].join("");

  // Table 1
  const CORPUS = [
    ["Scenes with ≥ 5 faces", 6.3, 0.1, 69.6],
    ["Scenes with ≤ 3 faces", 88.8, 80.0, 15.4],
    ["Competing negatives", 19.4, 79.7, 76.2],
    ["Overlapping speech", 0.5, 9.9, 12.0],
  ];
  $("t-corpus").tBodies[0].innerHTML = CORPUS.map((r) => `<tr><td>${r[0]}</td>${r.slice(1).map((x) =>
    `<td class="bar-cell">${x.toFixed(1)}<span class="bar"><i style="width:${x}%"></i></span></td>`).join("")}</tr>`).join("");

  // ---------- svg helpers ----------
  const el = (name, attrs = {}, parent) => {
    const node = document.createElementNS(NS, name);
    for (const [k, val] of Object.entries(attrs)) node.setAttribute(k, val);
    if (parent) parent.appendChild(node);
    return node;
  };
  const text = (parent, x, y, str, attrs = {}) => { const t = el("text", { x, y, ...attrs }, parent); t.textContent = str; return t; };
  const tip = (node, str) => { const t = el("title", {}, node); t.textContent = str; };
  const linear = (d0, d1, r0, r1) => (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);

  // ---------- chart: scene width ----------
  let occMetric = "fa";
  function drawOcc() {
    const svg = $("c-occ"); svg.innerHTML = "";
    const W = 560, H = 330, m = { l: 48, r: 92, t: 16, b: 78 };
    const key = occMetric;
    const all = OCC.series.flatMap((s) => s[key]);
    const lo = key === "fa" ? 4 : 82, hi = key === "fa" ? 18 : 98;
    const x = (i) => m.l + 30 + i * ((W - m.l - m.r - 60) / 3);
    const y = linear(lo, hi, H - m.b, m.t);
    const grid = el("g", { class: "grid" }, svg);
    for (let t = lo; t <= hi; t += key === "fa" ? 2 : 2) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, grid);
      text(svg, m.l - 8, y(t) + 4, t, { "text-anchor": "end" });
    }
    text(svg, 14, (m.t + H - m.b) / 2, key === "fa" ? "FA_c (%) ↓" : "mAP ↑",
      { transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", class: "ttl" });
    ["2", "3", "4", "5+"].forEach((lab, i) => {
      text(svg, x(i), H - m.b + 18, lab, { "text-anchor": "middle", class: "ttl" });
      const bw = 44, share = OCC.share[i];
      el("rect", { x: x(i) - bw / 2, y: H - m.b + 30, width: bw, height: 6, rx: 3, fill: "var(--surface-2)" }, svg);
      el("rect", { x: x(i) - bw / 2, y: H - m.b + 30, width: bw * share / 100, height: 6, rx: 3, fill: "var(--muted)", opacity: 0.55 }, svg);
      text(svg, x(i), H - m.b + 52, `${share}% of frames`, { "text-anchor": "middle", "font-size": 11 });
    });
    text(svg, (m.l + W - m.r) / 2, H - 6, "faces visible at the frame", { "text-anchor": "middle", "font-size": 11.5 });
    OCC.series.forEach((s) => {
      const d = s[key].map((val, i) => `${i ? "L" : "M"}${x(i)},${y(val)}`).join("");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.ours ? 3 : 2, "stroke-linejoin": "round", "stroke-dasharray": s.ours ? "" : "5 4" }, svg);
      s[key].forEach((val, i) => {
        const c = el("circle", { cx: x(i), cy: y(val), r: s.ours ? 5 : 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        tip(c, `${s.name} · ${["2", "3", "4", "5+"][i]} faces: ${val.toFixed(2)}`);
      });
    });
    // annotate the 5+ gap between LoCoNet and CAST x1
    if (key === "fa") {
      const a = y(11.39), b = y(8.15), xx = x(3) + 14;
      el("path", { d: `M${xx},${a} L${xx + 6},${a} L${xx + 6},${b} L${xx},${b}`, fill: "none", stroke: "var(--text)", "stroke-width": 1.2 }, svg);
      text(svg, xx + 12, (a + b) / 2 - 3, "LoCoNet 11.39", { class: "ttl", "font-size": 11 });
      text(svg, xx + 12, (a + b) / 2 + 12, "CAST 8.15", { class: "ttl", "font-size": 11 });
    }
    $("occ-legend").innerHTML = OCC.series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  }
  document.querySelectorAll("#occ-metric button").forEach((b) => b.addEventListener("click", () => {
    occMetric = b.dataset.v;
    document.querySelectorAll("#occ-metric button").forEach((o) => o.setAttribute("aria-pressed", o === b));
    drawOcc();
  }));
  drawOcc();

  // ---------- chart: params vs FA_c ----------
  (function drawParams() {
    const svg = $("c-params");
    const W = 560, H = 340, m = { l: 48, r: 18, t: 16, b: 46 };
    const lx = (p) => Math.log10(p);
    const x = linear(lx(1), lx(50), m.l, W - m.r);
    const y = linear(7.6, 12.2, H - m.b, m.t);
    const grid = el("g", { class: "grid" }, svg);
    [1, 2, 5, 10, 20, 50].forEach((t) => {
      el("line", { x1: x(lx(t)), x2: x(lx(t)), y1: m.t, y2: H - m.b }, grid);
      text(svg, x(lx(t)), H - m.b + 18, `${t}M`, { "text-anchor": "middle" });
    });
    for (let t = 8; t <= 12; t += 1) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, grid);
      text(svg, m.l - 8, y(t) + 4, t, { "text-anchor": "end" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 8, "scene-mixer parameters (log scale)", { "text-anchor": "middle", "font-size": 11.5 });
    text(svg, 14, (m.t + H - m.b) / 2, "FA_c on many.val (%) ↓", { transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", class: "ttl" });

    // 24x annotation: CAST x1 (Mimi) to LoCoNet s=5 x3 (Mimi)
    const a = [x(lx(1.57)), y(8.61)], b = [x(lx(37.57)), y(9.42)];
    el("path", { d: `M${a[0] + 10},${a[1] + 18} Q${(a[0] + b[0]) / 2},${a[1] + 46} ${b[0] - 6},${b[1] + 14}`, fill: "none", stroke: "var(--faint)", "stroke-width": 1.3, "stroke-dasharray": "4 4" }, svg);
    text(svg, (a[0] + b[0]) / 2 + 6, a[1] + 48, "24× the parameters, more false alarms", { "text-anchor": "middle", "font-size": 11.5 });

    const pts = OP.map((r) => ({ audio: r[0], mixer: r[1], s: r[2], blocks: r[3], p: r[4], fa: parseFloat(r[7]), sd: parseFloat("0" + r[8]) }));
    pts.forEach((q) => {
      const cx = x(lx(q.p)), cy = y(q.fa);
      const color = q.mixer === "CAST" ? "var(--accent)" : "#f08c00";
      el("line", { x1: cx, x2: cx, y1: y(q.fa - q.sd), y2: y(q.fa + q.sd), stroke: color, "stroke-width": 1.5, opacity: 0.6 }, svg);
      const label = `${q.mixer}${q.s ? ` s=${q.s}` : ""} ×${q.blocks} (${q.audio}): ${q.p.toFixed(2)}M, FA_c ${q.fa.toFixed(2)} ± ${q.sd.toFixed(2)}`;
      let mark;
      if (q.audio === "Mimi") mark = el("circle", { cx, cy, r: 6, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      else mark = el("rect", { x: cx - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: "var(--surface)", stroke: color, "stroke-width": 2.2 }, svg);
      tip(mark, label);
      if (q.audio === "Mimi") {
        const name = q.mixer === "CAST" ? `CAST ×${q.blocks}` : `s=${q.s} ×${q.blocks}`;
        text(svg, cx + 9, cy - 8, name, { "font-size": 11.5, class: q.mixer === "CAST" ? "ttl" : "" });
      }
    });
  })();

  // ---------- chart: cost of causality ----------
  (function drawCost() {
    const svg = $("c-cost");
    const W = 560, H = 290, m = { l: 150, r: 56, t: 14, b: 40 };
    const x = linear(84, 93, m.l, W - m.r);
    const row = (H - m.t - m.b) / COST.length;
    const grid = el("g", { class: "grid" }, svg);
    for (let t = 84; t <= 93; t += 1) {
      el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }, grid);
      if (t % 2 === 0) text(svg, x(t), H - m.b + 18, t, { "text-anchor": "middle" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 6, "many.val mAP", { "text-anchor": "middle", "font-size": 11.5 });
    COST.forEach((r, i) => {
      const cy = m.t + row * (i + 0.5);
      const ours = !!r[5];
      const color = ours ? "var(--accent)" : "var(--muted)";
      text(svg, m.l - 12, cy + 4, r[0], { "text-anchor": "end", class: ours ? "ttl" : "" });
      el("line", { x1: x(r[1]), x2: x(r[2]), y1: cy, y2: cy, stroke: color, "stroke-width": ours ? 4 : 3, "stroke-linecap": "round", opacity: 0.35 }, svg);
      const c1 = el("circle", { cx: x(r[1]), cy, r: 6.5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
      const c2 = el("circle", { cx: x(r[2]), cy, r: 6, fill: "var(--surface)", stroke: color, "stroke-width": 2.4 }, svg);
      tip(c1, `${r[0]} causal: ${r[1].toFixed(2)}`); tip(c2, `${r[0]} offline: ${r[2].toFixed(2)}`);
      text(svg, x(r[2]) + 12, cy + 4, `Δ ${r[3]}`, { class: ours ? "ttl" : "", "font-size": 12 });
    });
  })();

  window.addEventListener("themechange", drawOcc);

  // ---------- receptive field widget ----------
  (function receptiveField() {
    const F = 5, T = 16;
    const grid = $("rf-grid"), caption = $("rf-caption");
    grid.style.gridTemplateColumns = `46px repeat(${T}, minmax(0, 1fr))`;
    const cells = [];
    for (let i = 0; i < F; i++) {
      const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = `face ${i + 1}`; grid.appendChild(lab);
      cells.push([]);
      for (let t = 0; t < T; t++) {
        const c = document.createElement("div"); c.className = "cell";
        c.addEventListener("click", () => { target = [i, t]; auto = false; paint(); });
        grid.appendChild(c); cells[i].push(c);
      }
    }
    $("rf-axis").style.paddingLeft = "50px";
    let target = [1, 10], step = 0, auto = true;
    const CAPTIONS = [
      "Speaker pass. At each frame, a face is compared with every face present at that same frame. The pass has no time axis, so nothing from the future can enter.",
      "Time pass. The result is then propagated along the face's own row under a causal mask: frames after t are hidden.",
      "Together, the output for face i at frame t draws on every face in the scene, but only on frames up to t. That holds for any number of faces.",
    ];
    function paint() {
      const [ti, tt] = target;
      for (let i = 0; i < F; i++) for (let t = 0; t < T; t++) {
        const c = cells[i][t]; c.className = "cell";
        if (t > tt) c.classList.add("future");
        if (step === 0 && t === tt) c.classList.add("col");
        if (step === 1 && i === ti && t <= tt) c.classList.add("row");
        if (step === 2 && t <= tt) c.classList.add("field");
        if (i === ti && t === tt) c.classList.add("target");
      }
      caption.textContent = CAPTIONS[step];
      document.querySelectorAll("#rf-steps button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.step === step));
    }
    document.querySelectorAll("#rf-steps button").forEach((b) => b.addEventListener("click", () => { step = +b.dataset.step; auto = false; paint(); }));
    paint();
    setInterval(() => { if (auto && document.visibilityState === "visible") { step = (step + 1) % 3; paint(); } }, 2600);
  })();
})();
