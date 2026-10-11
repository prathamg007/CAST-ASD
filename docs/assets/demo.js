/* Interactive demo: video with live speaker boxes, diarization tracks and speaker-pass attention.
 * Per-clip data (demo/<id>/data.js) comes from the released checkpoints via tools/make_demo.py. */
(() => {
  const CLIPS = [
    { id: "01242_1", faces: 5, lang: "English" },
    { id: "01984_0", faces: 5, lang: "Chinese" },
    { id: "00934_0", faces: 9, lang: "Korean" },
    { id: "00776_0", faces: 5, lang: "English" },
    { id: "02552_0", faces: 9, lang: "Korean" },
    { id: "00706_0", faces: 4, lang: "Filipino" },
    { id: "00884_1", faces: 5, lang: "Japanese" },
    { id: "03141_1", faces: 5, lang: "English" },
    { id: "02808_1", faces: 6, lang: "English" },
    { id: "00727_0", faces: 5, lang: "Chinese" },
    { id: "01036_0", faces: 5, lang: "Hindi" },
    { id: "02904_2", faces: 4, lang: "English" },
    { id: "00134_0", faces: 4, lang: "Hindi" },
    { id: "00897_0", faces: 2, lang: "English" },
  ];
  const DEFAULT = "01242_1";
  const COLORS = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#f06595", "#94d82d"];
  const ROW = 38, AXIS = 22;

  const $ = (id) => document.getElementById(id);
  const video = $("video"), stage = $("stage"), overlay = $("overlay"), octx = overlay.getContext("2d");
  const tl = $("tl"), tctx = tl.getContext("2d"), tlWrap = $("tl-wrap");
  const state = { clip: null, mode: "causal", hl: "pred", err: false, cmp: false, frame: -1, layers: null, token: 0 };
  const LANE = 13, GUTTER = 66;
  const DEFAULT_MODELS = [{ key: "causal", label: "CAST-ASD (causal)", causal: true }, { key: "offline", label: "CAST-ASD (offline)", causal: false }];
  const shortName = (m) => m.label.split(" (")[0];
  const plainName = (m) => m.label.replace(/[()]/g, "");
  const comparing = () => state.cmp && state.clip && state.clip.compare.length > 1;
  const rowH = () => (comparing() ? 14 + LANE * state.clip.compare.length : ROW);
  const gutter = () => (comparing() ? GUTTER : 0);

  // ---------- data ----------
  const b64 = (s) => { const bin = atob(s); const a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; };
  const mask = (runs, n) => { const m = new Uint8Array(n); runs.forEach(([a, b]) => m.fill(1, a, b)); return m; };
  const loadScript = (src) => new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); });

  async function getClip(id) {
    window.CAST_DEMO = window.CAST_DEMO || {};
    if (!window.CAST_DEMO[id]) await loadScript(`demo/${id}/data.js`);
    const d = window.CAST_DEMO[id];
    if (!d._ready) {
      d.tracks.forEach((t, k) => {
        t.p = {};
        (d.models || DEFAULT_MODELS).forEach((m) => { t.p[m.key] = b64(t[m.key]); });
        t.ref = mask(t.gt, d.frames);
        t.color = COLORS[k % COLORS.length];
        t.thumb = `demo/${id}/face_${k}.jpg`;
      });
      d.models = d.models || DEFAULT_MODELS;
      d.compare = d.models.filter((m) => m.causal);        // systems shown side by side
      d.off = mask(d.offscreen, d.frames);
      d.id = id;
      d._ready = true;
    }
    return d;
  }

  // ---------- clip picker ----------
  const picker = $("clips");
  picker.innerHTML = CLIPS.map((c) => `
    <button class="clip-card" type="button" data-id="${c.id}" aria-pressed="false">
      <div class="thumb" style="background-image:url('demo/${c.id}/poster.jpg')"></div>
      <div class="meta"><div class="name">${c.faces} faces · ${c.lang}</div><div class="split">MSDWild ${c.id.split("_")[0]}</div></div>
    </button>`).join("");
  picker.addEventListener("click", (e) => { const b = e.target.closest(".clip-card"); if (b) select(b.dataset.id, true); });

  async function select(id, userAction) {
    const token = ++state.token;
    picker.querySelectorAll(".clip-card").forEach((b) => b.setAttribute("aria-pressed", b.dataset.id === id));
    $("loading").hidden = false;
    video.pause();
    const d = await getClip(id);
    if (token !== state.token) return;
    state.clip = d; state.frame = -1;
    $("clip-tag").textContent = `${d.tracks.length} faces`;
    video.src = `demo/${id}/clip.mp4`;
    video.poster = `demo/${id}/poster.jpg`;
    video.load();
    buildPanels(d);
    buildModes(d);
    buildMetrics(d);
    resize();
    render(0, true);
    if (userAction) video.play().catch(() => {});
  }
  video.addEventListener("loadeddata", () => { $("loading").hidden = true; render(currentFrame(), true); });
  video.addEventListener("waiting", () => { $("loading").hidden = false; });
  video.addEventListener("playing", () => { $("loading").hidden = true; });

  // ---------- side panels ----------
  function buildPanels(d) {
    $("faces").innerHTML = d.tracks.map((t, k) => `
      <div class="face-row" style="--c:${t.color}" data-k="${k}">
        <img src="${t.thumb}" alt="Face ${k + 1}">
        <div><div class="meter"><i></i><b></b></div>
          <div class="small muted" style="margin-top:3px;font-size:11.5px">Face ${k + 1}<span class="ref" title="Ground truth: speaking"></span></div></div>
        <span class="pct">0%</span>
      </div>`).join("");

    state.faceRows = [...$("faces").querySelectorAll(".face-row")];

    $("tl-faces").innerHTML = d.tracks.map((t, k) => `<div style="--c:${t.color}"><img src="${t.thumb}" alt="Face ${k + 1}"></div>`).join("") +
      `<div><span class="off">off-screen</span></div>`;
  }

  function buildModes(d) {
    if (!d.models.some((m) => m.key === state.mode)) state.mode = "causal";
    const two = d.models.length === 2;
    $("mode").innerHTML = d.models.map((m) => `<button type="button" data-v="${m.key}" aria-pressed="${m.key === state.mode}">` +
      `${two ? (m.causal ? "Causal" : "Offline") : plainName(m)}</button>`).join("");
    $("cmp-wrap").hidden = d.compare.length < 2;
    $("cmp-note").hidden = d.compare.length < 2;
    setMode(state.mode);
  }

  function buildMetrics(d) {
    const f = (x, s = 1) => (x == null || Number.isNaN(x) ? "–" : (x * s).toFixed(1));
    $("cm").tBodies[0].innerHTML = d.models.map((m) => {
      const v = d.metrics[m.key];
      return `<tr data-mode="${m.key}"${state.mode === m.key ? ' style="background:var(--accent-soft)"' : ""}><td>${m.label}</td>` +
        `<td>${f(v.ap)}</td><td>${f(v.ap_competing)}</td><td>${f(v["fa@0.5"], 100)}</td><td>${f(v["fa_competing@0.5"], 100)}</td>` +
        `<td>${f(v.oracle_der)}</td><td>${f(v.f1, 100)}</td></tr>`;
    }).join("");
  }

  // ---------- geometry ----------
  let dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = stage.getBoundingClientRect();
    overlay.width = Math.round(r.width * dpr); overlay.height = Math.round(r.height * dpr);
    const d = state.clip;
    if (d) {
      const rows = d.tracks.length + 1;
      const w = tlWrap.clientWidth, h = rows * rowH() + AXIS;
      $("tl-faces").style.setProperty("--rh", `${rowH()}px`);
      tl.width = Math.round(w * dpr); tl.height = Math.round(h * dpr); tl.style.height = `${h}px`;
      buildLayers();
    }
    render(currentFrame(), true);
  }
  new ResizeObserver(resize).observe(stage);
  new ResizeObserver(() => { if (state.clip) resize(); }).observe(tlWrap);
  window.addEventListener("themechange", () => { buildLayers(); render(currentFrame(), true); });

  function contentRect() {
    const d = state.clip, W = overlay.width, H = overlay.height;
    const ar = d.width / d.height;
    let w = W, h = W / ar;
    if (h > H) { h = H; w = H * ar; }
    return { x: (W - w) / 2, y: (H - h) / 2, w, h };
  }

  // ---------- timeline layers (pre-rendered) ----------
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function rgba(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }

  function buildLayers() {
    const d = state.clip; if (!d || !tl.width) return;
    const W = tl.width, H = tl.height, n = d.frames, s = dpr;
    const ROW = rowH(), G = gutter() * s, cmp = comparing();
    const xs = (f) => G + (f / n) * (W - G);
    const make = () => { const c = document.createElement("canvas"); c.width = W; c.height = H; return c; };
    const ref = make(), rc = ref.getContext("2d");
    const refColor = css("--ref"), surf2 = css("--surface-2");
    const rowsAll = d.tracks.length + 1;
    for (let r = 0; r < rowsAll; r++) {
      rc.fillStyle = surf2;
      roundRect(rc, G, (r * ROW + 2) * s, W - G, (ROW - 4) * s, 6 * s); rc.fill();
    }
    if (cmp) {
      rc.font = `600 ${10 * s}px Inter, system-ui, sans-serif`; rc.textBaseline = "middle"; rc.textAlign = "right";
      rc.fillStyle = css("--muted");
      d.tracks.forEach((t, r) => d.compare.forEach((m, j) =>
        rc.fillText(shortName(m), G - 6 * s, (r * ROW + 13 + j * LANE + 5) * s)));
    }
    const strip = (r, m, color) => {
      rc.fillStyle = color;
      let a = -1;
      for (let f = 0; f <= n; f++) {
        const on = f < n && m[f];
        if (on && a < 0) a = f;
        if (!on && a >= 0) { roundRect(rc, xs(a), (r * ROW + (cmp ? 5 : 6)) * s, Math.max(1, xs(f) - xs(a)), (cmp ? 4 : 5) * s, 2 * s); rc.fill(); a = -1; }
      }
    };
    d.tracks.forEach((t, r) => strip(r, t.ref, rgba(hexOf(refColor), 0.75)));
    // off-screen row: reference only, drawn taller
    rc.fillStyle = rgba(hexOf(refColor), 0.45);
    runs(d.off).forEach(([a, b]) => { roundRect(rc, xs(a), (d.tracks.length * ROW + 10) * s, Math.max(1, xs(b) - xs(a)), (ROW - 20) * s, 3 * s); rc.fill(); });

    const pred = {}, err = {};
    for (const mode of d.models.map((m) => m.key)) {
      const pc = make(), pctx = pc.getContext("2d");
      const ec = make(), ectx = ec.getContext("2d");
      d.tracks.forEach((t, r) => {
        const p = t.p[mode], top = (r * ROW + 13) * s, hh = (ROW - 18) * s;
        // probability area
        pctx.beginPath(); pctx.moveTo(0, top + hh);
        for (let f = 0; f < n; f++) pctx.lineTo(xs(f + 0.5), top + hh - (p[f] / 255) * hh);
        pctx.lineTo(W, top + hh); pctx.closePath();
        pctx.fillStyle = rgba(t.color, 0.22); pctx.fill();
        // decisions
        pctx.fillStyle = t.color;
        const dec = new Uint8Array(n); for (let f = 0; f < n; f++) dec[f] = p[f] >= 128;
        runs(dec).forEach(([a, b]) => { roundRect(pctx, xs(a), top + hh - 9 * s, Math.max(1.5, xs(b) - xs(a)), 9 * s, 2.5 * s); pctx.fill(); });
        ectx.fillStyle = css("--bad");
        const bad = new Uint8Array(n); for (let f = 0; f < n; f++) bad[f] = dec[f] !== t.ref[f];
        runs(bad).forEach(([a, b]) => ectx.fillRect(xs(a), (r * ROW + ROW - 4) * s, Math.max(1, xs(b) - xs(a)), 3 * s));
      });
      pred[mode] = pc; err[mode] = ec;
    }
    // side by side: one lane per causal system; correct detections in the face colour, false alarms in red
    let lanes = null;
    if (cmp) {
      lanes = make();
      const lc = lanes.getContext("2d");
      // false alarms are hatched as well as red: some face colours are close to red
      const tile = document.createElement("canvas"); tile.width = tile.height = Math.round(6 * s);
      const tc = tile.getContext("2d");
      tc.fillStyle = css("--bad"); tc.fillRect(0, 0, tile.width, tile.height);
      tc.strokeStyle = "rgba(0,0,0,0.55)"; tc.lineWidth = 1.6 * s;
      for (const o of [-tile.width, 0, tile.width]) { tc.beginPath(); tc.moveTo(o, tile.height); tc.lineTo(o + tile.width, 0); tc.stroke(); }
      const bad = lc.createPattern(tile, "repeat");
      d.tracks.forEach((t, r) => d.compare.forEach((m, j) => {
        const p = t.p[m.key], y = (r * ROW + 13 + j * LANE) * s, hh = (LANE - 3) * s;
        const hit = new Uint8Array(n), fa = new Uint8Array(n);
        for (let f = 0; f < n; f++) { const on = p[f] >= 128; hit[f] = on && t.ref[f]; fa[f] = on && !t.ref[f]; }
        lc.fillStyle = t.color;
        runs(hit).forEach(([a, b]) => { roundRect(lc, xs(a), y, Math.max(1.5, xs(b) - xs(a)), hh, 2 * s); lc.fill(); });
        lc.fillStyle = bad;
        runs(fa).forEach(([a, b]) => { roundRect(lc, xs(a), y, Math.max(1.5, xs(b) - xs(a)), hh, 2 * s); lc.fill(); });
      }));
    }
    state.layers = { ref, pred, err, lanes };
  }
  function hexOf(c) {
    if (c.startsWith("#") && c.length === 7) return c;
    if (c.startsWith("#") && c.length === 4) return "#" + [...c.slice(1)].map((x) => x + x).join("");
    return "#8a93a6";
  }
  function runs(m) { const out = []; let a = -1; for (let f = 0; f <= m.length; f++) { const on = f < m.length && m[f]; if (on && a < 0) a = f; if (!on && a >= 0) { out.push([a, f]); a = -1; } } return out; }
  function roundRect(c, x, y, w, h, r) { w = Math.max(0, w); h = Math.max(0, h); r = Math.max(0, Math.min(r, w / 2, h / 2)); c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

  // ---------- rendering ----------
  const currentFrame = () => { const d = state.clip; if (!d) return 0; return Math.max(0, Math.min(d.frames - 1, Math.floor(video.currentTime * d.fps + 1e-3))); };
  const fmt = (t) => { const m = Math.floor(t / 60), s = t - m * 60; return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`; };

  function render(f, force) {
    const d = state.clip; if (!d) return;
    if (f === state.frame && !force) return;
    state.frame = f;
    drawOverlay(d, f);
    drawTimeline(d, f);
    updatePanels(d, f);
    $("time").textContent = `${fmt(f / d.fps)} / ${fmt(d.frames / d.fps)} · frame ${f}`;
  }

  function drawOverlay(d, f) {
    const c = octx, W = overlay.width, H = overlay.height;
    c.clearRect(0, 0, W, H);
    const R = contentRect();
    const base = Math.max(10, Math.min(15, R.w / dpr / 64)) * dpr;
    c.font = `700 ${base}px Inter, system-ui, sans-serif`;
    c.textBaseline = "middle";
    // draw quiet faces first so speaking boxes sit on top
    const order = d.tracks.map((t, k) => k).sort((a, b) => d.tracks[a].p[state.mode][f] - d.tracks[b].p[state.mode][f]);
    for (const k of order) {
      const t = d.tracks[k], b = t.boxes[f];
      if (!b) continue;
      const p = t.p[state.mode][f] / 255;
      const on = state.hl === "pred" ? p >= 0.5 : !!t.ref[f];
      const x = R.x + b[0] * R.w, y = R.y + b[1] * R.h, w = (b[2] - b[0]) * R.w, h = (b[3] - b[1]) * R.h;
      const rad = Math.min(w, h) * 0.12;
      c.save();
      if (on) {
        c.shadowColor = t.color; c.shadowBlur = 22 * dpr;
        c.lineWidth = 3.5 * dpr; c.strokeStyle = t.color;
        c.fillStyle = rgba(t.color, 0.1);
        roundRect(c, x, y, w, h, rad); c.fill(); c.stroke();
        c.shadowBlur = 0;
      } else {
        c.lineWidth = 1.6 * dpr; c.strokeStyle = "rgba(255,255,255,0.55)";
        c.setLineDash([6 * dpr, 4 * dpr]);
        roundRect(c, x, y, w, h, rad); c.stroke();
        c.setLineDash([]);
      }
      const gt = state.hl === "gt";
      // probability bar along the bottom edge (model output only)
      const bw = w - 8 * dpr, bh = 4 * dpr, bx = x + 4 * dpr, by = y + h - bh - 4 * dpr;
      if (!gt) {
        c.fillStyle = "rgba(0,0,0,0.45)"; roundRect(c, bx, by, bw, bh, bh / 2); c.fill();
        c.fillStyle = on ? t.color : "rgba(255,255,255,0.75)"; roundRect(c, bx, by, Math.max(bh, bw * p), bh, bh / 2); c.fill();
      }
      // label chip
      const label = gt ? (on ? "speaking" : "silent") : `P ${Math.round(p * 100)}%`;
      const pad = 7 * dpr, dot = 7 * dpr, tw = c.measureText(label).width;
      const cw = tw + pad * 2 + dot + 5 * dpr, ch = base + 9 * dpr;
      let cx = x, cy = y - ch - 5 * dpr;
      if (cy < R.y + 2) cy = y + 5 * dpr;
      cx = Math.min(Math.max(cx, R.x + 2), R.x + R.w - cw - 2);
      c.fillStyle = on ? t.color : "rgba(10,12,18,0.68)";
      roundRect(c, cx, cy, cw, ch, ch / 2); c.fill();
      c.fillStyle = on ? "#0b0e15" : t.color;
      c.beginPath(); c.arc(cx + pad + dot / 2, cy + ch / 2, dot / 2, 0, Math.PI * 2); c.fill();
      c.fillStyle = on ? "#0b0e15" : "#fff";
      c.fillText(label, cx + pad + dot + 5 * dpr, cy + ch / 2 + 0.5);
      c.restore();
    }
  }

  function drawTimeline(d, f) {
    const L = state.layers; if (!L) return;
    const c = tctx, W = tl.width, H = tl.height, s = dpr;
    c.clearRect(0, 0, W, H);
    c.drawImage(L.ref, 0, 0);
    const ROW = rowH(), G = gutter() * s, cmp = comparing() && L.lanes;
    const px = G + ((f + 1) / d.frames) * (W - G);
    const streaming = cmp || d.models.find((m) => m.key === state.mode).causal;
    const visible = streaming ? px : W;
    c.save(); c.beginPath(); c.rect(0, 0, visible, H); c.clip();
    if (cmp) c.drawImage(L.lanes, 0, 0);
    else {
      c.drawImage(L.pred[state.mode], 0, 0);
      if (state.err) c.drawImage(L.err[state.mode], 0, 0);
    }
    c.restore();
    // axis
    const rowsH = (d.tracks.length + 1) * ROW * s;
    c.fillStyle = css("--faint"); c.font = `500 ${11 * s}px Inter, system-ui, sans-serif`; c.textBaseline = "top";
    const secs = d.frames / d.fps;
    for (let t = 0; t <= secs + 1e-6; t += 5) {
      const x = G + (t / secs) * (W - G);
      c.fillRect(Math.min(x, W - s), rowsH + 1 * s, s, 4 * s);
      c.textAlign = t === 0 ? "left" : t >= secs - 0.01 ? "right" : "center";
      c.fillText(`${t}s`, x, rowsH + 7 * s);
    }
    // playhead
    const accent = css("--accent");
    c.fillStyle = accent;
    c.fillRect(px - 1 * s, 0, 2 * s, rowsH);
    if (streaming) {
      const g = c.createLinearGradient(px - 40 * s, 0, px, 0);
      g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, rgba(hexOf(accent), 0.12));
      c.fillStyle = g; c.fillRect(px - 40 * s, 0, 40 * s, rowsH);
    }
  }

  function updatePanels(d, f) {
    d.tracks.forEach((t, k) => {
      const p = t.p[state.mode][f] / 255;
      const on = state.hl === "pred" ? p >= 0.5 : !!t.ref[f];
      const row = state.faceRows[k]; if (!row) return;
      row.classList.toggle("on", on);
      const gt = state.hl === "gt";
      row.querySelector(".meter i").style.width = gt ? (on ? "100%" : "0%") : `${(p * 100).toFixed(1)}%`;
      row.querySelector(".pct").textContent = gt ? (on ? "yes" : "no") : `${Math.round(p * 100)}%`;
      row.querySelector(".ref").classList.toggle("on", !!t.ref[f]);
    });
  }

  // ---------- playback loop ----------
  const hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  function onFrame() { render(currentFrame()); if (hasRVFC) video.requestVideoFrameCallback(onFrame); }
  if (hasRVFC) video.requestVideoFrameCallback(onFrame);
  (function raf() { if (!hasRVFC && !video.paused) render(currentFrame()); requestAnimationFrame(raf); })();
  ["seeked", "timeupdate", "loadedmetadata"].forEach((ev) => video.addEventListener(ev, () => render(currentFrame(), ev !== "timeupdate")));

  // ---------- controls ----------
  const setIcon = (btn, id) => btn.querySelector("use").setAttribute("href", id);
  const togglePlay = () => (video.paused || video.ended ? video.play().catch(() => {}) : video.pause());
  $("play").addEventListener("click", togglePlay);
  $("big-play").addEventListener("click", togglePlay);
  video.addEventListener("click", togglePlay);
  video.addEventListener("play", () => { stage.classList.add("playing"); setIcon($("play"), "#i-pause"); });
  video.addEventListener("pause", () => { stage.classList.remove("playing"); setIcon($("play"), "#i-play"); });
  video.addEventListener("ended", () => { stage.classList.remove("playing"); setIcon($("play"), "#i-play"); });

  const step = (k) => {
    const d = state.clip; if (!d) return;
    video.pause();
    const f = Math.max(0, Math.min(d.frames - 1, currentFrame() + k));
    video.currentTime = (f + 0.5) / d.fps;
  };
  $("step-back").addEventListener("click", () => step(-1));
  $("step-fwd").addEventListener("click", () => step(1));
  $("mute").addEventListener("click", () => { video.muted = !video.muted; setIcon($("mute"), video.muted ? "#i-mute" : "#i-vol"); });

  const seg = (id, fn) => $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll(`#${id} button`).forEach((o) => o.setAttribute("aria-pressed", o === b));
    fn(b.dataset.v);
  });
  seg("speed", (v) => { video.playbackRate = parseFloat(v); });
  const setMode = (v) => {
    state.mode = v;
    const models = state.clip ? state.clip.models : DEFAULT_MODELS;
    const m = models.find((x) => x.key === v) || models[0];
    const who = models.length === 2 ? (m.causal ? "Causal" : "Offline") : plainName(m);
    $("mode-note").textContent = m.causal
      ? `${who}: each score uses only video and audio up to the current frame, so the tracks fill in as the clip plays.`
      : `${who}: the model reads the whole 30 s clip before labelling any frame, so all of its tracks are available at once.`;
    const tag = $("mode-tag");
    tag.textContent = models.length === 2 ? (m.causal ? "Causal · streaming" : "Offline · whole clip") : `${shortName(m)} · ${m.causal ? "causal" : "offline"}`;
    tag.classList.toggle("live", m.causal);
    if (state.clip) buildMetrics(state.clip);
    render(currentFrame(), true);
  };
  seg("mode", setMode);
  const setHl = (v) => {
    state.hl = v;
    $("faces-unit").textContent = v === "gt" ? "ground truth" : "P(speaking)";
    $("box-note").textContent = v === "gt"
      ? "Boxes show the ground-truth annotation: solid when the person is labelled as speaking. This view does not depend on the model, so it is the same whichever model is selected."
      : "The number on each box is the model's P(speaking) for that face at the current frame. Solid coloured box: P ≥ 0.5. Dashed box: P < 0.5.";
    render(currentFrame(), true);
  };
  seg("hl", setHl);
  setHl("pred");
  $("show-err").addEventListener("change", (e) => { state.err = e.target.checked; $("err-legend").hidden = !state.err; render(currentFrame(), true); });
  $("cmp").addEventListener("change", (e) => {
    state.cmp = e.target.checked;
    document.querySelectorAll(".cmp-only").forEach((n) => { n.hidden = !state.cmp; });
    document.querySelectorAll(".single-only").forEach((n) => { n.hidden = state.cmp; });
    resize();
  });
  setMode("causal");

  // scrub on the timeline
  let dragging = false;
  const seekTo = (e) => {
    const d = state.clip; if (!d) return;
    const r = tl.getBoundingClientRect();
    const g = gutter();
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left - g) / (r.width - g)));
    const f = Math.min(d.frames - 1, Math.floor(frac * d.frames));
    video.currentTime = (f + 0.5) / d.fps;
  };
  tlWrap.addEventListener("pointerdown", (e) => { dragging = true; tlWrap.setPointerCapture(e.pointerId); seekTo(e); });
  tlWrap.addEventListener("pointermove", (e) => { if (dragging) seekTo(e); });
  tlWrap.addEventListener("pointerup", () => { dragging = false; });

  // keyboard: space / arrows while the demo is on screen
  let demoVisible = false;
  new IntersectionObserver(([e]) => { demoVisible = e.isIntersecting; }, { threshold: 0.3 }).observe(stage);
  document.addEventListener("keydown", (e) => {
    if (!demoVisible || e.target.closest("input, textarea, select, button") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  });

  select(DEFAULT, false);
})();
