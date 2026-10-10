/* Interactive demo: video with live speaker boxes, diarization tracks and speaker-pass attention.
 * Per-clip data (demo/<id>/data.js) comes from the released checkpoints via tools/make_demo.py. */
(() => {
  const CLIPS = [
    { id: "01242_1", faces: 5, split: "many.val", lang: "English", ap: "95.9" },
    { id: "00776_0", faces: 5, split: "many.val", lang: "English", ap: "94.3" },
    { id: "03141_1", faces: 5, split: "many.val", lang: "English", ap: "93.3" },
    { id: "00934_0", faces: 9, split: "many.val", lang: "Korean", ap: "97.6" },
    { id: "02808_1", faces: 6, split: "many.val", lang: "English", ap: "87.2" },
    { id: "00706_0", faces: 4, split: "few.val", lang: "Filipino", ap: "95.0" },
    { id: "01036_0", faces: 5, split: "many.val", lang: "Hindi", ap: "93.6" },
    { id: "00134_0", faces: 4, split: "few.val", lang: "Hindi", ap: "92.5" },
    { id: "00897_0", faces: 2, split: "few.val", lang: "English", ap: "98.5" },
  ];
  const DEFAULT = "01242_1";
  const COLORS = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#f06595", "#94d82d"];
  const ROW = 38, AXIS = 22;

  const $ = (id) => document.getElementById(id);
  const video = $("video"), stage = $("stage"), overlay = $("overlay"), octx = overlay.getContext("2d");
  const tl = $("tl"), tctx = tl.getContext("2d"), tlWrap = $("tl-wrap");
  const state = { clip: null, mode: "causal", hl: "pred", err: false, frame: -1, layers: null, token: 0 };

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
        t.p = { causal: b64(t.causal), offline: b64(t.offline) };
        t.ref = mask(t.gt, d.frames);
        t.color = COLORS[k % COLORS.length];
        t.thumb = `demo/${id}/face_${k}.jpg`;
      });
      d.off = mask(d.offscreen, d.frames);
      d.attn = b64(d.attention);
      d.id = id;
      d._ready = true;
    }
    return d;
  }

  // ---------- clip picker ----------
  const picker = $("clips");
  picker.innerHTML = CLIPS.map((c) => `
    <button class="clip-card" type="button" data-id="${c.id}" aria-pressed="false">
      <div class="thumb" style="background-image:url('demo/${c.id}/poster.jpg')"><span class="badge">mAP ${c.ap}</span></div>
      <div class="meta"><div class="name">${c.faces} faces · ${c.lang}</div><div class="split">${c.split} · ${c.id.split("_")[0]}</div></div>
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
    const meta = CLIPS.find((c) => c.id === id);
    $("clip-tag").textContent = `${meta.split} · ${d.tracks.length} faces`;
    video.src = `demo/${id}/clip.mp4`;
    video.poster = `demo/${id}/poster.jpg`;
    video.load();
    buildPanels(d);
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

    const F = d.tracks.length, attn = $("attn");
    attn.style.gridTemplateColumns = `24px repeat(${F}, minmax(0, 1fr))`;
    let html = "<div></div>" + d.tracks.map((t) => `<div class="h" style="--c:${t.color}"><img src="${t.thumb}" alt=""></div>`).join("");
    for (let i = 0; i < F; i++) {
      html += `<div class="r" style="--c:${d.tracks[i].color}"><img src="${d.tracks[i].thumb}" alt=""></div>`;
      for (let j = 0; j < F; j++) html += `<div class="cell" data-i="${i}" data-j="${j}"><span></span></div>`;
    }
    attn.innerHTML = html;
    state.attnCells = [...attn.querySelectorAll(".cell span")];
    state.attnRows = [...attn.querySelectorAll(".r")];
    state.faceRows = [...$("faces").querySelectorAll(".face-row")];

    $("tl-faces").innerHTML = d.tracks.map((t, k) => `<div style="--c:${t.color}"><img src="${t.thumb}" alt="Face ${k + 1}"></div>`).join("") +
      `<div><span class="off">off-screen</span></div>`;
  }

  function buildMetrics(d) {
    const f = (x, s = 1) => (x == null || Number.isNaN(x) ? "–" : (x * s).toFixed(1));
    const row = (name, m, key) => `<tr data-mode="${key}"${state.mode === key ? ' style="background:var(--accent-soft)"' : ""}><td>${name}</td>` +
      `<td>${f(m.ap)}</td><td>${f(m.ap_competing)}</td><td>${f(m.oracle_der)}</td><td>${f(m.f1, 100)}</td><td>${f(m.precision, 100)}</td><td>${f(m.recall, 100)}</td></tr>`;
    $("cm").tBodies[0].innerHTML = row("CAST-ASD causal", d.metrics.causal, "causal") + row("CAST-ASD offline", d.metrics.offline, "offline");
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
      const w = tlWrap.clientWidth, h = rows * ROW + AXIS;
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
    const xs = (f) => (f / n) * W;
    const make = () => { const c = document.createElement("canvas"); c.width = W; c.height = H; return c; };
    const ref = make(), rc = ref.getContext("2d");
    const refColor = css("--ref"), surf2 = css("--surface-2");
    const rowsAll = d.tracks.length + 1;
    for (let r = 0; r < rowsAll; r++) {
      rc.fillStyle = surf2;
      roundRect(rc, 0, (r * ROW + 2) * s, W, (ROW - 4) * s, 6 * s); rc.fill();
    }
    const strip = (r, m, color) => {
      rc.fillStyle = color;
      let a = -1;
      for (let f = 0; f <= n; f++) {
        const on = f < n && m[f];
        if (on && a < 0) a = f;
        if (!on && a >= 0) { roundRect(rc, xs(a), (r * ROW + 6) * s, Math.max(1, xs(f) - xs(a)), 5 * s, 2.5 * s); rc.fill(); a = -1; }
      }
    };
    d.tracks.forEach((t, r) => strip(r, t.ref, rgba(hexOf(refColor), 0.75)));
    // off-screen row: reference only, drawn taller
    rc.fillStyle = rgba(hexOf(refColor), 0.45);
    runs(d.off).forEach(([a, b]) => { roundRect(rc, xs(a), (d.tracks.length * ROW + 10) * s, Math.max(1, xs(b) - xs(a)), 18 * s, 3 * s); rc.fill(); });

    const pred = {}, err = {};
    for (const mode of ["causal", "offline"]) {
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
    state.layers = { ref, pred, err };
  }
  function hexOf(c) {
    if (c.startsWith("#") && c.length === 7) return c;
    if (c.startsWith("#") && c.length === 4) return "#" + [...c.slice(1)].map((x) => x + x).join("");
    return "#8a93a6";
  }
  function runs(m) { const out = []; let a = -1; for (let f = 0; f <= m.length; f++) { const on = f < m.length && m[f]; if (on && a < 0) a = f; if (!on && a >= 0) { out.push([a, f]); a = -1; } } return out; }
  function roundRect(c, x, y, w, h, r) { r = Math.min(r, w / 2, h / 2); c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

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
    const px = ((f + 1) / d.frames) * W;
    const visible = state.mode === "causal" ? px : W;
    c.save(); c.beginPath(); c.rect(0, 0, visible, H); c.clip();
    c.drawImage(L.pred[state.mode], 0, 0);
    if (state.err) c.drawImage(L.err[state.mode], 0, 0);
    c.restore();
    // axis
    const rowsH = (d.tracks.length + 1) * ROW * s;
    c.fillStyle = css("--faint"); c.font = `500 ${11 * s}px Inter, system-ui, sans-serif`; c.textBaseline = "top";
    const secs = d.frames / d.fps;
    for (let t = 0; t <= secs + 1e-6; t += 5) {
      const x = (t / secs) * W;
      c.fillRect(Math.min(x, W - s), rowsH + 1 * s, s, 4 * s);
      c.textAlign = t === 0 ? "left" : t >= secs - 0.01 ? "right" : "center";
      c.fillText(`${t}s`, x, rowsH + 7 * s);
    }
    // playhead
    const accent = css("--accent");
    c.fillStyle = accent;
    c.fillRect(px - 1 * s, 0, 2 * s, rowsH);
    if (state.mode === "causal") {
      const g = c.createLinearGradient(px - 40 * s, 0, px, 0);
      g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, rgba(hexOf(accent), 0.12));
      c.fillStyle = g; c.fillRect(px - 40 * s, 0, 40 * s, rowsH);
    }
  }

  function updatePanels(d, f) {
    const F = d.tracks.length;
    let top = -1, topP = -1;
    d.tracks.forEach((t, k) => {
      const p = t.p[state.mode][f] / 255;
      const on = state.hl === "pred" ? p >= 0.5 : !!t.ref[f];
      const row = state.faceRows[k]; if (!row) return;
      row.classList.toggle("on", on);
      const gt = state.hl === "gt";
      row.querySelector(".meter i").style.width = gt ? (on ? "100%" : "0%") : `${(p * 100).toFixed(1)}%`;
      row.querySelector(".pct").textContent = gt ? (on ? "yes" : "no") : `${Math.round(p * 100)}%`;
      row.querySelector(".ref").classList.toggle("on", !!t.ref[f]);
      if (p > topP) { topP = p; top = k; }
    });
    const base = f * F * F;
    for (let i = 0; i < F; i++) {
      for (let j = 0; j < F; j++) {
        const w = d.attn[base + i * F + j] / 255;
        const cell = state.attnCells[i * F + j];
        cell.style.opacity = Math.pow(w, 0.75).toFixed(3);
        cell.parentElement.title = `Face ${i + 1} → face ${j + 1}: ${(w * 100).toFixed(0)}%`;
      }
      state.attnRows[i].classList.toggle("row-on", i === top && topP >= 0.5);
    }
    $("attn-frame").textContent = `frame ${f}`;
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

  const seg = (id, fn) => document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(`#${id} button`).forEach((o) => o.setAttribute("aria-pressed", o === b));
    fn(b.dataset.v);
  }));
  seg("speed", (v) => { video.playbackRate = parseFloat(v); });
  const NOTES = {
    causal: "Causal: each score uses only video and audio up to the current frame, so the tracks fill in as the clip plays.",
    offline: "Offline: the model reads the whole 30 s clip before labelling any frame, so all of its tracks are available at once.",
  };
  const setMode = (v) => {
    state.mode = v;
    $("mode-note").textContent = NOTES[v];
    const tag = $("mode-tag");
    tag.textContent = v === "causal" ? "Causal · streaming" : "Offline · whole clip";
    tag.classList.toggle("live", v === "causal");
    if (state.clip) buildMetrics(state.clip);
    render(currentFrame(), true);
  };
  seg("mode", setMode);
  const setHl = (v) => {
    state.hl = v;
    $("faces-unit").textContent = v === "gt" ? "ground truth" : "P(speaking)";
    $("box-note").textContent = v === "gt"
      ? "Boxes show the ground-truth annotation: solid when the person is labelled as speaking. This view does not depend on the model, so it is identical for Causal and Offline."
      : "The number on each box is the model's P(speaking) for that face at the current frame. Solid coloured box: P ≥ 0.5. Dashed box: P < 0.5.";
    render(currentFrame(), true);
  };
  seg("hl", setHl);
  setHl("pred");
  $("show-err").addEventListener("change", (e) => { state.err = e.target.checked; $("err-legend").hidden = !state.err; render(currentFrame(), true); });
  setMode("causal");

  // scrub on the timeline
  let dragging = false;
  const seekTo = (e) => {
    const d = state.clip; if (!d) return;
    const r = tl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
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
