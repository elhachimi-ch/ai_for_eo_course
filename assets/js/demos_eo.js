/* AI for Earth Observation — EO-practice demos (no dependencies)
   Clouds & compositing · tiling, padding & normalisation · object detection (IoU, NMS, PR / AP) ·
   SAR–optical fusion · spatial cross-validation · metrics under class imbalance ·
   stratified area estimation (Olofsson et al. 2014).
   Needs demos_ml.js (DemoKit) and demos_vision.js (scene) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, gauss, clamp, pick } = K;
  const INK = "#14213D", MUT = "#5B6478", SAF = "#C99A2E", TER = THEME.p, BLU = "#2E4BC6", GRN = THEME.s;
  const FONT = "Inter, Arial, sans-serif";
  const pct = (v, d = 1) => (100 * v).toFixed(d) + " %";
  const disp = (v) => Math.round(255 * Math.pow(clamp(v * 3.4, 0, 1), 0.8));
  const MAT2C = [0, 1, 2, 3, 3, 3];                     // water, vegetation, bare soil, roofs/asphalt → 4 classes
  const C4 = [[46, 75, 198], THEME.sRgb, [201, 154, 46], THEME.pRgb];
  const $ = (root, s) => root.querySelector(s);
  const $$ = (root, s) => Array.from(root.querySelectorAll(s));

  /* ------------------------------------------------------------ shared helpers */
  function put(ctx, N, f) {
    const id = ctx.createImageData(N, N), d = id.data;
    for (let i = 0; i < N * N; i++) { const c = f(i); d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255; }
    ctx.putImageData(id, 0, 0);
  }
  /* draw an N×N image f(i) -> [r,g,b] scaled to the whole canvas, pixels kept sharp */
  function blit(ctx, N, f) {
    const oc = document.createElement("canvas"); oc.width = oc.height = N; put(oc.getContext("2d"), N, f);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(oc, 0, 0, ctx.canvas.width, ctx.canvas.height); ctx.restore();
  }
  function noise(r, N, cell) {               // smooth value noise in [0, 1]
    const G = Math.ceil(N / cell) + 2, g = new Float32Array(G * G), out = new Float32Array(N * N);
    for (let k = 0; k < g.length; k++) g[k] = r();
    const sm = (t) => t * t * (3 - 2 * t);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const gx = x / cell, gy = y / cell, x0 = Math.floor(gx), y0 = Math.floor(gy), fx = sm(gx - x0), fy = sm(gy - y0);
      const a = g[y0 * G + x0], b = g[y0 * G + x0 + 1], c = g[(y0 + 1) * G + x0], d = g[(y0 + 1) * G + x0 + 1];
      out[y * N + x] = (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
    }
    return out;
  }
  function fbm(seed, N, cells = [42, 17, 7], w = [0.6, 0.28, 0.12]) {
    const r = rng(seed), out = new Float32Array(N * N);
    cells.forEach((c, k) => { const f = noise(r, N, c); for (let i = 0; i < out.length; i++) out[i] += w[k] * f[i]; });
    return out;
  }
  const quantile = (a, q) => { const s = Float32Array.from(a).sort(); return s[clamp(Math.floor(q * (s.length - 1)), 0, s.length - 1)]; };
  /* cloud opacity in [0,1]; exactly `cover` of the pixels have opacity > 0.5, thin edges below */
  function cloudField(seed, N, cover) {
    const a = new Float32Array(N * N);
    if (cover <= 0) return a;
    const f = fbm(seed, N), t = quantile(f, 1 - cover);
    for (let i = 0; i < a.length; i++) a[i] = clamp((f[i] - t) / 0.05 + 0.5, 0, 1);
    return a;
  }
  /* axes in a box; returns data→pixel mappers */
  function axes(ctx, L, T, W, H, o) {
    const X = (v) => L + ((v - o.x[0]) / (o.x[1] - o.x[0])) * W, Y = (v) => T + H - ((v - o.y[0]) / (o.y[1] - o.y[0])) * H;
    ctx.save(); ctx.lineWidth = 1; ctx.font = `12px ${FONT}`; ctx.fillStyle = MUT;
    (o.yt || []).forEach((v) => {
      ctx.strokeStyle = "rgba(20,33,61,.09)"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + W, Y(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText((o.fy || String)(v), L - 6, Y(v) + 4);
    });
    (o.xt || []).forEach((v) => { ctx.textAlign = "center"; ctx.fillText((o.fx || String)(v), X(v), T + H + 16); });
    ctx.strokeStyle = INK; ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + H); ctx.lineTo(L + W, T + H); ctx.stroke();
    ctx.fillStyle = INK; ctx.font = `600 13px ${FONT}`;
    if (o.xl) { ctx.textAlign = "center"; ctx.fillText(o.xl, L + W / 2, T + H + 34); }
    if (o.yl) { ctx.save(); ctx.translate(L - 42, T + H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(o.yl, 0, 0); ctx.restore(); }
    ctx.restore();
    return { X, Y };
  }
  function label(ctx, txt, x, y, col, align = "left", font = `600 13px ${FONT}`) {
    ctx.save(); ctx.font = font; ctx.fillStyle = col; ctx.textAlign = align; ctx.fillText(txt, x, y); ctx.restore();
  }
  function dilate(m, N, r) {
    const o = new Uint8Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (m[y * N + x])
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < N && Y < N) o[Y * N + X] = 1;
      }
    return o;
  }
  const shuffle = (a, r) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  Object.assign(K, { put, blit, noise, fbm, cloudField, axes, label, shuffle });

  /* ======================================================= 1 · clouds, masks and compositing */
  DEMOS.composite = function (root) {
    const S = K.scene(), N = S.N, n = S.n, T = 8, BANDS = [1, 2, 3, 7];   // B2 B3 B4 B8
    const COVER = [0.08, 0.62, 0.3, 0.88, 0.18, 0.47, 0.72, 0.26], CLOUD = [0.52, 0.54, 0.56, 0.6], SH = [7, 5];
    const thumbs = $(root, ".thumbs"), dc = $(root, "canvas.date").getContext("2d"), cc = $(root, "canvas.comp").getContext("2d");
    const out = $(root, ".readout"), dl = $(root, ".dl"), cl = $(root, "input[name=cl]"), mk = $(root, "input[name=mask]"), bf = $(root, "input[name=buf]");
    const NAMES = { last: "latest date", mean: "mean of all dates", median: "median of all dates", medmask: "median, cloud mask", medfull: "median, cloud + shadow mask", maxndvi: "greenest pixel (max NDVI)" };
    thumbs.innerHTML = Array.from({ length: T }, (_, t) => `<canvas width="${N}" height="${N}" data-t="${t}" title="date ${t + 1}"></canvas>`).join("");
    const tcs = $$(thumbs, "canvas");
    let method = "medfull", cur = 3, obs, cm, sm, cov;
    function simulate() {
      const f = +cl.value; obs = []; cm = []; sm = []; cov = [];
      for (let t = 0; t < T; t++) {
        const c0 = clamp(COVER[t] * f, 0, 0.97), a = cloudField(900 + t * 31, N, c0);
        const o = new Float32Array(n * 4), c = new Uint8Array(n), s = new Uint8Array(n), r = rng(77 + t);
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          const i = y * N + x, xs = x - SH[0], ys = y - SH[1], sa = xs >= 0 && ys >= 0 ? a[ys * N + xs] : 0;
          c[i] = a[i] > 0.5 ? 1 : 0; s[i] = sa > 0.5 && !c[i] ? 1 : 0;
          const g = 1 + 0.01 * gauss(r);
          for (let k = 0; k < 4; k++) o[i * 4 + k] = (S.B[i * 13 + BANDS[k]] * (1 - 0.6 * sa) * (1 - a[i]) + a[i] * CLOUD[k]) * g;
        }
        obs.push(o); cm.push(c); sm.push(s); cov.push(c0);
      }
    }
    function drawDate(ctx, t, overlay) {
      const o = obs[t];
      const f = (i) => {
        let r = disp(o[i * 4 + 2]), g = disp(o[i * 4 + 1]), b = disp(o[i * 4]);
        if (overlay && cm[t][i]) { r = 0.45 * r + 0.55 * 240; g = 0.45 * g + 0.55 * 196; b = 0.45 * b + 0.55 * 40; }
        else if (overlay && sm[t][i]) { r = 0.45 * r + 0.55 * 140; g = 0.45 * g + 0.55 * 70; b = 0.45 * b + 0.55 * 200; }
        return [r, g, b];
      };
      if (ctx.canvas.width === N) put(ctx, N, f); else blit(ctx, N, f);
    }
    function composite() {
      const buf = bf.checked ? 2 : 0;
      const bad = obs.map((_, t) => {
        if (method === "medmask") return buf ? dilate(cm[t], N, buf) : cm[t];
        if (method === "medfull") { const m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = cm[t][i] | sm[t][i]; return buf ? dilate(m, N, buf) : m; }
        return null;
      });
      const C = new Float32Array(n * 4), gap = new Uint8Array(n), vals = [];
      let nValid = 0, minV = T;
      for (let i = 0; i < n; i++) {
        let cnt = T;
        if (method === "last") for (let k = 0; k < 4; k++) C[i * 4 + k] = obs[T - 1][i * 4 + k];
        else if (method === "maxndvi") {
          let b = 0, bv = -9;
          for (let t = 0; t < T; t++) { const o = obs[t], v = (o[i * 4 + 3] - o[i * 4 + 2]) / (o[i * 4 + 3] + o[i * 4 + 2] + 1e-6); if (v > bv) { bv = v; b = t; } }
          for (let k = 0; k < 4; k++) C[i * 4 + k] = obs[b][i * 4 + k];
        } else {
          const use = []; for (let t = 0; t < T; t++) if (!bad[t] || !bad[t][i]) use.push(t);
          cnt = use.length;
          if (!cnt) gap[i] = 1;
          else for (let k = 0; k < 4; k++) {
            vals.length = 0; for (const t of use) vals.push(obs[t][i * 4 + k]);
            if (method === "mean") C[i * 4 + k] = vals.reduce((a, b) => a + b, 0) / cnt;
            else { vals.sort((p, q) => p - q); C[i * 4 + k] = cnt % 2 ? vals[(cnt - 1) / 2] : (vals[cnt / 2 - 1] + vals[cnt / 2]) / 2; }
          }
        }
        nValid += cnt; if (cnt < minV) minV = cnt;
      }
      return { C, gap, meanV: nValid / n, minV };
    }
    function render() {
      const R = composite();
      blit(cc, N, (i) => {
        if (R.gap[i]) { const x = i % N, y = (i / N) | 0; return (x + y) % 6 < 2 ? [20, 33, 61] : [200, 200, 200]; }
        return [disp(R.C[i * 4 + 2]), disp(R.C[i * 4 + 1]), disp(R.C[i * 4])];
      });
      let se = 0, m = 0, bad = 0, gaps = 0;
      for (let i = 0; i < n; i++) {
        if (R.gap[i]) { gaps++; continue; }
        let e = 0;
        for (let k = 0; k < 3; k++) { const d = R.C[i * 4 + k] - S.B[i * 13 + BANDS[k]]; se += d * d; m++; e = Math.max(e, Math.abs(d)); }
        if (e > 0.02) bad++;
      }
      const masked = method.startsWith("med") && method !== "median";
      out.textContent = `composite: ${NAMES[method]} · ` + (masked ? `clear looks per pixel: mean ${R.meanV.toFixed(1)}, min ${R.minV} · gaps ${pct(gaps / n)} · ` : `all ${T} dates used, no mask · `)
        + `RMSE vs clear sky ${Math.sqrt(se / Math.max(m, 1)).toFixed(4)} · residual contamination ${pct(bad / Math.max(n - gaps, 1))}`;
    }
    function showDate() {
      drawDate(dc, cur, mk.checked);
      tcs.forEach((c, t) => c.classList.toggle("on", t === cur));
      dl.textContent = `date ${cur + 1} of ${T} · cloud cover ${pct(cov[cur], 0)}`;
    }
    function all() { simulate(); tcs.forEach((c, t) => drawDate(c.getContext("2d"), t, false)); showDate(); render(); }
    tcs.forEach((c) => (c.onclick = () => { cur = +c.dataset.t; showDate(); }));
    $$(root, "[data-m]").forEach((b) => (b.onclick = () => { method = b.dataset.m; pick(root, "data-m", method); render(); }));
    cl.oninput = all; mk.onchange = showDate; bf.onchange = render;
    pick(root, "data-m", method); all();
  };

  /* ======================================================= 2 · tiles, padding and normalisation */
  DEMOS.tiling = function (root) {
    const S = K.scene(), N = S.N, n = S.n, SIZES = [20, 32, 40, 80];
    const sc = $(root, "canvas.src").getContext("2d"), pc = $(root, "canvas.pred").getContext("2d"), out = $(root, ".readout");
    const tsIn = $(root, "input[name=ts]"), ovIn = $(root, "input[name=ov]"), tv = $(root, ".tv"), ovv = $(root, ".ovv");
    const ndwi = new Float32Array(n); let mu = 0, sd = 0;
    for (let i = 0; i < n; i++) { const g = S.B[i * 13 + 2], nir = S.B[i * 13 + 7]; ndwi[i] = (g - nir) / (g + nir); mu += ndwi[i]; }
    mu /= n; for (let i = 0; i < n; i++) sd += (ndwi[i] - mu) ** 2; sd = Math.sqrt(sd / n);
    const SIG = 4, R = 12, KER = [];
    for (let k = -R; k <= R; k++) KER.push(Math.exp((-k * k) / (2 * SIG * SIG)));
    const ks = KER.reduce((a, b) => a + b, 0); for (let k = 0; k < KER.length; k++) KER[k] /= ks;
    const TAU = (0 - mu) / sd;                 // "water if smoothed NDWI > 0", in global z units
    let norm = "global";
    /* the toy segmenter: Gaussian smoothing (zero padding at the window edge) + threshold; writes the core */
    function predict(x0, y0, x1, y1, cx0, cy0, cx1, cy1, m, s, P) {
      const w = x1 - x0, h = y1 - y0, a = new Float32Array(w * h), b = new Float32Array(w * h);
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) a[(y - y0) * w + x - x0] = (ndwi[y * N + x] - m) / s;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let k = -R; k <= R; k++) { const X = x + k; if (X >= 0 && X < w) v += KER[k + R] * a[y * w + X]; } b[y * w + x] = v; }
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let k = -R; k <= R; k++) { const Y = y + k; if (Y >= 0 && Y < h) v += KER[k + R] * b[Y * w + x]; } a[y * w + x] = v; }
      for (let y = cy0; y < cy1; y++) for (let x = cx0; x < cx1; x++) P[y * N + x] = a[(y - y0) * w + x - x0] > TAU ? 1 : 0;
    }
    const ref = new Uint8Array(n); predict(0, 0, N, N, 0, 0, N, N, mu, sd, ref);
    function run() {
      const ts = SIZES[+tsIn.value], ov = +ovIn.value; tv.textContent = ts + " px"; ovv.textContent = ov + " px";
      const P = new Uint8Array(n);
      for (let ty = 0; ty < N; ty += ts) for (let tx = 0; tx < N; tx += ts) {
        const cx1 = Math.min(N, tx + ts), cy1 = Math.min(N, ty + ts);
        const x0 = Math.max(0, tx - ov), y0 = Math.max(0, ty - ov), x1 = Math.min(N, cx1 + ov), y1 = Math.min(N, cy1 + ov);
        let m = mu, s = sd;
        if (norm === "tile") {
          let c = 0; m = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { m += ndwi[y * N + x]; c++; } m /= c;
          s = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) s += (ndwi[y * N + x] - m) ** 2; s = Math.sqrt(s / c) || 1;
        }
        predict(x0, y0, x1, y1, tx, ty, cx1, cy1, m, s, P);
      }
      let diff = 0, inter = 0, uni = 0;
      for (let i = 0; i < n; i++) { if (P[i] !== ref[i]) diff++; const t = S.water[i] > 0.5; if (P[i] && t) inter++; if (P[i] || t) uni++; }
      blit(sc, N, (i) => [S.rgb8[i * 3], S.rgb8[i * 3 + 1], S.rgb8[i * 3 + 2]]);
      blit(pc, N, (i) => (P[i] !== ref[i] ? C4[3] : P[i] ? C4[0] : [232, 228, 218]));
      const k = sc.canvas.width / N;
      [sc, pc].forEach((ctx) => {
        ctx.save(); ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.5;
        for (let v = ts; v < N; v += ts) { ctx.beginPath(); ctx.moveTo(v * k, 0); ctx.lineTo(v * k, N * k); ctx.moveTo(0, v * k); ctx.lineTo(N * k, v * k); ctx.stroke(); }
        ctx.restore();
      });
      if (ov) {   // the window the model actually sees for one tile
        const tx = ts * Math.min(1, Math.floor(N / ts) - 1), ty = tx;
        sc.save(); sc.setLineDash([7, 5]); sc.strokeStyle = SAF; sc.lineWidth = 3;
        sc.strokeRect((tx - ov) * k, (ty - ov) * k, (ts + 2 * ov) * k, (ts + 2 * ov) * k); sc.restore();
      }
      const extra = ((ts + 2 * ov) ** 2 / ts ** 2 - 1) * 100;
      out.textContent = `tiles ${ts}×${ts} px, ${ov} px overlap (receptive field ±${R} px, +${extra.toFixed(0)} % compute) · normalisation: ${norm === "global" ? "dataset statistics" : "per tile"} · `
        + `pixels that differ from the whole-image prediction ${pct(diff / n, 2)} · water IoU ${(inter / uni).toFixed(3)}`;
    }
    $$(root, "[data-norm]").forEach((b) => (b.onclick = () => { norm = b.dataset.norm; pick(root, "data-norm", norm); run(); }));
    tsIn.oninput = run; ovIn.oninput = run;
    pick(root, "data-norm", norm); run();
  };

  /* ======================================================= 3 · object detection: IoU, NMS, precision–recall, AP */
  DEMOS.nms = function (root) {
    const cv = $(root, "canvas.det"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    const pc = $(root, "canvas.pr").getContext("2d"), out = $(root, ".readout"), hov = $(root, ".hover");
    const sIn = $(root, "input[name=score]"), nIn = $(root, "input[name=nms]"), on = $(root, "input[name=on]"), mSel = $(root, "select[name=match]");
    const sv = $(root, ".sv"), nv = $(root, ".nv");
    let seed = 3, gt, cand, bg, kept = [];
    const iou = (a, b) => {
      const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
      const I = Math.max(0, x1 - x0) * Math.max(0, y1 - y0); return I / (a.w * a.h + b.w * b.h - I);
    };
    function make() {
      const r = rng(seed * 131 + 7);
      bg = document.createElement("canvas"); bg.width = W; bg.height = H; const g = bg.getContext("2d");
      const grd = g.createLinearGradient(0, 0, 0, H); grd.addColorStop(0, "#3F6F8C"); grd.addColorStop(1, "#1F465C");
      g.fillStyle = grd; g.fillRect(0, 0, W, H);
      g.strokeStyle = "rgba(255,255,255,.08)"; g.lineWidth = 1.3;
      for (let k = 0; k < 260; k++) { const x = r() * W, y = 70 + r() * H, l = 6 + r() * 16; g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(x + l / 2, y - 2, x + l, y); g.stroke(); }
      g.fillStyle = "#BFB6A1"; g.fillRect(0, 0, W, 64); g.fillStyle = "#AAA18B"; g.fillRect(392, 64, 54, 150);
      const CC = [THEME.p, "#2E4BC6", "#C99A2E", THEME.s, "#8E8E8E", "#6E4B8E"];
      for (let row = 0; row < 3; row++) for (let col = 0; col < 10; col++) { g.fillStyle = CC[Math.floor(r() * CC.length)]; g.fillRect(16 + col * 34, 7 + row * 17, 30, 13); }
      g.fillStyle = "#8C846F"; for (let k = 0; k < 6; k++) g.fillRect(470 + k * 26, 14, 18, 36);
      // ships
      const ships = [], boxes = [];
      for (let a = 0; a < 500 && ships.length < 9; a++) {
        const L = 34 + r() * 90, B = L * (0.2 + 0.05 * r()), vert = r() < 0.3, th = (vert ? Math.PI / 2 : 0) + 0.18 * (r() - 0.5);
        const cx = 40 + r() * (W - 80), cy = 96 + r() * (H - 130);
        const c = Math.abs(Math.cos(th)), s = Math.abs(Math.sin(th)), bw = L * c + B * s + 4, bh = L * s + B * c + 4;
        const box = { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh };
        if (box.y < 70 || box.x < 4 || box.x + bw > W - 4 || box.y + bh > H - 4) continue;
        if (box.x < 452 && box.x + bw > 386 && box.y < 220) continue;
        if (boxes.some((b) => iou({ x: b.x - 10, y: b.y - 10, w: b.w + 20, h: b.h + 20 }, box) > 0)) continue;
        ships.push({ cx, cy, L, B, th }); boxes.push(box);
      }
      ships.forEach((sh) => {
        g.save(); g.translate(sh.cx, sh.cy); g.rotate(sh.th);
        g.fillStyle = "rgba(255,255,255,.1)"; g.beginPath(); g.moveTo(-sh.L / 2, -sh.B / 3); g.lineTo(-sh.L / 2 - sh.L * 0.45, -sh.B * 0.55); g.lineTo(-sh.L / 2 - sh.L * 0.45, sh.B * 0.55); g.lineTo(-sh.L / 2, sh.B / 3); g.fill();
        g.fillStyle = "#E7E3DA"; g.beginPath();
        g.moveTo(-sh.L / 2, -sh.B / 2); g.lineTo(sh.L / 2 - sh.B * 0.8, -sh.B / 2); g.lineTo(sh.L / 2, 0); g.lineTo(sh.L / 2 - sh.B * 0.8, sh.B / 2); g.lineTo(-sh.L / 2, sh.B / 2); g.closePath(); g.fill();
        g.fillStyle = "#8F949C"; g.fillRect(-sh.L / 2 + 3, -sh.B / 2 + 2, sh.L * 0.22, sh.B - 4);
        g.restore();
      });
      gt = boxes;
      // candidate detections around each ship, plus false alarms on containers, cranes and waves
      cand = [];
      ships.forEach((sh, j) => {
        const s0 = clamp(0.4 + (0.55 * (sh.L - 34)) / 90 + 0.08 * gauss(r), 0.18, 0.97), m = 3 + Math.floor(r() * 4), b = gt[j];
        cand.push({ x: b.x + 0.04 * b.w * gauss(r), y: b.y + 0.04 * b.h * gauss(r), w: b.w * (1 + 0.05 * gauss(r)), h: b.h * (1 + 0.05 * gauss(r)), s: s0 });
        for (let k = 0; k < m; k++) {
          const d = 0.04 + (r() < 0.8 ? 0.16 : 0.32) * r();
          cand.push({ x: b.x + d * b.w * (r() - 0.5) * 1.6, y: b.y + d * b.h * (r() - 0.5) * 1.6, w: b.w * (1 + d * (r() - 0.4)), h: b.h * (1 + d * (r() - 0.4)), s: clamp(s0 * (1 - 1.1 * d) + 0.03 * gauss(r), 0.05, 0.99) });
        }
      });
      for (let k = 0; k < 4; k++) cand.push({ x: 16 + r() * 300, y: 6 + r() * 20, w: 40 + r() * 30, h: 16 + r() * 10, s: 0.2 + 0.42 * r() });
      for (let k = 0; k < 2; k++) cand.push({ x: 396 + r() * 10, y: 90 + r() * 90, w: 40, h: 18 + r() * 14, s: 0.25 + 0.3 * r() });
      for (let k = 0; k < 4; k++) cand.push({ x: r() * (W - 60), y: 90 + r() * (H - 120), w: 28 + r() * 30, h: 12 + r() * 12, s: 0.06 + 0.3 * r() });
    }
    function run() {
      const ts = +sIn.value, tn = +nIn.value, tm = +mSel.value;
      sv.textContent = ts.toFixed(2); nv.textContent = tn.toFixed(2);
      const sorted = cand.slice().sort((a, b) => b.s - a.s);
      kept = [];
      sorted.forEach((c) => { c.sup = on.checked && kept.some((k) => iou(k, c) > tn); if (!c.sup) kept.push(c); });
      const used = new Set();
      kept.forEach((c) => {
        let b = -1, bv = tm;
        gt.forEach((g, j) => { if (used.has(j)) return; const v = iou(c, g); if (v >= bv) { bv = v; b = j; } });
        c.tp = b >= 0; if (b >= 0) used.add(b);
      });
      let tp = 0, fp = 0; const P = [], R = [];
      kept.forEach((c) => { c.tp ? tp++ : fp++; P.push(tp / (tp + fp)); R.push(tp / gt.length); });
      const env = P.slice(); for (let i = env.length - 2; i >= 0; i--) env[i] = Math.max(env[i], env[i + 1]);
      let ap = 0, pr = 0; R.forEach((rv, i) => { ap += (rv - pr) * env[i]; pr = rv; });
      const op = kept.filter((c) => c.s >= ts), TP = op.filter((c) => c.tp).length, FP = op.length - TP, FN = gt.length - TP;
      // scene
      ctx.drawImage(bg, 0, 0);
      ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = SAF; ctx.lineWidth = 2;
      gt.forEach((g) => ctx.strokeRect(g.x, g.y, g.w, g.h)); ctx.restore();
      sorted.forEach((c) => {
        if (c.s < ts) return;
        if (c.sup) { ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.strokeRect(c.x, c.y, c.w, c.h); ctx.restore(); return; }
        const col = c.tp ? "#2FBF71" : "#F06A3F";
        ctx.strokeStyle = col; ctx.lineWidth = 2.6; ctx.strokeRect(c.x, c.y, c.w, c.h);
        ctx.font = `600 11px ${FONT}`; const t = c.s.toFixed(2), tw = ctx.measureText(t).width + 6;
        ctx.fillStyle = col; ctx.fillRect(c.x, c.y - 14, tw, 14); ctx.fillStyle = "#fff"; ctx.fillText(t, c.x + 3, c.y - 3);
      });
      // precision–recall curve
      const PW = pc.canvas.width, PH = pc.canvas.height;
      pc.clearRect(0, 0, PW, PH);
      const A = axes(pc, 56, 14, PW - 72, PH - 62, { x: [0, 1], y: [0, 1], xt: [0, 0.5, 1], yt: [0, 0.5, 1], xl: "recall", yl: "precision" });
      if (R.length) {
        pc.beginPath(); pc.moveTo(A.X(0), A.Y(0)); let rp = 0;
        R.forEach((rv, i) => { pc.lineTo(A.X(rp), A.Y(env[i])); pc.lineTo(A.X(rv), A.Y(env[i])); rp = rv; });
        pc.lineTo(A.X(rp), A.Y(0)); pc.closePath(); pc.fillStyle = "rgba(46,75,198,.16)"; pc.fill();
        pc.beginPath(); R.forEach((rv, i) => (i ? pc.lineTo(A.X(rv), A.Y(P[i])) : pc.moveTo(A.X(rv), A.Y(P[i])))); pc.strokeStyle = BLU; pc.lineWidth = 2; pc.stroke();
        const opR = TP / gt.length, opP = op.length ? TP / op.length : 1;
        pc.beginPath(); pc.arc(A.X(opR), A.Y(opP), 7, 0, 7); pc.fillStyle = TER; pc.fill(); pc.strokeStyle = "#fff"; pc.lineWidth = 2; pc.stroke();
        label(pc, `AP@${tm} = ${ap.toFixed(2)}`, A.X(0.04), A.Y(0.1), INK, "left", `600 15px ${FONT}`);
      }
      out.textContent = `${cand.length} candidate boxes → ${kept.length} after NMS · at score ≥ ${ts.toFixed(2)}: TP ${TP} · FP ${FP} · FN ${FN} · `
        + `precision ${op.length ? (TP / op.length).toFixed(2) : "—"} · recall ${(TP / gt.length).toFixed(2)} · AP@IoU ${tm} = ${ap.toFixed(2)}`;
    }
    cv.addEventListener("mousemove", (e) => {
      const b = cv.getBoundingClientRect(), x = ((e.clientX - b.left) / b.width) * W, y = ((e.clientY - b.top) / b.height) * H;
      const c = kept.filter((k) => k.s >= +sIn.value && x >= k.x && x <= k.x + k.w && y >= k.y && y <= k.y + k.h).sort((a, b) => a.w * a.h - b.w * b.h)[0];
      if (!c) { hov.textContent = "hover a box: IoU with the nearest ship"; return; }
      const best = Math.max(0, ...gt.map((g) => iou(c, g)));
      hov.textContent = `box score ${c.s.toFixed(2)} · best IoU with a ship ${best.toFixed(2)} → ${c.tp ? "true positive" : best >= +mSel.value ? "duplicate → false positive" : "false positive"}`;
    });
    $(root, "[data-act=new]").onclick = () => { seed++; make(); run(); };
    sIn.oninput = run; nIn.oninput = run; on.onchange = run; mSel.onchange = run;
    make(); run();
  };

  /* ======================================================= 4 · SAR–optical fusion under clouds */
  DEMOS.fusion = function (root) {
    const S = K.scene(), N = S.N, n = S.n;
    const oc = $(root, "canvas.opt").getContext("2d"), scx = $(root, "canvas.sar").getContext("2d"), mc = $(root, "canvas.map").getContext("2d");
    const out = $(root, ".readout"), cIn = $(root, "input[name=cl]"), cv = $(root, ".cv"), eIn = $(root, "input[name=err]");
    const OB = [1, 3, 7, 11], CLOUD = [0.5, 0.53, 0.56, 0.42], VH = [0.0004, 0.03, 0.005, 0.012, 0.012, 0.0015];
    const NAMES = { opt: "optical only", optm: "optical, clouds masked", sar: "SAR only", early: "stacked (early) fusion", fused: "fusion, clouds masked" };
    const y = new Uint8Array(n); for (let i = 0; i < n; i++) y[i] = MAT2C[S.dom[i]];
    // SAR: VV from the scene's σ⁰, VH from cover fractions; 4-look speckle, then 3×3 boxcar
    const r = rng(404), g4 = () => { let s = 0; for (let k = 0; k < 4; k++) s += -Math.log(1 - r()); return s / 4; };
    const vv0 = new Float32Array(n), vh0 = new Float32Array(n);
    for (let i = 0; i < n; i++) { vv0[i] = S.sar0[i] * g4(); let s = 0; for (const [m, f] of S.comp[i]) s += f * VH[m]; vh0[i] = s * g4(); }
    const box = (a) => {
      const o = new Float32Array(n);
      for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) {
        let s = 0, c = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const X = x + dx, Y = yy + dy; if (X >= 0 && Y >= 0 && X < N && Y < N) { s += a[Y * N + X]; c++; } }
        o[yy * N + x] = s / c;
      }
      return o;
    };
    const vv = box(vv0), vh = box(vh0), db = (v) => 10 * Math.log10(Math.max(v, 1e-5));
    const SAR = new Float32Array(n * 2); for (let i = 0; i < n; i++) { SAR[i * 2] = db(vv[i]); SAR[i * 2 + 1] = db(vh[i]); }
    const CLR = new Float32Array(n * 4); for (let i = 0; i < n; i++) OB.forEach((b, k) => (CLR[i * 4 + k] = S.B[i * 13 + b]));
    // Gaussian naive Bayes, trained on 800 labelled clear-sky pixels
    const tr = rng(9), idx = Array.from({ length: 800 }, () => Math.floor(tr() * n));
    function fit(F, d, floor) {
      const M = [];
      for (let c = 0; c < 4; c++) {
        const I = idx.filter((i) => y[i] === c), mu = new Float64Array(d), va = new Float64Array(d);
        I.forEach((i) => { for (let j = 0; j < d; j++) mu[j] += F[i * d + j] / I.length; });
        I.forEach((i) => { for (let j = 0; j < d; j++) va[j] += (F[i * d + j] - mu[j]) ** 2 / I.length; });
        for (let j = 0; j < d; j++) va[j] = Math.max(va[j], floor);
        M.push({ mu, va, lp: Math.log(I.length / idx.length) });
      }
      return M;
    }
    const MO = fit(CLR, 4, 1e-4), MS = fit(SAR, 2, 0.3);
    const ll = (M, F, d, i, c) => { let s = 0; const m = M[c]; for (let j = 0; j < d; j++) s -= 0.5 * ((F[i * d + j] - m.mu[j]) ** 2 / m.va[j] + Math.log(m.va[j])); return s; };
    blit(scx, N, (i) => { const v = Math.round(255 * clamp((SAR[i * 2] + 25) / 25, 0, 1)); return [v, v, v]; });
    let mode = "fused", OPT, mask, cover;
    function clouds() {
      cover = +cIn.value; cv.textContent = pct(cover, 0);
      const a = cloudField(5150, N, cover); OPT = new Float32Array(n * 4); mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) { mask[i] = a[i] > 0.5 ? 1 : 0; for (let k = 0; k < 4; k++) OPT[i * 4 + k] = CLR[i * 4 + k] * (1 - a[i]) + a[i] * CLOUD[k]; }
      const d2 = (v) => Math.round(255 * Math.pow(clamp(v, 0, 1), 0.9));        // SWIR · NIR · red
      blit(oc, N, (i) => [d2(OPT[i * 4 + 3] * 1.8), d2(OPT[i * 4 + 2] * 1.5), d2(OPT[i * 4 + 1] * 2.6)]);
    }
    function run() {
      const pred = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const useO = mode === "opt" || mode === "early" || ((mode === "optm" || mode === "fused") && !mask[i]);
        const useS = mode === "sar" || mode === "early" || mode === "fused";
        if (!useO && !useS) { pred[i] = 255; continue; }
        let b = 0, bv = -Infinity;
        for (let c = 0; c < 4; c++) {
          let v = MO[c].lp; if (useO) v += ll(MO, OPT, 4, i, c); if (useS) v += ll(MS, SAR, 2, i, c);
          if (v > bv) { bv = v; b = c; }
        }
        pred[i] = b;
      }
      let okC = 0, nC = 0, okU = 0, nU = 0, mapped = 0;
      for (let i = 0; i < n; i++) {
        const ok = pred[i] === y[i] ? 1 : 0; if (pred[i] !== 255) mapped++;
        if (mask[i]) { nU++; okU += ok; } else { nC++; okC += ok; }
      }
      blit(mc, N, (i) => {
        if (pred[i] === 255) { const x = i % N, yy = (i / N) | 0; return (x + yy) % 6 < 2 ? [20, 33, 61] : [205, 205, 205]; }
        if (eIn.checked && pred[i] !== y[i]) return [20, 20, 20];
        return C4[pred[i]];
      });
      out.textContent = `${NAMES[mode]} · cloud cover ${pct(cover, 0)} · accuracy: clear sky ${nC ? pct(okC / nC) : "—"} · under cloud ${nU ? pct(okU / nU) : "—"} · `
        + `whole scene ${pct((okC + okU) / n)} · pixels mapped ${pct(mapped / n, 0)}`;
    }
    $$(root, "[data-m]").forEach((b) => (b.onclick = () => { mode = b.dataset.m; pick(root, "data-m", mode); run(); }));
    cIn.oninput = () => { clouds(); run(); }; eIn.onchange = run;
    pick(root, "data-m", mode); clouds(); run();
  };

  /* ======================================================= 5 · spatial autocorrelation and cross-validation */
  DEMOS.spcv = function (root) {
    const N = 120, n = N * N, mc = $(root, "canvas.map").getContext("2d"), bc = $(root, "canvas.bars").getContext("2d"), out = $(root, ".readout");
    const bIn = $(root, "input[name=block]"), bv = $(root, ".bv"), pIn = $(root, "input[name=pe]"), pv = $(root, ".pv");
    const MU = [[0, 0, 0], [1.1, 0.35, 0.2], [0.25, 1.2, -0.5], [-0.7, 0.45, 1.1]];
    const CC = [[218, 186, 98], [160, 172, 92], [30, 120, 96], [122, 104, 72]];
    const FOLD = ["#2E4BC6", THEME.p, "#C99A2E", THEME.s, "#7B3FB5"];
    let seed = 1, view = "random", samp = "survey";
    let parcel, pcls, poff, eps, surveyed, lab, testMap;
    const X = new Float32Array(n * 3);
    function world() {
      const r = rng(seed * 7919 + 3), G = 10, seeds = [];
      for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) seeds.push([((gx + 0.15 + 0.7 * r()) * N) / G, ((gy + 0.15 + 0.7 * r()) * N) / G]);
      parcel = new Uint16Array(n);
      for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) {
        let b = 0, bd = Infinity; seeds.forEach(([sx, sy], k) => { const d = (x - sx) ** 2 + (yy - sy) ** 2; if (d < bd) { bd = d; b = k; } });
        parcel[yy * N + x] = b;
      }
      pcls = seeds.map(() => Math.floor(r() * 4));
      poff = seeds.map(() => [gauss(r), gauss(r), gauss(r)]);
      eps = new Float32Array(n * 3); for (let k = 0; k < eps.length; k++) eps[k] = gauss(r);
      const cen = [[0.2 + 0.6 * r(), 0.2 + 0.6 * r()], [0.2 + 0.6 * r(), 0.2 + 0.6 * r()], [0.2 + 0.6 * r(), 0.2 + 0.6 * r()]].map(([a, b]) => [a * N, b * N]);
      surveyed = seeds.map(([sx, sy]) => cen.some(([cx, cy]) => Math.hypot(sx - cx, sy - cy) < 20));
      const pool = [];
      for (let i = 0; i < n; i++) if (samp === "random" || surveyed[parcel[i]]) pool.push(i);
      lab = shuffle(pool, r).slice(0, 900);
      const inLab = new Uint8Array(n); lab.forEach((i) => (inLab[i] = 1));
      const cand = []; for (let i = 0; i < n; i++) if (!inLab[i] && (samp === "random" || !surveyed[parcel[i]])) cand.push(i);
      testMap = shuffle(cand, r).slice(0, 2500);
    }
    function features() {
      const pe = +pIn.value; pv.textContent = pe.toFixed(2);
      for (let i = 0; i < n; i++) { const p = parcel[i], c = pcls[p]; for (let j = 0; j < 3; j++) X[i * 3 + j] = MU[c][j] + pe * poff[p][j] + 0.3 * eps[i * 3 + j]; }
    }
    const cls = (i) => pcls[parcel[i]];
    function knn(train, test) {       // 5-NN; returns [accuracy, median distance (px) to the nearest training pixel]
      let ok = 0; const dist = [];
      for (const i of test) {
        const bd = [Infinity, Infinity, Infinity, Infinity, Infinity], bc = [0, 0, 0, 0, 0]; let near = Infinity;
        const xi = i % N, yi = (i / N) | 0;
        for (const j of train) {
          const d = (X[i * 3] - X[j * 3]) ** 2 + (X[i * 3 + 1] - X[j * 3 + 1]) ** 2 + (X[i * 3 + 2] - X[j * 3 + 2]) ** 2;
          if (d < bd[4]) { let k = 4; while (k > 0 && bd[k - 1] > d) { bd[k] = bd[k - 1]; bc[k] = bc[k - 1]; k--; } bd[k] = d; bc[k] = cls(j); }
          const g = (xi - (j % N)) ** 2 + (yi - ((j / N) | 0)) ** 2; if (g < near) near = g;
        }
        const v = [0, 0, 0, 0]; bc.forEach((c) => v[c]++);
        if (v.indexOf(Math.max(...v)) === cls(i)) ok++;
        dist.push(Math.sqrt(near));
      }
      dist.sort((a, b) => a - b);
      return [ok / test.length, dist[dist.length >> 1]];
    }
    function cv(foldOf) {
      let ok = 0, tot = 0; const ds = [];
      for (let f = 0; f < 5; f++) {
        const te = lab.filter((i) => foldOf(i) === f), trn = lab.filter((i) => foldOf(i) !== f);
        if (!te.length || !trn.length) continue;
        const [a, d] = knn(trn, te); ok += a * te.length; tot += te.length; ds.push(d);
      }
      return [ok / tot, ds.reduce((a, b) => a + b, 0) / ds.length];
    }
    let randFold, blockFold, res;
    function evaluate() {
      const b = +bIn.value; bv.textContent = b + " px";
      const r = rng(seed + 99), order = shuffle(lab.slice(), r), rf = new Map(); order.forEach((i, k) => rf.set(i, k % 5));
      randFold = (i) => rf.get(i);
      const bid = (i) => Math.floor((i % N) / b) + 1000 * Math.floor(((i / N) | 0) / b);
      const blocks = shuffle([...new Set(lab.map(bid))], rng(seed + 7)), bf = new Map(); blocks.forEach((k, j) => bf.set(k, j % 5));
      blockFold = (i) => bf.get(bid(i));
      res = { rand: cv(randFold), block: cv(blockFold), map: knn(lab, testMap) };
    }
    function draw() {
      const k = mc.canvas.width / N;
      blit(mc, N, (i) => { const c = CC[cls(i)], s = samp === "survey" && surveyed[parcel[i]] ? 1 : 0.55; return c.map((v) => Math.round(253 + (v - 253) * s)); });
      mc.save(); mc.strokeStyle = "rgba(20,33,61,.18)"; mc.lineWidth = 1;
      for (let yy = 0; yy < N; yy++) for (let x = 0; x < N; x++) {
        const p = parcel[yy * N + x];
        if (x + 1 < N && parcel[yy * N + x + 1] !== p) { mc.beginPath(); mc.moveTo((x + 1) * k, yy * k); mc.lineTo((x + 1) * k, (yy + 1) * k); mc.stroke(); }
        if (yy + 1 < N && parcel[(yy + 1) * N + x] !== p) { mc.beginPath(); mc.moveTo(x * k, (yy + 1) * k); mc.lineTo((x + 1) * k, (yy + 1) * k); mc.stroke(); }
      }
      mc.restore();
      if (view === "blocks") {
        const b = +bIn.value; mc.save(); mc.strokeStyle = INK; mc.lineWidth = 1.5; mc.setLineDash([5, 4]);
        for (let v = b; v < N; v += b) { mc.beginPath(); mc.moveTo(v * k, 0); mc.lineTo(v * k, N * k); mc.moveTo(0, v * k); mc.lineTo(N * k, v * k); mc.stroke(); }
        mc.restore();
      }
      const fo = view === "blocks" ? blockFold : randFold;
      lab.forEach((i) => { const x = i % N, yy = (i / N) | 0; mc.fillStyle = FOLD[fo(i)]; mc.fillRect(x * k - 1, yy * k - 1, k + 2, k + 2); });
      // bars
      const W = bc.canvas.width, H = bc.canvas.height; bc.clearRect(0, 0, W, H);
      const A = axes(bc, 58, 16, W - 76, H - 74, { x: [0, 3], y: [0.4, 1], yt: [0.4, 0.6, 0.8, 1], fy: (v) => Math.round(v * 100) + " %", yl: "accuracy" });
      const bars = [["random 5-fold CV", res.rand[0], TER], ["spatial-block CV", res.block[0], BLU], [samp === "survey" ? "true: unvisited parcels" : "true: rest of map", res.map[0], GRN]];
      bars.forEach(([name, v, col], j) => {
        const x0 = A.X(j + 0.18), x1 = A.X(j + 0.82);
        bc.fillStyle = col; bc.fillRect(x0, A.Y(Math.max(v, 0.4)), x1 - x0, A.Y(0.4) - A.Y(Math.max(v, 0.4)));
        label(bc, pct(v), (x0 + x1) / 2, A.Y(Math.max(v, 0.4)) - 7, INK, "center", `700 15px ${FONT}`);
        label(bc, name, (x0 + x1) / 2, H - 38, INK, "center", `600 12.5px ${FONT}`);
      });
      out.textContent = `${lab.length} labelled pixels (${samp === "survey" ? "field survey: all in visited parcels" : "random over the map"}) · median distance from a test pixel to the nearest training pixel: `
        + `random CV ${res.rand[1].toFixed(1)} px · blocks ${res.block[1].toFixed(1)} px · map ${res.map[1].toFixed(1)} px`;
    }
    const all = () => { world(); features(); evaluate(); draw(); };
    $$(root, "[data-view]").forEach((b) => (b.onclick = () => { view = b.dataset.view; pick(root, "data-view", view); draw(); }));
    $$(root, "[data-samp]").forEach((b) => (b.onclick = () => { samp = b.dataset.samp; pick(root, "data-samp", samp); all(); }));
    $(root, "[data-act=new]").onclick = () => { seed++; all(); };
    bIn.oninput = () => { evaluate(); draw(); };
    pIn.oninput = () => { features(); evaluate(); draw(); };
    pick(root, "data-view", view); pick(root, "data-samp", samp); all();
  };

  /* ======================================================= 6 · metrics under class imbalance */
  DEMOS.metrics = function (root) {
    const hc = $(root, "canvas.hist"), hx = hc.getContext("2d"), pc = $(root, "canvas.pr").getContext("2d"), cm = $(root, ".cm"), out = $(root, ".readout");
    const pIn = $(root, "input[name=prev]"), tIn = $(root, "input[name=t]"), pv = $(root, ".pv"), tv = $(root, ".tv"), lg = $(root, "input[name=log]");
    const NT = 20000, sig = (z) => 1 / (1 + Math.exp(-z));
    let pos, neg;
    function data() {
      const prev = +pIn.value / 100, r = rng(2718), nP = Math.max(5, Math.round(prev * NT));
      pos = Float32Array.from({ length: nP }, () => sig(1.25 + 1.1 * gauss(r)));
      neg = Float32Array.from({ length: NT - nP }, () => sig(-1.6 + 1.1 * gauss(r)));
      pv.textContent = pct(nP / NT, 1);
    }
    const counts = (t) => { let TP = 0, FP = 0; for (const s of pos) if (s >= t) TP++; for (const s of neg) if (s >= t) FP++; return { TP, FN: pos.length - TP, FP, TN: neg.length - FP }; };
    function metrics(c) {
      const n = c.TP + c.FN + c.FP + c.TN, oa = (c.TP + c.TN) / n, prec = c.TP / Math.max(c.TP + c.FP, 1), rec = c.TP / Math.max(c.TP + c.FN, 1);
      const spec = c.TN / Math.max(c.TN + c.FP, 1), f1 = (2 * prec * rec) / Math.max(prec + rec, 1e-9), iou = c.TP / Math.max(c.TP + c.FP + c.FN, 1);
      const pe = ((c.TP + c.FP) * (c.TP + c.FN) + (c.FN + c.TN) * (c.FP + c.TN)) / (n * n), kappa = (oa - pe) / (1 - pe);
      return { oa, prec, rec, spec, f1, iou, kappa, bacc: (rec + spec) / 2 };
    }
    function draw() {
      const t = +tIn.value; tv.textContent = t.toFixed(2);
      const W = hc.width, H = hc.height, B = 40; hx.clearRect(0, 0, W, H);
      const hp = new Float64Array(B), hn = new Float64Array(B);
      pos.forEach((s) => hp[Math.min(B - 1, Math.floor(s * B))]++); neg.forEach((s) => hn[Math.min(B - 1, Math.floor(s * B))]++);
      const f = lg.checked ? (v) => Math.log10(1 + v) : (v) => v, mx = f(Math.max(...hp, ...hn));
      const A = axes(hx, 54, 12, W - 70, H - 58, { x: [0, 1], y: [0, mx], xt: [0, 0.25, 0.5, 0.75, 1], xl: "model score p(settlement)", yl: lg.checked ? "log₁₀(1 + count)" : "pixels" });
      for (let b = 0; b < B; b++) {
        const x0 = A.X(b / B) + 1, w = A.X(1 / B) - A.X(0) - 2;
        hx.fillStyle = "rgba(46,75,198,.55)"; hx.fillRect(x0, A.Y(f(hn[b])), w, A.Y(0) - A.Y(f(hn[b])));
        hx.fillStyle = `rgba(${THEME.pRgb},.8)`; hx.fillRect(x0 + w * 0.25, A.Y(f(hp[b])), w * 0.5, A.Y(0) - A.Y(f(hp[b])));
      }
      hx.strokeStyle = INK; hx.lineWidth = 2.5; hx.setLineDash([6, 4]); hx.beginPath(); hx.moveTo(A.X(t), 8); hx.lineTo(A.X(t), A.Y(0)); hx.stroke(); hx.setLineDash([]);
      label(hx, "■ background", A.X(0.62), 28, "rgba(46,75,198,.9)"); label(hx, "■ informal settlement", A.X(0.62), 46, TER);
      const c = counts(t), m = metrics(c), fmt = (v) => v.toLocaleString("en-US");
      cm.innerHTML = `<table class="tbl small cmt"><tr><th></th><th>pred. settlement</th><th>pred. background</th></tr>`
        + `<tr><th>true settlement</th><td class="ok">TP ${fmt(c.TP)}</td><td class="no">FN ${fmt(c.FN)}</td></tr>`
        + `<tr><th>true background</th><td class="no">FP ${fmt(c.FP)}</td><td class="ok">TN ${fmt(c.TN)}</td></tr></table>`
        + `<div class="mgrid"><span>overall accuracy</span><b>${pct(m.oa)}</b><span>“all background” baseline</span><b>${pct(neg.length / NT)}</b>`
        + `<span>balanced accuracy</span><b>${pct(m.bacc)}</b><span>Cohen's κ</span><b>${m.kappa.toFixed(3)}</b>`
        + `<span>precision (user's acc.)</span><b>${m.prec.toFixed(3)}</b><span>recall (producer's acc.)</span><b>${m.rec.toFixed(3)}</b>`
        + `<span>F1</span><b>${m.f1.toFixed(3)}</b><span>IoU</span><b>${m.iou.toFixed(3)}</b></div>`;
      // precision–recall curve
      const PW = pc.canvas.width, PH = pc.canvas.height; pc.clearRect(0, 0, PW, PH);
      const P = axes(pc, 50, 12, PW - 64, PH - 58, { x: [0, 1], y: [0, 1], xt: [0, 0.5, 1], yt: [0, 0.5, 1], xl: "recall", yl: "precision" });
      pc.beginPath(); let ap = 0, pr = 0;
      for (let k = 100; k >= 0; k--) { const q = metrics(counts(k / 100)); const pp = q.prec || 1; k === 100 ? pc.moveTo(P.X(q.rec), P.Y(pp)) : pc.lineTo(P.X(q.rec), P.Y(pp)); ap += (q.rec - pr) * pp; pr = q.rec; }
      pc.strokeStyle = BLU; pc.lineWidth = 2.2; pc.stroke();
      const prev = pos.length / NT; pc.setLineDash([5, 4]); pc.strokeStyle = MUT; pc.lineWidth = 1.2; pc.beginPath(); pc.moveTo(P.X(0), P.Y(prev)); pc.lineTo(P.X(1), P.Y(prev)); pc.stroke(); pc.setLineDash([]);
      pc.beginPath(); pc.arc(P.X(m.rec), P.Y(m.prec || 1), 7, 0, 7); pc.fillStyle = TER; pc.fill(); pc.strokeStyle = "#fff"; pc.lineWidth = 2; pc.stroke();
      label(pc, `AP ≈ ${ap.toFixed(2)}`, P.X(0.05), P.Y(0.08), INK, "left", `600 14px ${FONT}`);
      out.textContent = `prevalence ${pct(pos.length / NT)} · threshold ${t.toFixed(2)} · predicted settlement area ${pct((c.TP + c.FP) / NT)} vs true ${pct(pos.length / NT)} · dashed line in the PR plot = a random classifier`;
    }
    hc.addEventListener("click", (e) => { const b = hc.getBoundingClientRect(), x = (e.clientX - b.left) / b.width, v = clamp((x * hc.width - 54) / (hc.width - 70), 0.01, 0.99); tIn.value = v; draw(); });
    $(root, "[data-act=f1]").onclick = () => { let bt = 0.5, bf = -1; for (let k = 1; k < 100; k++) { const f = metrics(counts(k / 100)).f1; if (f > bf) { bf = f; bt = k / 100; } } tIn.value = bt; draw(); };
    $(root, "[data-act=half]").onclick = () => { tIn.value = 0.5; draw(); };
    pIn.oninput = () => { data(); draw(); }; tIn.oninput = draw; lg.onchange = draw;
    data(); draw();
  };

  /* ======================================================= 7 · area estimation from a stratified sample */
  DEMOS.area = function (root) {
    const N = 120, n = N * N, HA = 0.09;                          // 30 m pixels
    const mc = $(root, "canvas.map").getContext("2d"), ec = $(root, "canvas.est").getContext("2d"), out = $(root, ".readout");
    const nIn = $(root, "input[name=n]"), nv = $(root, ".nv"), eIn = $(root, "input[name=err]"), aSel = $(root, "select[name=alloc]"), ref = $(root, "input[name=ref]");
    const f1 = fbm(61, N, [26, 11, 5], [0.6, 0.28, 0.12]), f2 = fbm(62, N, [9, 4], [0.7, 0.3]), f3 = fbm(63, N, [14, 6], [0.7, 0.3]);
    const tq = quantile(f1, 0.93), truth = new Uint8Array(n); for (let i = 0; i < n; i++) truth[i] = f1[i] > tq ? 1 : 0;
    const q2 = Float32Array.from(f2).sort(), q3 = Float32Array.from(f3).sort(), rank = (q, v) => { let lo = 0, hi = q.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; q[m] < v ? (lo = m + 1) : (hi = m); } return lo / q.length; };
    const u2 = Float32Array.from(f2, (v) => rank(q2, v)), u3 = Float32Array.from(f3, (v) => rank(q3, v));
    let map, W1, sample = [], hist = [], seed = 1;
    const tTrue = truth.reduce((a, b) => a + b, 0) * HA;
    function makeMap() {
      const e = +eIn.value, rec = 0.92 - 0.45 * Math.max(0, -e), fpr = 0.006 + 0.07 * Math.max(0, e);
      map = new Uint8Array(n); let m1 = 0;
      for (let i = 0; i < n; i++) { map[i] = truth[i] ? (u2[i] < rec ? 1 : 0) : u3[i] < fpr ? 1 : 0; m1 += map[i]; }
      W1 = m1 / n; hist = []; sample = [];
    }
    function alloc() {
      const nt = +nIn.value; nv.textContent = nt;
      if (aSel.value === "prop") { const n1 = Math.max(2, Math.round(nt * W1)); return [nt - n1, n1]; }
      return [Math.round(nt / 2), nt - Math.round(nt / 2)];
    }
    const strata = () => { const s = [[], []]; for (let i = 0; i < n; i++) s[map[i]].push(i); return s; };
    function estimate(r, keep) {
      const [n0, n1] = alloc(), S = strata(), pick2 = [shuffle(S[0].slice(), r).slice(0, n0), shuffle(S[1].slice(), r).slice(0, n1)];
      const W = [1 - W1, W1]; let p = 0, v = 0, n11 = 0;
      pick2.forEach((I, h) => {
        const ph = I.filter((i) => truth[i]).length / I.length; if (h === 1) n11 = ph;
        p += W[h] * ph; v += (W[h] ** 2 * ph * (1 - ph)) / Math.max(I.length - 1, 1);
      });
      if (keep) sample = pick2.flat();
      const est = p * n * HA, se = Math.sqrt(v) * n * HA;
      return { est, se, ua: n11, pa: (W1 * n11) / Math.max(p, 1e-9) };
    }
    let cur = null;
    function draw() {
      const k = mc.canvas.width / N;
      blit(mc, N, (i) => {
        if (!ref.checked) return map[i] ? C4[3] : [236, 230, 214];
        return map[i] && truth[i] ? C4[3] : map[i] ? [232, 178, 60] : truth[i] ? C4[0] : [236, 230, 214];
      });
      sample.forEach((i) => {
        const x = i % N, y = (i / N) | 0; mc.beginPath(); mc.arc((x + 0.5) * k, (y + 0.5) * k, 3.2, 0, 7);
        mc.fillStyle = truth[i] ? "#fff" : INK; mc.fill(); mc.strokeStyle = truth[i] ? INK : "#fff"; mc.lineWidth = 1.2; mc.stroke();
      });
      const W = ec.canvas.width, H = ec.canvas.height; ec.clearRect(0, 0, W, H);
      const pix = W1 * n * HA, hi = Math.max(tTrue, pix, cur ? cur.est + 2.2 * cur.se : 0, ...hist.map((h) => h.est + 2 * h.se)) * 1.12;
      const A = axes(ec, 150, 10, W - 170, H - 56, { x: [0, hi], y: [0, 4], xt: [0, hi / 3, (2 * hi) / 3, hi].map((v) => Math.round(v)), xl: "change area (ha)" });
      const row = (j, txt) => label(ec, txt, 140, A.Y(3.5 - j) + 5, INK, "right", `600 13px ${FONT}`);
      row(0, "true area"); row(1, "pixel counting"); row(2, "stratified estimate"); row(3, hist.length ? `${hist.length} repeated samples` : "repeated samples");
      const bar = (j, v, col) => { ec.fillStyle = col; ec.fillRect(A.X(0), A.Y(3.5 - j) - 11, A.X(v) - A.X(0), 22); label(ec, Math.round(v) + " ha", A.X(v) + 6, A.Y(3.5 - j) + 5, INK); };
      bar(0, tTrue, GRN); bar(1, pix, TER);
      ec.strokeStyle = GRN; ec.setLineDash([4, 4]); ec.beginPath(); ec.moveTo(A.X(tTrue), 6); ec.lineTo(A.X(tTrue), A.Y(0)); ec.stroke(); ec.setLineDash([]);
      if (cur) {
        const yy = A.Y(1.5); ec.strokeStyle = BLU; ec.lineWidth = 3; ec.beginPath(); ec.moveTo(A.X(cur.est - 1.96 * cur.se), yy); ec.lineTo(A.X(cur.est + 1.96 * cur.se), yy); ec.stroke();
        ec.beginPath(); ec.arc(A.X(cur.est), yy, 7, 0, 7); ec.fillStyle = BLU; ec.fill(); ec.lineWidth = 1;
        label(ec, `${Math.round(cur.est)} ± ${Math.round(1.96 * cur.se)} ha`, A.X(cur.est), yy - 14, BLU, "center");
      }
      let cover = 0;
      hist.forEach((h, j) => {
        const yy = A.Y(0.5) - 12 + (24 * j) / Math.max(hist.length, 1), ok = Math.abs(h.est - tTrue) <= 1.96 * h.se; cover += ok;
        ec.strokeStyle = ok ? "rgba(46,75,198,.45)" : `rgba(${THEME.pRgb},.9)`; ec.beginPath(); ec.moveTo(A.X(h.est - 1.96 * h.se), yy); ec.lineTo(A.X(h.est + 1.96 * h.se), yy); ec.stroke();
      });
      const [n0, n1] = alloc();
      out.textContent = `mapped change ${Math.round(pix)} ha vs true ${Math.round(tTrue)} ha · sample n = ${n0 + n1} (${n1} in mapped change, ${n0} in no change)`
        + (cur ? ` · user's acc. ${cur.ua.toFixed(2)} · producer's acc. ${cur.pa.toFixed(2)}` : "")
        + (hist.length ? ` · 95 % CIs that cover the truth: ${pct(cover / hist.length, 0)}` : "");
    }
    $(root, "[data-act=draw]").onclick = () => { seed++; cur = estimate(rng(seed * 17), true); draw(); };
    $(root, "[data-act=rep]").onclick = () => { for (let k = 0; k < 100; k++) { seed++; hist.push(estimate(rng(seed * 17), false)); } hist = hist.slice(-100); cur = estimate(rng(++seed * 17), true); draw(); };
    eIn.oninput = () => { makeMap(); cur = estimate(rng(seed * 17), true); draw(); };
    nIn.oninput = aSel.onchange = () => { hist = []; cur = estimate(rng(seed * 17), true); draw(); };
    ref.onchange = draw;
    makeMap(); cur = estimate(rng(seed * 17), true); draw();
  };
})();
