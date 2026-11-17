/* AI for Earth Observation — trustworthy & responsible EO AI demos (no dependencies)
   Calibration & temperature scaling · deep-ensemble uncertainty · attributions & permutation importance ·
   domain shift & adaptation · physics emulator & hybrid retrieval · geographic bias · carbon footprint.
   Needs demos_ml.js (DemoKit, Plot), demos_vision.js (scene) and demos_eo.js (axes, blit) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, gauss, clamp, pick, Plot, axes, label, blit, shuffle } = K;
  const INK = "#14213D", MUT = "#5B6478", SAF = "#C99A2E", TER = "#B5502E", BLU = "#2E4BC6", GRN = "#0F6B5C";
  const FONT = "Inter, Arial, sans-serif";
  const pct = (v, d = 1) => (100 * v).toFixed(d) + " %";
  const $ = (root, s) => root.querySelector(s);
  const $$ = (root, s) => Array.from(root.querySelectorAll(s));
  const softmax = (z, s = 1) => { const m = Math.max(...z) * s, e = z.map((v) => Math.exp(v * s - m)), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); };
  /* solve (A) x = b, A n×n (Gaussian elimination with partial pivoting) */
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  /* ridge regression on feature rows Φ */
  function ridge(Phi, y, lam) {
    const d = Phi[0].length, A = Array.from({ length: d }, () => new Array(d).fill(0)), b = new Array(d).fill(0);
    Phi.forEach((f, i) => { for (let a = 0; a < d; a++) { b[a] += f[a] * y[i]; for (let c = a; c < d; c++) A[a][c] += f[a] * f[c]; } });
    for (let a = 0; a < d; a++) { A[a][a] += lam; for (let c = 0; c < a; c++) A[a][c] = A[c][a]; }
    return solve(A, b);
  }
  const dot = (w, f) => { let s = 0; for (let k = 0; k < w.length; k++) s += w[k] * f[k]; return s; };
  /* Gaussian maximum-likelihood classifier (full 2×2 covariance) */
  function gml(pts, K_ = 3, wOf = () => 1) {
    return Array.from({ length: K_ }, (_, c) => {
      const P = pts.filter((p) => p.c === c); let sw = 0, mx = 0, my = 0;
      P.forEach((p) => { const w = wOf(p); sw += w; mx += w * p.x; my += w * p.y; }); mx /= sw; my /= sw;
      let a = 1e-5, b = 0, d = 1e-5; P.forEach((p) => { const w = wOf(p); a += (w * (p.x - mx) ** 2) / sw; b += (w * (p.x - mx) * (p.y - my)) / sw; d += (w * (p.y - my) ** 2) / sw; });
      const det = a * d - b * b; return { mx, my, ia: d / det, ib: -b / det, id: a / det, ld: Math.log(det) };
    });
  }
  const maha = (g, x, y) => { const u = x - g.mx, v = y - g.my; return g.ia * u * u + 2 * g.ib * u * v + g.id * v * v; };
  const classify = (M, x, y) => { let b = 0, bv = Infinity; M.forEach((g, c) => { const v = maha(g, x, y) + g.ld; if (v < bv) { bv = v; b = c; } }); return b; };
  const COL3 = ["#2E4BC6", "#0F6B5C", "#B5502E"], RGB3 = [[46, 75, 198], [15, 107, 92], [181, 80, 46]];

  /* ======================================================= 1 · calibration and temperature scaling */
  DEMOS.calib = function (root) {
    const rc = $(root, "canvas.rel").getContext("2d"), out = $(root, ".readout"), tIn = $(root, "input[name=T]"), tv = $(root, ".tv");
    const C = 5, M = 4000, r = rng(17), Z = [], Y = [];
    for (let m = 0; m < M; m++) {
      const y0 = Math.floor(r() * C), z = Array.from({ length: C }, (_, c) => (c === y0 ? 1.7 : 0) + 0.95 * gauss(r));
      const p = softmax(z); let u = r(), y = 0; while (y < C - 1 && u > p[y]) { u -= p[y]; y++; }
      Z.push(z); Y.push(y);
    }
    const KAPPA = { over: 2.4, under: 0.55 }; let model = "over";
    const stats = (T) => {
      const k = KAPPA[model] / T, bins = Array.from({ length: 10 }, () => ({ n: 0, c: 0, a: 0 })); let nll = 0, acc = 0;
      Z.forEach((z, i) => {
        const p = softmax(z, k), j = p.indexOf(Math.max(...p)), conf = p[j], ok = j === Y[i] ? 1 : 0, b = bins[Math.min(9, Math.floor(conf * 10))];
        b.n++; b.c += conf; b.a += ok; acc += ok; nll -= Math.log(Math.max(p[Y[i]], 1e-12));
      });
      const ece = bins.reduce((s, b) => s + (b.n ? (b.n / M) * Math.abs(b.a / b.n - b.c / b.n) : 0), 0);
      return { bins, nll: nll / M, acc: acc / M, ece };
    };
    function draw() {
      const T = +tIn.value, s = stats(T); tv.textContent = T.toFixed(2);
      const W = rc.canvas.width, H = rc.canvas.height; rc.clearRect(0, 0, W, H);
      const A = axes(rc, 60, 14, 380, H - 60, { x: [0, 1], y: [0, 1], xt: [0, 0.5, 1], yt: [0, 0.25, 0.5, 0.75, 1], xl: "confidence (max probability)", yl: "accuracy" });
      s.bins.forEach((b, j) => {
        if (!b.n) return;
        const x0 = A.X(j / 10) + 2, x1 = A.X((j + 1) / 10) - 2, acc = b.a / b.n, conf = b.c / b.n;
        rc.fillStyle = "rgba(46,75,198,.75)"; rc.fillRect(x0, A.Y(acc), x1 - x0, A.Y(0) - A.Y(acc));
        rc.fillStyle = "rgba(181,80,46,.32)"; rc.fillRect(x0, Math.min(A.Y(acc), A.Y(conf)), x1 - x0, Math.abs(A.Y(acc) - A.Y(conf)));
      });
      rc.setLineDash([6, 4]); rc.strokeStyle = INK; rc.lineWidth = 1.5; rc.beginPath(); rc.moveTo(A.X(0), A.Y(0)); rc.lineTo(A.X(1), A.Y(1)); rc.stroke(); rc.setLineDash([]);
      label(rc, "perfect calibration", A.X(0.52), A.Y(0.6), MUT, "right", `12px ${FONT}`);
      // confidence histogram
      const L2 = 520, W2 = W - L2 - 20, mx = Math.max(...s.bins.map((b) => b.n));
      const B = axes(rc, L2, 14, W2, H - 60, { x: [0, 1], y: [0, mx], xt: [0, 0.5, 1], xl: "how often each confidence occurs" });
      s.bins.forEach((b, j) => { rc.fillStyle = "rgba(201,154,46,.85)"; rc.fillRect(B.X(j / 10) + 2, B.Y(b.n), B.X(0.1) - B.X(0) - 4, B.Y(0) - B.Y(b.n)); });
      out.textContent = `T = ${T.toFixed(2)} · accuracy ${pct(s.acc)} (T never changes it) · ECE ${pct(s.ece)} · negative log-likelihood ${s.nll.toFixed(3)} · mean confidence ${pct(s.bins.reduce((a, b) => a + b.c, 0) / M)}`;
    }
    $(root, "[data-act=fit]").onclick = () => {
      let a = Math.log(0.2), b = Math.log(6); const g = (Math.sqrt(5) - 1) / 2, f = (t) => stats(Math.exp(t)).nll;
      for (let k = 0; k < 28; k++) { const c = b - g * (b - a), d = a + g * (b - a); f(c) < f(d) ? (b = d) : (a = c); }
      tIn.value = Math.exp((a + b) / 2).toFixed(2); draw();
    };
    $$(root, "[data-model]").forEach((b) => (b.onclick = () => { model = b.dataset.model; pick(root, "data-model", model); tIn.value = 1; draw(); }));
    tIn.oninput = draw; pick(root, "data-model", model); draw();
  };

  /* ======================================================= 2 · ensembles: aleatoric vs epistemic uncertainty */
  DEMOS.ens = function (root) {
    const cv = $(root, "canvas"), ctx = cv.getContext("2d"), out = $(root, ".readout");
    const mIn = $(root, "input[name=m]"), mv = $(root, ".mv"), gap = $(root, "input[name=gap]"), nIn = $(root, "input[name=noise]"), nv = $(root, ".nv");
    const f = (x) => 300 * (1 - Math.exp(-2.4 * x)), sd = (x, s) => s * (8 + 55 * x);
    let seed = 1, data, members;
    const D = 30;
    function makeData() {
      const r = rng(seed * 31), s = +nIn.value; nv.textContent = s.toFixed(1); data = [];
      while (data.length < 70) { const x = 0.04 + 0.72 * r(); if (gap.checked && x > 0.33 && x < 0.5) continue; data.push({ x, y: f(x) + sd(x, s) * gauss(r) }); }
    }
    function train() {
      const M = +mIn.value; mv.textContent = M; members = [];
      for (let m = 0; m < M; m++) {
        const r = rng(seed * 1000 + m * 7 + 1), W = Array.from({ length: D }, () => 6 * gauss(r)), B = Array.from({ length: D }, () => 2 * Math.PI * r());
        const phi = (x) => [1, ...W.map((w, k) => Math.cos(w * x + B[k]))];
        const bs = Array.from({ length: data.length }, () => data[Math.floor(r() * data.length)]);
        const P = bs.map((p) => phi(p.x)), wm = ridge(P, bs.map((p) => p.y / 100), 0.05);
        const lr = bs.map((p, i) => Math.log((p.y / 100 - dot(wm, P[i])) ** 2 + 1e-4)), wv = ridge(P, lr, 1);
        // E[log r²] = log σ² − 1.27 for Gaussian residuals: add the offset back
        members.push({ mu: (x) => 100 * dot(wm, phi(x)), s2: (x) => 1e4 * Math.exp(clamp(dot(wv, phi(x)) + 1.27, -12, 3)) });
      }
    }
    const predict = (x) => {
      const mus = members.map((m) => m.mu(x)), mu = mus.reduce((a, b) => a + b, 0) / mus.length;
      const ep = mus.reduce((a, b) => a + (b - mu) ** 2, 0) / mus.length, al = members.reduce((a, m) => a + m.s2(x), 0) / members.length;
      return { mu, ep, al, mus };
    };
    function draw() {
      const W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
      const A = axes(ctx, 66, 12, W - 86, H - 60, { x: [0, 1], y: [-50, 450], xt: [0, 0.25, 0.5, 0.75, 1], yt: [0, 100, 200, 300, 400], xl: "L-band HV backscatter (normalised) →", yl: "above-ground biomass (t/ha)" });
      ctx.save(); ctx.beginPath(); ctx.rect(A.X(0), A.Y(450), A.X(1) - A.X(0), A.Y(-50) - A.Y(450)); ctx.clip();
      const xs = Array.from({ length: 161 }, (_, i) => i / 160), P = xs.map(predict);
      const band = (k, col) => { ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(P[i].mu + k(P[i])))); for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(A.X(xs[i]), A.Y(P[i].mu - k(P[i]))); ctx.closePath(); ctx.fillStyle = col; ctx.fill(); };
      band((p) => 1.96 * Math.sqrt(p.ep + p.al), "rgba(46,75,198,.13)");
      band((p) => 1.96 * Math.sqrt(p.ep), "rgba(46,75,198,.33)");
      ctx.fillStyle = "rgba(201,154,46,.12)"; ctx.fillRect(A.X(0.76), A.Y(450), A.X(1) - A.X(0.76), A.Y(-50) - A.Y(450));
      if (members.length <= 12) members.forEach((m, j) => { ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(P[i].mus[j]))); ctx.strokeStyle = "rgba(46,75,198,.35)"; ctx.lineWidth = 1; ctx.stroke(); });
      ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(f(x)))); ctx.setLineDash([6, 4]); ctx.strokeStyle = GRN; ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); xs.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(P[i].mu))); ctx.strokeStyle = BLU; ctx.lineWidth = 3; ctx.stroke();
      data.forEach((p) => { ctx.beginPath(); ctx.arc(A.X(p.x), A.Y(p.y), 4.5, 0, 7); ctx.fillStyle = TER; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2; ctx.stroke(); });
      ctx.restore();
      label(ctx, "no training data →", A.X(0.78), A.Y(420), "#8A6512", "left", `600 12px ${FONT}`);
      const avg = (a, b) => { let s = 0, c = 0; xs.forEach((x, i) => { if (x >= a && x <= b) { s += Math.sqrt(P[i].ep); c++; } }); return s / c; };
      const r = rng(seed * 5 + 3), ts = +nIn.value; let cov = 0;
      for (let k = 0; k < 400; k++) { const x = 0.04 + 0.72 * r(), y = f(x) + sd(x, ts) * gauss(r), p = predict(x); if (Math.abs(y - p.mu) <= 1.96 * Math.sqrt(p.ep + p.al)) cov++; }
      out.textContent = `${members.length} member${members.length > 1 ? "s" : ""} · epistemic σ: with data ${avg(0.05, 0.3).toFixed(0)} t/ha · ${gap.checked ? `in the gap ${avg(0.36, 0.47).toFixed(0)} t/ha · ` : ""}beyond the data ${avg(0.82, 1).toFixed(0)} t/ha · `
        + `95 % interval covers ${pct(cov / 400, 0)} of new field plots (dark band: epistemic, light band: + aleatoric)`;
    }
    const all = () => { makeData(); train(); draw(); };
    mIn.oninput = () => { train(); draw(); }; gap.onchange = all; nIn.oninput = all;
    $(root, "[data-act=new]").onclick = () => { seed++; all(); };
    all();
  };

  /* ======================================================= 3 · explaining a classifier: attributions and permutation importance */
  DEMOS.xai = function (root) {
    const S = K.scene(), N = S.N, n = S.n, BI = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12];
    const BN = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], CN = ["water", "vegetation", "bare soil", "built-up"], MAT2C = [0, 1, 2, 3, 3, 3];
    const C4 = [[46, 75, 198], [15, 107, 92], [201, 154, 46], [181, 80, 46]];
    const ic = $(root, "canvas.img").getContext("2d"), ac = $(root, "canvas.att").getContext("2d"), bc = $(root, "canvas.bars").getContext("2d"), out = $(root, ".readout");
    const d = BI.length, Z = new Float32Array(n * d), y = new Uint8Array(n), mu = new Float64Array(d), sdv = new Float64Array(d);
    const r = rng(77);
    for (let i = 0; i < n; i++) { y[i] = MAT2C[S.dom[i]]; BI.forEach((b, j) => (Z[i * d + j] = S.B[i * 13 + b] * (1 + 0.04 * gauss(r)))); }
    for (let j = 0; j < d; j++) { for (let i = 0; i < n; i++) mu[j] += Z[i * d + j] / n; for (let i = 0; i < n; i++) sdv[j] += (Z[i * d + j] - mu[j]) ** 2 / n; sdv[j] = Math.sqrt(sdv[j]); }
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Z[i * d + j] = (Z[i * d + j] - mu[j]) / sdv[j];
    // softmax regression, full-batch gradient descent on 3000 pixels
    const tr = Array.from({ length: 3000 }, () => Math.floor(r() * n)), Wt = Array.from({ length: 4 }, () => new Float64Array(d + 1));
    for (let it = 0; it < 250; it++) {
      const G = Array.from({ length: 4 }, () => new Float64Array(d + 1));
      tr.forEach((i) => {
        const z = Wt.map((w) => { let s = w[d]; for (let j = 0; j < d; j++) s += w[j] * Z[i * d + j]; return s; }), p = softmax(z);
        for (let c = 0; c < 4; c++) { const g = p[c] - (y[i] === c ? 1 : 0); for (let j = 0; j < d; j++) G[c][j] += g * Z[i * d + j]; G[c][d] += g; }
      });
      for (let c = 0; c < 4; c++) for (let j = 0; j <= d; j++) Wt[c][j] -= 0.8 * (G[c][j] / tr.length + (j < d ? 1e-3 * Wt[c][j] : 0));
    }
    const logits = (row) => Wt.map((w) => { let s = w[d]; for (let j = 0; j < d; j++) s += w[j] * row[j]; return s; });
    const pred = new Uint8Array(n), prob = new Float32Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) { const p = softmax(logits(Z.subarray(i * d, i * d + d))); const k = p.indexOf(Math.max(...p)); pred[i] = k; prob[i] = p[k]; acc += k === y[i]; }
    acc /= n;
    let view = "rgb", band = 8, mode = "local", px = S.vegPx.slice(), perm = null;
    function permutation() {
      const rr = rng(5), te = Array.from({ length: 2500 }, () => Math.floor(rr() * n)), row = new Float64Array(d);
      const base = te.reduce((s, i) => s + (pred[i] === y[i] ? 1 : 0), 0) / te.length;
      return BI.map((_, j) => {
        const sh = shuffle(te.slice(), rr); let ok = 0;
        te.forEach((i, k) => { for (let q = 0; q < d; q++) row[q] = Z[i * d + q]; row[j] = Z[sh[k] * d + j]; const z = logits(row); if (z.indexOf(Math.max(...z)) === y[i]) ok++; });
        return base - ok / te.length;
      });
    }
    function drawImg() {
      blit(ic, N, (i) => (view === "rgb" ? [S.rgb8[i * 3], S.rgb8[i * 3 + 1], S.rgb8[i * 3 + 2]] : C4[pred[i]]));
      const k = ic.canvas.width / N; ic.strokeStyle = SAF; ic.lineWidth = 3; ic.strokeRect((px[0] - 2) * k, (px[1] - 2) * k, 5 * k, 5 * k);
      const att = new Float32Array(n); let mx = 1e-6;
      for (let i = 0; i < n; i++) { att[i] = Wt[pred[i]][band] * Z[i * d + band]; mx = Math.max(mx, Math.abs(att[i])); }
      blit(ac, N, (i) => {
        const v = clamp(att[i] / (0.6 * mx), -1, 1), t = Math.abs(v), c = v > 0 ? [181, 80, 46] : [46, 75, 198];
        return [253 + (c[0] - 253) * t, 251 + (c[1] - 251) * t, 247 + (c[2] - 247) * t].map(Math.round);
      });
    }
    function drawBars() {
      const W = bc.canvas.width, H = bc.canvas.height; bc.clearRect(0, 0, W, H);
      const i = px[1] * N + px[0], k = pred[i];
      const vals = mode === "local" ? BI.map((_, j) => Wt[k][j] * Z[i * d + j]) : perm || (perm = permutation());
      const mx = Math.max(0.05, ...vals.map(Math.abs)), lo = mode === "local" ? -mx : 0;
      const A = axes(bc, 62, 12, W - 76, H - 50, { x: [0, d], y: [lo * 1.1, mx * 1.1], yt: mode === "local" ? [-mx, 0, mx].map((v) => +v.toFixed(2)) : [0, +(mx / 2).toFixed(2), +mx.toFixed(2)], yl: mode === "local" ? "contribution to logit" : "accuracy drop" });
      vals.forEach((v, j) => {
        const x0 = A.X(j + 0.15), x1 = A.X(j + 0.85);
        bc.fillStyle = mode === "global" ? SAF : v >= 0 ? TER : BLU; bc.fillRect(x0, Math.min(A.Y(0), A.Y(v)), x1 - x0, Math.abs(A.Y(v) - A.Y(0)));
        label(bc, BN[j], (x0 + x1) / 2, H - 22, j === band ? TER : INK, "center", `600 12px ${FONT}`);
      });
      if (mode === "local") {
        const top = vals.map((v, j) => [v, BN[j]]).sort((a, b) => Math.abs(b[0]) - Math.abs(a[0])).slice(0, 3).map(([v, b]) => `${b} ${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`).join(", ");
        out.textContent = `model: softmax regression on 10 bands, accuracy ${pct(acc)} · pixel (${px[1]}, ${px[0]}): true ${CN[y[i]]}, predicted ${CN[k]} (p = ${prob[i].toFixed(2)}) · strongest evidence: ${top} · map: ${BN[band]} contribution`;
      } else {
        out.textContent = `permutation importance: shuffle one band across 2 500 pixels and measure the accuracy drop · correlated red-edge/NIR bands (B5–B8A) share credit, so each looks less important than the group`;
      }
    }
    const redraw = () => { drawImg(); drawBars(); };
    [ic.canvas, ac.canvas].forEach((c) => c.addEventListener("click", (e) => { const b = c.getBoundingClientRect(); px = [clamp(Math.floor(((e.clientX - b.left) / b.width) * N), 0, N - 1), clamp(Math.floor(((e.clientY - b.top) / b.height) * N), 0, N - 1)]; mode = "local"; pick(root, "data-mode", mode); redraw(); }));
    $$(root, "[data-view]").forEach((b) => (b.onclick = () => { view = b.dataset.view; pick(root, "data-view", view); drawImg(); }));
    $$(root, "[data-band]").forEach((b) => (b.onclick = () => { band = +b.dataset.band; pick(root, "data-band", band); redraw(); }));
    $$(root, "[data-mode]").forEach((b) => (b.onclick = () => { mode = b.dataset.mode; pick(root, "data-mode", mode); drawBars(); }));
    pick(root, "data-view", view); pick(root, "data-band", band); pick(root, "data-mode", mode); redraw();
  };

  /* ======================================================= 4 · domain shift and adaptation */
  DEMOS.shift = function (root) {
    const cv = $(root, "canvas"), P = Plot(cv), out = $(root, ".readout"), sIn = $(root, "input[name=s]"), sv = $(root, ".sv");
    const src = K.spectral3(11, 45), base = K.spectral3(12, 45);
    let adapt = "none";
    const move = (p, s) => ({ x: clamp((1 + 0.75 * s) * p.x + 0.06 * s + (p.c === 2 ? 0.06 * s : 0), 0.01, 0.99), y: clamp((1 - 0.3 * s) * p.y + 0.03 * s - (p.c === 1 ? 0.3 * s * p.y : 0), 0.01, 0.99), c: p.c });
    const stats = (pts) => { const m = [0, 0], v = [0, 0]; pts.forEach((p) => { m[0] += p.x / pts.length; m[1] += p.y / pts.length; }); pts.forEach((p) => { v[0] += (p.x - m[0]) ** 2 / pts.length; v[1] += (p.y - m[1]) ** 2 / pts.length; }); return { m, s: v.map(Math.sqrt) }; };
    const zs = (p, st) => ({ x: (p.x - st.m[0]) / st.s[0], y: (p.y - st.m[1]) / st.s[1], c: p.c });
    function draw() {
      const s = +sIn.value; sv.textContent = s.toFixed(2);
      const tgt = base.map((p) => move(p, s)), ss = stats(src), st = stats(tgt);
      const lab = []; [0, 1, 2].forEach((c) => tgt.filter((p) => p.c === c).slice(0, c === 0 ? 3 : 4).forEach((p) => lab.push(p)));
      let M, f;
      if (adapt === "std") { M = gml(src.map((p) => zs(p, ss))); f = (x, y) => { const q = zs({ x, y }, st); return classify(M, q.x, q.y); }; }
      else if (adapt === "ft") { const pts = src.concat(lab), w = new Set(lab); M = gml(pts, 3, (p) => (w.has(p) ? 8 : 1)); f = (x, y) => classify(M, x, y); }
      else { M = gml(src); f = (x, y) => classify(M, x, y); }
      const M0 = gml(src), ood = tgt.filter((p) => Math.min(...M0.map((g) => maha(g, p.x, p.y))) > 9.21).length;
      P.field(160, 100, (i, j, x, y) => [...RGB3[f(x, y)], 0.16], false);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", K.SPEC_TICKS);
      P.hollow(src, (p) => COL3[p.c]);
      P.points(tgt, (p) => COL3[p.c], { r: () => 5.5, bad: (p) => f(p.x, p.y) !== p.c });
      if (adapt === "ft") lab.forEach((p) => { P.ctx.beginPath(); P.ctx.arc(P.X(p.x), P.Y(p.y), 11, 0, 7); P.ctx.strokeStyle = SAF; P.ctx.lineWidth = 3; P.ctx.stroke(); });
      const accS = src.filter((p) => (adapt === "std" ? classify(M, ...Object.values(zs(p, ss)).slice(0, 2)) : classify(M, p.x, p.y)) === p.c).length / src.length;
      const accT = tgt.filter((p) => f(p.x, p.y) === p.c).length / tgt.length;
      const AD = { none: "no adaptation", std: "per-domain standardisation (no target labels)", ft: "fine-tune with 11 target labels (gold rings)" };
      out.textContent = `shift ${s.toFixed(2)} · ${AD[adapt]} · accuracy: source ${pct(accS, 0)} → target ${pct(accT, 0)} · target pixels flagged out-of-distribution (Mahalanobis² > 9.21) ${pct(ood / tgt.length, 0)}`;
    }
    $$(root, "[data-a]").forEach((b) => (b.onclick = () => { adapt = b.dataset.a; pick(root, "data-a", adapt); draw(); }));
    sIn.oninput = draw; pick(root, "data-a", adapt); draw();
  };

  /* ======================================================= 5 · physics emulator and hybrid retrieval of LAI */
  DEMOS.emu = function (root) {
    const fc = $(root, "canvas.fwd").getContext("2d"), sc = $(root, "canvas.sct").getContext("2d"), out = $(root, ".readout");
    const nIn = $(root, "input[name=noise]"), nv = $(root, ".nv"), vIn = $(root, "input[name=soil]"), vv = $(root, ".vv");
    const RINF = [0.025, 0.48], RS = [0.14, 0.2], KB = [0.65, 0.42];
    const fwd = (lai, s) => [0, 1].map((b) => RINF[b] + (s * RS[b] - RINF[b]) * Math.exp(-KB[b] * lai));
    const ndvi = (o) => (o[1] - o[0]) / (o[1] + o[0]);
    let method = "ml", seed = 1;
    // LUT of simulations (physics) and a hybrid model trained on them
    const r0 = rng(2), LUT = Array.from({ length: 3000 }, () => { const lai = 7 * r0(), s = 0.4 + 1.2 * r0(); return { lai, o: fwd(lai, s) }; });
    function trainML(noise) {
      const r = rng(3), D = 60, Wf = Array.from({ length: D }, () => [9 * gauss(r), 9 * gauss(r)]), B = Array.from({ length: D }, () => 2 * Math.PI * r());
      const phi = (o) => [1, ...Wf.map((w, k) => Math.cos(w[0] * o[0] + w[1] * o[1] + B[k]))];
      const rows = LUT.slice(0, 1500).map((e) => phi(e.o.map((v) => v + noise * gauss(r))));
      const w = ridge(rows, LUT.slice(0, 1500).map((e) => e.lai), 0.3);
      return (o) => clamp(dot(w, phi(o)), 0, 8);
    }
    function lut(o, noise) {
      const s = Math.max(noise, 0.004), best = [];
      LUT.forEach((e) => { const d = ((e.o[0] - o[0]) / s) ** 2 + ((e.o[1] - o[1]) / s) ** 2; best.push([d, e.lai]); });
      best.sort((a, b) => a[0] - b[0]); return best.slice(0, 15).reduce((a, b) => a + b[1], 0) / 15;
    }
    function empirical() {     // field campaign: 25 plots on one soil type, cubic fit LAI ~ NDVI
      const r = rng(4), rows = [], ys = [];
      for (let k = 0; k < 25; k++) { const lai = 6 * r(), o = fwd(lai, 1 + 0.05 * gauss(r)).map((v) => v + 0.004 * gauss(r)), v = ndvi(o); rows.push([1, v, v * v, v * v * v]); ys.push(lai); }
      const w = ridge(rows, ys, 1e-4); return (o) => { const v = ndvi(o); return clamp(dot(w, [1, v, v * v, v * v * v]), 0, 8); };
    }
    const EMP = empirical();
    let ML = null, mlNoise = -1;
    function run() {
      const noise = +nIn.value, sv = +vIn.value; nv.textContent = noise.toFixed(3); vv.textContent = "±" + Math.round(sv * 100) + " %";
      if (Math.abs(noise - mlNoise) > 1e-9) { ML = trainML(Math.max(noise, 0.002)); mlNoise = noise; }
      const r = rng(seed * 13), test = Array.from({ length: 260 }, () => { const lai = 7 * r(), s = 1 + sv * (2 * r() - 1); return { lai, o: fwd(lai, s).map((v) => v + noise * gauss(r)) }; });
      const M = { ndvi: EMP, lut: (o) => lut(o, noise), ml: ML }, rm = {};
      Object.keys(M).forEach((k) => { let se = 0; test.forEach((t) => { t[k] = M[k](t.o); se += (t[k] - t.lai) ** 2; }); rm[k] = Math.sqrt(se / test.length); });
      // forward model
      let W = fc.canvas.width, H = fc.canvas.height; fc.clearRect(0, 0, W, H);
      let A = axes(fc, 58, 12, W - 74, H - 58, { x: [0, 7], y: [0, 0.6], xt: [0, 1, 2, 3, 4, 5, 6, 7], yt: [0, 0.2, 0.4, 0.6], xl: "leaf area index (m²/m²)", yl: "reflectance" });
      [[1, BLU, "NIR"], [0, TER, "red"]].forEach(([b, col, nm]) => {
        fc.beginPath(); for (let l = 0; l <= 7; l += 0.05) fc.lineTo(A.X(l), A.Y(fwd(l, 1 - sv)[b])); for (let l = 7; l >= 0; l -= 0.05) fc.lineTo(A.X(l), A.Y(fwd(l, 1 + sv)[b]));
        fc.closePath(); fc.fillStyle = col === BLU ? "rgba(46,75,198,.15)" : "rgba(181,80,46,.18)"; fc.fill();
        fc.beginPath(); for (let l = 0; l <= 7; l += 0.05) fc.lineTo(A.X(l), A.Y(fwd(l, 1)[b])); fc.strokeStyle = col; fc.lineWidth = 2.5; fc.stroke();
        label(fc, nm, A.X(5.6), A.Y(fwd(5.6, 1)[b]) - 8, col);
      });
      label(fc, "band = soil brightness range", A.X(3.5), A.Y(0.3), MUT, "center", `12px ${FONT}`);
      // retrieval scatter
      W = sc.canvas.width; H = sc.canvas.height; sc.clearRect(0, 0, W, H);
      A = axes(sc, 58, 12, W - 74, H - 58, { x: [0, 7], y: [0, 8], xt: [0, 2, 4, 6], yt: [0, 2, 4, 6, 8], xl: "true LAI", yl: "retrieved LAI" });
      sc.setLineDash([5, 4]); sc.strokeStyle = MUT; sc.beginPath(); sc.moveTo(A.X(0), A.Y(0)); sc.lineTo(A.X(7), A.Y(7)); sc.stroke(); sc.setLineDash([]);
      const col = { ndvi: TER, lut: SAF, ml: BLU }[method];
      test.forEach((t) => { sc.beginPath(); sc.arc(A.X(t.lai), A.Y(t[method]), 3.4, 0, 7); sc.fillStyle = col; sc.globalAlpha = 0.7; sc.fill(); sc.globalAlpha = 1; });
      label(sc, `RMSE ${rm[method].toFixed(2)}`, A.X(0.3), A.Y(7.4), INK, "left", `600 15px ${FONT}`);
      out.textContent = `RMSE (LAI): empirical NDVI fit ${rm.ndvi.toFixed(2)} · LUT inversion ${rm.lut.toFixed(2)} · hybrid ML on simulations ${rm.ml.toFixed(2)} · test: 260 pixels, soil ${vv.textContent}, noise σ = ${noise.toFixed(3)}`;
    }
    $$(root, "[data-m]").forEach((b) => (b.onclick = () => { method = b.dataset.m; pick(root, "data-m", method); run(); }));
    nIn.oninput = run; vIn.oninput = run;
    $(root, "[data-act=new]").onclick = () => { seed++; run(); };
    pick(root, "data-m", method); run();
  };

  /* ======================================================= 6 · geographic bias in training data */
  DEMOS.region = function (root) {
    const bc = $(root, "canvas.bars").getContext("2d"), cc = $(root, "canvas.mix").getContext("2d"), out = $(root, ".readout");
    const aIn = $(root, "input[name=africa]"), av = $(root, ".av");
    const RG = [
      { n: "Europe", t: [1, 0, 1, 0, 0, 0], w: 0.5, col: "#2E4BC6" },
      { n: "North America", t: [1.05, 0.005, 0.97, 0, 0.02, 0], w: 0.33, col: "#5B7BD6" },
      { n: "East Asia", t: [1.12, 0.012, 0.94, 0.005, 0.05, 0.01], w: 0.12, col: "#7B3FB5" },
      { n: "South America", t: [0.96, 0, 1.04, 0, 0, 0.01], w: 0.05, col: "#0F6B5C" },
      { n: "North Africa", t: [1.4, 0.05, 0.84, 0.02, 0.2, 0.08], w: 0, col: "#C99A2E" },
      { n: "Sub-Saharan Africa", t: [1.28, 0.04, 0.8, 0.01, 0.24, 0.05], w: 0, col: "#B5502E" },
    ];
    const tf = (p, t) => ({ x: clamp(t[0] * p.x + t[1] + (p.c === 2 ? t[5] : 0), 0.01, 0.99), y: clamp(t[2] * p.y + t[3] - (p.c === 1 ? t[4] * p.y : 0), 0.01, 0.99), c: p.c });
    const pool = RG.map((g, k) => K.spectral3(300 + k, 120).map((p) => tf(p, g.t)));
    const test = RG.map((g, k) => K.spectral3(400 + k, 60).map((p) => tf(p, g.t)));
    function run() {
      const a = +aIn.value / 100; av.textContent = pct(a, 0);
      const share = RG.map((g, k) => (k >= 4 ? a / 2 : g.w * (1 - a)));
      const r = rng(9), train = [];
      share.forEach((s, k) => { const m = Math.round(s * 600); shuffle(pool[k].slice(), r).slice(0, m).forEach((p) => train.push(p)); });
      const M = gml(train), acc = test.map((T) => T.filter((p) => classify(M, p.x, p.y) === p.c).length / T.length);
      let W = bc.canvas.width, H = bc.canvas.height; bc.clearRect(0, 0, W, H);
      const A = axes(bc, 58, 28, W - 74, H - 76, { x: [0, RG.length], y: [0.4, 1], yt: [0.4, 0.6, 0.8, 1], fy: (v) => Math.round(v * 100) + " %", yl: "test accuracy" });
      RG.forEach((g, k) => {
        const x0 = A.X(k + 0.15), x1 = A.X(k + 0.85), v = Math.max(acc[k], 0.4);
        bc.fillStyle = g.col; bc.fillRect(x0, A.Y(v), x1 - x0, A.Y(0.4) - A.Y(v));
        label(bc, pct(acc[k], 0), (x0 + x1) / 2, A.Y(v) - 6, INK, "center", `700 14px ${FONT}`);
        label(bc, g.n, (x0 + x1) / 2, H - 30, INK, "center", `600 12px ${FONT}`);
      });
      W = cc.canvas.width; H = cc.canvas.height; cc.clearRect(0, 0, W, H);
      let x = 0; share.forEach((s, k) => { cc.fillStyle = RG[k].col; cc.fillRect(x, 6, s * W, H - 12); if (s > 0.06) label(cc, pct(s, 0), x + (s * W) / 2, H / 2 + 5, "#fff", "center"); x += s * W; });
      const lo = acc.indexOf(Math.min(...acc)), hi = acc.indexOf(Math.max(...acc));
      out.textContent = `training labels: ${pct(a, 0)} from Africa · best ${RG[hi].n} ${pct(acc[hi], 0)} · worst ${RG[lo].n} ${pct(acc[lo], 0)} · gap ${Math.round(100 * (acc[hi] - acc[lo]))} points · mean ${pct(acc.reduce((p, q) => p + q, 0) / acc.length, 0)}`;
    }
    $$(root, "[data-preset]").forEach((b) => (b.onclick = () => { aIn.value = b.dataset.preset; run(); }));
    aIn.oninput = run; run();
  };

  /* ======================================================= 7 · carbon footprint of training and inference */
  DEMOS.carbon = function (root) {
    const HW = { rtx: ["RTX 4090", 0.45], a100: ["A100", 0.4], h100: ["H100", 0.7], cpu: ["CPU server", 0.25] };
    const GRID = { low: 0.03, eu: 0.25, world: 0.47, coal: 0.8 };
    const NG = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], HR = [1, 2, 4, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160];
    const hw = $(root, "select[name=hw]"), grid = $(root, "select[name=grid]"), gIn = $(root, "input[name=g]"), hIn = $(root, "input[name=h]"), pIn = $(root, "input[name=pue]");
    const gv = $(root, ".gv"), hv = $(root, ".hv"), pv = $(root, ".pv"), big = $(root, ".big"), cv = $(root, "canvas"), ctx = cv.getContext("2d"), out = $(root, ".readout");
    const PRE = { unet: ["rtx", 0, 3], fm: ["a100", 6, 9], map: ["a100", 3, 6], agent: ["h100", 3, 12] };
    function run() {
      const g = NG[+gIn.value], h = HR[+hIn.value], pue = +pIn.value, P = HW[hw.value][1], ci = GRID[grid.value];
      gv.textContent = g; hv.textContent = h >= 48 ? `${h} h (${(h / 24).toFixed(h % 24 ? 1 : 0)} days)` : `${h} h`; pv.textContent = pue.toFixed(2);
      const kwh = g * P * h * pue, kg = kwh * ci;
      big.innerHTML = `<b>${kwh < 10 ? kwh.toFixed(1) : Math.round(kwh).toLocaleString("en-US")}</b> kWh &nbsp;→&nbsp; <b>${kg < 10 ? kg.toFixed(1) : Math.round(kg).toLocaleString("en-US")}</b> kg CO₂e`;
      const refs = [["your run", kg, TER], ["100 km by petrol car", 17, MUT], ["one long-haul return flight, 1 passenger", 1000, MUT], ["average person, 1 year (world)", 4700, MUT]];
      const W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
      const lo = -1, hi = 6, X = (v) => 300 + ((Math.log10(Math.max(v, 0.1)) - lo) / (hi - lo)) * (W - 330);
      ctx.font = `12px ${FONT}`; ctx.fillStyle = MUT; ctx.textAlign = "center";
      for (let e = lo; e <= hi; e++) { ctx.strokeStyle = "rgba(20,33,61,.1)"; ctx.beginPath(); ctx.moveTo(X(10 ** e), 4); ctx.lineTo(X(10 ** e), H - 22); ctx.stroke(); ctx.fillText(e < 3 ? `${10 ** e} kg` : `${10 ** (e - 3)} t`, X(10 ** e), H - 6); }
      refs.forEach(([nm, v, col], j) => {
        const yy = 14 + j * ((H - 40) / refs.length);
        ctx.fillStyle = col; ctx.fillRect(X(0.1), yy, X(v) - X(0.1), 22);
        label(ctx, nm, 290, yy + 16, INK, "right", `600 13px ${FONT}`);
      });
      out.textContent = `E = ${g} × ${P} kW × ${h} h × PUE ${pue.toFixed(2)} = ${kwh < 10 ? kwh.toFixed(1) : kwh.toFixed(0)} kWh · × ${ci} kg CO₂e/kWh · ≈ ${(kg / 0.17).toFixed(0)} km by car · hardware manufacturing (embodied carbon) not included`;
    }
    $$(root, "[data-pre]").forEach((b) => (b.onclick = () => { const [h, gi, hi] = PRE[b.dataset.pre]; hw.value = h; gIn.value = gi; hIn.value = hi; pick(root, "data-pre", b.dataset.pre); run(); }));
    [hw, grid].forEach((s) => (s.onchange = run)); [gIn, hIn, pIn].forEach((s) => (s.oninput = run));
    run();
  };
})();
