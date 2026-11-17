/* Deep Learning for Earth Observation — unsupervised-learning demos (no dependencies)
   k-means (step by step) · k-means unsupervised classification of the synthetic scene ·
   Gaussian mixture + EM · DBSCAN · PCA (2-D and 13-band scene).
   Needs demos_ml.js (DemoKit, Plot) and demos_vision.js (scene) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, gauss, clamp, pick, argmax, Plot } = K;
  const PAL = [[46, 75, 198], [15, 107, 92], [181, 80, 46], [201, 154, 46], [123, 63, 181], [27, 150, 190], [200, 80, 140], [110, 110, 110]];
  const col = (k, a = 1) => `rgba(${PAL[k % PAL.length].join(",")},${a})`;
  const INK = "#14213D", MUT = "#5B6478", SAF = "#C99A2E";
  const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  const pct = (v) => (100 * v).toFixed(1) + " %";

  function blobs(seed, spec, n) {          // spec: [mx, my, sx, sy, rho]
    const r = rng(seed), pts = [];
    spec.forEach(([mx, my, sx, sy, rho = 0]) => {
      for (let i = 0; i < n; i++) {
        const a = gauss(r), b = gauss(r);
        pts.push({ x: clamp(mx + sx * a, 0.01, 0.99), y: clamp(my + sy * (rho * a + Math.sqrt(1 - rho * rho) * b), 0.01, 0.99) });
      }
    });
    return pts;
  }
  /* random or k-means++ (D² sampling) initial centres */
  function initCenters(pts, k, plus, r) {
    const p0 = pts[Math.floor(r() * pts.length)], C = [{ x: p0.x, y: p0.y }];
    while (C.length < k) {
      if (!plus) { const p = pts[Math.floor(r() * pts.length)]; C.push({ x: p.x, y: p.y }); continue; }
      const D = pts.map((p) => Math.min(...C.map((c) => d2(p, c)))), S = D.reduce((a, b) => a + b, 0);
      let u = r() * S, i = 0;
      while (u > D[i] && i < D.length - 1) { u -= D[i]; i++; }
      C.push({ x: pts[i].x, y: pts[i].y });
    }
    return C;
  }
  function ellipse(P, g, s) {                // g: {mx, my, sxx, syy, sxy}
    const ctx = P.ctx, tr = g.sxx + g.syy, det = g.sxx * g.syy - g.sxy * g.sxy, d = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const l1 = tr / 2 + d, l2 = Math.max(1e-9, tr / 2 - d);
    let e1 = Math.abs(g.sxy) > 1e-12 ? [l1 - g.syy, g.sxy] : g.sxx >= g.syy ? [1, 0] : [0, 1];
    const nn = Math.hypot(e1[0], e1[1]); e1 = [e1[0] / nn, e1[1] / nn];
    const e2 = [-e1[1], e1[0]];
    ctx.beginPath();
    for (let t = 0; t <= 72; t++) {
      const a = (t / 72) * 2 * Math.PI, u = s * Math.sqrt(l1) * Math.cos(a), v = s * Math.sqrt(l2) * Math.sin(a);
      const px = P.X(g.mx + u * e1[0] + v * e2[0]), py = P.Y(g.my + u * e1[1] + v * e2[1]);
      t ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
  }
  function centroid(P, c, k) {
    const ctx = P.ctx, x = P.X(c.x), y = P.Y(c.y);
    ctx.beginPath(); ctx.arc(x, y, 12, 0, 7); ctx.fillStyle = col(k); ctx.fill(); ctx.lineWidth = 3.5; ctx.strokeStyle = INK; ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
  }

  /* ======================================================= 1 · k-means, step by step */
  DEMOS.kmeans = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const kIn = root.querySelector("input[name=k]"), kv = root.querySelector(".kv"), initSel = root.querySelector("select[name=init]");
    const pts = blobs(31, [[0.11, 0.1, 0.035, 0.03], [0.14, 0.72, 0.04, 0.06], [0.33, 0.43, 0.05, 0.05], [0.63, 0.5, 0.06, 0.045, 0.6]], 30);
    let C, assigned, it, phase, seed = 1, timer = null, msg = "";
    const J = () => pts.reduce((s, p) => s + d2(p, C[p.k]), 0);
    function reset() {
      clearInterval(timer);
      const k = +kIn.value; kv.textContent = k;
      C = initCenters(pts, k, initSel.value === "pp", rng(seed * 97 + k));
      assigned = false; it = 0; phase = "assign"; msg = "centroids initialised — press Assign";
    }
    function assign() {
      let changed = 0;
      pts.forEach((p) => {
        let b = 0, bd = Infinity;
        C.forEach((c, k) => { const d = d2(p, c); if (d < bd) { bd = d; b = k; } });
        if (!assigned || p.k !== b) changed++;
        p.k = b;
      });
      assigned = true; phase = "update"; msg = `assignment step: ${changed} pixels changed cluster`;
      return changed;
    }
    function update() {
      C = C.map((c, k) => {
        const S = pts.filter((p) => p.k === k);
        return S.length ? { x: S.reduce((s, p) => s + p.x, 0) / S.length, y: S.reduce((s, p) => s + p.y, 0) / S.length } : c;
      });
      it++; phase = "assign"; msg = `update step: each centroid moved to the mean of its pixels (iteration ${it})`;
    }
    function draw() {
      P.field(165, 100, (i, j, x, y) => {
        let b = 0, bd = Infinity;
        C.forEach((c, k) => { const d = (x - c.x) ** 2 + (y - c.y) ** 2; if (d < bd) { bd = d; b = k; } });
        return [...PAL[b], 0.15];
      }, false);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", K.SPEC_TICKS);
      if (assigned) {
        ctx.strokeStyle = "rgba(20,33,61,.16)"; ctx.lineWidth = 1;
        pts.forEach((p) => { const c = C[p.k]; ctx.beginPath(); ctx.moveTo(P.X(p.x), P.Y(p.y)); ctx.lineTo(P.X(c.x), P.Y(c.y)); ctx.stroke(); });
      }
      P.points(pts, (p) => (assigned ? col(p.k) : "#9AA3B5"), { r: () => 5 });
      C.forEach((c, k) => centroid(P, c, k));
      out.textContent = `k = ${C.length} · iteration ${it} · J = Σ‖x − μ‖² = ${assigned ? J().toFixed(3) : "—"} · ${msg}`;
    }
    root.querySelector("[data-act=assign]").onclick = () => { clearInterval(timer); assign(); draw(); };
    root.querySelector("[data-act=update]").onclick = () => { clearInterval(timer); if (assigned) update(); draw(); };
    root.querySelector("[data-act=run]").onclick = () => {
      clearInterval(timer);
      timer = setInterval(() => {
        if (phase === "assign") { const ch = assign(); if (ch === 0 && it > 0) { clearInterval(timer); msg = `converged after ${it} iterations — J never increases`; } }
        else update();
        draw();
      }, 350);
    };
    root.querySelector("[data-act=new]").onclick = () => { seed++; reset(); draw(); };
    kIn.oninput = () => { reset(); draw(); };
    initSel.onchange = () => { reset(); draw(); };
    reset(); assign(); update(); assign(); draw();
  };

  /* ======================================================= 2 · k-means on the scene (unsupervised classification) */
  DEMOS.ukscene = function (root) {
    const S = K.scene(), N = S.N, n = S.n, out = root.querySelector(".readout"), legend = root.querySelector(".legend");
    const sc = root.querySelector("canvas.src").getContext("2d"), dc = root.querySelector("canvas.dst").getContext("2d");
    const kIn = root.querySelector("input[name=k]"), kv = root.querySelector(".kv");
    const FEAT = { rgb: { b: [1, 2, 3], name: "B2 B3 B4 (visible only)" }, s2: { b: [1, 2, 3, 7, 11, 12], name: "B2 B3 B4 B8 B11 B12" } };
    // colours ordered from water-like to vegetation-like (clusters are ranked by mean NDVI)
    const SCPAL = [[46, 75, 198], [95, 100, 115], [181, 80, 46], [201, 154, 46], [150, 165, 75], [15, 107, 92], [8, 66, 52]];
    let feat = "s2";
    const band = (i, b) => S.B[i * 13 + b];
    const idImg = sc.createImageData(N, N);
    for (let i = 0; i < n; i++) { idImg.data[i * 4] = S.rgb8[i * 3]; idImg.data[i * 4 + 1] = S.rgb8[i * 3 + 1]; idImg.data[i * 4 + 2] = S.rgb8[i * 3 + 2]; idImg.data[i * 4 + 3] = 255; }
    sc.putImageData(idImg, 0, 0);
    function run() {
      const k = +kIn.value, f = FEAT[feat].b, d = f.length, r = rng(5 + k);
      kv.textContent = k;
      const X = new Float32Array(n * d);
      f.forEach((b, j) => {
        let m = 0, s = 0; for (let i = 0; i < n; i++) m += band(i, b); m /= n;
        for (let i = 0; i < n; i++) s += (band(i, b) - m) ** 2; s = Math.sqrt(s / n) || 1;
        for (let i = 0; i < n; i++) X[i * d + j] = (band(i, b) - m) / s;
      });
      const dist = (i, c) => { let s = 0; for (let j = 0; j < d; j++) s += (X[i * d + j] - c[j]) ** 2; return s; };
      // k-means++ init
      const i0 = Math.floor(r() * n), C = [Array.from(X.subarray(i0 * d, i0 * d + d))];
      const D = new Float64Array(n).fill(Infinity);
      while (C.length < k) {
        let S2 = 0; const c = C[C.length - 1];
        for (let i = 0; i < n; i++) { D[i] = Math.min(D[i], dist(i, c)); S2 += D[i]; }
        let u = r() * S2, i = 0; while (u > D[i] && i < n - 1) { u -= D[i]; i++; }
        C.push(Array.from(X.subarray(i * d, i * d + d)));
      }
      const lab = new Uint8Array(n);
      let iters = 0, J = 0;
      for (; iters < 30; iters++) {
        let ch = 0; J = 0;
        for (let i = 0; i < n; i++) { let b = 0, bd = Infinity; for (let c = 0; c < k; c++) { const v = dist(i, C[c]); if (v < bd) { bd = v; b = c; } } if (lab[i] !== b) ch++; lab[i] = b; J += bd; }
        const sum = Array.from({ length: k }, () => new Float64Array(d)), cnt = new Float64Array(k);
        for (let i = 0; i < n; i++) { cnt[lab[i]]++; for (let j = 0; j < d; j++) sum[lab[i]][j] += X[i * d + j]; }
        for (let c = 0; c < k; c++) if (cnt[c]) C[c] = Array.from(sum[c], (v) => v / cnt[c]);
        if (ch === 0 && iters > 0) break;
      }
      // describe clusters
      const st = Array.from({ length: k }, () => ({ n: 0, ndvi: 0, ndwi: 0, mat: new Float64Array(6) }));
      for (let i = 0; i < n; i++) {
        const s = st[lab[i]], nir = band(i, 7), red = band(i, 3), gr = band(i, 2);
        s.n++; s.ndvi += (nir - red) / (nir + red); s.ndwi += (gr - nir) / (gr + nir); s.mat[S.dom[i]]++;
      }
      st.forEach((s) => { s.ndvi /= s.n || 1; s.ndwi /= s.n || 1; });
      const order = st.map((s, c) => c).sort((a, b) => st[a].ndvi - st[b].ndvi), rank = new Array(k);
      order.forEach((c, i) => (rank[c] = i));
      const colOf = (c) => SCPAL[k > 1 ? Math.round((rank[c] * (SCPAL.length - 1)) / (k - 1)) : 0];
      const id = dc.createImageData(N, N);
      for (let i = 0; i < n; i++) { const c = colOf(lab[i]); id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255; }
      dc.putImageData(id, 0, 0);
      const guess = (s) => (s.ndwi > 0.15 ? "water" : s.ndvi > 0.45 ? "dense vegetation" : s.ndvi > 0.25 ? "crops / sparse vegetation" : "bare soil or built-up");
      legend.innerHTML = order.map((c) => {
        const s = st[c], co = colOf(c);
        return `<div><i style="background:rgb(${co})"></i><span><b>${(100 * s.n / n).toFixed(0)} %</b> · NDVI ${s.ndvi.toFixed(2)}<br><span class="xs">analyst: ${guess(s)}</span></span></div>`;
      }).join("");
      const purity = st.reduce((a, s) => a + Math.max(...s.mat), 0) / n;
      out.textContent = `k = ${k} · features ${FEAT[feat].name} (standardised) · converged in ${iters + 1} iterations · purity vs true cover ${pct(purity)}`;
    }
    root.querySelectorAll("[data-feat]").forEach((b) => (b.onclick = () => { feat = b.dataset.feat; pick(root, "data-feat", feat); run(); }));
    kIn.oninput = run;
    pick(root, "data-feat", feat); run();
  };

  /* ======================================================= 3 · Gaussian mixture + EM */
  DEMOS.gmm = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const kIn = root.querySelector("input[name=k]"), kv = root.querySelector(".kv");
    const pts = K.spectral3(5, 40, 0).map((p) => ({ x: p.x, y: p.y }));
    let G, R = null, it = 0, ll = NaN, seed = 3, timer = null, step = "";
    const logN = (g, x, y) => {
      const dx = x - g.mx, dy = y - g.my, det = g.sxx * g.syy - g.sxy * g.sxy;
      return -0.5 * (g.syy * dx * dx - 2 * g.sxy * dx * dy + g.sxx * dy * dy) / det - 0.5 * Math.log(det) - Math.log(2 * Math.PI);
    };
    function post(x, y) {
      const l = G.map((g) => Math.log(g.pi) + logN(g, x, y)), m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), Z = e.reduce((a, b) => a + b, 0);
      return { p: e.map((v) => v / Z), lse: m + Math.log(Z) };
    }
    function reset() {
      clearInterval(timer);
      const k = +kIn.value, r = rng(seed * 13 + k); kv.textContent = k;
      G = initCenters(pts, k, true, r).map((c) => ({ pi: 1 / k, mx: c.x, my: c.y, sxx: 0.004, syy: 0.004, sxy: 0 }));
      R = null; it = 0; ll = NaN; step = "random means, round covariances — press E-step";
    }
    function E() { ll = 0; R = pts.map((q) => { const o = post(q.x, q.y); ll += o.lse; return o.p; }); step = "E-step: soft responsibilities γ updated (colours)"; }
    function M() {
      if (!R) E();
      G = G.map((g, k) => {
        let Nk = 0, mx = 0, my = 0; pts.forEach((q, i) => { const w = R[i][k]; Nk += w; mx += w * q.x; my += w * q.y; });
        Nk = Math.max(Nk, 1e-9); mx /= Nk; my /= Nk;
        let sxx = 0, syy = 0, sxy = 0; pts.forEach((q, i) => { const w = R[i][k]; sxx += w * (q.x - mx) ** 2; syy += w * (q.y - my) ** 2; sxy += w * (q.x - mx) * (q.y - my); });
        return { pi: Nk / pts.length, mx, my, sxx: sxx / Nk + 1e-4, syy: syy / Nk + 1e-4, sxy: sxy / Nk };
      });
      it++; step = `M-step: means, covariances and weights re-estimated (iteration ${it})`;
    }
    function draw() {
      P.field(165, 100, (i, j, x, y) => { const p = post(x, y).p, k = argmax(p); return [...PAL[k], 0.06 + 0.24 * p[k]]; }, true);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", K.SPEC_TICKS);
      P.points(pts, (q) => {
        if (!R) return "#9AA3B5";
        const w = R[pts.indexOf(q)], c = [0, 0, 0]; w.forEach((v, k) => PAL[k].forEach((x, j) => (c[j] += v * x)));
        return `rgb(${c.map(Math.round)})`;
      }, { r: () => 5 });
      G.forEach((g, k) => {
        ctx.strokeStyle = col(k); ctx.lineWidth = 2.6; ctx.setLineDash([]); ellipse(P, g, 1);
        ctx.lineWidth = 1.5; ctx.setLineDash([6, 5]); ellipse(P, g, 2); ctx.setLineDash([]);
        centroid(P, { x: g.mx, y: g.my }, k);
      });
      out.textContent = `K = ${G.length} · iteration ${it} · log-likelihood ${isNaN(ll) ? "—" : ll.toFixed(1)} · π = (${G.map((g) => g.pi.toFixed(2)).join(", ")}) · ${step}`;
    }
    root.querySelector("[data-act=e]").onclick = () => { clearInterval(timer); E(); draw(); };
    root.querySelector("[data-act=m]").onclick = () => { clearInterval(timer); M(); draw(); };
    root.querySelector("[data-act=run]").onclick = () => {
      clearInterval(timer); let c = 0;
      timer = setInterval(() => { E(); M(); E(); draw(); if (++c >= 30) clearInterval(timer); }, 160);
    };
    root.querySelector("[data-act=new]").onclick = () => { seed++; reset(); draw(); };
    kIn.oninput = () => { reset(); draw(); };
    reset(); for (let i = 0; i < 25; i++) { E(); M(); } E(); draw();
  };

  /* ======================================================= 4 · DBSCAN */
  DEMOS.dbscan = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const eIn = root.querySelector("input[name=eps]"), mIn = root.querySelector("input[name=minpts]");
    const eV = root.querySelector(".ev"), mV = root.querySelector(".mv");
    // detections (e.g. buildings or fire hotspots): a settlement strung along a river, a second arc, a compact village, false alarms
    const r = rng(17), pts = [];
    for (let i = 0; i < 80; i++) { const t = r(); pts.push({ x: 0.08 + 0.62 * t + 0.012 * gauss(r), y: 0.68 + 0.12 * Math.sin(t * 5.5) + 0.018 * gauss(r) }); }
    for (let i = 0; i < 55; i++) { const a = Math.PI * (0.15 + 0.85 * r()); pts.push({ x: 0.36 + 0.24 * Math.cos(a) + 0.014 * gauss(r), y: 0.42 - 0.24 * Math.sin(a) + 0.014 * gauss(r) }); }
    for (let i = 0; i < 45; i++) pts.push({ x: 0.8 + 0.035 * gauss(r), y: 0.28 + 0.035 * gauss(r) });
    for (let i = 0; i < 30; i++) pts.push({ x: 0.03 + 0.94 * r(), y: 0.03 + 0.94 * r() });
    const n = pts.length;
    let mode = "db", lab = [], core = [], hover = null;
    function dbscan(eps, minPts) {
      const e2 = eps * eps, nb = pts.map((p) => { const L = []; pts.forEach((q, j) => { if (d2(p, q) <= e2) L.push(j); }); return L; });
      core = nb.map((L) => L.length >= minPts);
      lab = new Array(n).fill(-2); let c = 0;
      for (let i = 0; i < n; i++) {
        if (lab[i] !== -2) continue;
        if (!core[i]) { lab[i] = -1; continue; }
        lab[i] = c; const queue = [...nb[i]];
        while (queue.length) {
          const j = queue.pop();
          if (lab[j] === -1) lab[j] = c;
          if (lab[j] !== -2) continue;
          lab[j] = c;
          if (core[j]) queue.push(...nb[j]);
        }
        c++;
      }
      return c;
    }
    function kmeans3() {
      let C = initCenters(pts, 3, true, rng(4));
      for (let t = 0; t < 30; t++) {
        lab = pts.map((p) => { let b = 0, bd = Infinity; C.forEach((c, k) => { const d = d2(p, c); if (d < bd) { bd = d; b = k; } }); return b; });
        C = C.map((c, k) => { const S = pts.filter((_, i) => lab[i] === k); return S.length ? { x: S.reduce((s, p) => s + p.x, 0) / S.length, y: S.reduce((s, p) => s + p.y, 0) / S.length } : c; });
      }
      core = pts.map(() => true);
    }
    function draw() {
      const eps = +eIn.value, minPts = +mIn.value;
      eV.textContent = eps.toFixed(3); mV.textContent = minPts;
      let nc = 3;
      if (mode === "db") nc = dbscan(eps, minPts); else kmeans3();
      P.field(4, 4, () => [255, 255, 255, 0], false);
      P.frame("easting →", "northing →", null);
      if (hover && mode === "db") {
        ctx.beginPath(); ctx.ellipse(P.X(hover.x), P.Y(hover.y), eps * P.W, eps * P.H, 0, 0, 7);
        ctx.fillStyle = "rgba(201,154,46,.18)"; ctx.fill(); ctx.strokeStyle = SAF; ctx.lineWidth = 2; ctx.stroke();
      }
      pts.forEach((p, i) => {
        const x = P.X(p.x), y = P.Y(p.y);
        if (lab[i] < 0) { ctx.strokeStyle = "#8A93A6"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke(); return; }
        ctx.beginPath(); ctx.arc(x, y, core[i] ? 5.5 : 3.8, 0, 7); ctx.fillStyle = col(lab[i], core[i] ? 1 : 0.55); ctx.fill();
        ctx.lineWidth = 1.3; ctx.strokeStyle = "#fff"; ctx.stroke();
      });
      if (mode === "db") {
        const noise = lab.filter((v) => v < 0).length, nCore = core.filter(Boolean).length;
        let s = `DBSCAN · ε = ${eps.toFixed(3)} · minPts = ${minPts} · clusters ${nc} · core ${nCore} · border ${n - nCore - noise} · noise ${noise}`;
        if (hover) { const c = pts.filter((q) => d2(q, hover) <= eps * eps).length; s += ` · ${c} points within ε of the cursor`; }
        out.textContent = s;
      } else out.textContent = "k-means (k = 3): spherical clusters cut the curved settlements and absorb every false alarm";
    }
    eIn.oninput = draw; mIn.oninput = draw;
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { mode = b.dataset.mode; pick(root, "data-mode", mode); draw(); }));
    cv.addEventListener("mousemove", (e) => { hover = P.at(e); draw(); });
    cv.addEventListener("mouseleave", () => { hover = null; draw(); });
    pick(root, "data-mode", mode); draw();
  };

  /* ======================================================= 5 · PCA (2-D intuition + 13-band scene) */
  function jacobi(A) {
    const n = A.length, a = A.map((r) => r.slice()), v = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
      }
    }
    return a.map((r, i) => ({ val: r[i], vec: v.map((row) => row[i]) })).sort((x, y) => y.val - x.val);
  }
  K.jacobi = jacobi;
  DEMOS.pca = function (root) {
    const S = K.scene(), N = S.N, n = S.n;
    const cv = root.querySelector("canvas.p2"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height, out = root.querySelector(".readout");
    const aIn = root.querySelector("input[name=angle]"), aV = root.querySelector(".av");
    const ic = root.querySelector("canvas.pcimg").getContext("2d"), bc = root.querySelector("canvas.bars"), bx = bc.getContext("2d");
    // 2-D: red (B4) vs SWIR-1 (B11) of 450 random pixels; equal scales on both axes
    const r = rng(8), P2 = [];
    for (let k = 0; k < 450; k++) { const i = Math.floor(r() * n); P2.push([S.B[i * 13 + 3], S.B[i * 13 + 11]]); }
    const mu = [0, 1].map((j) => P2.reduce((s, p) => s + p[j], 0) / P2.length);
    // equal scales on both axes (so angles are true), view centred on the data in x
    const YR = 0.48, XR = (YR * (W - 50)) / (H - 40), X0 = mu[0] - XR / 2;
    const X = (v) => 40 + ((v - X0) / XR) * (W - 50), Y = (v) => H - 30 - (v / YR) * (H - 40);
    const cov = [[0, 0], [0, 0]];
    P2.forEach((p) => { for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) cov[a][b] += ((p[a] - mu[a]) * (p[b] - mu[b])) / P2.length; });
    const eig2 = jacobi(cov), tot2 = eig2[0].val + eig2[1].val;
    const ang1 = ((Math.atan2(eig2[0].vec[1], eig2[0].vec[0]) * 180) / Math.PI + 180) % 180;
    function draw2() {
      const th = (+aIn.value * Math.PI) / 180, u = [Math.cos(th), Math.sin(th)];
      aV.textContent = `${(+aIn.value).toFixed(0)}°`;
      ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#FDFBF7"; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(20,33,61,.1)"; ctx.fillStyle = MUT; ctx.font = "12px Inter, Arial, sans-serif"; ctx.textAlign = "center";
      for (let v = Math.ceil(Math.max(0, X0) * 10) / 10 || 0.1; v < X0 + XR; v += 0.1) { if (v <= 0) continue; ctx.beginPath(); ctx.moveTo(X(v), 10); ctx.lineTo(X(v), H - 30); ctx.stroke(); ctx.fillText(v.toFixed(1), X(v), H - 14); }
      ctx.textAlign = "right"; for (let v = 0.1; v < YR; v += 0.1) { ctx.beginPath(); ctx.moveTo(40, Y(v)); ctx.lineTo(W - 10, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), 34, Y(v) + 4); }
      ctx.fillStyle = INK; ctx.font = "600 13px Inter, Arial, sans-serif"; ctx.textAlign = "right"; ctx.fillText("Red (B4) →", W - 14, H - 36);
      ctx.save(); ctx.translate(54, 16); ctx.rotate(-Math.PI / 2); ctx.fillText("SWIR-1 (B11) →", 0, 0); ctx.restore();
      // projection line and residuals
      const L = 0.6, a = [mu[0] - L * u[0], mu[1] - L * u[1]], b = [mu[0] + L * u[0], mu[1] + L * u[1]];
      let pv = 0;
      ctx.strokeStyle = "rgba(46,75,198,.18)"; ctx.lineWidth = 1;
      P2.forEach((p) => {
        const t = (p[0] - mu[0]) * u[0] + (p[1] - mu[1]) * u[1]; pv += (t * t) / P2.length;
        ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(mu[0] + t * u[0]), Y(mu[1] + t * u[1])); ctx.stroke();
      });
      P2.forEach((p) => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.6, 0, 7); ctx.fillStyle = "rgba(20,33,61,.55)"; ctx.fill(); });
      ctx.strokeStyle = "#2E4BC6"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke();
      [[eig2[0], "#C99A2E", "PC1"], [eig2[1], "#B5502E", "PC2"]].forEach(([e, c, lb]) => {
        const s = 2 * Math.sqrt(e.val), ex = mu[0] + s * e.vec[0], ey = mu[1] + s * e.vec[1];
        ctx.strokeStyle = c; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(X(mu[0]), Y(mu[1])); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
        ctx.fillStyle = c; ctx.font = "700 14px Inter, Arial, sans-serif"; ctx.textAlign = "left"; ctx.fillText(lb, X(ex) + 6, Y(ey) - 4);
      });
      out.textContent = `variance along the blue direction ${(1e4 * pv).toFixed(2)}·10⁻⁴ = ${pct(pv / tot2)} of the total · maximum = λ₁ = ${pct(eig2[0].val / tot2)} at ${ang1.toFixed(0)}° (PC1)`;
    }
    // 13-band PCA of the whole scene
    const d = 13, mean = new Float64Array(d), C13 = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += S.B[i * d + j] / n;
    for (let i = 0; i < n; i += 2) for (let a = 0; a < d; a++) { const va = S.B[i * d + a] - mean[a]; for (let b = a; b < d; b++) C13[a][b] += va * (S.B[i * d + b] - mean[b]); }
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { C13[a][b] /= n / 2; C13[b][a] = C13[a][b]; }
    const eig = jacobi(C13), tot = eig.reduce((s, e) => s + e.val, 0);
    const score = (i, k) => { let s = 0; for (let j = 0; j < d; j++) s += (S.B[i * d + j] - mean[j]) * eig[k].vec[j]; return s; };
    const stretch = (k) => {
      const v = new Float32Array(n); for (let i = 0; i < n; i++) v[i] = score(i, k);
      const s = Float32Array.from(v).sort(), lo = s[Math.floor(0.02 * n)], hi = s[Math.floor(0.98 * n)];
      return v.map((x) => clamp((x - lo) / (hi - lo), 0, 1) * 255);
    };
    const PCS = [0, 1, 2, 3].map(stretch);
    let show = "rgb";
    function drawImg() {
      const id = ic.createImageData(N, N);
      for (let i = 0; i < n; i++) {
        const c = show === "rgb" ? [PCS[0][i], PCS[1][i], PCS[2][i]] : [PCS[+show][i], PCS[+show][i], PCS[+show][i]];
        id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255;
      }
      ic.putImageData(id, 0, 0);
      const BW = bc.width, BH = bc.height, L = 34, B = 26, w = (BW - L - 8) / d;
      bx.clearRect(0, 0, BW, BH); bx.fillStyle = "#FDFBF7"; bx.fillRect(0, 0, BW, BH);
      bx.font = "11px Inter, Arial, sans-serif"; bx.fillStyle = MUT; bx.textAlign = "right";
      [0, 50, 100].forEach((v) => bx.fillText(v + "%", L - 4, BH - B - (v / 100) * (BH - B - 16) + 4));
      let cum = 0; const pts = [];
      eig.forEach((e, k) => {
        const f = e.val / tot, h = f * (BH - B - 16); cum += f;
        bx.fillStyle = k < 3 ? ["#C99A2E", "#B5502E", "#2E4BC6"][k] : "#C9D2F3";
        bx.fillRect(L + k * w + 2, BH - B - h, w - 4, h);
        bx.fillStyle = MUT; bx.textAlign = "center"; bx.fillText(k + 1, L + k * w + w / 2, BH - 10);
        pts.push([L + k * w + w / 2, BH - B - cum * (BH - B - 16)]);
      });
      bx.strokeStyle = INK; bx.lineWidth = 1.6; bx.beginPath(); pts.forEach(([x, y], k) => (k ? bx.lineTo(x, y) : bx.moveTo(x, y))); bx.stroke();
      bx.fillStyle = INK; bx.textAlign = "left"; bx.font = "600 12px Inter, Arial, sans-serif";
      bx.fillText(`PC1 ${pct(eig[0].val / tot)} · PC1–3 ${pct((eig[0].val + eig[1].val + eig[2].val) / tot)} (line = cumulative)`, L, 12);
    }
    aIn.oninput = draw2;
    root.querySelector("[data-act=snap]").onclick = () => { aIn.value = ang1.toFixed(0); draw2(); };
    root.querySelectorAll("[data-pc]").forEach((b) => (b.onclick = () => { show = b.dataset.pc; pick(root, "data-pc", show); drawImg(); }));
    pick(root, "data-pc", show); draw2(); drawImg();
  };
})();
