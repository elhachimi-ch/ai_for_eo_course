/* Deep Learning for Earth Observation — the four learning paradigms side by side (no dependencies)
   One demo, four modes on EO toy data:
     supervised      k-NN on labelled (Red, NIR) pixels
     unsupervised    k-means++ on the same pixels, no labels
     self-supervised predict hidden Sentinel-2 bands from the visible ones (masked-band pretext task)
     reinforcement   tabular Q-learning: a survey drone flies to a flooded village around storm cells
   Needs demos_ml.js (DemoKit, Plot, spectral3) and demos_vision.js (scene) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, clamp, pick, argmax, Plot } = K;
  const INK = "#14213D", MUT = "#5B6478", SAF = "#C99A2E", IVORY = "#FDFBF7";
  const CLS = [[46, 75, 198], THEME.sRgb, THEME.pRgb];                       // water, vegetation, soil
  const CLU = [[123, 63, 181], [27, 150, 190], [201, 154, 46], [200, 80, 140], [110, 110, 110], [96, 140, 60]];
  const rgba = (c, a = 1) => `rgba(${c.join(",")},${a})`;
  const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  const pct = (v) => (100 * v).toFixed(0) + " %";
  const BN = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"];
  const WL = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190];

  DEMOS.paradigms = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, W = cv.width, H = cv.height;
    const out = root.querySelector(".readout"), slide = root.closest("section") || root;
    const $ = (s) => root.querySelector(s);
    let mode = "sup";

    /* ---------------------------------------------------------- shared pixels (Red, NIR) */
    const pts = K.spectral3(11, 40, 0);
    const order = pts.map((_, i) => i);
    { const r = rng(7); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } }

    /* ---------------------------------------------------------- 1 · supervised */
    const mIn = $("input[name=m]"), mv = $(".mv");
    function sup() {
      const m = +mIn.value; mv.textContent = m;
      const cnt = [0, 0, 0], lab = [];
      order.forEach((i) => { const p = pts[i]; if (cnt[p.c] < m) { cnt[p.c]++; lab.push(p); } });
      const L = new Set(lab), kk = m >= 3 ? 3 : 1;
      const knn = (x, y) => {
        const ds = lab.map((p) => [(p.x - x) ** 2 + (p.y - y) ** 2, p.c]).sort((a, b) => a[0] - b[0]), v = [0, 0, 0];
        for (let j = 0; j < kk; j++) v[ds[j][1]] += 1 + 1e-6 * (kk - j);      // nearest breaks ties
        return argmax(v);
      };
      P.field(132, 80, (i, j, x, y) => [...CLS[knn(x, y)], 0.17], false);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", K.SPEC_TICKS);
      const test = pts.filter((p) => !L.has(p)), ok = test.filter((p) => knn(p.x, p.y) === p.c).length;
      P.hollow(test, (p) => rgba(CLS[p.c], 0.55));
      P.points(lab, (p) => rgba(CLS[p.c]), { r: () => 8, bad: () => true });
      legend(["water", "vegetation", "soil / built-up"], CLS);
      out.innerHTML = `<b>${lab.length}</b> labelled pixels (ringed) · ${kk}-nearest-neighbour classifier · ` +
        `accuracy on the other ${test.length} pixels: <b>${pct(ok / test.length)}</b><br>feedback = a human label for every training pixel`;
    }

    /* ---------------------------------------------------------- 2 · unsupervised */
    const kIn = $("input[name=k]"), kv = $(".kv");
    let useed = 1;
    function unsup() {
      const k = +kIn.value; kv.textContent = k;
      const r = rng(useed * 131 + k), p0 = pts[Math.floor(r() * pts.length)], C = [{ x: p0.x, y: p0.y }];
      while (C.length < k) {                                             // k-means++ (D² sampling)
        const D = pts.map((p) => Math.min(...C.map((c) => d2(p, c))));
        let u = r() * D.reduce((a, b) => a + b, 0), i = 0;
        while (u > D[i] && i < D.length - 1) { u -= D[i]; i++; }
        C.push({ x: pts[i].x, y: pts[i].y });
      }
      const lab = new Array(pts.length).fill(-1);
      let it = 0;
      for (; it < 60; it++) {
        let ch = 0;
        pts.forEach((p, i) => {
          let b = 0, bd = Infinity;
          C.forEach((c, j) => { const d = d2(p, c); if (d < bd) { bd = d; b = j; } });
          if (lab[i] !== b) { ch++; lab[i] = b; }
        });
        C.forEach((c, j) => {
          const S = pts.filter((_, i) => lab[i] === j);
          if (S.length) { c.x = S.reduce((s, p) => s + p.x, 0) / S.length; c.y = S.reduce((s, p) => s + p.y, 0) / S.length; }
        });
        if (!ch) break;
      }
      const near = (x, y) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = (x - c.x) ** 2 + (y - c.y) ** 2; if (d < bd) { bd = d; b = j; } }); return b; };
      P.field(132, 80, (i, j, x, y) => [...CLU[near(x, y)], 0.15], false);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", K.SPEC_TICKS);
      P.points(pts, (p) => rgba(CLU[lab[pts.indexOf(p)]]), { r: () => 6 });
      C.forEach((c, j) => {
        const x = P.X(c.x), y = P.Y(c.y);
        ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.fillStyle = rgba(CLU[j]); ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "700 13px Inter, Arial, sans-serif"; ctx.textAlign = "center"; ctx.fillText(j + 1, x, y + 4.5);
      });
      const J = pts.reduce((s, p, i) => s + d2(p, C[lab[i]]), 0);
      let pure = 0;
      C.forEach((_, j) => { const c = [0, 0, 0]; pts.forEach((p, i) => { if (lab[i] === j) c[p.c]++; }); pure += Math.max(...c); });
      legend(C.map((_, j) => `cluster ${j + 1}`), CLU);
      out.innerHTML = `${k} clusters in ${it + 1} iterations · J = ${J.toFixed(3)} · purity vs the hidden true cover <b>${pct(pure / pts.length)}</b>` +
        `<br>feedback = none; the clusters have no names until an analyst gives them`;
    }

    /* ---------------------------------------------------------- 3 · self-supervised (masked bands) */
    const S = K.scene(), mkIn = $("input[name=mask]"), mkv = $(".mkv");
    const pool = []; { const r = rng(99); for (let t = 0; t < 3000; t++) pool.push(Math.floor(r() * S.n)); }
    const thumb = document.createElement("canvas"); thumb.width = S.N; thumb.height = S.N;
    { const tc = thumb.getContext("2d"), id = tc.createImageData(S.N, S.N);
      for (let i = 0; i < S.n; i++) { id.data[i * 4] = S.rgb8[i * 3]; id.data[i * 4 + 1] = S.rgb8[i * 3 + 1]; id.data[i * 4 + 2] = S.rgb8[i * 3 + 2]; id.data[i * 4 + 3] = 255; }
      tc.putImageData(id, 0, 0); }
    let px = 0, hidden = new Set(), pred = null, base = null, sseed = 3;
    const val = (i, b) => S.B[i * 13 + b];
    function mask() {
      const r = rng(sseed * 7919 + +mkIn.value), idx = [...Array(13).keys()];
      for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      hidden = new Set(idx.slice(0, +mkIn.value)); pred = base = null;
    }
    function newPixel() {
      const r = rng(sseed++ * 104729);
      // a pixel with some vegetation: its red-edge jump is what a naive guess cannot reproduce
      do { px = Math.floor(r() * S.n); } while (pool.includes(px) || !S.comp[px].some(([m, f]) => m === 1 && f >= 0.3));
      mask();
    }
    function predict() {
      const vis = [...Array(13).keys()].filter((b) => !hidden.has(b));
      const nn = pool.map((i) => { let s = 0; vis.forEach((b) => (s += (val(i, b) - val(px, b)) ** 2)); return [s, i]; })
        .sort((a, b) => a[0] - b[0]).slice(0, 5);
      pred = {}; base = {};
      hidden.forEach((b) => {
        pred[b] = nn.reduce((s, [, i]) => s + val(i, b), 0) / nn.length;
        let lo = -1, hi = -1;                                            // naive baseline: line between visible bands
        for (let v = b - 1; v >= 0; v--) if (!hidden.has(v)) { lo = v; break; }
        for (let v = b + 1; v < 13; v++) if (!hidden.has(v)) { hi = v; break; }
        base[b] = lo < 0 ? val(px, hi) : hi < 0 ? val(px, lo) : val(px, lo) + ((val(px, hi) - val(px, lo)) * (WL[b] - WL[lo])) / (WL[hi] - WL[lo]);
      });
    }
    function ssl() {
      mkv.textContent = mkIn.value;
      ctx.fillStyle = IVORY; ctx.fillRect(0, 0, W, H);
      // the unlabelled scene, with the chosen pixel
      const tx = 16, ty = 34, ts = 190;
      ctx.imageSmoothingEnabled = false; ctx.drawImage(thumb, tx, ty, ts, ts);
      ctx.strokeStyle = "rgba(20,33,61,.3)"; ctx.lineWidth = 1; ctx.strokeRect(tx, ty, ts, ts);
      const cx = tx + ((px % S.N) + 0.5) * ts / S.N, cy = ty + (Math.floor(px / S.N) + 0.5) * ts / S.N;
      ctx.strokeStyle = SAF; ctx.lineWidth = 3; ctx.strokeRect(cx - 7, cy - 7, 14, 14);
      ctx.fillStyle = INK; ctx.font = "600 13px Inter, Arial, sans-serif"; ctx.textAlign = "left";
      ctx.fillText("unlabelled scene", tx, 22);
      ctx.fillStyle = MUT; ctx.font = "12px Inter, Arial, sans-serif";
      wrap(S.describe(px), tx, ty + ts + 20, ts, 15);
      // spectrum plot
      const x0 = 250, x1 = 646, y0 = 26, y1 = 340, ymax = 0.55;
      const X = (b) => x0 + 12 + (b * (x1 - x0 - 24)) / 12, Y = (v) => y1 - (clamp(v, 0, ymax) / ymax) * (y1 - y0);
      ctx.strokeStyle = "rgba(20,33,61,.1)"; ctx.fillStyle = MUT; ctx.textAlign = "right";
      [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((v) => { ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), x0 - 4, Y(v) + 4); });
      hidden.forEach((b) => {
        ctx.fillStyle = "rgba(201,154,46,.16)"; ctx.fillRect(X(b) - 13, y0, 26, y1 - y0);
        if (!pred) { ctx.fillStyle = "#8A6512"; ctx.font = "700 18px Inter, Arial, sans-serif"; ctx.textAlign = "center"; ctx.fillText("?", X(b), (y0 + y1) / 2); }
      });
      ctx.font = "11px Inter, Arial, sans-serif"; ctx.textAlign = "center";
      BN.forEach((n, b) => { ctx.fillStyle = hidden.has(b) ? "#8A6512" : MUT; ctx.fillText(n, X(b), y1 + 16); });
      ctx.fillStyle = INK; ctx.font = "600 13px Inter, Arial, sans-serif"; ctx.textAlign = "left";
      ctx.fillText("reflectance", x0, 16);
      ctx.textAlign = "right"; ctx.fillText("Sentinel-2 band →", x1, y1 + 36);
      // true spectrum: full line once revealed, else only the visible part
      ctx.lineWidth = 2.2; ctx.strokeStyle = pred ? "rgba(20,33,61,.55)" : INK;
      if (pred) ctx.setLineDash([6, 5]);
      ctx.beginPath(); let started = false;
      for (let b = 0; b < 13; b++) {
        if (!pred && hidden.has(b)) { started = false; continue; }
        started ? ctx.lineTo(X(b), Y(val(px, b))) : ctx.moveTo(X(b), Y(val(px, b))); started = true;
      }
      ctx.stroke(); ctx.setLineDash([]);
      for (let b = 0; b < 13; b++) if (!hidden.has(b)) { ctx.beginPath(); ctx.arc(X(b), Y(val(px, b)), 5, 0, 7); ctx.fillStyle = INK; ctx.fill(); }
      if (pred) {
        hidden.forEach((b) => {
          ctx.beginPath(); ctx.arc(X(b), Y(base[b]), 6, 0, 7); ctx.lineWidth = 2.2; ctx.strokeStyle = "#8C94A6"; ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke();
          const x = X(b), y = Y(pred[b]);
          ctx.beginPath(); ctx.moveTo(x, y - 8); ctx.lineTo(x + 8, y); ctx.lineTo(x, y + 8); ctx.lineTo(x - 8, y); ctx.closePath();
          ctx.fillStyle = THEME.p; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
        });
        const rm = (f) => Math.sqrt([...hidden].reduce((s, b) => s + (f[b] - val(px, b)) ** 2, 0) / hidden.size);
        legend(["true spectrum (revealed)", "learned guess", "straight-line guess"], [[20, 33, 61], THEME.pRgb, [140, 148, 166]]);
        out.innerHTML = `hidden-band error (RMSE): learned from the unlabelled pool <b>${rm(pred).toFixed(3)}</b> · straight line ${rm(base).toFixed(3)}` +
          `<br>feedback = the hidden bands themselves · human labels used: <b>0</b>`;
      } else {
        legend(["visible bands", "hidden bands"], [[20, 33, 61], [201, 154, 46]]);
        out.innerHTML = `pretext task: predict the ${hidden.size} hidden bands from the other ${13 - hidden.size}, using ${pool.length.toLocaleString("en")} unlabelled pixels` +
          `<br>press <b>Predict hidden bands</b> to reveal the truth`;
      }
    }
    function wrap(text, x, y, w, lh) {
      let line = "";
      text.split(" ").forEach((word) => {
        const t = line ? line + " " + word : word;
        if (ctx.measureText(t).width > w && line) { ctx.fillText(line, x, y); y += lh; line = word; } else line = t;
      });
      if (line) ctx.fillText(line, x, y);
    }

    /* ---------------------------------------------------------- 4 · reinforcement (Q-learning) */
    const GW = 11, GH = 7, CS = 56, OX = (W - GW * CS) / 2, OY = (H - GH * CS) / 2;
    const START = [0, 6], GOAL = [10, 1];
    const STORM = new Set(["3,2", "3,3", "3,4", "3,5", "6,0", "6,1", "6,2", "6,3", "7,5", "8,5", "8,6", "9,2", "9,3"]);
    const ACT = [[1, 0], [0, 1], [-1, 0], [0, -1]];                      // E S W N
    const ALPHA = 0.5, GAMMA = 0.95, EPS = 0.15;
    let Q, eps, path, rets, lastRet, rr = rng(5);
    const sid = (x, y) => y * GW + x;
    function rlReset() { Q = new Float64Array(GW * GH * 4); eps = 0; path = null; rets = []; lastRet = null; rr = rng(5); }
    function greedy(s, r) {
      const q = Q.subarray(s * 4, s * 4 + 4), m = Math.max(...q), best = [0, 1, 2, 3].filter((a) => q[a] === m);
      return best[Math.floor(r() * best.length)];
    }
    function episode() {
      let [x, y] = START, G = 0, end = "";
      const p = [[x, y]];
      for (let t = 0; t < 120; t++) {
        const s = sid(x, y), a = rr() < EPS ? Math.floor(rr() * 4) : greedy(s, rr);
        const nx = clamp(x + ACT[a][0], 0, GW - 1), ny = clamp(y + ACT[a][1], 0, GH - 1);
        let rew = -1, done = false;
        if (STORM.has(nx + "," + ny)) { rew = -10; done = true; end = "lost in a storm"; }
        else if (nx === GOAL[0] && ny === GOAL[1]) { rew = 20; done = true; end = "reached the village"; }
        const s2 = sid(nx, ny), target = rew + (done ? 0 : GAMMA * Math.max(...Q.subarray(s2 * 4, s2 * 4 + 4)));
        Q[s * 4 + a] += ALPHA * (target - Q[s * 4 + a]);
        G += rew; x = nx; y = ny; p.push([x, y]);
        if (done) break;
      }
      eps++; path = p; lastRet = { G, end: end || "ran out of battery" };
      rets.push(end === "reached the village" ? 1 : 0); if (rets.length > 50) rets.shift();
    }
    function greedyRoute() {
      let [x, y] = START; const seen = new Set();
      for (let t = 0; t < 60; t++) {
        if (x === GOAL[0] && y === GOAL[1]) return t;
        const s = sid(x, y); if (seen.has(s) || STORM.has(x + "," + y)) return -1; seen.add(s);
        const q = Q.subarray(s * 4, s * 4 + 4); if (Math.max(...q) === 0 && Math.min(...q) === 0) return -1;
        const a = argmax(q); x = clamp(x + ACT[a][0], 0, GW - 1); y = clamp(y + ACT[a][1], 0, GH - 1);
      }
      return -1;
    }
    function rl() {
      ctx.fillStyle = IVORY; ctx.fillRect(0, 0, W, H);
      let vmin = Infinity, vmax = -Infinity;
      for (let s = 0; s < GW * GH; s++) { const v = Math.max(...Q.subarray(s * 4, s * 4 + 4)); if (v !== 0) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); } }
      for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
        const px0 = OX + x * CS, py0 = OY + y * CS, s = sid(x, y), q = Q.subarray(s * 4, s * 4 + 4), v = Math.max(...q);
        const storm = STORM.has(x + "," + y), goal = x === GOAL[0] && y === GOAL[1];
        ctx.fillStyle = storm ? "#3A4256" : goal ? SAF : v !== 0 && vmax > vmin ? rgba(THEME.sRgb, 0.08 + 0.55 * (v - vmin) / (vmax - vmin)) : "#fff";
        ctx.fillRect(px0, py0, CS, CS);
        ctx.strokeStyle = "rgba(20,33,61,.14)"; ctx.lineWidth = 1; ctx.strokeRect(px0 + 0.5, py0 + 0.5, CS - 1, CS - 1);
        ctx.textAlign = "center"; ctx.font = "600 11px Inter, Arial, sans-serif";
        if (storm) { ctx.fillStyle = "#C9D2F3"; ctx.fillText("storm", px0 + CS / 2, py0 + CS / 2 + 4); continue; }
        if (goal) { ctx.fillStyle = INK; ctx.fillText("flooded", px0 + CS / 2, py0 + CS / 2 - 3); ctx.fillText("village", px0 + CS / 2, py0 + CS / 2 + 11); continue; }
        if (v !== 0 || Math.min(...q) !== 0) {                         // greedy-policy arrow
          const a = argmax(q), cx = px0 + CS / 2, cy = py0 + CS / 2, dx = ACT[a][0] * 14, dy = ACT[a][1] * 14;
          ctx.strokeStyle = INK; ctx.fillStyle = INK; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(cx - dx, cy - dy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(cx + dx + ACT[a][0] * 4, cy + dy + ACT[a][1] * 4);
          ctx.lineTo(cx + dx - ACT[a][1] * 5 - ACT[a][0] * 3, cy + dy + ACT[a][0] * 5 - ACT[a][1] * 3);
          ctx.lineTo(cx + dx + ACT[a][1] * 5 - ACT[a][0] * 3, cy + dy - ACT[a][0] * 5 - ACT[a][1] * 3); ctx.fill();
        }
      }
      const [sx, sy] = START;
      ctx.fillStyle = INK; ctx.font = "700 11px Inter, Arial, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("BASE", OX + sx * CS + CS / 2, OY + sy * CS + CS - 6);
      if (path) {
        ctx.strokeStyle = THEME.p; ctx.lineWidth = 3.5; ctx.lineJoin = "round"; ctx.globalAlpha = 0.85;
        ctx.beginPath(); path.forEach(([x, y], i) => { const X = OX + x * CS + CS / 2, Y = OY + y * CS + CS / 2; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.stroke();
        ctx.globalAlpha = 1;
        const [lx, ly] = path[path.length - 1];
        ctx.beginPath(); ctx.arc(OX + lx * CS + CS / 2, OY + ly * CS + CS / 2, 7, 0, 7); ctx.fillStyle = THEME.p; ctx.fill();
      }
      legend([], []);
      const route = greedyRoute(), rate = rets.length ? rets.reduce((a, b) => a + b, 0) / rets.length : 0;
      out.innerHTML = eps === 0
        ? `untrained drone: Q(s, a) = 0 everywhere · press <b>1 episode</b> to let it fly (ε-greedy, ε = ${EPS})<br>feedback = a reward after each move, never the right route`
        : `episode ${eps} · last flight: ${lastRet.end}, return ${lastRet.G} · success over last ${rets.length}: <b>${pct(rate)}</b> · greedy route: ` +
          `${route > 0 ? `<b>${route} moves</b>, safe` : "not found yet"}<br>feedback = a reward after each move, never the right route`;
    }

    /* ---------------------------------------------------------- legend drawn on the canvas */
    function legend(names, cols) {
      if (!names.length) return;
      ctx.font = "600 12px Inter, Arial, sans-serif"; ctx.textAlign = "left";
      const w = Math.max(...names.map((n) => ctx.measureText(n).width)) + 34, h = names.length * 19 + 10;
      const lx = mode === "ssl" ? 16 : W - w - 10, ly = mode === "ssl" ? 296 : 10;   // ssl: under the scene thumbnail
      ctx.fillStyle = "rgba(255,255,255,.88)"; ctx.fillRect(lx, ly, w, h);
      ctx.strokeStyle = "rgba(20,33,61,.15)"; ctx.strokeRect(lx + 0.5, ly + 0.5, w - 1, h - 1);
      names.forEach((n, i) => {
        ctx.fillStyle = rgba(cols[i]); ctx.fillRect(lx + 8, ly + 8 + i * 19, 14, 12);
        ctx.fillStyle = INK; ctx.fillText(n, lx + 28, ly + 18 + i * 19);
      });
    }

    /* ---------------------------------------------------------- wiring */
    const DRAW = { sup, unsup, ssl, rl };
    function setMode(m) {
      mode = m;
      pick(root, "data-mode", m);
      root.querySelectorAll("[data-for]").forEach((g) => (g.style.display = g.dataset.for === m ? "contents" : "none"));
      slide.querySelectorAll("[data-panel]").forEach((p) => (p.style.display = p.dataset.panel === m ? "" : "none"));
      DRAW[m]();
    }
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
    mIn.oninput = sup;
    kIn.oninput = unsup;
    $("[data-act=reinit]").onclick = () => { useed++; unsup(); };
    mkIn.oninput = () => { mask(); ssl(); };
    $("[data-act=pixel]").onclick = () => { newPixel(); ssl(); };
    $("[data-act=predict]").onclick = () => { predict(); ssl(); };
    $("[data-act=ep]").onclick = () => { episode(); rl(); };
    $("[data-act=ep100]").onclick = () => { for (let i = 0; i < 100; i++) episode(); rl(); };
    $("[data-act=rlreset]").onclick = () => { rlReset(); rl(); };
    rlReset(); newPixel();
    setMode("sup");
  };
})();
