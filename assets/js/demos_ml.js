/* Deep Learning for Earth Observation — classical machine-learning demos (no dependencies)
   Naive Bayes / Gaussian maximum-likelihood classifier · decision tree · bagging & random forest ·
   AdaBoost · gradient boosting · support vector machine.
   Each demo registers in window.DECK_DEMOS; deck.js initialises it when its slide is first shown. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});

  /* ================================================================ shared kit */
  const Kit = (window.DemoKit = window.DemoKit || {});
  Kit.rng = function (seed) {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  };
  Kit.gauss = function (r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  Kit.clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  Kit.pick = function (root, attr, val) {
    root.querySelectorAll(`[${attr}]`).forEach((b) => b.classList.toggle("gold", b.getAttribute(attr) === String(val)));
  };
  Kit.argmax = (a) => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
  const { rng, gauss, clamp, pick, argmax } = Kit;

  const INK = "#14213D", MUT = "#5B6478", SAF = "#C99A2E";
  const COL = ["#2E4BC6", "#0F6B5C", "#B5502E"];
  const RGB = [[46, 75, 198], [15, 107, 92], [181, 80, 46]];
  const NAMES = ["water", "vegetation", "soil / built-up"];
  const RED_MAX = 0.4, NIR_MAX = 0.6;           // axis ranges of the (Red, NIR) feature space
  const IVORY = [253, 251, 247];
  const pct = (v) => (100 * v).toFixed(1) + " %";

  /* 2-D plot on a canvas: data space [0,1]² */
  function Plot(cv) {
    const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    const X = (x) => x * W, Y = (y) => (1 - y) * H;
    return {
      ctx, W, H, X, Y,
      at(e) { const b = cv.getBoundingClientRect(); return { x: clamp((e.clientX - b.left) / b.width, 0, 1), y: clamp(1 - (e.clientY - b.top) / b.height, 0, 1) }; },
      /* paint a gw×gh field; fn(i, j, x, y) -> [r, g, b, alpha] blended over ivory */
      field(gw, gh, fn, smooth) {
        const id = new ImageData(gw, gh), d = id.data;
        for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
          const c = fn(i, j, (i + 0.5) / gw, 1 - (j + 0.5) / gh), a = c[3], k = (j * gw + i) * 4;
          d[k] = IVORY[0] + (c[0] - IVORY[0]) * a; d[k + 1] = IVORY[1] + (c[1] - IVORY[1]) * a; d[k + 2] = IVORY[2] + (c[2] - IVORY[2]) * a; d[k + 3] = 255;
        }
        const oc = document.createElement("canvas"); oc.width = gw; oc.height = gh; oc.getContext("2d").putImageData(id, 0, 0);
        ctx.save(); ctx.imageSmoothingEnabled = !!smooth; ctx.drawImage(oc, 0, 0, W, H); ctx.restore();
      },
      frame(xl, yl, ticks) {
        ctx.save();
        ctx.strokeStyle = "rgba(20,33,61,.09)"; ctx.lineWidth = 1;
        for (let k = 1; k < 4; k++) {
          ctx.beginPath(); ctx.moveTo(X(k / 4), 0); ctx.lineTo(X(k / 4), H); ctx.moveTo(0, Y(k / 4)); ctx.lineTo(W, Y(k / 4)); ctx.stroke();
        }
        ctx.fillStyle = MUT; ctx.font = "12px Inter, Arial, sans-serif";
        if (ticks) {
          ctx.textAlign = "center"; for (let k = 1; k < 4; k++) ctx.fillText(ticks[0](k / 4), X(k / 4), H - 6);
          ctx.textAlign = "left"; for (let k = 1; k < 4; k++) ctx.fillText(ticks[1](k / 4), 30, Y(k / 4) - 4);
        }
        ctx.font = "600 14px Inter, Arial, sans-serif"; ctx.fillStyle = INK;
        ctx.textAlign = "right"; ctx.fillText(xl, W - 10, H - 22);
        ctx.translate(20, 12); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right"; ctx.fillText(yl, 0, 0);
        ctx.restore();
      },
      points(pts, colorOf, o = {}) {
        pts.forEach((p, i) => {
          const r = o.r ? o.r(p, i) : 6;
          if (o.bad && o.bad(p, i)) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r + 3.5, 0, 7); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke(); }
          ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r, 0, 7); ctx.fillStyle = colorOf(p); ctx.fill();
          ctx.lineWidth = 1.6; ctx.strokeStyle = "#fff"; ctx.stroke();
        });
      },
      hollow(pts, colorOf) {
        pts.forEach((p) => { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 5, 0, 7); ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.fill(); ctx.lineWidth = 2.4; ctx.strokeStyle = colorOf(p); ctx.stroke(); });
      },
    };
  }
  const SPEC_TICKS = [(v) => (v * RED_MAX).toFixed(2), (v) => (v * NIR_MAX).toFixed(2)];

  /* ================================================================ datasets */
  /* three land-cover classes in (Red, NIR): compact water, vegetation along the vigour gradient,
     soil / built-up along the "soil line" (both strongly correlated, as in real data) */
  function spectral3(seed, nPer, labelNoise) {
    const r = rng(seed), pts = [];
    for (let i = 0; i < nPer; i++) {
      pts.push({ x: 0.11 + 0.035 * gauss(r), y: 0.09 + 0.03 * gauss(r), c: 0 });
      let t = r(); pts.push({ x: 0.31 - 0.22 * t + 0.035 * gauss(r), y: 0.36 + 0.42 * t + 0.035 * gauss(r), c: 1 });
      t = r(); pts.push({ x: 0.33 + 0.40 * t + 0.03 * gauss(r), y: 0.27 + 0.34 * t + 0.03 * gauss(r), c: 2 });
    }
    pts.forEach((p) => {
      p.x = clamp(p.x, 0.01, 0.99); p.y = clamp(p.y, 0.01, 0.99);
      if (labelNoise && r() < labelNoise) p.c = (p.c + 1 + Math.floor(r() * 2)) % 3;
    });
    return pts;
  }
  /* binary problems: c = 1 (positive) / 0 (negative) */
  const BIN = {
    linear: { axes: ["Red (B4) →", "NIR (B8) →"], ticks: SPEC_TICKS, neg: 0, names: ["water", "vegetation"] },
    xor: { axes: ["NDVI in March →", "NDVI in July →"], ticks: [(v) => v.toFixed(2), (v) => v.toFixed(2)], neg: 2, names: ["bare / orchard", "single-season crop"] },
    ring: { axes: ["feature x₁ →", "feature x₂ →"], ticks: null, neg: 2, names: ["outside", "inside"] },
  };
  function binary(kind, seed, n) {
    const r = rng(seed), pts = [];
    if (kind === "linear") {
      for (let i = 0; i < n; i++) {
        pts.push({ x: 0.17 + 0.06 * gauss(r), y: 0.2 + 0.07 * gauss(r), c: 0 });
        const t = r(); pts.push({ x: 0.32 - 0.2 * t + 0.05 * gauss(r), y: 0.47 + 0.35 * t + 0.05 * gauss(r), c: 1 });
      }
    } else if (kind === "xor") {
      // rain-fed cereal (green in March), summer crop (green in July)  vs  bare land, evergreen orchard
      const C = [[0.76, 0.22, 1], [0.24, 0.72, 1], [0.22, 0.2, 0], [0.74, 0.74, 0]];
      for (let i = 0; i < n / 2; i++) C.forEach(([a, b, c]) => pts.push({ x: a + 0.075 * gauss(r), y: b + 0.075 * gauss(r), c }));
    } else {
      for (let i = 0; i < n; i++) {
        let a = 2 * Math.PI * r(), d = 0.19 * Math.sqrt(r());
        pts.push({ x: 0.5 + d * Math.cos(a), y: 0.5 + d * Math.sin(a), c: 1 });
        a = 2 * Math.PI * r(); d = 0.28 + 0.15 * r();
        pts.push({ x: 0.5 + d * Math.cos(a), y: 0.5 + d * Math.sin(a), c: 0 });
      }
    }
    pts.forEach((p) => { p.x = clamp(p.x, 0.01, 0.99); p.y = clamp(p.y, 0.01, 0.99); });
    return pts;
  }
  const binCol = (kind) => (p) => COL[p.c ? 1 : BIN[kind].neg];
  const binRGB = (kind, c) => RGB[c ? 1 : BIN[kind].neg];

  /* ================================================================ CART trees */
  const IMP = {
    gini(c, t) { if (t <= 0) return 0; let s = 1; for (const v of c) s -= (v / t) ** 2; return s; },
    entropy(c, t) { if (t <= 0) return 0; let s = 0; for (const v of c) if (v > 0) { const p = v / t; s -= p * Math.log2(p); } return s; },
  };
  /* o: {K, maxDepth, minLeaf, crit, mtry (1 = one random feature per split), r} ; w: sample weights or null */
  function growTree(P, idx, w, o, depth) {
    const K = o.K, cnt = new Float64Array(K);
    let tot = 0;
    for (const i of idx) { const wi = w ? w[i] : 1; cnt[P[i].c] += wi; tot += wi; }
    const node = { n: idx.length, dist: Array.from(cnt, (v) => v / (tot || 1)), depth };
    const imp0 = IMP[o.crit](cnt, tot);
    if (depth >= o.maxDepth || idx.length < 2 * o.minLeaf || imp0 <= 1e-12) return node;
    let feats = [0, 1];
    if (o.mtry === 1 && o.r) feats = o.r() < 0.5 ? [0, 1] : [1, 0];
    let best = null;
    for (let fi = 0; fi < feats.length; fi++) {
      const f = feats[fi], key = f ? "y" : "x";
      const s = idx.slice().sort((a, b) => P[a][key] - P[b][key]);
      const L = new Float64Array(K), R = new Float64Array(K);
      let lt = 0;
      for (let k = 0; k < s.length - 1; k++) {
        const i = s[k], wi = w ? w[i] : 1;
        L[P[i].c] += wi; lt += wi;
        const v = P[i][key], v2 = P[s[k + 1]][key];
        if (v2 - v < 1e-12 || k + 1 < o.minLeaf || s.length - k - 1 < o.minLeaf) continue;
        for (let j = 0; j < K; j++) R[j] = cnt[j] - L[j];
        const rt = tot - lt;
        const gain = imp0 - (lt / tot) * IMP[o.crit](L, lt) - (rt / tot) * IMP[o.crit](R, rt);
        if (!best || gain > best.gain + 1e-12) best = { gain, f, t: (v + v2) / 2, k, s };
      }
      // random forest: only look at the second feature if the first gave no useful split
      if (o.mtry === 1 && best && best.gain > 1e-12) break;
    }
    if (!best || best.gain <= 1e-12) return node;
    node.f = best.f; node.t = best.t;
    node.l = growTree(P, best.s.slice(0, best.k + 1), w, o, depth + 1);
    node.r = growTree(P, best.s.slice(best.k + 1), w, o, depth + 1);
    return node;
  }
  function leaf(n, x, y) { while (n.l) n = (n.f ? y : x) <= n.t ? n.l : n.r; return n; }
  function treeStats(n) {
    if (!n.l) return { leaves: 1, depth: n.depth };
    const a = treeStats(n.l), b = treeStats(n.r);
    return { leaves: a.leaves + b.leaves, depth: Math.max(a.depth, b.depth) };
  }
  function drawSplits(P, n, x0, x1, y0, y1) {
    if (!n.l) return;
    const c = P.ctx; c.beginPath();
    if (n.f === 0) { c.moveTo(P.X(n.t), P.Y(y0)); c.lineTo(P.X(n.t), P.Y(y1)); c.stroke(); drawSplits(P, n.l, x0, n.t, y0, y1); drawSplits(P, n.r, n.t, x1, y0, y1); }
    else { c.moveTo(P.X(x0), P.Y(n.t)); c.lineTo(P.X(x1), P.Y(n.t)); c.stroke(); drawSplits(P, n.l, x0, x1, y0, n.t); drawSplits(P, n.r, x0, x1, n.t, y1); }
  }

  /* ======================================================= 1 · naive Bayes / MLC */
  DEMOS.nb = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    let pts, full = false, addC = 1, G = null, hover = null;
    const reset = () => { pts = spectral3(5, 26, 0); };
    function fit() {
      G = [0, 1, 2].map((k) => {
        const S = pts.filter((p) => p.c === k), n = S.length;
        if (n < 2) return null;
        let mx = 0, my = 0; S.forEach((p) => { mx += p.x; my += p.y; }); mx /= n; my /= n;
        let sxx = 0, syy = 0, sxy = 0;
        S.forEach((p) => { sxx += (p.x - mx) ** 2; syy += (p.y - my) ** 2; sxy += (p.x - mx) * (p.y - my); });
        return { mx, my, sxx: sxx / n + 2e-4, syy: syy / n + 2e-4, sxy: full ? sxy / n : 0, prior: n / pts.length };
      });
    }
    function logp(g, x, y) {
      const dx = x - g.mx, dy = y - g.my, det = g.sxx * g.syy - g.sxy * g.sxy;
      return Math.log(g.prior) - 0.5 * (g.syy * dx * dx - 2 * g.sxy * dx * dy + g.sxx * dy * dy) / det - 0.5 * Math.log(det);
    }
    function post(x, y) {
      const l = G.map((g) => (g ? logp(g, x, y) : -Infinity)), m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), Z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / Z);
    }
    function ellipse(g, s) {
      const tr = g.sxx + g.syy, det = g.sxx * g.syy - g.sxy * g.sxy, d = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
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
    function readout() {
      const acc = pts.filter((p) => argmax(post(p.x, p.y)) === p.c).length / pts.length;
      let s = `${full ? "Gaussian maximum-likelihood · full Σ" : "Gaussian naive Bayes · diagonal Σ"} · training accuracy ${pct(acc)}`;
      if (hover) { const p = post(hover.x, hover.y); s += ` · P(class | pixel) = ${p.map((v, k) => `${NAMES[k].split(" ")[0]} ${v.toFixed(2)}`).join(", ")}`; }
      else s += " · hover: P(class | pixel)";
      out.textContent = s;
    }
    function draw() {
      fit();
      P.field(165, 105, (i, j, x, y) => { const p = post(x, y), k = argmax(p); return [...RGB[k], 0.08 + 0.36 * (p[k] - 1 / 3) * 1.5]; }, true);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", SPEC_TICKS);
      G.forEach((g, k) => {
        if (!g) return;
        ctx.strokeStyle = COL[k]; ctx.lineWidth = 2.5; ctx.setLineDash([]); ellipse(g, 1);
        ctx.lineWidth = 1.5; ctx.setLineDash([6, 5]); ellipse(g, 2); ctx.setLineDash([]);
        ctx.strokeStyle = INK; ctx.lineWidth = 3; const x = P.X(g.mx), y = P.Y(g.my);
        ctx.beginPath(); ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y); ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8); ctx.stroke();
      });
      P.points(pts, (p) => COL[p.c], { bad: (p) => argmax(post(p.x, p.y)) !== p.c, r: () => 5 });
      readout();
    }
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { full = b.dataset.mode === "full"; pick(root, "data-mode", b.dataset.mode); draw(); }));
    root.querySelectorAll("[data-c]").forEach((b) => (b.onclick = () => { addC = +b.dataset.c; pick(root, "data-c", addC); }));
    root.querySelector("[data-act=reset]").onclick = () => { reset(); draw(); };
    cv.addEventListener("mousedown", (e) => { const p = P.at(e); pts.push({ x: p.x, y: p.y, c: addC }); draw(); });
    cv.addEventListener("mousemove", (e) => { hover = P.at(e); readout(); });
    cv.addEventListener("mouseleave", () => { hover = null; readout(); });
    pick(root, "data-mode", "naive"); pick(root, "data-c", addC);
    reset(); draw();
  };

  /* ======================================================= 2 · decision tree */
  DEMOS.tree = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const rulesEl = root.closest(".slide").querySelector(".rules");
    const dIn = root.querySelector("input[name=depth]"), dv = root.querySelector(".dv");
    const crit = root.querySelector("select[name=crit]"), testCb = root.querySelector("input[name=test]");
    let seed = 3, train, test;
    const data = () => { train = spectral3(seed, 30, 0.07); test = spectral3(seed + 1000, 60, 0.07); };
    const cond = (n) => (n.f ? `NIR ≤ ${(n.t * NIR_MAX).toFixed(3)}` : `Red ≤ ${(n.t * RED_MAX).toFixed(3)}`);
    function rules(root) {
      const L = [];
      (function rec(n, ind) {
        if (!n.l) { L.push(`${ind}→ ${NAMES[argmax(n.dist)]}   [${n.n} px]`); return; }
        if (n.depth >= 2) { L.push(`${ind}→ … deeper subtree   [${n.n} px]`); return; }
        L.push(`${ind}if ${cond(n)}:`); rec(n.l, ind + "    "); L.push(`${ind}else:`); rec(n.r, ind + "    ");
      })(root, "");
      return L.join("\n");
    }
    function draw() {
      const D = +dIn.value; dv.textContent = D;
      const tree = growTree(train, train.map((_, i) => i), null, { K: 3, maxDepth: D, minLeaf: 1, crit: crit.value, mtry: 2 }, 0);
      P.field(165, 105, (i, j, x, y) => { const n = leaf(tree, x, y), k = argmax(n.dist); return [...RGB[k], 0.14 + 0.24 * n.dist[k]]; }, false);
      ctx.strokeStyle = "rgba(20,33,61,.45)"; ctx.lineWidth = 1.2; drawSplits(P, tree, 0, 1, 0, 1);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", SPEC_TICKS);
      const pred = (p) => argmax(leaf(tree, p.x, p.y).dist);
      if (testCb.checked) P.hollow(test, (p) => COL[p.c]);
      P.points(train, (p) => COL[p.c], { bad: (p) => pred(p) !== p.c, r: () => 5 });
      const acc = (S) => S.filter((p) => pred(p) === p.c).length / S.length, st = treeStats(tree);
      out.textContent = `max depth ${D} · ${st.leaves} leaves · training accuracy ${pct(acc(train))} · test accuracy ${pct(acc(test))}`;
      if (rulesEl) rulesEl.textContent = rules(tree);
    }
    dIn.oninput = draw; crit.onchange = draw; testCb.onchange = draw;
    root.querySelector("[data-act=new]").onclick = () => { seed += 17; data(); draw(); };
    data(); draw();
  };

  /* ======================================================= 3 · bagging & random forest */
  DEMOS.forest = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), out = root.querySelector(".readout");
    const bIn = root.querySelector("input[name=B]"), bv = root.querySelector(".bv");
    const GW = 132, GH = 84, NT = 100;
    let seed = 8, mode = "rf", train, test, trees = [], oob = [], gridPred = [], testPred = [], trainPred = [];
    const data = () => { train = spectral3(seed, 30, 0.08); test = spectral3(seed + 500, 60, 0.08); };
    function fitAll() {
      const r = rng(seed * 31 + (mode === "rf" ? 7 : 3)), n = train.length;
      trees = []; oob = []; gridPred = []; testPred = []; trainPred = [];
      const T = mode === "tree" ? 1 : NT;
      for (let b = 0; b < T; b++) {
        let idx, mask = new Uint8Array(n).fill(1);
        if (mode === "tree") { idx = train.map((_, i) => i); mask.fill(0); }
        else { idx = []; for (let i = 0; i < n; i++) { const k = Math.floor(r() * n); idx.push(k); mask[k] = 0; } }
        const t = growTree(train, idx, null, { K: 3, maxDepth: 40, minLeaf: 1, crit: "gini", mtry: mode === "rf" ? 1 : 2, r }, 0);
        trees.push(t); oob.push(mask);
        const g = new Uint8Array(GW * GH);
        for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) g[j * GW + i] = argmax(leaf(t, (i + 0.5) / GW, 1 - (j + 0.5) / GH).dist);
        gridPred.push(g);
        testPred.push(test.map((p) => argmax(leaf(t, p.x, p.y).dist)));
        trainPred.push(train.map((p) => argmax(leaf(t, p.x, p.y).dist)));
      }
    }
    function draw() {
      const B = mode === "tree" ? 1 : +bIn.value;
      bv.textContent = mode === "tree" ? "—" : B;
      bIn.disabled = mode === "tree";
      const v = new Uint16Array(3);
      P.field(GW, GH, (i, j) => {
        v.fill(0); for (let b = 0; b < B; b++) v[gridPred[b][j * GW + i]]++;
        const k = argmax(v); return [...RGB[k], 0.08 + 0.36 * (v[k] / B)];
      }, false);
      P.frame("Red (B4) reflectance →", "NIR (B8) reflectance →", SPEC_TICKS);
      const vote = (preds, i, filt) => { v.fill(0); let m = 0; for (let b = 0; b < B; b++) if (!filt || oob[b][i]) { v[preds[b][i]]++; m++; } return m ? argmax(v) : -1; };
      const ens = train.map((_, i) => vote(trainPred, i));
      P.points(train, (p) => COL[p.c], { bad: (p, i) => ens[i] !== p.c, r: () => 5 });
      const testAcc = test.filter((p, i) => vote(testPred, i) === p.c).length / test.length;
      let oobTxt = "OOB —";
      if (mode !== "tree") {
        let hit = 0, m = 0;
        train.forEach((p, i) => { const k = vote(trainPred, i, true); if (k >= 0) { m++; hit += k === p.c; } });
        oobTxt = m ? `OOB accuracy ${pct(hit / m)}` : "OOB —";
      }
      const label = { tree: "one fully grown tree", bag: `bagging · B = ${B} trees`, rf: `random forest · B = ${B} trees, 1 random band per split` }[mode];
      out.textContent = `${label} · ${oobTxt} · test accuracy ${pct(testAcc)}`;
    }
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { mode = b.dataset.mode; pick(root, "data-mode", mode); fitAll(); draw(); }));
    bIn.oninput = draw;
    root.querySelector("[data-act=new]").onclick = () => { seed += 13; data(); fitAll(); draw(); };
    pick(root, "data-mode", mode); data(); fitAll(); draw();
  };

  /* ======================================================= 4 · AdaBoost */
  DEMOS.adaboost = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const GW = 132, GH = 84;
    let kind = "ring", depth = 1, pts, w, learners, Fg, Fp, last = null, timer = null, note = "";
    function reset() {
      clearInterval(timer);
      pts = binary(kind, kind === "xor" ? 21 : 4, kind === "xor" ? 36 : 60);
      w = new Float64Array(pts.length).fill(1 / pts.length);
      learners = []; Fg = new Float32Array(GW * GH); Fp = new Float64Array(pts.length); last = null; note = "";
    }
    const h = (t, x, y) => (leaf(t, x, y).dist[1] >= 0.5 ? 1 : -1);
    function step() {
      const tree = growTree(pts, pts.map((_, i) => i), w, { K: 2, maxDepth: depth, minLeaf: 1, crit: "gini", mtry: 2 }, 0);
      const hp = pts.map((p) => h(tree, p.x, p.y));
      let eps = 0; pts.forEach((p, i) => { if (hp[i] !== (p.c ? 1 : -1)) eps += w[i]; });
      if (eps >= 0.5 - 1e-9) { note = " · weak learner no better than chance — stopped"; return false; }
      eps = Math.max(eps, 1e-6);
      const alpha = 0.5 * Math.log((1 - eps) / eps);
      let Z = 0;
      pts.forEach((p, i) => { w[i] *= Math.exp(-alpha * (p.c ? 1 : -1) * hp[i]); Z += w[i]; Fp[i] += alpha * hp[i]; });
      for (let i = 0; i < w.length; i++) w[i] /= Z;
      for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) Fg[j * GW + i] += alpha * h(tree, (i + 0.5) / GW, 1 - (j + 0.5) / GH);
      learners.push({ tree, alpha, eps }); last = { tree, alpha, eps };
      return true;
    }
    function draw() {
      const A = learners.reduce((s, l) => s + l.alpha, 0) || 1;
      P.field(GW, GH, (i, j) => { const f = Fg[j * GW + i] / A; return learners.length ? [...binRGB(kind, f > 0 ? 1 : 0), 0.1 + 0.3 * Math.min(1, Math.abs(f) * 1.5)] : [255, 255, 255, 0]; }, false);
      if (last) { ctx.save(); ctx.strokeStyle = SAF; ctx.lineWidth = 3.5; ctx.setLineDash([10, 7]); drawSplits(P, last.tree, 0, 1, 0, 1); ctx.restore(); }
      P.frame(BIN[kind].axes[0], BIN[kind].axes[1], BIN[kind].ticks);
      const n = pts.length;
      P.points(pts, binCol(kind), { r: (p, i) => clamp(2.5 + 4.2 * Math.sqrt(w[i] * n), 2.5, 17), bad: (p, i) => learners.length && (Fp[i] > 0 ? 1 : 0) !== p.c });
      const err = learners.length ? pts.filter((p, i) => (Fp[i] > 0 ? 1 : 0) !== p.c).length : n;
      out.textContent = learners.length
        ? `T = ${learners.length} ${depth === 1 ? "stumps" : "depth-2 trees"} · last: ε = ${last.eps.toFixed(3)}, α = ${last.alpha.toFixed(2)} · training errors ${err}/${n}${note}`
        : `T = 0 · all ${n} weights equal (1/n) — press “+1 learner”`;
    }
    root.querySelector("[data-act=step]").onclick = () => { clearInterval(timer); step(); draw(); };
    root.querySelector("[data-act=run]").onclick = () => {
      clearInterval(timer); let k = 0;
      timer = setInterval(() => { if (!step() || ++k >= 25) clearInterval(timer); draw(); }, 90);
    };
    root.querySelector("[data-act=reset]").onclick = () => { reset(); draw(); };
    root.querySelectorAll("[data-ds]").forEach((b) => (b.onclick = () => { kind = b.dataset.ds; pick(root, "data-ds", kind); reset(); for (let i = 0; i < 3; i++) step(); draw(); }));
    root.querySelectorAll("[data-wl]").forEach((b) => (b.onclick = () => { depth = +b.dataset.wl; pick(root, "data-wl", depth); reset(); for (let i = 0; i < 3; i++) step(); draw(); }));
    pick(root, "data-ds", kind); pick(root, "data-wl", depth);
    reset(); for (let i = 0; i < 15; i++) step(); draw();
  };

  /* ======================================================= 5 · gradient boosting (1-D NDVI profile) */
  DEMOS.gboost = function (root) {
    const cv = root.querySelector("canvas"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height, out = root.querySelector(".readout");
    const nuIn = root.querySelector("input[name=nu]"), nuV = root.querySelector(".nv");
    const dSel = root.querySelector("select[name=depth]"), lSel = root.querySelector("select[name=loss]");
    // synthetic NDVI of a rain-fed cereal parcel, days since 1 September; a few cloud-contaminated dips
    const r = rng(77), xs = [], ys = [];
    const ndvi = (d) => 0.16 + 0.6 * (1 / (1 + Math.exp(-(d - 118) / 13)) - 1 / (1 + Math.exp(-(d - 236) / 14)));
    for (let d = 2; d <= 362; d += 5) { xs.push(d); ys.push(ndvi(d) + 0.025 * gauss(r)); }
    [9, 15, 29, 34, 40, 58].forEach((i) => (ys[i] = 0.06 + 0.07 * r()));
    const n = xs.length, MINL = 3, GRID = 366;
    let F, Fg, m, lastH;
    const L = 48, R = 12, T1 = 14, B1 = 286, T2 = 312, B2 = H - 30;
    const X = (d) => L + (d / 365) * (W - L - R), Y1 = (v) => B1 - (v / 0.9) * (B1 - T1), Y2 = (v) => (T2 + B2) / 2 - (v / 0.9) * (B2 - T2);
    const median = (a) => { const s = a.slice().sort((p, q) => p - q), k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
    function fitTree(res, a, b, d) {
      let s = 0, s2 = 0; for (let i = a; i < b; i++) { s += res[i]; s2 += res[i] * res[i]; }
      const node = { a, b, v: s / (b - a) };
      if (d === 0 || b - a < 2 * MINL) return node;
      let best = null, ls = 0, ls2 = 0;
      for (let k = a; k < b - 1; k++) {
        ls += res[k]; ls2 += res[k] * res[k];
        const nl = k - a + 1, nr = b - a - nl;
        if (nl < MINL || nr < MINL) continue;
        const sse = ls2 - (ls * ls) / nl + (s2 - ls2) - ((s - ls) * (s - ls)) / nr;
        if (!best || sse < best.sse) best = { sse, k };
      }
      if (!best) return node;
      node.t = (xs[best.k] + xs[best.k + 1]) / 2;
      node.l = fitTree(res, a, best.k + 1, d - 1); node.r = fitTree(res, best.k + 1, b, d - 1);
      return node;
    }
    function leaves(nd, f) { if (nd.l) { leaves(nd.l, f); leaves(nd.r, f); } else f(nd); }
    const ev = (nd, x) => { while (nd.l) nd = x <= nd.t ? nd.l : nd.r; return nd.v; };
    function reset() {
      const f0 = lSel.value === "l2" ? ys.reduce((a, b) => a + b, 0) / n : median(ys);
      F = new Float64Array(n).fill(f0); Fg = new Float64Array(GRID).fill(f0); m = 0; lastH = null;
    }
    function step() {
      const nu = +nuIn.value, l2 = lSel.value === "l2";
      const res = ys.map((y, i) => (l2 ? y - F[i] : Math.sign(y - F[i])));
      const tree = fitTree(res, 0, n, +dSel.value);
      if (!l2) leaves(tree, (nd) => { const v = []; for (let i = nd.a; i < nd.b; i++) v.push(ys[i] - F[i]); nd.v = median(v); });
      for (let i = 0; i < n; i++) F[i] += nu * ev(tree, xs[i]);
      lastH = new Float64Array(GRID);
      for (let d = 0; d < GRID; d++) { lastH[d] = ev(tree, d); Fg[d] += nu * lastH[d]; }
      m++;
    }
    function draw() {
      nuV.textContent = (+nuIn.value).toFixed(2);
      ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#FDFBF7"; ctx.fillRect(0, 0, W, H);
      ctx.font = "12px Inter, Arial, sans-serif"; ctx.fillStyle = MUT; ctx.strokeStyle = "rgba(20,33,61,.09)"; ctx.lineWidth = 1;
      "SONDJFMAMJJA".split("").forEach((c, k) => { const x = X(k * 30.4 + 15); ctx.textAlign = "center"; ctx.fillText(c, x, H - 10); });
      for (let k = 0; k <= 12; k++) { const x = X(k * 30.4); ctx.beginPath(); ctx.moveTo(x, T1); ctx.lineTo(x, B2); ctx.stroke(); }
      ctx.textAlign = "right";
      [0, 0.2, 0.4, 0.6, 0.8].forEach((v) => { ctx.fillText(v.toFixed(1), L - 6, Y1(v) + 4); ctx.beginPath(); ctx.moveTo(L, Y1(v)); ctx.lineTo(W - R, Y1(v)); ctx.stroke(); });
      ctx.strokeStyle = "rgba(20,33,61,.3)"; ctx.beginPath(); ctx.moveTo(L, Y2(0)); ctx.lineTo(W - R, Y2(0)); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = "600 13px Inter, Arial, sans-serif"; ctx.textAlign = "left";
      ctx.fillText("NDVI · observations and model F_m(t)", L + 6, T1 + 14);
      ctx.fillText("residuals y − F  ·  new tree h_m", L + 6, T2 + 4);
      // residual bars
      ctx.fillStyle = "rgba(46,75,198,.55)";
      for (let i = 0; i < n; i++) { const v = ys[i] - F[i], y0 = Y2(0), y1 = Y2(clamp(v, -0.45, 0.45)); ctx.fillRect(X(xs[i]) - 2, Math.min(y0, y1), 4, Math.abs(y1 - y0)); }
      if (lastH) { ctx.strokeStyle = SAF; ctx.lineWidth = 3; ctx.beginPath(); for (let d = 0; d < GRID; d++) { const y = Y2(clamp(lastH[d], -0.45, 0.45)); d ? ctx.lineTo(X(d), y) : ctx.moveTo(X(d), y); } ctx.stroke(); }
      // data + model
      for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(X(xs[i]), Y1(ys[i]), 4.2, 0, 7); ctx.fillStyle = "#0F6B5C"; ctx.fill(); }
      ctx.strokeStyle = "#B5502E"; ctx.lineWidth = 3.2; ctx.beginPath();
      for (let d = 0; d < GRID; d++) { const y = Y1(clamp(Fg[d], 0, 0.9)); d ? ctx.lineTo(X(d), y) : ctx.moveTo(X(d), y); }
      ctx.stroke();
      let mse = 0, mae = 0; for (let i = 0; i < n; i++) { const e = ys[i] - F[i]; mse += e * e; mae += Math.abs(e); }
      out.textContent = `M = ${m} trees · ν = ${(+nuIn.value).toFixed(2)} · depth ${dSel.value} · ${lSel.value === "l2" ? "squared" : "absolute"} loss · training MSE ${(mse / n).toFixed(4)} · MAE ${(mae / n).toFixed(3)}`;
    }
    let timer = null;
    root.querySelector("[data-act=step]").onclick = () => { clearInterval(timer); step(); draw(); };
    root.querySelector("[data-act=run]").onclick = () => { clearInterval(timer); let k = 0; timer = setInterval(() => { step(); draw(); if (++k >= 20) clearInterval(timer); }, 90); };
    root.querySelector("[data-act=reset]").onclick = () => { clearInterval(timer); reset(); draw(); };
    nuIn.oninput = () => (nuV.textContent = (+nuIn.value).toFixed(2));
    dSel.onchange = lSel.onchange = () => { clearInterval(timer); reset(); for (let i = 0; i < 4; i++) step(); draw(); };
    reset(); for (let i = 0; i < 4; i++) step(); draw();
  };

  /* ======================================================= 6 · support vector machine (simplified SMO) */
  DEMOS.svm = function (root) {
    const cv = root.querySelector("canvas"), P = Plot(cv), ctx = P.ctx, out = root.querySelector(".readout");
    const cIn = root.querySelector("input[name=C]"), gIn = root.querySelector("input[name=g]");
    const cV = root.querySelector(".cv"), gV = root.querySelector(".gv");
    const GW = 220, GH = 140;
    let kind = "linear", kern = "lin", pts, M = null;
    const load = () => { pts = binary(kind, kind === "xor" ? 9 : 12, kind === "xor" ? 28 : 30); };
    function train() {
      const n = pts.length, C = 10 ** +cIn.value, g = 10 ** +gIn.value;
      const Kf = kern === "rbf" ? (a, b) => Math.exp(-g * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)) : (a, b) => a.x * b.x + a.y * b.y;
      const K = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = i; j < n; j++) K[i * n + j] = K[j * n + i] = Kf(pts[i], pts[j]);
      const yv = pts.map((p) => (p.c ? 1 : -1)), al = new Float64Array(n);
      let b = 0;
      const f = (i) => { let s = b; for (let k = 0; k < n; k++) if (al[k] > 0) s += al[k] * yv[k] * K[k * n + i]; return s; };
      const r = rng(7), tol = 1e-3;
      let passes = 0, it = 0;
      while (passes < 8 && it < 400) {
        it++; let changed = 0;
        for (let i = 0; i < n; i++) {
          const Ei = f(i) - yv[i];
          if ((yv[i] * Ei < -tol && al[i] < C) || (yv[i] * Ei > tol && al[i] > 0)) {
            let j = Math.floor(r() * (n - 1)); if (j >= i) j++;
            const Ej = f(j) - yv[j], ai = al[i], aj = al[j];
            let lo, hi;
            if (yv[i] !== yv[j]) { lo = Math.max(0, aj - ai); hi = Math.min(C, C + aj - ai); } else { lo = Math.max(0, ai + aj - C); hi = Math.min(C, ai + aj); }
            if (hi - lo < 1e-10) continue;
            const eta = 2 * K[i * n + j] - K[i * n + i] - K[j * n + j];
            if (eta >= 0) continue;
            const nj = clamp(aj - (yv[j] * (Ei - Ej)) / eta, lo, hi);
            if (Math.abs(nj - aj) < 1e-7) continue;
            const ni = ai + yv[i] * yv[j] * (aj - nj);
            const b1 = b - Ei - yv[i] * (ni - ai) * K[i * n + i] - yv[j] * (nj - aj) * K[i * n + j];
            const b2 = b - Ej - yv[i] * (ni - ai) * K[i * n + j] - yv[j] * (nj - aj) * K[j * n + j];
            b = ni > 0 && ni < C ? b1 : nj > 0 && nj < C ? b2 : (b1 + b2) / 2;
            al[i] = ni; al[j] = nj; changed++;
          }
        }
        passes = changed ? 0 : passes + 1;
      }
      const sv = []; for (let i = 0; i < n; i++) if (al[i] > 1e-7) sv.push(i);
      const dec = (x, y) => { let s = b; const q = { x, y }; for (const i of sv) s += al[i] * yv[i] * Kf(pts[i], q); return s; };
      let wn = null;
      if (kern === "lin") { let w0 = 0, w1 = 0; for (const i of sv) { w0 += al[i] * yv[i] * pts[i].x; w1 += al[i] * yv[i] * pts[i].y; } wn = Math.hypot(w0, w1); }
      M = { dec, sv, C, g, wn, yv };
    }
    function draw() {
      cV.textContent = fmt(10 ** +cIn.value); gV.textContent = fmt(10 ** +gIn.value);
      gIn.disabled = kern !== "rbf";
      train();
      const grid = new Float32Array(GW * GH);
      for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) grid[j * GW + i] = M.dec((i + 0.5) / GW, 1 - (j + 0.5) / GH);
      P.field(GW, GH, (i, j) => { const f = grid[j * GW + i]; return [...binRGB(kind, f > 0 ? 1 : 0), Math.abs(f) < 1 ? 0.13 : 0.3]; }, false);
      const sx = P.W / GW, sy = P.H / GH;
      [[0, INK, 1], [1, MUT, 0], [-1, MUT, 0]].forEach(([lev, col, solid]) => {
        ctx.fillStyle = col;
        for (let j = 0; j < GH - 1; j++) for (let i = 0; i < GW - 1; i++) {
          const v = grid[j * GW + i] - lev, a = grid[j * GW + i + 1] - lev, c = grid[(j + 1) * GW + i] - lev;
          if (v * a <= 0 || v * c <= 0) { if (solid || (i + j) % 8 < 4) ctx.fillRect(i * sx, j * sy, sx + (solid ? 1 : 0.5), sy + (solid ? 1 : 0.5)); }
        }
      });
      P.frame(BIN[kind].axes[0], BIN[kind].axes[1], BIN[kind].ticks);
      const pred = (p) => (M.dec(p.x, p.y) > 0 ? 1 : 0);
      M.sv.forEach((i) => { const p = pts[i]; ctx.beginPath(); ctx.arc(P.X(p.x), P.Y(p.y), 11, 0, 7); ctx.lineWidth = 3.5; ctx.strokeStyle = SAF; ctx.stroke(); });
      P.points(pts, binCol(kind), { r: () => 5.5 });
      const acc = pts.filter((p) => pred(p) === p.c).length / pts.length;
      out.textContent = `${kern === "rbf" ? "RBF kernel" : "linear kernel"} · C = ${fmt(M.C)}${kern === "rbf" ? ` · γ = ${fmt(M.g)}` : ""} · support vectors ${M.sv.length}/${pts.length} · training accuracy ${pct(acc)}`
        + (M.wn ? ` · margin 2/‖w‖ = ${(2 / M.wn).toFixed(3)}` : "");
    }
    const fmt = (v) => (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
    root.querySelectorAll("[data-ds]").forEach((b) => (b.onclick = () => { kind = b.dataset.ds; pick(root, "data-ds", kind); load(); draw(); }));
    root.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { kern = b.dataset.k; pick(root, "data-k", kern); draw(); }));
    cIn.oninput = draw; gIn.oninput = draw;
    cv.addEventListener("mousedown", (e) => { const p = P.at(e); pts.push({ x: p.x, y: p.y, c: e.shiftKey || e.button === 2 ? 0 : 1 }); draw(); });
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    pick(root, "data-ds", kind); pick(root, "data-k", kern);
    load(); draw();
  };
})();
