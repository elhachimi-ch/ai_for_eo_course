/* Deep Learning for Earth Observation — reinforcement-learning demos (no dependencies)
   A survey-drone gridworld (reach the flooded village, avoid storm cells, mountains block):
   value iteration · tabular Q-learning · multi-armed bandits (ε-greedy, UCB1, Thompson sampling).
   Needs demos_ml.js (DemoKit) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, gauss, clamp, pick } = K;
  const INK = "#14213D", MUT = "#5B6478";

  /* ============================================================ gridworld */
  // S start · G goal (+10) · X storm cell (−10) · # mountain (blocked) · . free (−0.1 battery per move)
  const MAP = ["........G", ".##..X.#.", "..#.XX.#.", ".........", "X#.##.X..", "S.....X.."];
  const ACT = [[-1, 0], [0, 1], [1, 0], [0, -1]], ARROW = ["↑", "→", "↓", "←"];
  const STEP = -0.1, GOAL = 10, STORM = -10;
  function World() {
    this.R = MAP.length; this.C = MAP[0].length;
    this.g = MAP.map((r) => r.split(""));
    this.n = this.R * this.C;
    this.start = this.g.flat().indexOf("S");
  }
  World.prototype.type = function (s) { return this.g[Math.floor(s / this.C)][s % this.C]; };
  World.prototype.terminal = function (s) { const t = this.type(s); return t === "G" || t === "X"; };
  World.prototype.move = function (s, a) {
    const r = Math.floor(s / this.C), c = s % this.C, nr = r + ACT[a][0], nc = c + ACT[a][1];
    if (nr < 0 || nc < 0 || nr >= this.R || nc >= this.C || this.g[nr][nc] === "#") return s;
    return nr * this.C + nc;
  };
  World.prototype.reward = function (s2) { const t = this.type(s2); return t === "G" ? GOAL : t === "X" ? STORM : STEP; };
  /* intended move with prob 1 − slip, otherwise one of the two perpendicular moves */
  World.prototype.trans = function (s, a, slip) {
    return [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]].filter(([, p]) => p > 0)
      .map(([b, p]) => { const s2 = this.move(s, b); return { p, s2, r: this.reward(s2), done: this.terminal(s2) }; });
  };
  World.prototype.sample = function (s, a, slip, r) {
    const u = r(), b = u < 1 - slip ? a : u < 1 - slip / 2 ? (a + 1) % 4 : (a + 3) % 4, s2 = this.move(s, b);
    return { s2, r: this.reward(s2), done: this.terminal(s2) };
  };

  function drawGrid(cv, w, val, pol, o = {}) {
    const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, cw = W / w.C, ch = H / w.R;
    ctx.clearRect(0, 0, W, H);
    for (let s = 0; s < w.n; s++) {
      const x = (s % w.C) * cw, y = Math.floor(s / w.C) * ch, t = w.type(s);
      let bg = "#FDFBF7";
      if (t === "#") bg = "#22325A"; else if (t === "X") bg = "#B5502E"; else if (t === "G") bg = "#0F6B5C";
      ctx.fillStyle = bg; ctx.fillRect(x, y, cw, ch);
      if (t === "." || t === "S") {
        const v = val ? val[s] : 0, a = Math.min(1, Math.abs(v) / 8) * 0.55;
        if (val && o.show !== false && (o.seen ? o.seen[s] : true)) { ctx.fillStyle = v >= 0 ? `rgba(15,107,92,${a})` : `rgba(181,80,46,${a})`; ctx.fillRect(x, y, cw, ch); }
      }
      ctx.strokeStyle = "rgba(20,33,61,.15)"; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      if (t === "#") { ctx.fillStyle = "#C9D2F3"; ctx.font = "26px Inter, Arial"; ctx.fillText("▲", x + cw / 2, y + ch / 2); }
      else if (t === "X") { ctx.fillStyle = "#fff"; ctx.font = "700 14px Inter, Arial"; ctx.fillText("storm", x + cw / 2, y + ch / 2 - 8); ctx.fillText("−10", x + cw / 2, y + ch / 2 + 10); }
      else if (t === "G") { ctx.fillStyle = "#fff"; ctx.font = "700 13px Inter, Arial"; ctx.fillText("flooded", x + cw / 2, y + ch / 2 - 15); ctx.fillText("village", x + cw / 2, y + ch / 2); ctx.fillText("+10", x + cw / 2, y + ch / 2 + 16); }
      else {
        const seen = o.seen ? o.seen[s] : true;
        if (pol && seen) { ctx.fillStyle = INK; ctx.font = "700 22px Inter, Arial"; ctx.fillText(ARROW[pol[s]], x + cw / 2, y + ch / 2 - 6); }
        if (val && seen) { ctx.fillStyle = "#22325A"; ctx.font = "12px DejaVu Mono, Consolas, monospace"; ctx.fillText(val[s].toFixed(2), x + cw / 2, y + ch - 12); }
        if (t === "S") { ctx.fillStyle = "#8A6512"; ctx.font = "700 12px Inter, Arial"; ctx.textAlign = "left"; ctx.fillText("start", x + 4, y + 11); }
      }
    }
    if (o.path && o.path.length > 1) {
      ctx.strokeStyle = "rgba(201,154,46,.9)"; ctx.lineWidth = 4; ctx.lineJoin = "round"; ctx.beginPath();
      o.path.forEach((s, k) => { const x = (s % w.C + 0.5) * cw, y = (Math.floor(s / w.C) + 0.5) * ch; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    if (o.agent != null) {
      const x = (o.agent % w.C + 0.5) * cw, y = (Math.floor(o.agent / w.C) + 0.5) * ch;
      ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.fillStyle = "#C99A2E"; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke();
      ctx.fillStyle = INK; ctx.font = "700 12px Inter, Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("✈", x, y + 1);
    }
    ctx.textBaseline = "alphabetic";
  }
  const cellAt = (cv, w, e) => { const b = cv.getBoundingClientRect(); return Math.floor(((e.clientY - b.top) / b.height) * w.R) * w.C + Math.floor(((e.clientX - b.left) / b.width) * w.C); };

  /* ============================================================ 1 · value iteration */
  DEMOS.vi = function (root) {
    const cv = root.querySelector("canvas"), out = root.querySelector(".readout");
    const gIn = root.querySelector("input[name=gamma]"), sIn = root.querySelector("input[name=slip]");
    const gV = root.querySelector(".gv"), sV = root.querySelector(".sv");
    const w = new World();
    let V, pol, sweeps, delta, timer = null;
    const Q = (s, a, gam, slip) => w.trans(s, a, slip).reduce((t, o) => t + o.p * (o.r + (o.done ? 0 : gam * V[o.s2])), 0);
    function reset() { clearInterval(timer); V = new Float64Array(w.n); pol = new Uint8Array(w.n); sweeps = 0; delta = NaN; }
    function sweep() {
      const gam = +gIn.value, slip = +sIn.value, V2 = new Float64Array(w.n); delta = 0;
      for (let s = 0; s < w.n; s++) {
        const t = w.type(s); if (t === "#" || w.terminal(s)) continue;
        let best = -Infinity, ba = 0;
        for (let a = 0; a < 4; a++) { const q = Q(s, a, gam, slip); if (q > best + 1e-12) { best = q; ba = a; } }
        V2[s] = best; pol[s] = ba; delta = Math.max(delta, Math.abs(best - V[s]));
      }
      V = V2; sweeps++;
      return delta;
    }
    function draw() {
      gV.textContent = (+gIn.value).toFixed(2); sV.textContent = (+sIn.value).toFixed(2);
      drawGrid(cv, w, V, sweeps ? pol : null, {});
      out.textContent = sweeps
        ? `sweep ${sweeps} · max |ΔV| = ${delta.toFixed(4)}${delta < 1e-3 ? " — converged" : ""} · V(start) = ${V[w.start].toFixed(2)} · γ = ${(+gIn.value).toFixed(2)}, slip = ${(+sIn.value).toFixed(2)}`
        : "V = 0 everywhere — press “1 sweep” (click a free cell to add a mountain or a storm)";
    }
    root.querySelector("[data-act=sweep]").onclick = () => { clearInterval(timer); sweep(); draw(); };
    root.querySelector("[data-act=run]").onclick = () => { clearInterval(timer); let k = 0; timer = setInterval(() => { const d = sweep(); draw(); if (d < 1e-3 || ++k > 200) clearInterval(timer); }, 120); };
    root.querySelector("[data-act=reset]").onclick = () => { reset(); draw(); };
    gIn.oninput = sIn.oninput = () => { reset(); draw(); };
    cv.addEventListener("mousedown", (e) => {
      const s = cellAt(cv, w, e), r = Math.floor(s / w.C), c = s % w.C, t = w.g[r][c];
      if (t === "S" || t === "G") return;
      w.g[r][c] = t === "." ? "#" : t === "#" ? "X" : ".";
      reset(); for (let k = 0; k < 60; k++) sweep(); draw();
    });
    reset(); for (let k = 0; k < 80; k++) sweep(); draw();
  };

  /* ============================================================ 2 · Q-learning */
  DEMOS.qlearn = function (root) {
    const cv = root.querySelector("canvas.grid"), ch = root.querySelector("canvas.ret"), out = root.querySelector(".readout");
    const eIn = root.querySelector("input[name=eps]"), aIn = root.querySelector("input[name=alpha]");
    const eV = root.querySelector(".ev"), aV = root.querySelector(".av");
    const w = new World(), GAM = 0.95, SLIP = 0.1, MAXT = 100;
    let Q, seen, rets, wins, path, r, timer = null;
    function reset() { clearInterval(timer); Q = Array.from({ length: w.n }, () => new Float64Array(4)); seen = new Uint8Array(w.n); rets = []; wins = []; path = []; r = rng(11); }
    const greedy = (s) => { const q = Q[s], m = Math.max(...q), best = [0, 1, 2, 3].filter((a) => q[a] >= m - 1e-12); return best[Math.floor(r() * best.length)]; };
    function* episode() {
      const eps = +eIn.value, alpha = +aIn.value;
      let s = w.start, G = 0, t = 0; path = [s];
      for (; t < MAXT; t++) {
        const a = r() < eps ? Math.floor(r() * 4) : greedy(s);
        const o = w.sample(s, a, SLIP, r);
        const target = o.r + (o.done ? 0 : GAM * Math.max(...Q[o.s2]));
        Q[s][a] += alpha * (target - Q[s][a]); seen[s] = 1;
        G += o.r; s = o.s2; path.push(s);
        yield s;
        if (o.done) break;
      }
      rets.push(G); wins.push(w.type(s) === "G" ? 1 : 0);
    }
    const runN = (k) => { for (let i = 0; i < k; i++) for (const _ of episode()); };
    function drawChart() {
      const c = ch.getContext("2d"), W = ch.width, H = ch.height, L = 34, B = 20;
      c.clearRect(0, 0, W, H); c.fillStyle = "#FDFBF7"; c.fillRect(0, 0, W, H);
      const Y = (v) => H - B - ((clamp(v, -14, 12) + 14) / 26) * (H - B - 10), n = rets.length, X = (i) => L + (i / Math.max(1, n - 1)) * (W - L - 8);
      c.font = "11px Inter, Arial"; c.fillStyle = MUT; c.textAlign = "right";
      [-10, 0, 10].forEach((v) => { c.fillText(v, L - 4, Y(v) + 4); c.strokeStyle = "rgba(20,33,61,.1)"; c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(W - 8, Y(v)); c.stroke(); });
      c.textAlign = "left"; c.fillStyle = INK; c.font = "600 12px Inter, Arial"; c.fillText("return per episode (dots) and 20-episode mean", L + 4, 12);
      c.fillStyle = "rgba(46,75,198,.35)"; rets.forEach((v, i) => c.fillRect(X(i) - 1, Y(v) - 1, 2.5, 2.5));
      c.strokeStyle = "#B5502E"; c.lineWidth = 2.4; c.beginPath();
      rets.forEach((_, i) => { const a = Math.max(0, i - 19), m = rets.slice(a, i + 1).reduce((s, v) => s + v, 0) / (i + 1 - a); i ? c.lineTo(X(i), Y(m)) : c.moveTo(X(i), Y(m)); });
      c.stroke();
    }
    function draw(agent) {
      eV.textContent = (+eIn.value).toFixed(2); aV.textContent = (+aIn.value).toFixed(2);
      const V = Q.map((q) => Math.max(...q)), pol = Q.map((q) => q.indexOf(Math.max(...q)));
      drawGrid(cv, w, V, pol, { seen, path, agent });
      drawChart();
      const n = rets.length, last = rets.slice(-50);
      out.textContent = n
        ? `episodes ${n} · last return ${rets[n - 1].toFixed(1)} · mean of last ${last.length} ${(last.reduce((s, v) => s + v, 0) / last.length).toFixed(2)} · reached the village in ${(100 * wins.slice(-50).reduce((s, v) => s + v, 0) / last.length).toFixed(0)} % of them`
        : "Q = 0 everywhere — the drone knows nothing; press “1 episode”";
    }
    root.querySelector("[data-act=one]").onclick = () => {
      clearInterval(timer); const it = episode();
      timer = setInterval(() => { const o = it.next(); if (o.done) { clearInterval(timer); draw(); } else draw(o.value); }, 70);
    };
    root.querySelector("[data-act=fifty]").onclick = () => { clearInterval(timer); runN(50); draw(); };
    root.querySelector("[data-act=many]").onclick = () => { clearInterval(timer); runN(500); draw(); };
    root.querySelector("[data-act=reset]").onclick = () => { reset(); draw(); };
    eIn.oninput = aIn.oninput = () => { eV.textContent = (+eIn.value).toFixed(2); aV.textContent = (+aIn.value).toFixed(2); };
    reset(); runN(150); draw();
  };

  /* ============================================================ 3 · multi-armed bandits */
  function gammaS(k, r) {
    if (k < 1) return gammaS(k + 1, r) * Math.pow(r(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v; do { x = gauss(r); v = 1 + c * x; } while (v <= 0);
      v = v * v * v; const u = r();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  const betaS = (a, b, r) => { const x = gammaS(a, r), y = gammaS(b, r); return x / (x + y); };
  DEMOS.bandit = function (root) {
    const rc = root.querySelector("canvas.reg"), ac = root.querySelector("canvas.arms"), out = root.querySelector(".readout");
    const eIn = root.querySelector("input[name=eps]"), cIn = root.querySelector("input[name=c]"), rev = root.querySelector("input[name=reveal]");
    const eV = root.querySelector(".ev"), cV = root.querySelector(".cv");
    const P = [0.3, 0.55, 0.42, 0.68, 0.2], NAMES = ["A", "B", "C", "D", "E"], BEST = Math.max(...P);
    const ST = [["eg", "ε-greedy", "#2E4BC6"], ["ucb", "UCB1", "#B5502E"], ["ts", "Thompson", "#0F6B5C"]];
    let A, t, show = "ts", timer = null;
    function reset() {
      clearInterval(timer); t = 0;
      A = {};
      ST.forEach(([k], j) => (A[k] = { N: new Float64Array(5), S: new Float64Array(5), reg: [0], r: rng(40 + j), env: rng(7) }));
    }
    function choose(k, s) {
      const n = s.N, est = (a) => (n[a] ? s.S[a] / n[a] : 0);
      if (k === "eg") { if (s.r() < +eIn.value) return Math.floor(s.r() * 5); let b = 0; for (let a = 1; a < 5; a++) if (est(a) > est(b)) b = a; return b; }
      if (k === "ucb") { for (let a = 0; a < 5; a++) if (!n[a]) return a; let b = 0, bv = -Infinity; for (let a = 0; a < 5; a++) { const v = est(a) + +cIn.value * Math.sqrt(Math.log(t + 1) / n[a]); if (v > bv) { bv = v; b = a; } } return b; }
      let b = 0, bv = -1; for (let a = 0; a < 5; a++) { const v = betaS(1 + s.S[a], 1 + n[a] - s.S[a], s.r); if (v > bv) { bv = v; b = a; } } return b;
    }
    function pull(T) {
      for (let i = 0; i < T; i++) {
        t++;
        ST.forEach(([k]) => { const s = A[k], a = choose(k, s), win = s.env() < P[a] ? 1 : 0; s.N[a]++; s.S[a] += win; s.reg.push(s.reg[s.reg.length - 1] + BEST - P[a]); });
      }
    }
    function draw() {
      eV.textContent = (+eIn.value).toFixed(2); cV.textContent = (+cIn.value).toFixed(1);
      // regret curves
      let c = rc.getContext("2d"), W = rc.width, H = rc.height, L = 40, B = 22;
      c.clearRect(0, 0, W, H); c.fillStyle = "#FDFBF7"; c.fillRect(0, 0, W, H);
      const ymax = Math.max(5, ...ST.map(([k]) => A[k].reg[t])) * 1.1, X = (i) => L + (i / Math.max(1, t)) * (W - L - 10), Y = (v) => H - B - (v / ymax) * (H - B - 26);
      c.font = "11px Inter, Arial"; c.fillStyle = MUT; c.textAlign = "right";
      [0, 0.5, 1].forEach((f) => { c.fillText((f * ymax).toFixed(0), L - 5, Y(f * ymax) + 4); c.strokeStyle = "rgba(20,33,61,.1)"; c.beginPath(); c.moveTo(L, Y(f * ymax)); c.lineTo(W - 10, Y(f * ymax)); c.stroke(); });
      c.textAlign = "center"; c.fillText(`t = ${t} visits`, W / 2, H - 5);
      ST.forEach(([k, name, co], j) => {
        c.strokeStyle = co; c.lineWidth = 2.6; c.beginPath();
        const step = Math.max(1, Math.floor(t / 400));
        for (let i = 0; i <= t; i += step) { const y = Y(A[k].reg[i]); i ? c.lineTo(X(i), y) : c.moveTo(X(i), y); }
        c.stroke();
        c.fillStyle = co; c.font = "600 12px Inter, Arial"; c.textAlign = "left"; c.fillText(`${name} ${A[k].reg[t].toFixed(1)}`, L + 8 + j * 150, 14);
      });
      // arms of the selected strategy
      c = ac.getContext("2d"); W = ac.width; H = ac.height;
      c.clearRect(0, 0, W, H); c.fillStyle = "#FDFBF7"; c.fillRect(0, 0, W, H);
      const s = A[show], co = ST.find(([k]) => k === show)[2], maxN = Math.max(1, ...s.N), gw = (W - 20) / 5;
      c.font = "600 12px Inter, Arial"; c.fillStyle = INK; c.textAlign = "left";
      c.fillText(`${ST.find(([k]) => k === show)[1]}: visits per site (bars) · estimated detection rate (●)${rev.checked ? " · true rate (|)" : ""}`, 10, 14);
      for (let a = 0; a < 5; a++) {
        const x = 10 + a * gw, base = H - 22, h = (s.N[a] / maxN) * (H - 52);
        c.fillStyle = co + "55"; c.fillRect(x + 10, base - h, gw - 20, h);
        c.fillStyle = MUT; c.font = "12px Inter, Arial"; c.textAlign = "center"; c.fillText(`site ${NAMES[a]} · ${s.N[a]}`, x + gw / 2, H - 6);
        const yv = (v) => base - v * (H - 52);
        if (s.N[a]) { c.beginPath(); c.arc(x + gw / 2, yv(s.S[a] / s.N[a]), 5, 0, 7); c.fillStyle = co; c.fill(); }
        if (rev.checked) { c.strokeStyle = INK; c.lineWidth = 3; c.beginPath(); c.moveTo(x + 14, yv(P[a])); c.lineTo(x + gw - 14, yv(P[a])); c.stroke(); }
      }
      out.textContent = `t = ${t} · cumulative regret (expected detections lost vs always visiting the best site): `
        + ST.map(([k, nm]) => `${nm} ${A[k].reg[t].toFixed(1)}`).join(" · ");
    }
    root.querySelector("[data-act=ten]").onclick = () => { clearInterval(timer); pull(10); draw(); };
    root.querySelector("[data-act=run]").onclick = () => { clearInterval(timer); let k = 0; timer = setInterval(() => { pull(20); draw(); if (++k >= 50) clearInterval(timer); }, 50); };
    root.querySelector("[data-act=reset]").onclick = () => { reset(); draw(); };
    root.querySelectorAll("[data-st]").forEach((b) => (b.onclick = () => { show = b.dataset.st; pick(root, "data-st", show); draw(); }));
    eIn.oninput = cIn.oninput = () => { reset(); pull(300); draw(); };
    rev.onchange = draw;
    pick(root, "data-st", show); reset(); pull(300); draw();
  };
})();
