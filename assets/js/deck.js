/* Deep Learning for Earth Observation — slide engine (no dependencies) */
(function () {
  "use strict";
  const params = new URLSearchParams(location.search);
  const EXPORT = params.has("export") || params.has("print-pdf");
  const slides = Array.from(document.querySelectorAll("#stage > .slide"));
  const stage = document.getElementById("stage");
  let cur = 0;

  /* =====================================================================
     Interactive demos — initialised lazily when their slide is shown
     ===================================================================== */
  function initDemos(slide) {
    slide.querySelectorAll("[data-demo]").forEach((el) => {
      if (el.dataset.ready) return;
      el.dataset.ready = "1";
      const f = DEMOS[el.dataset.demo];
      if (f) { try { f(el); } catch (err) { console.error("demo", el.dataset.demo, err); } }
    });
  }
  window.__initDemos = initDemos;

  /* Perceptron playground: click = class A, shift/right-click = class B */
  const DEMOS = {};
  DEMOS.perceptron = function (root) {
    const cv = root.querySelector("canvas"), ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height, out = root.querySelector(".readout");
    let pts = [], w = [0.2, -0.3, 0.05], epoch = 0, timer = null;
    const lr = 0.05;
    // seed: "water" (low NIR, low red) vs "vegetation" (high NIR, low/mid red)
    function seed() {
      pts = [];
      let s = 7;
      const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
      for (let i = 0; i < 22; i++) pts.push({ x: 0.15 + 0.3 * rnd(), y: 0.08 + 0.22 * rnd(), c: -1 });
      for (let i = 0; i < 22; i++) pts.push({ x: 0.10 + 0.35 * rnd(), y: 0.52 + 0.4 * rnd(), c: 1 });
      w = [0.2, -0.3, 0.05]; epoch = 0;
    }
    function predict(p) { return w[0] * p.x + w[1] * p.y + w[2] >= 0 ? 1 : -1; }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      // shaded half-planes
      const img = ctx.createImageData(W, H);
      for (let yy = 0; yy < H; yy += 1) for (let xx = 0; xx < W; xx += 1) {
        const p = { x: xx / W, y: 1 - yy / H }, c = predict(p), i = (yy * W + xx) * 4;
        if (c > 0) { img.data[i] = 226; img.data[i + 1] = 241; img.data[i + 2] = 236; }
        else { img.data[i] = 230; img.data[i + 1] = 234; img.data[i + 2] = 250; }
        img.data[i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      // axes labels
      ctx.fillStyle = "#5B6478"; ctx.font = "14px Inter";
      ctx.fillText("Red reflectance →", W - 140, H - 8);
      ctx.save(); ctx.translate(14, 130); ctx.rotate(-Math.PI / 2); ctx.fillText("NIR reflectance →", 0, 0); ctx.restore();
      // boundary
      if (Math.abs(w[1]) > 1e-6) {
        ctx.strokeStyle = "#B5502E"; ctx.lineWidth = 3; ctx.beginPath();
        const y0 = -(w[0] * 0 + w[2]) / w[1], y1 = -(w[0] * 1 + w[2]) / w[1];
        ctx.moveTo(0, H - y0 * H); ctx.lineTo(W, H - y1 * H); ctx.stroke();
      }
      pts.forEach((p) => {
        ctx.beginPath(); ctx.arc(p.x * W, H - p.y * H, 7, 0, 7);
        ctx.fillStyle = p.c > 0 ? "#0F6B5C" : "#2E4BC6"; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = predict(p) === p.c ? "#fff" : "#B5502E"; ctx.stroke();
      });
      const err = pts.filter((p) => predict(p) !== p.c).length;
      out.innerHTML = `epoch ${epoch} · w = [${w.map((v) => v.toFixed(2)).join(", ")}] · errors ${err}/${pts.length}`;
      return err;
    }
    function step() {
      // one epoch of the Rosenblatt rule
      pts.forEach((p) => {
        const y = predict(p);
        if (y !== p.c) { w[0] += lr * p.c * p.x; w[1] += lr * p.c * p.y; w[2] += lr * p.c; }
      });
      epoch++;
      return draw();
    }
    cv.addEventListener("mousedown", (e) => {
      const r = cv.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width, y = 1 - (e.clientY - r.top) / r.height;
      pts.push({ x, y, c: e.shiftKey || e.button === 2 ? -1 : 1 }); draw();
    });
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    root.querySelector("[data-act=step]").onclick = step;
    root.querySelector("[data-act=train]").onclick = () => {
      clearInterval(timer); let n = 0;
      timer = setInterval(() => { const err = step(); if (err === 0 || ++n > 200) clearInterval(timer); }, 120);
    };
    root.querySelector("[data-act=xor]").onclick = () => {
      clearInterval(timer); pts = []; w = [0.2, -0.3, 0.05]; epoch = 0;
      [[.2,.2,-1],[.8,.8,-1],[.2,.8,1],[.8,.2,1]].forEach(([a,b,c])=>{for(let i=0;i<6;i++)pts.push({x:a+(i%3-1)*.04,y:b+(Math.floor(i/3)-.5)*.06,c});});
      draw();
    };
    root.querySelector("[data-act=reset]").onclick = () => { clearInterval(timer); seed(); draw(); };
    seed(); draw();
  };

  /* Gradient descent on a 1-D non-convex loss */
  DEMOS.gd = function (root) {
    const cv = root.querySelector("canvas"), ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height, out = root.querySelector(".readout");
    const lrIn = root.querySelector("input[name=lr]"), lrOut = root.querySelector(".lrv");
    const L = (x) => 0.08 * x ** 4 - 0.6 * x ** 2 + 0.25 * x + 1.6;
    const dL = (x) => 0.32 * x ** 3 - 1.2 * x + 0.25;
    let x = 2.6, path = [x], timer = null;
    const X = (v) => ((v + 3.2) / 6.4) * W, Y = (v) => H - 20 - (v / 3.2) * (H - 40);
    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "#14213D"; ctx.lineWidth = 3; ctx.beginPath();
      for (let i = 0; i <= 300; i++) { const v = -3.2 + (6.4 * i) / 300; i ? ctx.lineTo(X(v), Y(L(v))) : ctx.moveTo(X(v), Y(L(v))); }
      ctx.stroke();
      ctx.strokeStyle = "#C99A2E"; ctx.lineWidth = 2; ctx.beginPath();
      path.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(L(v))) : ctx.moveTo(X(v), Y(L(v)))));
      ctx.stroke();
      path.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(v), Y(L(v)), i === path.length - 1 ? 9 : 4, 0, 7); ctx.fillStyle = i === path.length - 1 ? "#B5502E" : "#C99A2E"; ctx.fill(); });
      out.textContent = `step ${path.length - 1} · θ = ${x.toFixed(3)} · L(θ) = ${L(x).toFixed(3)} · ∂L/∂θ = ${dL(x).toFixed(3)}`;
    }
    function step() { const lr = parseFloat(lrIn.value); x = x - lr * dL(x); x = Math.max(-3.2, Math.min(3.2, x)); path.push(x); draw(); }
    lrIn.oninput = () => (lrOut.textContent = parseFloat(lrIn.value).toFixed(2));
    root.querySelector("[data-act=step]").onclick = step;
    root.querySelector("[data-act=run]").onclick = () => { clearInterval(timer); let n = 0; timer = setInterval(() => { step(); if (++n > 40) clearInterval(timer); }, 110); };
    root.querySelector("[data-act=reset]").onclick = () => { clearInterval(timer); x = parseFloat(root.querySelector("select[name=x0]").value); path = [x]; draw(); };
    root.querySelector("select[name=x0]").onchange = root.querySelector("[data-act=reset]").onclick;
    lrOut.textContent = parseFloat(lrIn.value).toFixed(2);
    draw();
  };

  /* Convolution on a synthetic "satellite" scene */
  DEMOS.conv = function (root) {
    const src = root.querySelector("canvas.src"), dst = root.querySelector("canvas.dst");
    const sc = src.getContext("2d"), dc = dst.getContext("2d");
    const N = src.width;
    const kOut = root.closest(".slide").querySelector(".kernel");
    // build scene: fields (parcels), river, road, buildings
    const img = new Float32Array(N * N);
    let s = 11; const rnd = () => ((s = (s * 9301 + 49297) % 233280) / 233280);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const px = Math.floor(x / 48), py = Math.floor(y / 40);
      const base = 0.25 + 0.5 * (((px * 7 + py * 13) % 5) / 5);
      img[y * N + x] = base + 0.04 * Math.sin(x * 0.9 + py) ;
    }
    for (let x = 0; x < N; x++) { const yc = 0.55 * N + 22 * Math.sin(x / 30); for (let y = 0; y < N; y++) if (Math.abs(y - yc) < 7) img[y * N + x] = 0.08; }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (Math.abs(x - 0.68 * N - 0.25 * y) < 3) img[y * N + x] = 0.92;
    for (let b = 0; b < 26; b++) { const bx = Math.floor(rnd() * (N * 0.35)) + 10, by = Math.floor(rnd() * (N * 0.35)) + 8; for (let y = by; y < by + 7; y++) for (let x = bx; x < bx + 9; x++) img[y * N + x] = 0.98; }
    const K = {
      identity: [[0,0,0],[0,1,0],[0,0,0]],
      blur: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
      sharpen: [[0,-1,0],[-1,5,-1],[0,-1,0]],
      sobelX: [[-1,0,1],[-2,0,2],[-1,0,1]],
      sobelY: [[-1,-2,-1],[0,0,0],[1,2,1]],
      laplace: [[0,1,0],[1,-4,1],[0,1,0]],
    };
    function paint(ctx, arr, abs) {
      const id = ctx.createImageData(N, N);
      let mn = Infinity, mx = -Infinity;
      for (const v of arr) { const t = abs ? Math.abs(v) : v; mn = Math.min(mn, t); mx = Math.max(mx, t); }
      for (let i = 0; i < arr.length; i++) {
        const t = ((abs ? Math.abs(arr[i]) : arr[i]) - mn) / (mx - mn + 1e-9);
        // indigo → ivory ramp
        id.data[i * 4] = 20 + t * 231; id.data[i * 4 + 1] = 33 + t * 214; id.data[i * 4 + 2] = 61 + t * 179; id.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(id, 0, 0);
    }
    function apply(name) {
      const k = K[name], o = new Float32Array(N * N);
      for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
        let a = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) a += k[j + 1][i + 1] * img[(y + j) * N + x + i];
        o[y * N + x] = a;
      }
      paint(dc, o, name.startsWith("sobel") || name === "laplace");
      kOut.innerHTML = k.map((r) => r.map((v) => (Math.abs(v - 1 / 9) < 1e-6 ? "1/9" : v)).join("&nbsp;&nbsp;")).join("<br>");
      root.querySelectorAll("button[data-k]").forEach((b) => b.classList.toggle("gold", b.dataset.k === name));
    }
    paint(sc, img, false);
    root.querySelectorAll("button[data-k]").forEach((b) => (b.onclick = () => apply(b.dataset.k)));
    apply("sobelX");
  };

  /* Scaled dot-product attention explorer */
  DEMOS.attn = function (root) {
    const toks = root.dataset.tokens.split("|");
    const grid = root.querySelector(".agrid"), out = root.querySelector(".readout");
    // hand-crafted 4-d embeddings so the pattern is interpretable
    const E = {
      "Sentinel-2": [1, .2, 0, .1], "sees": [.1, .1, .9, 0], "flooded": [.2, 1, .1, .2], "fields": [.9, .6, 0, .1],
      "near": [0, .1, .3, .9], "the": [0, 0, .1, .2], "river": [.3, .9, 0, .5], "after": [0, .2, .4, .8], "storm": [.1, .9, .2, .6],
    };
    const v = (t) => E[t] || [0.2, 0.2, 0.2, 0.2];
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    let temp = 1, cur = 2;
    grid.innerHTML = toks.map((t, i) => `<span class="atok" data-i="${i}">${t}<small></small></span>`).join("");
    const els = Array.from(grid.querySelectorAll(".atok"));
    function render(qi) {
      cur = qi;
      const d = 4, q = v(toks[qi]);
      const sc = toks.map((t) => dot(q, v(t)) / Math.sqrt(d) / temp);
      const m = Math.max(...sc), ex = sc.map((s) => Math.exp(s - m)), Z = ex.reduce((a, b) => a + b, 0), p = ex.map((e) => e / Z);
      els.forEach((el, i) => {
        el.classList.toggle("q", i === qi);
        el.style.background = `rgba(181,80,46,${Math.min(1, p[i] * 1.6).toFixed(3)})`;
        el.querySelector("small").textContent = p[i].toFixed(2);
      });
      out.textContent = `query = "${toks[qi]}" · softmax(q·kᵀ/√d) shown under each token`;
    }
    els.forEach((el, i) => { el.onmouseenter = () => render(i); el.onclick = () => render(i); });
    const tIn = root.querySelector("input[name=temp]");
    if (tIn) tIn.oninput = () => { temp = parseFloat(tIn.value); root.querySelector(".tv").textContent = temp.toFixed(1); render(cur); };
    render(2);
  };

  if (EXPORT) {
    document.body.classList.add("export");
    slides.forEach((s, i) => { s.classList.add("active"); initDemos(s); });
    window.__deckReady = true;
    return;
  }

  /* ---------- scaling ---------- */
  function fit() {
    const W = 1600, H = 900, pad = 70;
    const s = Math.min(window.innerWidth / W, (window.innerHeight - pad) / H);
    stage.style.transform = `translate(-50%,-50%) scale(${s})`;
  }
  window.addEventListener("resize", fit); fit();

  /* ---------- navigation ---------- */
  const counter = document.getElementById("counter");
  const progress = document.getElementById("progress");
  function go(n, push = true) {
    n = Math.max(0, Math.min(slides.length - 1, n));
    slides[cur].classList.remove("active");
    cur = n;
    const s = slides[cur];
    s.classList.add("active");
    initDemos(s);
    counter.textContent = `${cur + 1} / ${slides.length}`;
    progress.style.width = `${(100 * (cur + 1)) / slides.length}%`;
    if (push) history.replaceState(null, "", `#/${cur + 1}`);
    document.querySelectorAll("#toc a").forEach((a, i) => a.classList.toggle("cur", i === cur));
    const notes = s.querySelector(".notes");
    document.getElementById("notesBody").innerHTML = notes ? notes.innerHTML : "<i>No notes for this slide.</i>";
    document.title = (s.dataset.title || "Slide") + " · DL for Earth Observation";
  }
  const next = () => go(cur + 1), prev = () => go(cur - 1);
  window.deck = { go, next, prev };

  document.getElementById("btnNext").onclick = next;
  document.getElementById("btnPrev").onclick = prev;
  document.getElementById("btnFirst").onclick = () => go(0);
  document.getElementById("btnMenu").onclick = () => document.getElementById("toc").classList.toggle("open");
  document.getElementById("btnNotes").onclick = () => document.getElementById("notesPanel").classList.toggle("open");
  document.getElementById("btnFull").onclick = toggleFull;
  document.getElementById("btnPrint").onclick = () => { window.open(location.pathname + "?export", "_blank"); };
  document.getElementById("btnHelp").onclick = () => document.getElementById("help").classList.add("open");
  document.getElementById("help").onclick = () => document.getElementById("help").classList.remove("open");

  function toggleFull() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input,textarea,select")) return;
    const k = e.key;
    if (["ArrowRight", "PageDown", " ", "Enter"].includes(k)) { e.preventDefault(); next(); }
    else if (["ArrowLeft", "PageUp", "Backspace"].includes(k)) { e.preventDefault(); prev(); }
    else if (k === "Home") go(0);
    else if (k === "End") go(slides.length - 1);
    else if (k === "m" || k === "M") document.getElementById("toc").classList.toggle("open");
    else if (k === "n" || k === "N") document.getElementById("notesPanel").classList.toggle("open");
    else if (k === "f" || k === "F") toggleFull();
    else if (k === "?" || k === "h") document.getElementById("help").classList.toggle("open");
    else if (k === "Escape") { document.getElementById("toc").classList.remove("open"); document.getElementById("help").classList.remove("open"); }
  });

  // swipe
  let tx = null;
  document.addEventListener("touchstart", (e) => { tx = e.touches[0].clientX; }, { passive: true });
  document.addEventListener("touchend", (e) => {
    if (tx === null) return;
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 60) (dx < 0 ? next : prev)();
    tx = null;
  });

  /* ---------- table of contents ---------- */
  const toc = document.getElementById("tocList");
  let lastMod = null;
  slides.forEach((s, i) => {
    const mod = s.dataset.module || "";
    if (mod && mod !== lastMod) {
      const d = document.createElement("div"); d.className = "mod"; d.textContent = mod; toc.appendChild(d); lastMod = mod;
    }
    const a = document.createElement("a");
    a.href = `#/${i + 1}`;
    const kind = s.dataset.kind || "";
    a.innerHTML = `<span class="n">${i + 1}</span><span>${s.dataset.title || "—"}</span>${kind && kind !== "theory" ? `<span class="k ${kind}">${kind}</span>` : ""}`;
    a.onclick = (e) => { e.preventDefault(); go(i); document.getElementById("toc").classList.remove("open"); };
    toc.appendChild(a);
  });

  /* ---------- copy buttons ---------- */
  document.querySelectorAll(".code .bar button.copy").forEach((b) => {
    b.onclick = () => {
      const txt = b.closest(".code").querySelector("pre").innerText;
      navigator.clipboard?.writeText(txt).then(() => { b.textContent = "Copied ✓"; setTimeout(() => (b.textContent = "Copy"), 1500); });
    };
  });

  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/#\/(\d+)/); if (m) go(parseInt(m[1], 10) - 1, false);
  });
  const m = location.hash.match(/#\/(\d+)/);
  go(m ? parseInt(m[1], 10) - 1 : 0, false);

})();
