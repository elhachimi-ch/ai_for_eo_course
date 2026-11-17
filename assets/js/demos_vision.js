/* Deep Learning for Earth Observation — computer-vision fundamentals demos (no dependencies)
   A synthetic Sentinel-2-like scene (fields, river, reservoir, town, road) built from smoothed
   reflectance spectra, and its Sentinel-1-like σ⁰ twin. Demos: pixels & RGB · multispectral /
   hyperspectral spectra and composites · grayscale · thresholding (Otsu) · convolution step by step ·
   SAR speckle filtering (mean, median, Lee). Registered in window.DECK_DEMOS. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const Kit = window.DemoKit || {};
  const rng = Kit.rng || ((seed) => { let s = seed >>> 0 || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); });
  const gauss = Kit.gauss || ((r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); });
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const pick = (root, attr, val) => root.querySelectorAll(`[${attr}]`).forEach((b) => b.classList.toggle("gold", b.getAttribute(attr) === String(val)));
  const INK = "#14213D", MUT = "#5B6478";

  /* ============================================================ spectral library
     smoothed, illustrative surface-reflectance curves (wavelength nm, reflectance) */
  const LIB = [
    { name: "water", pts: [[400, .065], [450, .07], [500, .062], [550, .05], [600, .033], [650, .024], [700, .014], [750, .008], [800, .005], [900, .003], [1000, .002], [2500, .001]] },
    { name: "vegetation", pts: [[400, .03], [450, .035], [500, .045], [550, .105], [600, .07], [650, .045], [680, .038], [700, .08], [720, .19], [750, .37], [800, .45], [900, .47], [1000, .46], [1100, .45], [1200, .41], [1300, .39], [1400, .24], [1450, .17], [1500, .25], [1650, .31], [1800, .26], [1900, .1], [1950, .08], [2050, .13], [2200, .17], [2300, .14], [2500, .08]] },
    { name: "bare soil", pts: [[400, .08], [450, .1], [500, .13], [550, .17], [600, .22], [650, .26], [700, .28], [800, .31], [900, .33], [1000, .35], [1200, .38], [1400, .36], [1450, .32], [1600, .41], [1800, .42], [1900, .36], [2000, .38], [2200, .35], [2300, .32], [2500, .29]] },
    { name: "concrete roof", pts: [[400, .2], [500, .24], [600, .27], [700, .28], [800, .29], [1000, .3], [1400, .29], [1600, .32], [1900, .28], [2200, .3], [2500, .27]] },
    { name: "tile roof", pts: [[400, .07], [500, .08], [550, .1], [600, .2], [650, .29], [700, .32], [800, .34], [1000, .36], [1400, .35], [1600, .41], [1900, .34], [2200, .37], [2500, .33]] },
    { name: "asphalt", pts: [[400, .05], [700, .065], [1000, .08], [1600, .1], [2200, .095], [2500, .09]] },
  ];
  function refl(m, l) {
    const p = LIB[m].pts;
    if (l <= p[0][0]) return p[0][1];
    for (let i = 1; i < p.length; i++) if (l <= p[i][0]) { const [a, va] = p[i - 1], [b, vb] = p[i]; return va + ((vb - va) * (l - a)) / (b - a); }
    return p[p.length - 1][1];
  }
  const bandAvg = (m, lo, hi) => { let s = 0, k = 0; for (let l = lo; l <= hi; l += 2.5) { s += refl(m, l); k++; } return s / k; };
  /* Sentinel-2 MSI: centre and width (nm) */
  const S2 = [[443, 20], [490, 65], [560, 35], [665, 30], [705, 15], [740, 15], [783, 20], [842, 115], [865, 20], [945, 20], [1375, 30], [1610, 90], [2190, 180]];
  const S2N = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"];
  const LIBB = LIB.map((_, m) => S2.map(([c, w]) => bandAvg(m, c - w / 2, c + w / 2)));
  const SIG = [0.004, 0.11, 0.045, 0.25, 0.25, 0.006];      // σ⁰ (linear) per material, C-band VV-like

  /* ============================================================ the scene */
  let SC = null;
  function scene() {
    if (SC) return SC;
    const N = 160, n = N * N, r = rng(2024);
    const comp = new Array(n), tex = new Float32Array(n), rough = new Float32Array(n).fill(1), town = new Uint8Array(n), water = new Float32Array(n);
    const parcels = [];
    (function split(x0, y0, x1, y1) {
      const w = x1 - x0, h = y1 - y0;
      if (w * h < 700 + r() * 900 || (w < 30 && h < 30)) { parcels.push({ x0, y0, x1, y1 }); return; }
      if (w >= h) { const s = x0 + Math.round(w * (0.3 + 0.4 * r())); split(x0, y0, s, y1); split(s, y0, x1, y1); }
      else { const s = y0 + Math.round(h * (0.3 + 0.4 * r())); split(x0, y0, x1, s); split(x0, s, x1, y1); }
    })(0, 0, N, N);
    parcels.forEach((p) => {
      const t = r();
      p.v = t < 0.55 ? 0.45 + 0.55 * r() : t < 0.85 ? 0 : 0.12 + 0.1 * r();
      p.rough = 0.7 + 0.6 * r();
      for (let y = p.y0; y < p.y1; y++) for (let x = p.x0; x < p.x1; x++) {
        const i = y * N + x, v = x === p.x0 || y === p.y0 ? 0.25 : p.v;
        comp[i] = v > 0 ? [[1, v], [2, 1 - v]] : [[2, 1]];
        rough[i] = p.rough;
      }
    });
    const hash = (a, b) => (((Math.imul(a, 73856093) ^ Math.imul(b, 19349663)) >>> 0) % 1024) / 1024;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      // town (top-left): streets, buildings with concrete or tile roofs
      if ((x / 64) ** 2 + (y / 58) ** 2 < 1 + 0.08 * Math.sin(x / 5 + y / 7)) {
        town[i] = 1;
        if (x % 12 < 2 || y % 11 < 2) comp[i] = [[5, 1]];
        else if (x % 12 === 2 || y % 11 === 2) comp[i] = [[2, 0.5], [3, 0.5]];
        else comp[i] = [[hash(Math.floor(x / 12), Math.floor(y / 11)) < 0.6 ? 3 : 4, 1]];
      }
      // national road
      if (Math.abs(160 * (x - 70) - 42 * y) / Math.hypot(160, 42) < 1.3) comp[i] = [[5, 1]];
      tex[i] = 1 + 0.035 * gauss(r);
    }
    // river with riparian vegetation, and a reservoir
    for (let x = 0; x < N; x++) {
      const yc = 104 + 13 * Math.sin(x / 23) + 5 * Math.sin(x / 8.5), hw = 3.2 + 1.2 * Math.sin(x / 31);
      for (let y = 0; y < N; y++) {
        const i = y * N + x, d = Math.abs(y - yc), fw = clamp(hw + 0.5 - d, 0, 1);
        if (fw <= 0 && d < hw + 3.5 && !town[i]) comp[i] = [[1, 0.85], [2, 0.15]];
        if (fw > 0) { comp[i] = [[0, fw], ...comp[i].map(([m, f]) => [m, f * (1 - fw)])]; water[i] = fw; }
      }
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x, q = Math.hypot((x - 126) / 24, (y - 38) / 15), fw = clamp((1 - q) * 14 + 0.5, 0, 1);
      if (fw > 0) { comp[i] = [[0, fw], ...comp[i].filter(([m]) => m !== 0).map(([m, f]) => [m, f * (1 - fw)])]; water[i] = Math.max(water[i], fw); }
    }
    // bands, display RGB, σ⁰
    const B = new Float32Array(n * 13), rgb8 = new Uint8ClampedArray(n * 3), sar0 = new Float32Array(n);
    const disp = (v) => Math.round(255 * Math.pow(clamp(v * 3.4, 0, 1), 0.8));
    for (let i = 0; i < n; i++) {
      for (let b = 0; b < 13; b++) { let s = 0; for (const [m, f] of comp[i]) s += f * LIBB[m][b]; B[i * 13 + b] = s * tex[i]; }
      rgb8[i * 3] = disp(B[i * 13 + 3]); rgb8[i * 3 + 1] = disp(B[i * 13 + 2]); rgb8[i * 3 + 2] = disp(B[i * 13 + 1]);
      let s = 0; for (const [m, f] of comp[i]) s += f * SIG[m] * (m === 1 || m === 2 ? rough[i] : 1);
      if (town[i] && r() < 0.035) s *= 15;                               // double-bounce point targets
      sar0[i] = s;
    }
    // homogeneous field for ENL; a vegetated field for the default spectrum
    const area = (p) => (p.x1 - p.x0) * (p.y1 - p.y0);
    const free = (p) => { for (let y = p.y0; y < p.y1; y++) for (let x = p.x0; x < p.x1; x++) if (water[y * N + x] > 0 || town[y * N + x] || comp[y * N + x].some(([m]) => m === 5)) return false; return true; };
    const big = parcels.filter((p) => area(p) > 500 && free(p)).sort((a, b) => area(b) - area(a));
    const hp = big[0] || parcels[0];
    const homog = { x0: hp.x0 + 4, y0: hp.y0 + 4, x1: hp.x1 - 4, y1: hp.y1 - 4 };
    const vp = big.find((p) => p.v > 0.6) || hp;
    const vegPx = [Math.floor((vp.x0 + vp.x1) / 2), Math.floor((vp.y0 + vp.y1) / 2)];
    const describe = (i) => comp[i].slice().sort((a, b) => b[1] - a[1]).filter(([, f]) => f >= 0.05)
      .map(([m, f]) => `${LIB[m].name} ${Math.round(f * 100)} %`).join(" + ");
    const spectrum = (i, l) => { let s = 0; for (const [m, f] of comp[i]) s += f * refl(m, l); return s * tex[i]; };
    const dom = new Uint8Array(n);   // dominant cover class per pixel
    for (let i = 0; i < n; i++) { let b = comp[i][0]; for (const c of comp[i]) if (c[1] > b[1]) b = c; dom[i] = b[0]; }
    SC = { N, n, comp, B, rgb8, sar0, water, homog, vegPx, describe, spectrum, dom, names: LIB.map((l) => l.name) };
    return SC;
  }
  if (window.DemoKit) window.DemoKit.scene = scene;
  const cssPx = (cv, e, N) => { const b = cv.getBoundingClientRect(); return [clamp(Math.floor(((e.clientX - b.left) / b.width) * N), 0, N - 1), clamp(Math.floor(((e.clientY - b.top) / b.height) * N), 0, N - 1)]; };
  function frameAt(ctx, x, y, h, col) {
    ctx.fillStyle = col; const s = 2 * h + 3;
    ctx.fillRect(x - h - 1, y - h - 1, s, 1); ctx.fillRect(x - h - 1, y + h + 1, s, 1); ctx.fillRect(x - h - 1, y - h - 1, 1, s); ctx.fillRect(x + h + 1, y - h - 1, 1, s);
  }
  function putRGB(ctx, N, f) {     // f(i) -> [r, g, b]
    const id = ctx.createImageData(N, N);
    for (let i = 0; i < N * N; i++) { const c = f(i); id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255; }
    ctx.putImageData(id, 0, 0);
  }

  /* ============================================================ 1 · pixels & RGB */
  DEMOS.pixels = function (root) {
    const S = scene(), N = S.N, cv = root.querySelector("canvas.scene"), ctx = cv.getContext("2d");
    const grid = root.querySelector(".pxgrid"), out = root.querySelector(".readout");
    let ch = "rgb", px = [101, 38];
    grid.innerHTML = Array.from({ length: 25 }, (_, k) => `<div${k === 12 ? ' class="c"' : ""}></div>`).join("");
    const cells = Array.from(grid.children);
    const col = (i) => { const r = S.rgb8[i * 3], g = S.rgb8[i * 3 + 1], b = S.rgb8[i * 3 + 2]; if (ch === "rgb") return [r, g, b]; const v = S.rgb8[i * 3 + +ch]; return [v, v, v]; };
    function paint() {
      putRGB(ctx, N, col);
      frameAt(ctx, px[0], px[1], 2, "#C99A2E");
      cells.forEach((el, k) => {
        const x = clamp(px[0] + (k % 5) - 2, 0, N - 1), y = clamp(px[1] + Math.floor(k / 5) - 2, 0, N - 1), i = y * N + x, c = col(i);
        el.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
        el.style.color = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] > 140 ? INK : "#fff";
        el.innerHTML = ch === "rgb"
          ? `<span>R ${S.rgb8[i * 3]}</span><span>G ${S.rgb8[i * 3 + 1]}</span><span>B ${S.rgb8[i * 3 + 2]}</span>`
          : `<span style="font-size:19px">${S.rgb8[i * 3 + +ch]}</span>`;
      });
      const i = px[1] * N + px[0];
      out.textContent = `pixel (row ${px[1]}, col ${px[0]}) · ${S.describe(i)} · 8-bit (R, G, B) = (${S.rgb8[i * 3]}, ${S.rgb8[i * 3 + 1]}, ${S.rgb8[i * 3 + 2]})`;
    }
    cv.addEventListener("mousemove", (e) => { px = cssPx(cv, e, N); paint(); });
    root.querySelectorAll("[data-ch]").forEach((b) => (b.onclick = () => { ch = b.dataset.ch; pick(root, "data-ch", ch); paint(); }));
    pick(root, "data-ch", ch); paint();
  };

  /* ============================================================ 2 · multispectral / hyperspectral */
  DEMOS.spectra = function (root) {
    const S = scene(), N = S.N, cv = root.querySelector("canvas.scene"), ctx = cv.getContext("2d");
    const sp = root.querySelector("canvas.spec"), sx = sp.getContext("2d"), W = sp.width, H = sp.height, out = root.querySelector(".readout");
    let comp = "true", sensor = "s2", px = S.vegPx.slice();
    const g = (v, k) => Math.round(255 * Math.pow(clamp(v * k, 0, 1), 0.8));
    const STOPS = [[-0.7, [46, 75, 198]], [0, [232, 220, 181]], [0.35, [190, 176, 96]], [0.85, [15, 107, 92]]];
    const ramp = (v) => { if (v <= STOPS[0][0]) return STOPS[0][1]; for (let k = 1; k < STOPS.length; k++) if (v <= STOPS[k][0]) { const [a, ca] = STOPS[k - 1], [b, cb] = STOPS[k], t = (v - a) / (b - a); return ca.map((c, j) => Math.round(c + (cb[j] - c) * t)); } return STOPS[STOPS.length - 1][1]; };
    const band = (i, b) => S.B[i * 13 + b];
    const COMP = {
      true: (i) => [S.rgb8[i * 3], S.rgb8[i * 3 + 1], S.rgb8[i * 3 + 2]],
      nir: (i) => [g(band(i, 7), 2.0), g(band(i, 3), 3.4), g(band(i, 2), 3.4)],
      swir: (i) => [g(band(i, 12), 2.6), g(band(i, 8), 2.0), g(band(i, 3), 3.4)],
      ndvi: (i) => ramp((band(i, 7) - band(i, 3)) / (band(i, 7) + band(i, 3))),
    };
    const L = 52, R = 14, T = 14, Bm = 40;
    const X = (l) => L + ((l - 400) / 2100) * (W - L - R), Y = (v) => H - Bm - (v / 0.6) * (H - T - Bm);
    function drawSpec() {
      const i = px[1] * N + px[0];
      sx.clearRect(0, 0, W, H); sx.fillStyle = "#FDFBF7"; sx.fillRect(0, 0, W, H);
      sx.fillStyle = "rgba(20,33,61,.07)";
      [[1340, 1460], [1790, 1960]].forEach(([a, b]) => sx.fillRect(X(a), T, X(b) - X(a), H - T - Bm));
      sx.font = "11px Inter, Arial, sans-serif"; sx.fillStyle = MUT; sx.textAlign = "center";
      sx.fillText("H₂O", X(1400), T + 12); sx.fillText("H₂O", X(1875), T + 12);
      sx.strokeStyle = "rgba(20,33,61,.1)"; sx.lineWidth = 1; sx.textAlign = "right"; sx.font = "12px Inter, Arial, sans-serif";
      [0, 0.2, 0.4, 0.6].forEach((v) => { sx.beginPath(); sx.moveTo(L, Y(v)); sx.lineTo(W - R, Y(v)); sx.stroke(); sx.fillText(v.toFixed(1), L - 6, Y(v) + 4); });
      sx.textAlign = "center";
      [500, 1000, 1500, 2000, 2500].forEach((l) => sx.fillText(l + " nm", X(l) - (l === 2500 ? 18 : 0), H - 8));
      const gr = sx.createLinearGradient(X(400), 0, X(700), 0);
      ["#7B3FB5", "#2E4BC6", "#1BA0C8", "#0F9B5C", "#D9C232", "#E58A2E", "#C0392B"].forEach((c, k) => gr.addColorStop(k / 6, c));
      sx.fillStyle = gr; sx.fillRect(X(400), H - Bm + 3, X(700) - X(400), 6);
      sx.save(); sx.translate(14, T + 4); sx.rotate(-Math.PI / 2); sx.textAlign = "right"; sx.fillStyle = INK; sx.font = "600 12px Inter, Arial, sans-serif"; sx.fillText("reflectance", 0, 0); sx.restore();
      // the true (continuous) spectrum
      sx.strokeStyle = "rgba(20,33,61,.38)"; sx.lineWidth = 1.6; sx.beginPath();
      for (let l = 400; l <= 2500; l += 5) { const y = Y(S.spectrum(i, l)); l === 400 ? sx.moveTo(X(l), y) : sx.lineTo(X(l), y); }
      sx.stroke();
      let nb;
      if (sensor === "rgb") {
        [[400, 500, "46,75,198"], [500, 600, "15,107,92"], [600, 700, "181,80,46"]].forEach(([a, b, c]) => {
          let s = 0, k = 0; for (let l = a; l <= b; l += 5) { s += S.spectrum(i, l); k++; } const v = s / k;
          sx.fillStyle = `rgba(${c},.25)`; sx.fillRect(X(a), Y(v), X(b) - X(a), Y(0) - Y(v));
          sx.fillStyle = `rgb(${c})`; sx.fillRect(X(a), Y(v) - 2, X(b) - X(a), 4);
        });
        nb = "RGB camera: 3 broad bands, visible only (400–700 nm)";
      } else if (sensor === "s2") {
        S2.forEach(([c, w], b) => {
          const v = band(i, b);
          sx.fillStyle = "rgba(46,75,198,.16)"; sx.fillRect(X(c - w / 2), Y(v), Math.max(2, X(c + w / 2) - X(c - w / 2)), Y(0) - Y(v));
          sx.fillStyle = "#2E4BC6"; sx.fillRect(X(c - w / 2), Y(v) - 2, Math.max(3, X(c + w / 2) - X(c - w / 2)), 4);
          if ([1, 2, 3, 7, 11, 12].includes(b)) { sx.fillStyle = INK; sx.font = "600 11px Inter, Arial, sans-serif"; sx.fillText(S2N[b], X(c), Y(v) - 7); }
        });
        nb = "Sentinel-2 MSI: 13 bands with gaps (443–2190 nm)";
      } else {
        sx.strokeStyle = "#2E4BC6"; sx.lineWidth = 1.4; sx.beginPath(); let k = 0;
        for (let l = 400; l <= 2500; l += 10) { const y = Y(S.spectrum(i, l)); k++ ? sx.lineTo(X(l), y) : sx.moveTo(X(l), y); }
        sx.stroke(); sx.fillStyle = "#2E4BC6";
        for (let l = 400; l <= 2500; l += 10) { sx.beginPath(); sx.arc(X(l), Y(S.spectrum(i, l)), 1.9, 0, 7); sx.fill(); }
        nb = `hyperspectral: ${k} contiguous 10-nm bands (400–2500 nm)`;
      }
      const ndvi = (band(i, 7) - band(i, 3)) / (band(i, 7) + band(i, 3));
      out.textContent = `pixel (row ${px[1]}, col ${px[0]}) · ${S.describe(i)} · ${nb} · NDVI = ${ndvi.toFixed(2)}`;
    }
    function paint() { putRGB(ctx, N, COMP[comp]); frameAt(ctx, px[0], px[1], 1, "#C99A2E"); }
    cv.addEventListener("mousemove", (e) => { px = cssPx(cv, e, N); paint(); drawSpec(); });
    root.querySelectorAll("[data-comp]").forEach((b) => (b.onclick = () => { comp = b.dataset.comp; pick(root, "data-comp", comp); paint(); }));
    root.querySelectorAll("[data-sensor]").forEach((b) => (b.onclick = () => { sensor = b.dataset.sensor; pick(root, "data-sensor", sensor); drawSpec(); }));
    pick(root, "data-comp", comp); pick(root, "data-sensor", sensor); paint(); drawSpec();
  };

  /* ============================================================ 3 · grayscale */
  DEMOS.gray = function (root) {
    const S = scene(), N = S.N, out = root.querySelector(".readout");
    const sc = root.querySelector("canvas.src").getContext("2d"), dc = root.querySelector("canvas.dst").getContext("2d");
    const sl = ["wr", "wg", "wb"].map((k) => root.querySelector(`input[name=${k}]`));
    const PRE = { mean: [1 / 3, 1 / 3, 1 / 3], luma: [0.299, 0.587, 0.114], r: [1, 0, 0], b: [0, 0, 1] };
    function draw() {
      const w = sl.map((s) => +s.value), z = w[0] + w[1] + w[2] || 1, q = w.map((v) => v / z);
      putRGB(dc, N, (i) => { const y = Math.round(q[0] * S.rgb8[i * 3] + q[1] * S.rgb8[i * 3 + 1] + q[2] * S.rgb8[i * 3 + 2]); return [y, y, y]; });
      out.textContent = `Y = ${q[0].toFixed(3)}·R + ${q[1].toFixed(3)}·G + ${q[2].toFixed(3)}·B   (weights normalised to sum to 1)`;
    }
    function preset(name) { PRE[name].forEach((v, k) => (sl[k].value = v)); pick(root, "data-preset", name); draw(); }
    root.querySelectorAll("[data-preset]").forEach((b) => (b.onclick = () => preset(b.dataset.preset)));
    sl.forEach((s) => (s.oninput = () => { pick(root, "data-preset", "—"); draw(); }));
    putRGB(sc, N, (i) => [S.rgb8[i * 3], S.rgb8[i * 3 + 1], S.rgb8[i * 3 + 2]]);
    preset("luma");
  };

  /* ============================================================ 4 · thresholding + Otsu */
  DEMOS.threshold = function (root) {
    const S = scene(), N = S.N, n = S.n, out = root.querySelector(".readout");
    const sc = root.querySelector("canvas.src").getContext("2d"), dc = root.querySelector("canvas.dst").getContext("2d");
    const hc = root.querySelector("canvas.hist"), hx = hc.getContext("2d"), W = hc.width, H = hc.height;
    const tIn = root.querySelector("input[name=t]");
    const band = (i, b) => S.B[i * 13 + b];
    const IN = {
      gray: { lo: 0, hi: 1, t: 0.5, f: (i) => (0.299 * S.rgb8[i * 3] + 0.587 * S.rgb8[i * 3 + 1] + 0.114 * S.rgb8[i * 3 + 2]) / 255 },
      ndwi: { lo: -1, hi: 1, t: 0, f: (i) => (band(i, 2) - band(i, 7)) / (band(i, 2) + band(i, 7)) },
      ndvi: { lo: -1, hi: 1, t: 0.3, f: (i) => (band(i, 7) - band(i, 3)) / (band(i, 7) + band(i, 3)) },
    };
    Object.values(IN).forEach((o) => { o.v = new Float32Array(n); for (let i = 0; i < n; i++) o.v[i] = o.f(i); });
    const NB = 100, L = 10, R = 10, T = 22, Bm = 24;
    const X = (v, o) => L + ((v - o.lo) / (o.hi - o.lo)) * (W - L - R);
    let inp = "ndwi", t = 0;
    function hist(o) { const h = new Float64Array(NB); for (const v of o.v) h[clamp(Math.floor(((v - o.lo) / (o.hi - o.lo)) * NB), 0, NB - 1)]++; return h; }
    function otsu(o) {
      const h = hist(o); let tot = 0, sum = 0; for (let k = 0; k < NB; k++) { tot += h[k]; sum += k * h[k]; }
      let wB = 0, sB = 0, best = -1, bk = 0;
      for (let k = 0; k < NB - 1; k++) {
        wB += h[k]; sB += k * h[k]; const wF = tot - wB; if (!wB || !wF) continue;
        const d = sB / wB - (sum - sB) / wF, s = wB * wF * d * d; if (s > best) { best = s; bk = k; }
      }
      return o.lo + ((bk + 1) / NB) * (o.hi - o.lo);
    }
    function sep(o) {     // σ²_B / σ²_total at the current threshold
      let n0 = 0, n1 = 0, s0 = 0, s1 = 0, s = 0, s2 = 0;
      for (const v of o.v) { s += v; s2 += v * v; if (v > t) { n1++; s1 += v; } else { n0++; s0 += v; } }
      const mu = s / n, vt = s2 / n - mu * mu; if (!n0 || !n1) return 0;
      return ((n0 / n) * (n1 / n) * (s0 / n0 - s1 / n1) ** 2) / vt;
    }
    function draw() {
      const o = IN[inp];
      let mn = Infinity, mx = -Infinity; for (const v of o.v) { if (v < mn) mn = v; if (v > mx) mx = v; }
      putRGB(sc, N, (i) => { const g = Math.round((255 * (o.v[i] - mn)) / (mx - mn)); return [g, g, g]; });
      let ones = 0, inter = 0, uni = 0;
      putRGB(dc, N, (i) => {
        const b = o.v[i] > t, w = S.water[i] > 0.5; ones += b; inter += b && w; uni += b || w;
        return b ? [251, 247, 240] : [20, 33, 61];
      });
      const h = hist(o), hm = Math.max(...h);
      hx.clearRect(0, 0, W, H); hx.fillStyle = "#FDFBF7"; hx.fillRect(0, 0, W, H);
      const bw = (W - L - R) / NB;
      for (let k = 0; k < NB; k++) {
        const v = o.lo + ((k + 0.5) / NB) * (o.hi - o.lo), hh = Math.sqrt(h[k] / hm) * (H - T - Bm);
        hx.fillStyle = v > t ? "#C99A2E" : "#22325A"; hx.fillRect(L + k * bw + 0.5, H - Bm - hh, bw - 1, hh);
      }
      hx.fillStyle = MUT; hx.font = "12px Inter, Arial, sans-serif"; hx.textAlign = "center";
      for (let k = 0; k <= 4; k++) { const v = o.lo + (k / 4) * (o.hi - o.lo); hx.fillText(v.toFixed(o.hi === 1 && o.lo === 0 ? 2 : 1), clamp(X(v, o), 14, W - 14), H - 6); }
      hx.textAlign = "left"; hx.fillText("pixel count (√ scale)", L, 14);
      hx.strokeStyle = "#B5502E"; hx.lineWidth = 3; hx.beginPath(); hx.moveTo(X(t, o), T - 6); hx.lineTo(X(t, o), H - Bm); hx.stroke();
      hx.fillStyle = "#B5502E"; hx.font = "600 13px Inter, Arial, sans-serif"; hx.textAlign = X(t, o) > W - 90 ? "right" : "left";
      hx.fillText(` t = ${t.toFixed(3)} `, X(t, o) + (hx.textAlign === "right" ? -4 : 4), T + 6);
      tIn.min = o.lo; tIn.max = o.hi; tIn.step = (o.hi - o.lo) / 400; tIn.value = t;
      out.textContent = `${{ gray: "grayscale", ndwi: "NDWI", ndvi: "NDVI" }[inp]} > ${t.toFixed(3)} · ${((100 * ones) / n).toFixed(1)} % of pixels = 1 · separability σ²_B/σ² = ${sep(o).toFixed(2)}`
        + (inp === "ndwi" ? ` · IoU with the true water mask ${(inter / (uni || 1)).toFixed(2)}` : "");
    }
    const setFromHist = (e) => { const b = hc.getBoundingClientRect(), o = IN[inp], px = ((e.clientX - b.left) / b.width) * W; t = clamp(o.lo + ((px - L) / (W - L - R)) * (o.hi - o.lo), o.lo, o.hi); draw(); };
    hc.addEventListener("mousedown", (e) => { setFromHist(e); const mv = (ev) => setFromHist(ev), up = () => { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); }; window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up); });
    tIn.oninput = () => { t = +tIn.value; draw(); };
    root.querySelectorAll("[data-in]").forEach((b) => (b.onclick = () => { inp = b.dataset.in; t = IN[inp].t; pick(root, "data-in", inp); draw(); }));
    root.querySelector("[data-act=otsu]").onclick = () => { t = otsu(IN[inp]); draw(); };
    pick(root, "data-in", inp); t = IN[inp].t; draw();
  };

  /* ============================================================ 5 · convolution step by step */
  DEMOS.convstep = function (root) {
    const IMG = [
      [1, 1, 1, 1, 8, 8, 8, 8], [1, 1, 1, 1, 8, 8, 8, 8], [1, 1, 1, 1, 8, 8, 8, 8], [1, 1, 1, 1, 8, 8, 8, 8],
      [1, 9, 9, 1, 8, 8, 8, 8], [1, 9, 9, 1, 8, 8, 8, 8], [1, 1, 1, 1, 8, 8, 8, 8], [1, 1, 1, 1, 8, 8, 8, 8]];
    const KS = {
      sobelX: { k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], s: 1, lab: "" },
      blur: { k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], s: 1 / 9, lab: "× 1/9" },
      sharpen: { k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], s: 1, lab: "" },
      laplace: { k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], s: 1, lab: "" },
    };
    const slide = root.closest(".slide");
    const gin = root.querySelector(".cgrid.in"), gk = root.querySelector(".cgrid.k"), gout = root.querySelector(".cgrid.out");
    const calc = root.querySelector(".calc"), osz = slide.querySelector(".osz"), outL = root.querySelector(".out-l"), kLab = root.querySelector(".klab");
    let kn = "sobelX", Pd = 0, St = 1, pos = 0, timer = null, Wi, Wo, maxAbs;
    const val = (r, c) => (r < 0 || c < 0 || r >= 8 || c >= 8 ? 0 : IMG[r][c]);
    const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
    function conv(i, j) {
      const K = KS[kn], terms = []; let s = 0;
      for (let m = 0; m < 3; m++) for (let q = 0; q < 3; q++) { const a = val(i * St + m - Pd, j * St + q - Pd), b = K.k[m][q]; s += a * b; terms.push(`${a}·${b < 0 ? `(${b})` : b}`); }
      return { v: s * K.s, terms, raw: s };
    }
    const shade = (v) => { const t = v / 9, c = [253 + (34 - 253) * t, 251 + (50 - 251) * t, 247 + (90 - 247) * t].map(Math.round); return `rgb(${c})`; };
    function build() {
      clearInterval(timer); pos = 0;
      Wi = 8 + 2 * Pd; Wo = Math.floor((8 - 3 + 2 * Pd) / St) + 1;
      gin.style.gridTemplateColumns = `repeat(${Wi}, 32px)`;
      let h = "";
      for (let r = 0; r < Wi; r++) for (let c = 0; c < Wi; c++) {
        const rr = r - Pd, cc = c - Pd, pad = rr < 0 || cc < 0 || rr >= 8 || cc >= 8, v = val(rr, cc);
        h += `<div class="${pad ? "pad" : ""}" style="background:${pad ? "transparent" : shade(v)};color:${v > 5 ? "#fff" : INK}">${v}</div>`;
      }
      gin.innerHTML = h;
      gk.innerHTML = KS[kn].k.flat().map((v) => `<div class="kc">${v}</div>`).join("");
      gout.style.gridTemplateColumns = `repeat(${Wo}, 32px)`;
      gout.innerHTML = Array.from({ length: Wo * Wo }, () => "<div></div>").join("");
      maxAbs = 1e-9; for (let p = 0; p < Wo * Wo; p++) maxAbs = Math.max(maxAbs, Math.abs(conv(Math.floor(p / Wo), p % Wo).v));
      outL.textContent = `output (${Wo} × ${Wo})`;
      if (osz) osz.textContent = `here: ⌊(8 − 3 + 2·${Pd}) / ${St}⌋ + 1 = ${Wo}`;
      kLab.textContent = KS[kn].lab;
      calc.textContent = "Step: the 3 × 3 window slides over the input; each output value = sum of element-wise products.";
      pick(root, "data-k", kn); pick(root, "data-p", Pd); pick(root, "data-s", St);
    }
    function show(p) {
      const i = Math.floor(p / Wo), j = p % Wo, r = conv(i, j), cell = gout.children[p], t = r.v / maxAbs;
      cell.textContent = fmt(r.v);
      cell.style.background = t >= 0 ? `rgba(181,80,46,${(Math.abs(t) * 0.85).toFixed(3)})` : `rgba(46,75,198,${(Math.abs(t) * 0.85).toFixed(3)})`;
      cell.style.color = Math.abs(t) > 0.55 ? "#fff" : INK;
      gin.querySelectorAll(".win").forEach((e) => e.classList.remove("win"));
      for (let m = 0; m < 3; m++) for (let q = 0; q < 3; q++) gin.children[(i * St + m) * Wi + (j * St + q)].classList.add("win");
      gout.querySelectorAll(".cur").forEach((e) => e.classList.remove("cur")); cell.classList.add("cur");
      calc.textContent = `out[${i},${j}] = ${r.terms.join(" + ")}${KS[kn].s !== 1 ? ` = ${r.raw}, × 1/9` : ""} = ${fmt(r.v)}`;
    }
    const step = () => { if (pos < Wo * Wo) show(pos++); };
    root.querySelector("[data-act=step]").onclick = () => { clearInterval(timer); step(); };
    root.querySelector("[data-act=play]").onclick = () => { clearInterval(timer); timer = setInterval(() => { if (pos >= Wo * Wo) clearInterval(timer); else step(); }, 380); };
    root.querySelector("[data-act=all]").onclick = () => { clearInterval(timer); while (pos < Wo * Wo) step(); };
    root.querySelector("[data-act=reset]").onclick = build;
    root.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { kn = b.dataset.k; build(); }));
    root.querySelectorAll("[data-p]").forEach((b) => (b.onclick = () => { Pd = +b.dataset.p; build(); }));
    root.querySelectorAll("[data-s]").forEach((b) => (b.onclick = () => { St = +b.dataset.s; build(); }));
    build(); for (let k = 0; k < 9; k++) step();
  };

  /* ============================================================ 6 · SAR speckle filtering */
  DEMOS.sar = function (root) {
    const S = scene(), N = S.N, n = S.n, out = root.querySelector(".readout");
    const sc = root.querySelector("canvas.src").getContext("2d"), dc = root.querySelector("canvas.dst").getContext("2d");
    const winIn = root.querySelector("input[name=win]"), wv = root.querySelector(".wv"), maskCb = root.querySelector("input[name=mask]");
    const lin = root.querySelector(".lin"), lout = root.querySelector(".lout"), tBtn = root.querySelector("[data-act=truth]");
    const TH = -18;
    let L = 1, f = "lee", truth = false, I = null;
    function speckle() {
      const r = rng(99 + 7 * L); I = new Float32Array(n);
      for (let i = 0; i < n; i++) { let g = 0; for (let k = 0; k < L; k++) { let u = r(); while (u === 0) u = r(); g -= Math.log(u); } I[i] = (S.sar0[i] * g) / L; }
    }
    function filter(win) {
      if (f === "none") return I;
      const h = (win - 1) / 2, o = new Float32Array(n);
      if (f === "median") {
        const buf = new Float32Array(win * win);
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          let k = 0;
          for (let yy = Math.max(0, y - h); yy <= Math.min(N - 1, y + h); yy++) for (let xx = Math.max(0, x - h); xx <= Math.min(N - 1, x + h); xx++) buf[k++] = I[yy * N + xx];
          const s = buf.subarray(0, k).sort(); o[y * N + x] = k % 2 ? s[k >> 1] : (s[(k >> 1) - 1] + s[k >> 1]) / 2;
        }
        return o;
      }
      const Wd = N + 1, s1 = new Float64Array(Wd * Wd), s2 = new Float64Array(Wd * Wd);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const v = I[y * N + x], a = (y + 1) * Wd + x + 1;
        s1[a] = v + s1[a - Wd] + s1[a - 1] - s1[a - Wd - 1]; s2[a] = v * v + s2[a - Wd] + s2[a - 1] - s2[a - Wd - 1];
      }
      const Cu2 = 1 / L;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const x0 = Math.max(0, x - h), x1 = Math.min(N - 1, x + h) + 1, y0 = Math.max(0, y - h), y1 = Math.min(N - 1, y + h) + 1, c = (x1 - x0) * (y1 - y0);
        const sum = s1[y1 * Wd + x1] - s1[y0 * Wd + x1] - s1[y1 * Wd + x0] + s1[y0 * Wd + x0];
        const sq = s2[y1 * Wd + x1] - s2[y0 * Wd + x1] - s2[y1 * Wd + x0] + s2[y0 * Wd + x0];
        const m = sum / c, v = Math.max(0, sq / c - m * m), i = y * N + x;
        if (f === "mean") o[i] = m;
        else { const Ci2 = v / (m * m + 1e-20), k = Ci2 > 0 ? clamp((1 - Cu2 / Ci2) / (1 + Cu2), 0, 1) : 0; o[i] = m + k * (I[i] - m); }
      }
      return o;
    }
    const db = (v) => 10 * Math.log10(Math.max(v, 1e-6));
    function paint(ctx, a, mask) {
      putRGB(ctx, N, (i) => {
        const d = db(a[i]), g = Math.round(255 * clamp((d + 26) / 28, 0, 1));
        return mask && d < TH ? [Math.round(g * 0.3 + 46 * 0.7), Math.round(g * 0.3 + 75 * 0.7), Math.round(g * 0.3 + 198 * 0.7)] : [g, g, g];
      });
      const H = S.homog; ctx.save(); ctx.strokeStyle = "#C99A2E"; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.strokeRect(H.x0 + 0.5, H.y0 + 0.5, H.x1 - H.x0 - 1, H.y1 - H.y0 - 1); ctx.restore();
    }
    function enl(a) {
      const H = S.homog; let s = 0, s2 = 0, c = 0;
      for (let y = H.y0; y < H.y1; y++) for (let x = H.x0; x < H.x1; x++) { const v = a[y * N + x]; s += v; s2 += v * v; c++; }
      const m = s / c; return (m * m) / Math.max(1e-20, s2 / c - m * m);
    }
    function iou(a) { let it = 0, un = 0; for (let i = 0; i < n; i++) { const p = db(a[i]) < TH, w = S.water[i] > 0.5; it += p && w; un += p || w; } return it / (un || 1); }
    function draw() {
      const win = +winIn.value; wv.textContent = `${win} × ${win}`;
      const o = filter(win), mask = maskCb.checked;
      paint(sc, truth ? S.sar0 : I, mask); paint(dc, o, mask);
      lin.textContent = truth ? "noise-free σ⁰ (dB)" : `speckled σ⁰ (dB), L = ${L}`;
      const fname = { none: "no filter", mean: `mean ${win}×${win}`, median: `median ${win}×${win}`, lee: `Lee ${win}×${win}` }[f];
      lout.textContent = `filtered: ${fname}`;
      tBtn.classList.toggle("gold", truth);
      out.textContent = `ENL in the dashed field: ${enl(I).toFixed(1)} → ${enl(o).toFixed(1)}`
        + (mask ? ` · water mask (σ⁰ < ${TH} dB) IoU: ${iou(I).toFixed(2)} → ${iou(o).toFixed(2)}` : " · tick the water mask to threshold the result");
    }
    root.querySelectorAll("[data-looks]").forEach((b) => (b.onclick = () => { L = +b.dataset.looks; pick(root, "data-looks", L); speckle(); draw(); }));
    root.querySelectorAll("[data-f]").forEach((b) => (b.onclick = () => { f = b.dataset.f; pick(root, "data-f", f); draw(); }));
    winIn.oninput = draw; maskCb.onchange = draw;
    tBtn.onclick = () => { truth = !truth; draw(); };
    pick(root, "data-looks", L); pick(root, "data-f", f); speckle(); draw();
  };
})();
