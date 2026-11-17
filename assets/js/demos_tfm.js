/* AI for Earth Observation — tabular foundation model demos (no dependencies)
   In-context learning as Bayesian model averaging under a prior · TabArena / BeyondArena leaderboard explorer ·
   proper scoring rules and how they re-rank probabilistic regressors.
   Needs demos_ml.js (DemoKit) and demos_eo.js (axes, label) loaded first. */
(function () {
  "use strict";
  const DEMOS = (window.DECK_DEMOS = window.DECK_DEMOS || {});
  const K = window.DemoKit;
  const { rng, gauss, clamp, pick, axes, label } = K;
  const INK = "#14213D", MUT = "#5B6478", TER = "#B5502E", BLU = "#2E4BC6", GRN = "#0F6B5C";
  const FONT = "Inter, Arial, sans-serif";
  const $ = (root, s) => root.querySelector(s);
  const $$ = (root, s) => Array.from(root.querySelectorAll(s));
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const phi = (z) => Math.exp(-0.5 * z * z) / SQ2PI;
  function Phi(z) { // Abramowitz & Stegun 7.1.26
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const e = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + e) : 0.5 * (1 - e);
  }
  /* E|X| for X ~ N(m, s²) */
  const absMean = (m, s) => s * 2 * phi(m / s) + m * (2 * Phi(m / s) - 1);
  /* CRPS of N(mu, s²) at y (closed form) */
  const crpsN = (mu, s, y) => { const z = (y - mu) / s; return s * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / Math.sqrt(Math.PI)); };
  const canvasXY = (cv, e) => { const b = cv.getBoundingClientRect(); return [((e.clientX - b.left) / b.width) * cv.width, ((e.clientY - b.top) / b.height) * cv.height]; };

  /* ======================================================= 1 · in-context learning = Bayesian model averaging under p(D) */
  DEMOS.icl = function (root) {
    const cv = $(root, "canvas.fx"), ctx = cv.getContext("2d"), wc = $(root, "canvas.w"), wx = wc.getContext("2d"), out = $(root, ".readout");
    const nIn = $(root, "input[name=n]"), nv = $(root, ".nv"), drawsIn = $(root, "input[name=draws]");
    const TRUTH = {
      smooth: (x) => 0.8 * Math.sin(6.5 * x) + 0.3 * Math.cos(13 * x),
      step: (x) => (x < 0.5 ? -0.65 : 0.75) + 0.15 * x,
      sat: (x) => 1.2 * (1 - Math.exp(-5 * x)) - 0.45,
    };
    const NOISE = 0.12, SF2 = 0.8;
    const LS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8], SN = [0.05, 0.12, 0.3];
    const PRIORS = {
      broad: { ls: LS, sn: SN, txt: "broad prior: rough and smooth functions, low and high noise" },
      smooth: { ls: [0.4, 0.8], sn: [0.05, 0.12], txt: "narrow prior: smooth functions only (ℓ ≥ 0.4), low noise" },
    };
    let truth = "step", prior = "broad", seed = 3, pool = [], extra = [], test = [], draws = [];
    const XS = Array.from({ length: 161 }, (_, i) => i / 160);

    function makeData() {
      const r = rng(seed * 97 + 5), f = TRUTH[truth];
      pool = Array.from({ length: 60 }, () => { const x = 0.02 + 0.96 * r(); return { x, y: f(x) + NOISE * gauss(r) }; });
      const rt = rng(seed * 13 + 77);
      test = Array.from({ length: 300 }, () => { const x = rt(); return { x, y: f(x) + NOISE * gauss(rt) }; });
    }
    function makeDraws() {
      const P = PRIORS[prior], r = rng(seed * 7 + 11); draws = [];
      for (let k = 0; k < 7; k++) {
        const l = P.ls[Math.floor(r() * P.ls.length)], D = 80, W = [], B = [];
        for (let j = 0; j < D; j++) { W.push(gauss(r) / l); B.push(2 * Math.PI * r()); }
        draws.push(XS.map((x) => { let s = 0; for (let j = 0; j < D; j++) s += Math.cos(W[j] * x + B[j]); return Math.sqrt(SF2) * Math.sqrt(2 / D) * s; }));
      }
    }
    /* exact GP posterior for one hypothesis φ = (ℓ, σn) */
    function gp(ctxPts, l, sn) {
      const n = ctxPts.length, kf = (a, b) => SF2 * Math.exp(-((a - b) ** 2) / (2 * l * l));
      if (!n) return { lml: 0, pred: (x) => [0, SF2 + sn * sn] };
      const L = Array.from({ length: n }, () => new Float64Array(n));
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
        let s = kf(ctxPts[i].x, ctxPts[j].x) + (i === j ? sn * sn + 1e-8 : 0);
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
      }
      const fw = (b) => { const v = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * v[k]; v[i] = s / L[i][i]; } return v; };
      const bw = (b) => { const v = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * v[k]; v[i] = s / L[i][i]; } return v; };
      const y = ctxPts.map((p) => p.y), a1 = fw(y), alpha = bw(a1);
      let lml = 0; for (let i = 0; i < n; i++) lml += -0.5 * a1[i] * a1[i] - Math.log(L[i][i]) - 0.5 * Math.log(2 * Math.PI);
      return { lml, pred: (x) => { const ks = ctxPts.map((p) => kf(x, p.x)); let m = 0; for (let i = 0; i < n; i++) m += ks[i] * alpha[i]; const v = fw(ks); let q = 0; for (let i = 0; i < n; i++) q += v[i] * v[i]; return [m, Math.max(SF2 - q, 1e-9) + sn * sn]; } };
    }
    let comps = [], best = null;
    function fit() {
      const ctxPts = pool.slice(0, +nIn.value).concat(extra), P = PRIORS[prior];
      nv.textContent = +nIn.value + extra.length;
      const H = []; P.ls.forEach((l) => P.sn.forEach((sn) => H.push({ l, sn, g: gp(ctxPts, l, sn) })));
      const mx = Math.max(...H.map((h) => h.g.lml));
      H.forEach((h) => (h.w = Math.exp(h.g.lml - mx))); const tw = H.reduce((a, h) => a + h.w, 0); H.forEach((h) => (h.w /= tw));
      best = H.reduce((a, h) => (h.w > a.w ? h : a), H[0]);
      comps = H.filter((h) => h.w > 1e-4); const t2 = comps.reduce((a, h) => a + h.w, 0); comps.forEach((h) => (h.w /= t2));
      return ctxPts;
    }
    const mixAt = (x) => comps.map((h) => { const [m, v] = h.g.pred(x); return { w: h.w, m, s: Math.sqrt(v) }; });
    const mixQ = (C, q) => {
      let lo = Math.min(...C.map((c) => c.m - 5 * c.s)), hi = Math.max(...C.map((c) => c.m + 5 * c.s));
      for (let it = 0; it < 40; it++) { const md = (lo + hi) / 2; let F = 0; C.forEach((c) => (F += c.w * Phi((md - c.m) / c.s))); if (F < q) lo = md; else hi = md; }
      return (lo + hi) / 2;
    };
    const mixCRPS = (C, y) => {
      let a = 0, b = 0; C.forEach((c) => (a += c.w * absMean(c.m - y, c.s)));
      C.forEach((c) => C.forEach((d) => (b += c.w * d.w * absMean(c.m - d.m, Math.hypot(c.s, d.s)))));
      return a - 0.5 * b;
    };
    function draw() {
      const ctxPts = fit(), W = cv.width, H = cv.height; ctx.clearRect(0, 0, W, H);
      const A = axes(ctx, 56, 12, W - 72, H - 58, { x: [0, 1], y: [-2, 2], xt: [0, 0.25, 0.5, 0.75, 1], yt: [-2, -1, 0, 1, 2], xl: "feature x (e.g. NDWI) →", yl: "target y" });
      ctx.save(); ctx.beginPath(); ctx.rect(A.X(0), A.Y(2), A.X(1) - A.X(0), A.Y(-2) - A.Y(2)); ctx.clip();
      if (drawsIn.checked) draws.forEach((d) => { ctx.beginPath(); XS.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(d[i]))); ctx.strokeStyle = "rgba(91,100,120,.35)"; ctx.lineWidth = 1.2; ctx.stroke(); });
      const M = XS.map((x) => { const C = mixAt(x); let m = 0; C.forEach((c) => (m += c.w * c.m)); return { m, lo: mixQ(C, 0.05), hi: mixQ(C, 0.95) }; });
      ctx.beginPath(); XS.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(M[i].hi))); for (let i = XS.length - 1; i >= 0; i--) ctx.lineTo(A.X(XS[i]), A.Y(M[i].lo)); ctx.closePath();
      ctx.fillStyle = "rgba(46,75,198,.16)"; ctx.fill();
      ctx.beginPath(); XS.forEach((x) => ctx.lineTo(A.X(x), A.Y(TRUTH[truth](x)))); ctx.setLineDash([6, 4]); ctx.strokeStyle = GRN; ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); XS.forEach((x, i) => ctx.lineTo(A.X(x), A.Y(M[i].m))); ctx.strokeStyle = BLU; ctx.lineWidth = 2.5; ctx.stroke();
      ctxPts.forEach((p, i) => { ctx.beginPath(); ctx.arc(A.X(p.x), A.Y(p.y), 5, 0, 7); ctx.fillStyle = i >= ctxPts.length - extra.length ? "#C99A2E" : TER; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); });
      ctx.restore();
      label(ctx, "— posterior predictive mean   shaded: 90 % band   - - true function   ● context rows", A.X(0.01), 26, MUT, "left", `12px ${FONT}`);
      /* posterior over length-scales */
      const P = PRIORS[prior], ww = wc.width, wh = wc.height; wx.clearRect(0, 0, ww, wh);
      const ag = LS.map((l) => (P.ls.includes(l) ? comps.filter((h) => h.l === l).reduce((a, h) => a + h.w, 0) : null));
      const B = axes(wx, 44, 30, ww - 58, wh - 82, { x: [0, LS.length], y: [0, 1], yt: [0, 0.5, 1], fy: (v) => v.toFixed(1), xl: "length-scale ℓ of φ" });
      label(wx, "p(φ | D_train)", 8, 18, INK, "left", `600 13px ${FONT}`);
      LS.forEach((l, i) => {
        const x0 = B.X(i + 0.18), bw = B.X(i + 0.82) - x0;
        wx.fillStyle = MUT; wx.font = `10px ${FONT}`; wx.textAlign = "center"; wx.fillText(String(l).replace(/^0\./, "."), B.X(i + 0.5), B.Y(0) + 16);
        if (ag[i] === null) { wx.fillStyle = "rgba(91,100,120,.10)"; wx.fillRect(x0, B.Y(1), bw, B.Y(0) - B.Y(1)); wx.save(); wx.translate(B.X(i + 0.5), B.Y(0.5)); wx.rotate(-Math.PI / 2); wx.fillStyle = MUT; wx.font = `11px ${FONT}`; wx.fillText("not in prior", 0, 4); wx.restore(); return; }
        const h = Math.max(ag[i], 0.004); wx.fillStyle = BLU; wx.beginPath(); wx.roundRect ? wx.roundRect(x0, B.Y(h), bw, B.Y(0) - B.Y(h), [4, 4, 0, 0]) : wx.rect(x0, B.Y(h), bw, B.Y(0) - B.Y(h)); wx.fill();
      });
      /* scores on 300 new points from the same truth */
      let crB = 0, crS = 0, cvB = 0, cvS = 0;
      test.forEach((t) => {
        const C = mixAt(t.x); crB += mixCRPS(C, t.y);
        const lo = mixQ(C, 0.05), hi = mixQ(C, 0.95); if (t.y >= lo && t.y <= hi) cvB++;
        const [m, v] = best.g.pred(t.x), s = Math.sqrt(v); crS += crpsN(m, s, t.y); if (Math.abs(t.y - m) <= 1.645 * s) cvS++;
      });
      const nT = test.length;
      out.innerHTML = `${PRIORS[prior].txt} · <b>model average</b>: CRPS ${(crB / nT).toFixed(3)}, 90 % band covers ${Math.round((100 * cvB) / nT)} % of new points · `
        + `<b>single best φ</b> (ℓ = ${best.l}, σ = ${best.sn}): CRPS ${(crS / nT).toFixed(3)}, covers ${Math.round((100 * cvS) / nT)} % · click the plot to add a point`;
    }
    root.querySelectorAll("[data-truth]").forEach((b) => (b.onclick = () => { truth = b.dataset.truth; pick(root, "data-truth", truth); extra = []; makeData(); draw(); }));
    root.querySelectorAll("[data-prior]").forEach((b) => (b.onclick = () => { prior = b.dataset.prior; pick(root, "data-prior", prior); makeDraws(); draw(); }));
    $(root, "[data-act=new]").onclick = () => { seed++; extra = []; makeData(); makeDraws(); draw(); };
    nIn.oninput = draw; drawsIn.onchange = draw;
    cv.addEventListener("click", (e) => {
      const [px, py] = canvasXY(cv, e), x = (px - 56) / (cv.width - 72), y = 2 - (4 * (py - 12)) / (cv.height - 58);
      if (x >= 0 && x <= 1 && y >= -2 && y <= 2 && extra.length < 30) { extra.push({ x, y }); draw(); }
    });
    pick(root, "data-truth", truth); pick(root, "data-prior", prior);
    makeData(); makeDraws(); draw();
  };

  /* ======================================================= 2 · TabArena / BeyondArena leaderboard explorer */
  /* Snapshot of https://huggingface.co/spaces/TabArena/leaderboard (downloaded 17 Nov 2026), models-only entrants,
     imputation on. Row: [model, config (d = default, te = tuned + ensembled), family, Elo, CI low, CI high,
     median train time s/1K rows, % datasets imputed]. TabArena: RandomForest (default) = 1000; BeyondArena: XGBoost (default) = 1000. */
  const ARENA = {"all":[["LimiX-2-1016","d","F",1998,1911,2138,31.24,0.0],["Kumo-Tabular","d","F",1971,1873,2129,2.39,0.0],["Seldon","d","F",1904,1805,2054,1.78,0.0],["TabPFN-3.5","d","F",1867,1802,1965,1.84,0.0],["TabFM","d","F",1791,1693,1892,16.12,0.0],["Mitra-v2","d","F",1767,1701,1858,57.14,0.0],["EXAONE-Tabular","d","F",1746,1698,1814,2.0,0.0],["TabPFN-3","d","F",1630,1584,1689,1.29,0.0],["TabICLv2","d","F",1567,1517,1621,0.75,0.0],["RealTabPFN-2.5","te","F",1566,1523,1620,2040.22,0.0],["TabDPT-1.3","d","F",1522,1484,1580,0.57,0.0],["RealTabPFN-2.5","d","F",1504,1465,1553,5.81,0.0],["RealMLP","te","N",1481,1439,1522,2950.72,0.0],["TabM","te","N",1427,1392,1469,2450.13,0.0],["LightGBM","te","T",1411,1383,1437,417.05,0.0],["CatBoost","te","T",1401,1365,1435,1346.21,0.0],["CatBoost","d","T",1362,1324,1401,5.88,0.0],["XGBoost","te","T",1360,1331,1387,693.49,0.0],["TabM","d","N",1285,1245,1326,7.49,0.0],["SAP-RPT-OSS","d","F",1272,1210,1337,14.11,0.0],["TabPFNv2","d","F",1252,1186,1316,3.36,35.29],["RealMLP","d","N",1218,1178,1254,10.06,0.0],["XGBoost","d","T",1208,1170,1240,1.94,0.0],["LightGBM","d","T",1183,1151,1218,1.96,0.0],["TabSTAR-v1.1","d","F",1181,1095,1249,40.41,0.0],["RandomForest","d","T",1000,947,1043,0.43,0.0]],"small":[["LimiX-2-1016","d","F",1921,1841,2065,29.9,0.0],["Kumo-Tabular","d","F",1898,1812,2026,2.4,0.0],["Seldon","d","F",1831,1747,1967,1.95,0.0],["TabPFN-3.5","d","F",1791,1738,1887,2.02,0.0],["TabFM","d","F",1763,1669,1908,14.32,0.0],["Mitra-v2","d","F",1728,1668,1837,74.62,0.0],["EXAONE-Tabular","d","F",1706,1655,1771,1.85,0.0],["TabPFN-3","d","F",1612,1564,1674,1.91,0.0],["RealTabPFN-2.5","te","F",1576,1525,1634,2289.39,0.0],["TabICLv2","d","F",1561,1490,1652,0.96,0.0],["RealTabPFN-2.5","d","F",1528,1482,1592,6.76,0.0],["TabDPT-1.3","d","F",1512,1452,1575,0.7,0.0],["RealMLP","te","N",1460,1416,1505,3771.85,0.0],["TabM","te","N",1412,1368,1451,2585.58,0.0],["LightGBM","te","T",1377,1340,1405,892.49,0.0],["CatBoost","te","T",1354,1314,1385,1739.17,0.0],["TabPFNv2","d","F",1338,1270,1404,4.54,8.33],["XGBoost","te","T",1316,1280,1347,884.48,0.0],["SAP-RPT-OSS","d","F",1315,1261,1384,16.23,0.0],["CatBoost","d","T",1309,1272,1342,8.46,0.0],["TabM","d","N",1288,1234,1338,9.94,0.0],["RealMLP","d","N",1215,1172,1256,15.69,0.0],["TabSTAR-v1.1","d","F",1203,1126,1268,42.38,0.0],["LightGBM","d","T",1163,1114,1211,3.61,0.0],["XGBoost","d","T",1160,1122,1188,3.29,0.0],["RandomForest","d","T",1000,924,1057,0.51,0.0]],"medium":[["LimiX-2-1016","d","F",2460,2350,2693,32.93,0.0],["Kumo-Tabular","d","F",2404,2284,2666,2.09,0.0],["Seldon","d","F",2323,2258,2494,1.4,0.0],["TabPFN-3.5","d","F",2307,2214,2528,1.05,0.0],["Mitra-v2","d","F",1997,1897,2167,43.41,0.0],["TabFM","d","F",1977,1845,2257,25.96,0.0],["EXAONE-Tabular","d","F",1971,1919,2089,3.84,0.0],["TabPFN-3","d","F",1758,1642,1946,0.56,0.0],["TabICLv2","d","F",1654,1560,1810,0.41,0.0],["TabDPT-1.3","d","F",1617,1492,1775,0.44,0.0],["RealTabPFN-2.5","te","F",1610,1527,1710,735.58,0.0],["RealMLP","te","N",1606,1516,1697,1719.82,0.0],["CatBoost","te","T",1596,1506,1697,489.93,0.0],["CatBoost","d","T",1566,1451,1689,2.47,0.0],["LightGBM","te","T",1565,1512,1621,131.56,0.0],["XGBoost","te","T",1533,1466,1613,282.13,0.0],["TabM","te","N",1523,1442,1636,1236.88,0.0],["RealTabPFN-2.5","d","F",1505,1442,1607,1.88,0.0],["XGBoost","d","T",1360,1252,1443,0.49,0.0],["TabM","d","N",1313,1228,1401,2.13,0.0],["LightGBM","d","T",1253,1193,1289,0.29,0.0],["RealMLP","d","N",1252,1179,1318,6.75,0.0],["SAP-RPT-OSS","d","F",1183,1079,1265,11.44,0.0],["TabSTAR-v1.1","d","F",1137,1027,1213,26.72,0.0],["RandomForest","d","T",1000,904,1055,0.2,0.0]],"clf":[["LimiX-2-1016","d","F",1960,1856,2130,22.78,0.0],["Kumo-Tabular","d","F",1928,1822,2080,2.79,0.0],["Seldon","d","F",1884,1775,2045,2.09,0.0],["TabPFN-3.5","d","F",1847,1779,1952,1.85,0.0],["TabFM","d","F",1785,1669,1910,18.45,0.0],["Mitra-v2","d","F",1760,1678,1876,72.59,0.0],["EXAONE-Tabular","d","F",1754,1702,1833,1.53,0.0],["TabPFN-3","d","F",1625,1561,1696,1.28,0.0],["TabICLv2","d","F",1572,1510,1646,0.93,0.0],["RealTabPFN-2.5","te","F",1557,1497,1618,2046.25,0.0],["RealTabPFN-2.5","d","F",1524,1472,1577,5.76,0.0],["TabDPT-1.3","d","F",1518,1461,1590,0.62,0.0],["RealMLP","te","N",1466,1424,1512,2879.46,0.0],["TabM","te","N",1443,1404,1491,2168.86,0.0],["LightGBM","te","T",1414,1383,1456,382.09,0.0],["CatBoost","te","T",1400,1354,1448,1107.0,0.0],["CatBoost","d","T",1377,1333,1422,5.41,0.0],["XGBoost","te","T",1364,1317,1403,685.87,0.0],["TabM","d","N",1304,1259,1356,6.85,0.0],["SAP-RPT-OSS","d","F",1292,1219,1348,13.7,0.0],["TabPFNv2","d","F",1286,1205,1363,3.37,31.58],["RealMLP","d","N",1230,1186,1272,10.48,0.0],["XGBoost","d","T",1222,1176,1260,1.77,0.0],["TabSTAR-v1.1","d","F",1206,1130,1278,36.43,0.0],["LightGBM","d","T",1183,1139,1238,1.79,0.0],["RandomForest","d","T",1000,938,1064,0.38,0.0]],"reg":[["Kumo-Tabular","d","F",2417,2235,2679,2.04,0.0],["LimiX-2-1016","d","F",2406,2222,2720,37.45,0.0],["Seldon","d","F",2198,2112,2361,1.14,0.0],["TabPFN-3.5","d","F",2157,2041,2362,1.57,0.0],["TabFM","d","F",2004,1931,2150,14.28,0.0],["Mitra-v2","d","F",1983,1883,2151,43.41,0.0],["EXAONE-Tabular","d","F",1890,1816,2004,2.4,0.0],["TabPFN-3","d","F",1803,1691,1983,1.34,0.0],["RealTabPFN-2.5","te","F",1748,1668,1863,1709.05,0.0],["TabICLv2","d","F",1693,1567,1897,0.66,0.0],["TabDPT-1.3","d","F",1665,1551,1860,0.47,0.0],["RealMLP","te","N",1661,1611,1739,3995.01,0.0],["RealTabPFN-2.5","d","F",1556,1450,1666,7.04,0.0],["CatBoost","te","T",1505,1446,1580,2229.21,0.0],["LightGBM","te","T",1500,1432,1567,700.19,0.0],["TabM","te","N",1471,1395,1574,3013.86,0.0],["XGBoost","te","T",1428,1376,1484,834.93,0.0],["CatBoost","d","T",1399,1303,1478,8.73,0.0],["TabM","d","N",1292,1193,1399,9.19,0.0],["SAP-RPT-OSS","d","F",1277,1119,1434,20.24,0.0],["RealMLP","d","N",1228,1138,1328,8.9,0.0],["LightGBM","d","T",1218,1169,1245,2.11,0.0],["TabPFNv2","d","F",1198,1028,1326,2.8,46.15],["XGBoost","d","T",1196,1099,1285,2.24,0.0],["TabSTAR-v1.1","d","F",1141,995,1250,48.29,0.0],["RandomForest","d","T",1000,905,1051,0.53,0.0]],"iid":[["Kumo-Tabular","d","F",1576,1532,1639,7.44,0.0],["Seldon","d","F",1518,1476,1577,4.09,0.0],["TabPFN-3.5","d","F",1499,1467,1537,4.02,0.0],["LimiX-2-1016","d","F",1484,1429,1566,51.18,5.83],["TabFM","d","F",1406,1354,1460,41.11,5.83],["EXAONE-Tabular","d","F",1363,1324,1408,5.83,5.83],["TabPFN-3","d","F",1333,1306,1357,22.96,0.0],["TabICLv2","d","F",1280,1246,1317,13.08,0.0],["TabPFN-2.6","d","F",1256,1233,1284,26.97,5.83],["RealMLP","te","N",1251,1229,1274,1520.18,0.0],["TabM","te","N",1192,1166,1216,1258.32,0.0],["CatBoost","te","T",1191,1171,1212,507.95,0.0],["LightGBM","te","T",1153,1126,1181,221.36,0.0],["CatBoost","d","T",1135,1108,1162,14.92,0.0],["XGBoost","te","T",1122,1089,1155,336.08,0.0],["TabM","d","N",1112,1083,1141,32.41,0.0],["RealMLP","d","N",1056,1015,1084,58.98,0.0],["XGBoost","d","T",1000,962,1041,8.67,0.0],["LightGBM","d","T",989,954,1021,5.34,0.0],["RandomForest","d","T",882,838,924,10.82,0.0]],"grouped":[["RealMLP","te","N",1246,1191,1338,1789.75,0.0],["TabM","te","N",1242,1179,1327,896.5,0.0],["Kumo-Tabular","d","F",1220,1129,1301,11.81,0.0],["LightGBM","te","T",1216,1148,1298,205.25,0.0],["CatBoost","te","T",1184,1131,1251,1523.07,0.0],["TabM","d","N",1168,1097,1245,23.39,0.0],["LimiX-2-1016","d","F",1167,1097,1257,170.44,22.22],["TabPFN-3.5","d","F",1158,1091,1230,13.83,0.0],["TabFM","d","F",1151,1073,1226,70.35,22.22],["Seldon","d","F",1135,1046,1218,4.67,0.0],["CatBoost","d","T",1122,1068,1185,17.86,0.0],["EXAONE-Tabular","d","F",1111,1031,1189,13.82,22.22],["XGBoost","te","T",1094,1031,1180,321.55,0.0],["RealMLP","d","N",1087,1025,1162,59.94,0.0],["TabPFN-2.6","d","F",1071,999,1122,46.03,22.22],["TabICLv2","d","F",1055,977,1128,15.83,0.0],["TabPFN-3","d","F",1049,981,1135,23.11,0.0],["LightGBM","d","T",1020,960,1090,5.48,0.0],["XGBoost","d","T",1000,929,1075,9.42,0.0],["RandomForest","d","T",973,901,1022,11.04,0.0]],"temporal":[["Kumo-Tabular","d","F",1175,1051,1325,7.93,0.0],["TabPFN-3.5","d","F",1173,1109,1259,10.36,0.0],["RealMLP","te","N",1172,1090,1264,612.76,0.0],["CatBoost","te","T",1132,1057,1232,305.38,0.0],["XGBoost","te","T",1128,1062,1208,160.01,0.0],["LightGBM","te","T",1128,1067,1204,70.1,0.0],["TabM","te","N",1080,1018,1152,348.74,0.0],["CatBoost","d","T",1074,1018,1157,9.93,0.0],["Seldon","d","F",1052,968,1148,3.8,0.0],["TabPFN-3","d","F",1023,979,1081,14.81,0.0],["XGBoost","d","T",1000,942,1068,3.07,0.0],["RealMLP","d","N",990,935,1053,18.35,0.0],["TabM","d","N",974,928,1022,10.93,0.0],["LightGBM","d","T",962,912,1021,0.94,0.0],["TabPFN-2.6","d","F",950,870,1037,9.36,51.43],["LimiX-2-1016","d","F",948,847,1059,30.7,57.14],["EXAONE-Tabular","d","F",916,817,1014,5.47,57.14],["TabFM","d","F",913,831,997,15.46,57.14],["TabICLv2","d","F",817,733,885,8.33,0.0],["RandomForest","d","T",705,633,762,2.77,0.0]]};
  const VIEWS = {
    all: { bench: "TabArena-v0.1", n: 51, txt: "all 51 IID datasets" },
    small: { bench: "TabArena-v0.1", n: 36, txt: "36 small datasets" },
    medium: { bench: "TabArena-v0.1", n: 15, txt: "15 medium datasets" },
    clf: { bench: "TabArena-v0.1", n: 38, txt: "38 classification datasets" },
    reg: { bench: "TabArena-v0.1", n: 13, txt: "13 regression datasets" },
    iid: { bench: "BeyondArena", n: 103, txt: "103 datasets, random (IID) splits" },
    grouped: { bench: "BeyondArena", n: 18, txt: "18 datasets, grouped splits (unseen groups)" },
    temporal: { bench: "BeyondArena", n: 21, txt: "21 datasets, temporal splits (predict the future)" },
  };
  const FAMC = { F: "#2E4BC6", N: "#16876B", T: "#C99A2E", B: "#B5502E", O: "#5B6478" };
  const FAMN = { F: "foundation model", N: "neural network", T: "tree-based", B: "baseline", O: "other" };
  DEMOS.arena = function (root) {
    const cv = $(root, "canvas"), ctx = cv.getContext("2d"), out = $(root, ".readout"), defIn = $(root, "input[name=def]");
    let view = "all", mode = "rank", hover = -1, layout = [];
    const rows = () => ARENA[view].filter((r) => !defIn.checked || r[1] === "d").slice().sort((a, b) => b[3] - a[3]);
    const nm = (r) => `${r[0]}${r[1] === "te" ? " (tuned+ens.)" : ""}${r[7] > 0 ? " *" : ""}`;
    function hatch(x, y, w, h, col) {
      ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      for (let k = -h; k < w; k += 6) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
      ctx.restore();
    }
    function legend(x, y) {
      ctx.save(); ctx.font = `12px ${FONT}`; ctx.textAlign = "left";
      ["F", "N", "T", "B"].forEach((f) => { ctx.fillStyle = FAMC[f]; ctx.fillRect(x, y - 9, 12, 12); ctx.fillStyle = INK; ctx.fillText(FAMN[f], x + 17, y + 1); x += ctx.measureText(FAMN[f]).width + 34; });
      ctx.strokeStyle = MUT; ctx.strokeRect(x, y - 9, 12, 12);
      ctx.fillStyle = INK; ctx.fillText("hatched, * = partly imputed", x + 17, y + 1); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(x, y - 9, 12, 12); ctx.clip(); ctx.strokeStyle = MUT; for (let k = -12; k < 12; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + 3); ctx.lineTo(x + k + 12, y - 9); ctx.stroke(); } ctx.restore();
    }
    function tip(px, py, r) {
      const L = [nm(r).replace(" *", ""), `${FAMN[r[2]]} · Elo ${r[3]} (95 % CI ${r[4]}–${r[5]})`, `median train time ${r[6] < 10 ? r[6].toFixed(2) : Math.round(r[6])} s per 1 000 rows`];
      if (r[7] > 0) L.push(`${r[7].toFixed(0)} % of datasets imputed (outside its range)`);
      ctx.save(); ctx.font = `12px ${FONT}`; const w = Math.max(...L.map((s) => ctx.measureText(s).width)) + 20, h = 18 * L.length + 10;
      const x = clamp(px + 14, 4, cv.width - w - 4), y = clamp(py - h - 8, 4, cv.height - h - 4);
      ctx.fillStyle = "rgba(20,33,61,.94)"; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, 8) : ctx.rect(x, y, w, h); ctx.fill();
      L.forEach((s, i) => { ctx.fillStyle = i ? "#E5E9F4" : "#fff"; ctx.font = `${i ? "" : "600 "}12px ${FONT}`; ctx.fillText(s, x + 10, y + 20 + 18 * i); });
      ctx.restore();
    }
    function draw() {
      const R = rows(), W = cv.width, H = cv.height, V = VIEWS[view]; ctx.clearRect(0, 0, W, H); layout = [];
      const lo = Math.floor((Math.min(...R.map((r) => r[4])) - 40) / 100) * 100, hi = Math.ceil((Math.max(...R.map((r) => r[5])) + 20) / 100) * 100;
      const ticks = []; for (let v = lo; v <= hi; v += hi - lo > 900 ? 200 : 100) ticks.push(v);
      const anchor = 1000, anchorName = V.bench === "BeyondArena" ? "XGBoost default = 1000" : "RandomForest default = 1000";
      if (mode === "rank") {
        const L = 220, T = 34, A = axes(ctx, L, T, W - L - 18, H - T - 40, { x: [lo, hi], y: [0, 1], xt: ticks, xl: `Elo · ${V.bench}, ${V.txt} · dashed: ${anchorName}` });
        const rh = (H - T - 40) / R.length;
        ctx.save(); ctx.strokeStyle = "rgba(20,33,61,.12)"; ticks.forEach((v) => { ctx.beginPath(); ctx.moveTo(A.X(v), T); ctx.lineTo(A.X(v), H - 40); ctx.stroke(); }); ctx.restore();
        ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = MUT; ctx.beginPath(); ctx.moveTo(A.X(anchor), T); ctx.lineTo(A.X(anchor), H - 40); ctx.stroke(); ctx.restore();
        R.forEach((r, i) => {
          const y = T + i * rh, bh = Math.max(rh - 3, 4), x0 = A.X(lo), x1 = A.X(r[3]), c = FAMC[r[2]];
          if (i === hover) { ctx.fillStyle = "rgba(201,154,46,.14)"; ctx.fillRect(0, y, W, rh); }
          ctx.fillStyle = c; ctx.globalAlpha = r[7] > 0 ? 0.35 : 1; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x0, y + 1.5, x1 - x0, bh, [0, 4, 4, 0]) : ctx.rect(x0, y + 1.5, x1 - x0, bh); ctx.fill(); ctx.globalAlpha = 1;
          if (r[7] > 0) hatch(x0, y + 1.5, x1 - x0, bh, c);
          ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(A.X(r[4]), y + 1.5 + bh / 2); ctx.lineTo(A.X(r[5]), y + 1.5 + bh / 2); ctx.stroke();
          ctx.fillStyle = INK; ctx.font = `${r[1] === "d" && r[2] === "F" ? "600 " : ""}${rh < 15 ? 11 : 12}px ${FONT}`; ctx.textAlign = "right"; ctx.fillText(nm(r), L - 8, y + 1.5 + bh / 2 + 4);
          ctx.textAlign = "left"; ctx.fillStyle = MUT; ctx.font = `${rh < 15 ? 10 : 11}px ${FONT}`; ctx.fillText(String(r[3]), Math.max(A.X(r[5]), x1) + 5, y + 1.5 + bh / 2 + 4);
          layout.push({ r, x: 0, y, w: W, h: rh });
        });
        legend(L, 20);
        if (hover >= 0 && R[hover]) tip(A.X(R[hover][3]), T + hover * rh + rh / 2, R[hover]);
      } else {
        const L = 70, T = 34, lx = (t) => Math.log10(Math.max(t, 0.1));
        const A = axes(ctx, L, T, W - L - 24, H - T - 52, { x: [-1, 4.3], y: [lo, hi], xt: [-1, 0, 1, 2, 3, 4], fx: (v) => (v < 0 ? "0.1" : String(10 ** v)), yt: ticks, xl: "median training (or tuning) time, s per 1 000 rows — log scale →", yl: "Elo" });
        const lab = new Set(R.slice(0, 4).map((r) => r)); const bestNon = R.find((r) => r[2] !== "F"); if (bestNon) lab.add(bestNon);
        const tr = R.find((r) => r[2] === "T"); if (tr) lab.add(tr);
        R.forEach((r, i) => {
          const x = A.X(lx(r[6])), y = A.Y(r[3]), c = FAMC[r[2]];
          ctx.beginPath(); ctx.arc(x, y, i === hover ? 8 : 6, 0, 7); ctx.fillStyle = c; ctx.globalAlpha = r[7] > 0 ? 0.4 : 1; ctx.fill(); ctx.globalAlpha = 1;
          ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); if (r[1] === "te") { ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.stroke(); }
          layout.push({ r, cx: x, cy: y });
        });
        /* direct labels for the leaders, skipping any that would collide with a placed label or a mark */
        ctx.save(); ctx.font = `600 12px ${FONT}`; const placed = [];
        const hit = (a) => placed.some((b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h)
          || layout.some((m) => m.cx > a.x - 7 && m.cx < a.x + a.w + 7 && m.cy > a.y - 7 && m.cy < a.y + a.h + 7);
        layout.forEach((m) => {
          if (!lab.has(m.r)) return; const t = nm(m.r), w = ctx.measureText(t).width + 4;
          const opts = [{ x: m.cx + 11, y: m.cy - 8 }, { x: m.cx - 11 - w, y: m.cy - 8 }, { x: m.cx - w / 2, y: m.cy - 26 }, { x: m.cx - w / 2, y: m.cy + 12 }];
          const o = opts.map((q) => ({ x: q.x, y: q.y, w, h: 15 })).find((q) => q.x > L && q.x + q.w < W - 4 && !hit(q));
          if (!o) return; placed.push(o); ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.fillText(t, o.x + 2, o.y + 11);
        });
        ctx.restore();
        legend(L, 20);
        if (hover >= 0 && layout[hover]) tip(layout[hover].cx, layout[hover].cy, layout[hover].r);
      }
      const bestF = R.find((r) => r[2] === "F"), bestO = R.find((r) => r[2] !== "F");
      const d = bestF && bestO ? bestF[3] - bestO[3] : 0, p = 1 / (1 + 10 ** (-Math.abs(d) / 400));
      out.innerHTML = `<b>${V.bench}</b>, ${V.txt}: best foundation model <b>${nm(bestF)}</b> (${bestF[3]}) vs best non-foundation model <b>${nm(bestO)}</b> (${bestO[3]}) → `
        + `${d >= 0 ? "FM ahead" : "FM behind"} by ${Math.abs(d)} Elo, i.e. the leader wins ≈ ${Math.round(100 * p)} % of head-to-head dataset comparisons. ${mode === "rank" ? "Hover a bar for details." : "Ringed marks are tuned + ensembled (hours of search); the rest are one forward pass or a default fit."}`;
    }
    cv.addEventListener("mousemove", (e) => {
      const [px, py] = canvasXY(cv, e); let h = -1;
      if (mode === "rank") layout.forEach((b, i) => { if (py >= b.y && py < b.y + b.h) h = i; });
      else { let bd = 18 * 18; layout.forEach((b, i) => { const dd = (b.cx - px) ** 2 + (b.cy - py) ** 2; if (dd < bd) { bd = dd; h = i; } }); }
      if (h !== hover) { hover = h; draw(); }
    });
    cv.addEventListener("mouseleave", () => { hover = -1; draw(); });
    root.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => { view = b.dataset.view; hover = -1; pick(root, "data-view", view); draw(); }));
    root.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { mode = b.dataset.mode; hover = -1; pick(root, "data-mode", mode); draw(); }));
    defIn.onchange = () => { hover = -1; draw(); };
    pick(root, "data-view", view); pick(root, "data-mode", mode);
    draw();
  };

  /* ======================================================= 3 · proper scoring rules re-rank probabilistic regressors */
  DEMOS.scoring = function (root) {
    const cv = $(root, "canvas"), ctx = cv.getContext("2d"), tab = $(root, ".scoretab"), out = $(root, ".readout");
    const kIn = $(root, "input[name=k]"), kv = $(root, ".kv"), bIn = $(root, "input[name=b]"), bv = $(root, ".bv");
    const N = 2000, r = rng(2026), X = [], Y = [], M = [], S = [], ETA = [];
    for (let i = 0; i < N; i++) {
      const x = r(), m = 3 + 2 * Math.sin(5 * x), s = 0.25 + 1.0 * x; // heteroscedastic: uncertainty grows with x
      X.push(x); M.push(m); S.push(s); Y.push(m + s * gauss(r)); ETA.push(gauss(r));
    }
    const SBAR = Math.sqrt(S.reduce((a, s) => a + s * s, 0) / N);
    const FC = [
      { id: "A", name: "A · sharp, over-confident", col: BLU },
      { id: "B", name: "B · noisier mean, calibrated", col: "#16876B" },
      { id: "C", name: "C · one spread for all, biased", col: TER },
    ];
    const METRICS = [
      { k: "rmse", h: "RMSE", t: "point" }, { k: "mae", h: "MAE", t: "point" }, { k: "crps", h: "CRPS", t: "proper" },
      { k: "is", h: "IS 90 %", t: "proper" }, { k: "nll", h: "log score", t: "proper" }, { k: "cov", h: "coverage 90 %", t: "diagnostic" },
    ];
    function forecast(f, i, kA, b) {
      if (f === "A") return [M[i], kA * S[i]];
      if (f === "B") return [M[i] + 0.45 * S[i] * ETA[i], S[i] * Math.sqrt(1 + 0.45 ** 2)];
      return [M[i] + b * SBAR, SBAR];
    }
    function draw() {
      const kA = +kIn.value, b = +bIn.value; kv.textContent = kA.toFixed(2) + " × true σ"; bv.textContent = b.toFixed(2) + " σ̄";
      const res = FC.map((f) => {
        const s = { rmse: 0, mae: 0, crps: 0, is: 0, nll: 0, cov: 0 }, pit = new Array(10).fill(0);
        for (let i = 0; i < N; i++) {
          const [mu, sd] = forecast(f.id, i, kA, b), y = Y[i], z = (y - mu) / sd, l = mu - 1.645 * sd, u = mu + 1.645 * sd;
          s.rmse += (y - mu) ** 2; s.mae += Math.abs(y - mu); s.crps += crpsN(mu, sd, y);
          s.is += u - l + (y < l ? 20 * (l - y) : 0) + (y > u ? 20 * (y - u) : 0);
          s.nll += Math.log(sd * SQ2PI) + 0.5 * z * z; if (y >= l && y <= u) s.cov++;
          pit[Math.min(9, Math.floor(10 * Phi(z)))]++;
        }
        Object.keys(s).forEach((k) => (s[k] /= N)); s.rmse = Math.sqrt(s.rmse);
        return { f, s, pit };
      });
      const win = {}; METRICS.forEach((m) => { const v = res.map((q) => (m.k === "cov" ? Math.abs(q.s.cov - 0.9) : q.s[m.k])); win[m.k] = v.indexOf(Math.min(...v)); });
      tab.innerHTML = `<tr><th>forecaster</th>${METRICS.map((m) => `<th>${m.h}<br><span style="font-weight:400;text-transform:none;letter-spacing:0">${m.t}</span></th>`).join("")}</tr>`
        + res.map((q, j) => `<tr><td><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:${q.f.col};margin-right:7px"></span>${q.f.name}</td>`
          + METRICS.map((m) => { const v = m.k === "cov" ? Math.round(100 * q.s.cov) + " %" : q.s[m.k].toFixed(3); return `<td style="white-space:nowrap">${win[m.k] === j ? `<b>★ ${v}</b>` : v}</td>`; }).join("") + "</tr>").join("");
      /* PIT histograms */
      const W = cv.width, H = cv.height, pw = (W - 40) / 3; ctx.clearRect(0, 0, W, H);
      res.forEach((q, j) => {
        const x0 = 20 + j * pw + 30, w = pw - 50, y0 = 30, h = H - 70, mx = 3;
        const A = axes(ctx, x0, y0, w, h, { x: [0, 1], y: [0, mx], xt: [0, 0.5, 1], yt: [0, 1, 2, 3], xl: "PIT = F(y)" });
        q.pit.forEach((c, k) => { const d = Math.min((c / N) * 10, mx); ctx.fillStyle = q.f.col; ctx.beginPath(); const bx = A.X(k / 10) + 1, bw = A.X(0.1) - A.X(0) - 2; ctx.roundRect ? ctx.roundRect(bx, A.Y(d), bw, A.Y(0) - A.Y(d), [3, 3, 0, 0]) : ctx.rect(bx, A.Y(d), bw, A.Y(0) - A.Y(d)); ctx.fill(); });
        ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = INK; ctx.beginPath(); ctx.moveTo(A.X(0), A.Y(1)); ctx.lineTo(A.X(1), A.Y(1)); ctx.stroke(); ctx.restore();
        label(ctx, q.f.name, x0, 18, INK, "left", `600 12px ${FONT}`);
      });
      const order = (k) => res.map((q, j) => j).sort((a, c) => res[a].s[k] - res[c].s[k]).map((j) => res[j].f.id).join(" > ");
      out.innerHTML = `Ranking by RMSE: <b>${order("rmse")}</b> · by CRPS: <b>${order("crps")}</b> · by log score: <b>${order("nll")}</b>. `
        + `A flat PIT histogram (dashed line) means calibrated; a U shape means intervals too narrow, a slope means bias.`;
    }
    kIn.oninput = draw; bIn.oninput = draw;
    draw();
  };
})();
