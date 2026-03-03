"use strict";

/* ================================================================
   1.  Scroll-reveal via IntersectionObserver
   ================================================================ */
const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        revealObserver.unobserve(e.target);
      }
    }
  },
  { threshold: 0.15 }
);
document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

const animObserver = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("animate");
        animObserver.unobserve(e.target);
      }
    }
  },
  { threshold: 0.25 }
);
document.querySelectorAll(".s1-inject, .pipeline").forEach((el) =>
  animObserver.observe(el)
);

/* ================================================================
   2.  Hero floating-dots canvas
   ================================================================ */
(function initHeroDots() {
  const canvas = document.getElementById("hero-dots");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H, dots;
  const COLS = 14;
  const ROWS = 7;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    W = rect.width;
    H = rect.height;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildDots();
  }

  function buildDots() {
    dots = [];
    const gapX = W / (COLS + 1);
    const gapY = H / (ROWS + 1);
    for (let r = 1; r <= ROWS; r++) {
      for (let c = 1; c <= COLS; c++) {
        dots.push({
          x: gapX * c,
          y: gapY * r,
          baseY: gapY * r,
          phase: Math.random() * Math.PI * 2,
          speed: 0.3 + Math.random() * 0.4,
          amp: 3 + Math.random() * 5,
          r: 3 + Math.random() * 2.5,
          hue: 210 + Math.random() * 30,
          alpha: 0.08 + Math.random() * 0.12,
        });
      }
    }
  }

  let time = 0;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    time += 0.012;
    for (const d of dots) {
      d.y = d.baseY + Math.sin(time * d.speed + d.phase) * d.amp;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${d.hue}, 60%, 58%, ${d.alpha})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  draw();
})();

/* ================================================================
   3.  Stage 1 – ISAB two-step attention canvas
   ================================================================ */
(function initISABCanvas() {
  const canvas = document.getElementById("s1-isab");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = 440;
  const H = 260;

  const N_VALS = 3;
  const N_IND = 3;
  const colX = 60;
  const indX = 220;
  const outX = 380;
  const nodeR = 16;

  function yPos(count, i) {
    const gap = (H - 60) / (count + 1);
    return 30 + gap * (i + 1);
  }

  const vals = Array.from({ length: N_VALS }, (_, i) => ({ x: colX, y: yPos(N_VALS, i) }));
  const inds = Array.from({ length: N_IND }, (_, i) => ({ x: indX, y: yPos(N_IND, i) }));
  const outs = Array.from({ length: N_VALS }, (_, i) => ({ x: outX, y: yPos(N_VALS, i) }));

  let started = false;
  let startTime = 0;

  const obs = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting && !started) {
        started = true;
        startTime = performance.now();
        obs.unobserve(canvas);
      }
    },
    { threshold: 0.3 }
  );
  obs.observe(canvas);

  function ease(t) { return 1 - Math.pow(1 - t, 3); }

  function drawCurve(ax, ay, bx, by, alpha) {
    const mx = (ax + bx) / 2;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.quadraticCurveTo(mx, (ay + by) / 2 - 10, bx, by);
    ctx.strokeStyle = `rgba(0,113,227,${alpha})`;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  function drawNode(x, y, r, fill, stroke, label) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.stroke(); }
    if (label) {
      ctx.fillStyle = fill === "#0071e3" ? "#fff" : "#1d1d1f";
      ctx.font = "600 10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x, y);
    }
  }

  function drawLabel(x, y, text) {
    ctx.fillStyle = "#86868b";
    ctx.font = "600 10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(text, x, y);
  }

  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    const elapsed = started ? (now - startTime) / 1000 : 0;

    // Phase 1 (0–1.2s): show value nodes
    const valLabels = ["0.3", "\u22120.7", "0.1"];
    const valColors = ["#dbeafe", "#e8f0fe", "#fef9e7"];
    for (let i = 0; i < N_VALS; i++) {
      const t = ease(Math.min(1, Math.max(0, (elapsed - i * 0.08) / 0.4)));
      drawNode(vals[i].x, vals[i].y, nodeR * t, valColors[i], "#d2d2d7",
        t > 0.6 ? valLabels[i] : null);
    }
    if (elapsed > 0.2) drawLabel(colX, H - 18, "Column x\u2081 values");

    // Phase 2 (0.6–1.5s): show inducing points
    for (let i = 0; i < N_IND; i++) {
      const t = ease(Math.min(1, Math.max(0, (elapsed - 0.6 - i * 0.1) / 0.4)));
      drawNode(inds[i].x, inds[i].y, (nodeR - 2) * t, "#0071e3", null,
        t > 0.6 ? `I${i + 1}` : null);
    }
    if (elapsed > 0.8) drawLabel(indX, H - 18, "Inducing points");

    // Phase 3 (1.2–2.2s): step 1 curves – values → inducing points
    if (elapsed > 1.2) {
      ctx.save();
      ctx.setLineDash([4, 3]);
      for (let vi = 0; vi < N_VALS; vi++) {
        for (let ii = 0; ii < N_IND; ii++) {
          const delay = 1.2 + (vi * N_IND + ii) * 0.04;
          const t = ease(Math.min(1, Math.max(0, (elapsed - delay) / 0.5)));
          if (t > 0) drawCurve(vals[vi].x + nodeR, vals[vi].y,
            inds[ii].x - nodeR + 2, inds[ii].y, 0.18 * t);
        }
      }
      ctx.restore();
    }

    // Step 1 label
    if (elapsed > 1.6) {
      const t = Math.min(1, (elapsed - 1.6) / 0.4);
      ctx.globalAlpha = t;
      ctx.fillStyle = "#0071e3";
      ctx.font = "600 9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Step A: attend \u2192", (colX + indX) / 2, 18);
      ctx.globalAlpha = 1;
    }

    // Phase 4 (2.2–3.2s): step 2 curves – inducing → outputs
    if (elapsed > 2.2) {
      for (let ii = 0; ii < N_IND; ii++) {
        for (let oi = 0; oi < N_VALS; oi++) {
          const delay = 2.2 + (ii * N_VALS + oi) * 0.03;
          const t = ease(Math.min(1, Math.max(0, (elapsed - delay) / 0.5)));
          if (t > 0) drawCurve(inds[ii].x + nodeR - 2, inds[ii].y,
            outs[oi].x - nodeR, outs[oi].y, 0.22 * t);
        }
      }
    }

    // Step 2 label
    if (elapsed > 2.6) {
      const t = Math.min(1, (elapsed - 2.6) / 0.4);
      ctx.globalAlpha = t;
      ctx.fillStyle = "#0071e3";
      ctx.font = "600 9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Step B: attend back \u2192", (indX + outX) / 2, 18);
      ctx.globalAlpha = 1;
    }

    // Phase 5 (2.8–3.5s): output nodes
    const outLabels = ["e\u2081", "e\u2082", "e\u2083"];
    for (let i = 0; i < N_VALS; i++) {
      const t = ease(Math.min(1, Math.max(0, (elapsed - 2.8 - i * 0.06) / 0.4)));
      if (t > 0) {
        drawNode(outs[i].x, outs[i].y, nodeR * t, "#dbeafe", "#0071e3",
          t > 0.6 ? outLabels[i] : null);
      }
    }
    if (elapsed > 3.0) drawLabel(outX, H - 18, "Enriched vectors");

    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();

/* ================================================================
   5.  Row-interaction attention canvas animation (Stage 2)
   ================================================================ */
(function initAttnCanvas() {
  const canvas = document.getElementById("attn-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = 360;
  const H = 280;

  const clsLabels = ["CLS\u2081", "CLS\u2082", "CLS\u2083", "CLS\u2084"];
  const featureLabels = ["x\u2081", "x\u2082"];
  const nodeCount = clsLabels.length + featureLabels.length;
  const allLabels = [...clsLabels, ...featureLabels];

  const positions = [];
  const nodeR = 22;
  const centerY = 100;
  const spacing = W / (nodeCount + 1);
  for (let i = 0; i < nodeCount; i++) {
    positions.push({ x: spacing * (i + 1), y: centerY });
  }

  const connections = [
    [0, 4], [0, 5], [1, 4], [1, 5],
    [2, 4], [2, 5], [3, 4], [3, 5],
    [4, 5],
  ];

  let started = false;
  let startTime = 0;

  const obs = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting && !started) {
        started = true;
        startTime = performance.now();
        obs.unobserve(canvas);
      }
    },
    { threshold: 0.3 }
  );
  obs.observe(canvas);

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function draw(now) {
    ctx.clearRect(0, 0, W, H);
    const elapsed = started ? (now - startTime) / 1000 : 0;

    for (let ci = 0; ci < connections.length; ci++) {
      const [a, b] = connections[ci];
      const delay = ci * 0.08;
      const progress = Math.min(1, Math.max(0, (elapsed - 0.3 - delay) / 0.5));
      if (progress <= 0) continue;
      const t = easeOut(progress);
      const pa = positions[a];
      const pb = positions[b];
      const mx = (pa.x + pb.x) / 2;
      const my = pa.y - 28 - Math.abs(a - b) * 6;

      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y - nodeR);
      ctx.quadraticCurveTo(mx, my, pb.x, pb.y - nodeR);
      ctx.strokeStyle = `rgba(0,113,227,${0.22 * t})`;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    for (let i = 0; i < nodeCount; i++) {
      const p = positions[i];
      const isCLS = i < 4;
      const nodeDelay = i * 0.06;
      const progress = Math.min(1, Math.max(0, (elapsed - nodeDelay) / 0.45));
      const t = easeOut(progress);
      const r = nodeR * t;

      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isCLS ? "#0071e3" : "#f0f2f5";
      ctx.fill();
      if (!isCLS) {
        ctx.strokeStyle = "#d2d2d7";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      if (t > 0.5) {
        ctx.fillStyle = isCLS ? "#fff" : "#1d1d1f";
        ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.globalAlpha = Math.min(1, (t - 0.5) * 4);
        ctx.fillText(allLabels[i], p.x, p.y);
        ctx.globalAlpha = 1;
      }
    }

    const aggDelay = 1.3;
    const aggProgress = Math.min(1, Math.max(0, (elapsed - aggDelay) / 0.5));
    if (aggProgress > 0) {
      const t = easeOut(aggProgress);
      const aggY = centerY + 80;
      ctx.globalAlpha = t;

      for (let i = 0; i < 4; i++) {
        const p = positions[i];
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(p.x, p.y + nodeR);
        ctx.lineTo(p.x, aggY - 14);
        ctx.strokeStyle = "rgba(0,113,227,.35)";
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const aggW = 180;
      const aggH = 32;
      const aggX = (positions[0].x + positions[3].x) / 2 - aggW / 2;
      ctx.beginPath();
      ctx.roundRect(aggX, aggY - aggH / 2, aggW, aggH, 8);
      ctx.fillStyle = "#e8f0fe";
      ctx.fill();
      ctx.strokeStyle = "#0071e3";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.fillStyle = "#1d1d1f";
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("row representation", aggX + aggW / 2, aggY);
      ctx.globalAlpha = 1;
    }

    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
