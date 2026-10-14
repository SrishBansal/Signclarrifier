/**
 * ClarifySign Decoupled Avatar Renderer (HTML5 Canvas 2D)
 *
 * Design brief: a calm, grounded, realistic-proportioned signer who is recognisably South Asian
 * through real variation (skin tone range, dark hair, everyday clothing) and NOT through costume.
 *   - No default turban, bindi, jewellery or "ethnic" props: any single one would stand in for
 *     all of India. Style is selectable via the `style` option (skin / hair / outfit / garment).
 *   - Hands are the content of sign language, so they get the most care: skin-coloured, filled,
 *     outlined, depth-ordered fingers; skin vs garment contrast works for every skin tone.
 *   - The face is deliberately neutral. The landmark data has no face tracking, so the face
 *     only blinks lightly; it never fakes expressions that carry grammatical meaning in ISL.
 *
 * Playback logic, timeline, IK, smoothing and exports are unchanged from the previous version.
 * ZERO dependencies on NLU or planner modules.
 */

(function (global) {
  "use strict";

  // Real MediaPipe Connections
  const HAND_PALM_CONNECTIONS = [
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]
  ];
  // chain = landmark indices from knuckle to tip; w = relative width
  const HAND_FINGERS = [
    { name: "thumb", chain: [1, 2, 3, 4], w: 1.20 },
    { name: "index", chain: [5, 6, 7, 8], w: 1.00 },
    { name: "middle", chain: [9, 10, 11, 12], w: 1.03 },
    { name: "ring", chain: [13, 14, 15, 16], w: 0.96 },
    { name: "pinky", chain: [17, 18, 19, 20], w: 0.84 }
  ];

  // ---- Style presets -------------------------------------------------------------------
  // Skin tones span the range of South Asian complexions; none is "the" Indian skin tone.
  const SKINS = {
    light: { base: "#e2b790", shade: "#c99a75", line: "#8a5a3d", lip: "#b87565", iris: "#3b2a20" },
    medium: { base: "#c68e64", shade: "#aa734c", line: "#6f4228", lip: "#9a5646", iris: "#2e1f17" },
    deep: { base: "#8f5c3c", shade: "#744729", line: "#44270f", lip: "#6c382c", iris: "#1f140e" }
  };
  const HAIRS = {
    black: { base: "#1b1512", shine: "#4a3a32" },
    brown: { base: "#2f2019", shine: "#5c4337" }
  };
  // Plain, solid, everyday garments (no embroidery, trim, or pattern).
  const GARMENTS = {
    indigo: { base: "#4a5f9e", shade: "#3a4b82", line: "#232f58", trim: "#7184bd" },
    ecru: { base: "#e7dfcf", shade: "#cdc3ae", line: "#8f8570", trim: "#b8ac94" },
    clay: { base: "#b86a50", shade: "#9a523b", line: "#65311f", trim: "#d79479" },
    sage: { base: "#7d9983", shade: "#667f6c", line: "#415648", trim: "#a4bba9" }
  };
  // Trousers worn under the shirt / kurta (plain, dark enough for any skin tone to read against).
  const TROUSERS = {
    charcoal: { base: "#2f3541", line: "#1b1f27" },
    navy: { base: "#2a3552", line: "#171e33" },
    khaki: { base: "#8c7d5e", line: "#5d513a" }
  };
  const DEFAULT_STYLE = {
    skin: "medium", hair: "short", hairColor: "black", outfit: "kurta",
    garment: "indigo", trousers: "charcoal", build: "broad"
  };

  // Smoothstep ease-in-out
  function easeInOut(t) {
    t = Math.max(0, Math.min(1, t));
    return t * t * (3 - 2 * t);
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // Linear interpolation of 225-dim float arrays with easing
  function lerpFrame(f1, f2, alpha) {
    const out = new Float32Array(225);
    for (let i = 0; i < 225; i++) {
      out[i] = (1 - alpha) * f1[i] + alpha * f2[i];
    }
    return out;
  }

  // 3-frame moving-average smoothing per coordinate.
  // Skips frames where a hand block is all zeros so rest frames stay clean.
  // (Zero-block checks are computed once per frame instead of once per coordinate.)
  function smoothSequence(seq) {
    if (!seq || seq.length < 3) return seq;
    const blockIsZero = (frame, a, b) => { for (let i = a; i < b; i++) if (frame[i] !== 0) return false; return true; };
    const out = [];
    for (let i = 0; i < seq.length; i++) {
      const prev = seq[Math.max(0, i - 1)];
      const curr = seq[i];
      const next = seq[Math.min(seq.length - 1, i + 1)];
      const lhZero = blockIsZero(curr, 0, 63);
      const rhZero = blockIsZero(curr, 63, 126);
      const sm = new Float32Array(225);
      for (let d = 0; d < 225; d++) {
        if ((d < 63 && lhZero) || (d >= 63 && d < 126 && rhZero)) {
          sm[d] = curr[d];
        } else {
          sm[d] = (prev[d] + curr[d] + next[d]) / 3;
        }
      }
      out.push(sm);
    }
    return out;
  }

  // Canonical Rest Frame (225 dims)
  function buildRestFrame() {
    const f = new Float32Array(225);
    // Pose (indices 126..224): stored at TRAINING scale (y stretched 16/9).
    // renderFrame will unstretch with U = yUnstretch = 9/16.
    const setPose = (idx, x, y, z) => {
      const base = 126 + idx * 3;
      f[base] = x; f[base + 1] = y; f[base + 2] = z;
    };
    // y values here are in stretched space; renderFrame applies U before drawing
    setPose(0, 0.0, -1.0, -0.3);    // Nose
    setPose(7, 0.35, -1.0, 0.0);    // Left ear
    setPose(8, -0.35, -1.0, 0.0);    // Right ear
    setPose(9, 0.12, -0.85, -0.2);   // Mouth left
    setPose(10, -0.12, -0.85, -0.2);   // Mouth right
    setPose(11, 0.5, 0.0, 0.0);    // Left shoulder
    setPose(12, -0.5, 0.0, 0.0);    // Right shoulder
    setPose(13, 0.56, 1.30, 0.15);   // Left elbow  (arms slightly bent: a calm "ready" stance)
    setPose(14, -0.56, 1.30, 0.15);   // Right elbow
    setPose(15, 0.40, 2.30, 0.0);    // Left wrist  (hands held in front of the hips, clear of the captions)
    setPose(16, -0.40, 2.30, 0.0);    // Right wrist
    setPose(23, 0.28, 2.50, 0.0);    // Left hip
    setPose(24, -0.28, 2.50, 0.0);    // Right hip

    // Hands: a relaxed, slightly curled hand with ALL 21 landmarks (thumb toward the body).
    // x is mirrored per side; y points down. [x, y, z] in the hand block's own units.
    const HAND_REST = [
      [0.00, 0.00, 0.00],
      [0.14, 0.12, -0.05], [0.26, 0.26, -0.06], [0.31, 0.38, -0.06], [0.33, 0.50, -0.06],   // thumb
      [0.15, 0.42, -0.02], [0.16, 0.62, -0.02], [0.16, 0.76, -0.02], [0.15, 0.88, -0.02],   // index
      [0.04, 0.44, 0.00], [0.04, 0.66, 0.00], [0.04, 0.80, 0.00], [0.04, 0.92, 0.00],   // middle
      [-0.07, 0.43, 0.02], [-0.07, 0.63, 0.02], [-0.07, 0.76, 0.02], [-0.07, 0.86, 0.02],   // ring
      [-0.17, 0.40, 0.04], [-0.18, 0.56, 0.04], [-0.18, 0.67, 0.04], [-0.18, 0.76, 0.04]    // pinky
    ];
    [false, true].forEach(isRight => {
      const side = isRight ? 1.0 : -1.0;
      HAND_REST.forEach(([x, y, z], idx) => {
        const base = (isRight ? 63 : 0) + idx * 3;
        f[base] = side * x; f[base + 1] = y; f[base + 2] = z;
      });
    });
    return f;
  }

  const REST_FRAME = buildRestFrame();

  // Two-Bone Analytical Inverse Kinematics for arms
  // l1=0.85, l2=0.8 match typical normalized bone lengths in 16:9 training data
  function solveTwoBoneIK(shoulder, wrist, elbowHint, isLeft, l1, l2) {
    l1 = l1 !== undefined ? l1 : 0.85;
    l2 = l2 !== undefined ? l2 : 0.80;
    const dx = wrist[0] - shoulder[0];
    const dy = wrist[1] - shoulder[1];
    const d = Math.hypot(dx, dy);

    if (d < 1e-4) {
      return [shoulder[0] + (isLeft ? l1 : -l1), shoulder[1], shoulder[2]];
    }

    const ux = dx / d;
    const uy = dy / d;
    const normSign = isLeft ? 1.0 : -1.0;
    let vx = -uy * normSign;
    let vy = ux * normSign;

    if (elbowHint) {
      const ehx = elbowHint[0] - shoulder[0];
      const ehy = elbowHint[1] - shoulder[1];
      if (ehx * vx + ehy * vy < 0) { vx = -vx; vy = -vy; }
    }

    let ex, ey;
    if (d >= l1 + l2) {
      ex = shoulder[0] + (l1 / d) * dx;
      ey = shoulder[1] + (l1 / d) * dy;
    } else if (d <= Math.abs(l1 - l2)) {
      ex = shoulder[0] + l1 * ux;
      ey = shoulder[1] + l1 * uy;
    } else {
      const cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d);
      const clampedCosA = clamp(cosA, -1, 1);
      const sinA = Math.sqrt(Math.max(0, 1 - clampedCosA * clampedCosA));
      ex = shoulder[0] + l1 * (clampedCosA * ux + sinA * vx);
      ey = shoulder[1] + l1 * (clampedCosA * uy + sinA * vy);
    }
    const ez = (shoulder[2] + wrist[2]) * 0.5;
    return [ex, ey, ez];
  }

  // ---- Drawing helpers (plain Canvas 2D: paths, fills, strokes only) -----------------------
  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function mix(a, b, t) {
    const A = hexToRgb(a), B = hexToRgb(b);
    return "#" + [0, 1, 2].map(i => Math.round(A[i] + (B[i] - A[i]) * t).toString(16).padStart(2, "0")).join("");
  }
  function strokeSeg(g, a, b, color, width, cap) {
    g.strokeStyle = color; g.lineWidth = width; g.lineCap = cap || "round"; g.lineJoin = "round";
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
  }
  function disc(g, p, r, fill) {
    g.fillStyle = fill; g.beginPath(); g.arc(p[0], p[1], r, 0, Math.PI * 2); g.fill();
  }
  // A limb segment: outline pass, then fill pass.
  function limb(g, a, b, width, fill, line, outline, cap) {
    strokeSeg(g, a, b, line, width + outline * 2, cap);
    strokeSeg(g, a, b, fill, width, cap);
  }
  function lerp2(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }

  // Body proportions. "broad" is the default; "slim" is a narrower frame with a finer face.
  const BUILDS = {
    broad: {
      cap: 0.115, rise: 0.045, neckTop: 0.46, neckBase: 0.58, wUp: 0.215, wFore: 0.186, wSkin: 0.112,
      jaw: 1.0, brow: 0.095, browArch: 0.05, lip: 0.62, lipTone: 0.35, ear: 0.145, lid: 0.060
    },
    slim: {
      cap: 0.075, rise: 0.075, neckTop: 0.36, neckBase: 0.46, wUp: 0.190, wFore: 0.162, wSkin: 0.105,
      jaw: 0.0, brow: 0.075, browArch: 0.085, lip: 1.0, lipTone: 0.0, ear: 0.13, lid: 0.068
    }
  };

  function drawTorso(g, x11, x12, xNeck, S, st, canvasH) {
    const gar = st.garment, tr = st.trousers, pr = st.prop;
    const L = x12, R = x11;                        // canvas-left / canvas-right shoulder joints
    const cap = S * pr.cap, neckHalf = S * (pr.neckBase * 0.215 * 0.9);
    const hemY = xNeck[1] + S * 1.62;              // shirt / kurta ends at the hip
    const bottomY = canvasH + 8;
    const bl = [xNeck[0] + (L[0] - xNeck[0]) * 0.93 - S * 0.03, hemY];
    const br = [xNeck[0] + (R[0] - xNeck[0]) * 0.93 + S * 0.03, hemY];

    // trousers, behind the shirt, running off the bottom of the canvas
    g.beginPath();
    g.moveTo(bl[0] + S * 0.03, hemY - S * 0.08);
    g.lineTo(br[0] - S * 0.03, hemY - S * 0.08);
    g.lineTo(br[0] - S * 0.07, bottomY);
    g.lineTo(bl[0] + S * 0.07, bottomY);
    g.closePath();
    g.fillStyle = tr.base; g.fill();
    g.strokeStyle = tr.line; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();
    strokeSeg(g, [xNeck[0], hemY], [xNeck[0], bottomY], tr.line, 1.6);

    // shirt / kurta body
    g.beginPath();
    g.moveTo(L[0] - cap, L[1] + S * 0.05);
    g.quadraticCurveTo(L[0] - cap * 0.2, L[1] - S * pr.rise, xNeck[0] - neckHalf, xNeck[1] - S * 0.02);
    g.lineTo(xNeck[0] + neckHalf, xNeck[1] - S * 0.02);
    g.quadraticCurveTo(R[0] + cap * 0.2, R[1] - S * pr.rise, R[0] + cap, R[1] + S * 0.05);
    g.lineTo(br[0], br[1]);
    g.quadraticCurveTo(xNeck[0], hemY + S * 0.045, bl[0], bl[1]);
    g.closePath();
    g.fillStyle = gar.base; g.fill();
    g.strokeStyle = gar.line; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();

    // soft side shading gives the body volume without gradients
    [[L, bl, 1], [R, br, -1]].forEach(([sh, hem, dir]) => {
      g.fillStyle = gar.shade;
      g.beginPath();
      g.moveTo(sh[0] - dir * cap, sh[1] + S * 0.06);
      g.lineTo(hem[0], hem[1]);
      g.lineTo(hem[0] + dir * S * 0.13, hem[1]);
      g.quadraticCurveTo(sh[0] + dir * S * 0.10, sh[1] + S * 0.9, sh[0] + dir * S * 0.12, sh[1] + S * 0.12);
      g.closePath(); g.fill();
    });

    // hem stitch and small side vents
    g.strokeStyle = gar.trim; g.lineWidth = 1.6; g.lineCap = "round";
    g.beginPath();
    g.moveTo(bl[0] + S * 0.04, bl[1] - S * 0.07);
    g.quadraticCurveTo(xNeck[0], hemY - S * 0.02, br[0] - S * 0.04, br[1] - S * 0.07);
    g.stroke();
    strokeSeg(g, [bl[0] + S * 0.01, bl[1] - S * 0.16], [bl[0] + S * 0.01, bl[1] + S * 0.01], gar.line, 1.6);
    strokeSeg(g, [br[0] - S * 0.01, br[1] - S * 0.16], [br[0] - S * 0.01, br[1] + S * 0.01], gar.line, 1.6);
  }

  function drawNeckAndCollar(g, xNeck, cx, cy, rx, ry, S, st) {
    const skin = st.skin, gar = st.garment, pr = st.prop;
    const nw1 = rx * pr.neckTop, nw2 = rx * pr.neckBase;
    g.beginPath();
    g.moveTo(cx - nw1, cy + ry * 0.55);
    g.lineTo(cx + nw1, cy + ry * 0.55);
    g.lineTo(xNeck[0] + nw2, xNeck[1] + S * 0.01);
    g.lineTo(xNeck[0] - nw2, xNeck[1] + S * 0.01);
    g.closePath();
    g.fillStyle = skin.shade; g.fill();
    g.strokeStyle = skin.line; g.lineWidth = 1.6; g.stroke();

    if (st.outfit === "shirt") {
      // open collar: skin V with two flat collar flaps
      g.fillStyle = skin.shade; g.beginPath();
      g.moveTo(xNeck[0] - nw2 * 1.0, xNeck[1] - S * 0.02);
      g.lineTo(xNeck[0], xNeck[1] + S * 0.17);
      g.lineTo(xNeck[0] + nw2 * 1.0, xNeck[1] - S * 0.02);
      g.closePath(); g.fill();
      [-1, 1].forEach(s => {
        g.fillStyle = gar.base; g.strokeStyle = gar.line; g.lineWidth = 1.6;
        g.beginPath();
        g.moveTo(xNeck[0] + s * nw2 * 0.92, xNeck[1] - S * 0.045);
        g.lineTo(xNeck[0] + s * nw2 * 1.85, xNeck[1] + S * 0.10);
        g.lineTo(xNeck[0] + s * S * 0.012, xNeck[1] + S * 0.185);
        g.closePath(); g.fill(); g.stroke();
      });
      strokeSeg(g, [xNeck[0], xNeck[1] + S * 0.185], [xNeck[0], xNeck[1] + S * 0.62], gar.trim, 2);
      [0.30, 0.62, 0.94].forEach(t => disc(g, [xNeck[0], xNeck[1] + S * (0.19 + 0.43 * t)], 2.1, gar.trim));
    } else {
      // band (mandarin) collar and a short plain placket, as on an everyday kurta
      const cw = nw2 * 1.2;
      g.beginPath();
      g.moveTo(xNeck[0] - cw, xNeck[1] - S * 0.012);
      g.quadraticCurveTo(xNeck[0], xNeck[1] + S * 0.085, xNeck[0] + cw, xNeck[1] - S * 0.012);
      g.lineTo(xNeck[0] + cw * 0.9, xNeck[1] - S * 0.062);
      g.quadraticCurveTo(xNeck[0], xNeck[1] + S * 0.018, xNeck[0] - cw * 0.9, xNeck[1] - S * 0.062);
      g.closePath();
      g.fillStyle = gar.base; g.fill();
      g.strokeStyle = gar.line; g.lineWidth = 1.6; g.stroke();
      const top = xNeck[1] + S * 0.075;
      strokeSeg(g, [xNeck[0], top], [xNeck[0], top + S * 0.34], gar.trim, 2);
      [0.30, 0.62, 0.94].forEach(t => disc(g, [xNeck[0], top + S * 0.34 * t], 2.1, gar.trim));
    }
  }

  function drawBackHair(g, cx, cy, rx, ry, S, st) {
    if (st.hairStyle !== "long") return;
    const h = st.hairColors;
    g.beginPath();
    g.moveTo(cx - rx * 1.06, cy - ry * 0.25);
    g.bezierCurveTo(cx - rx * 1.40, cy + ry * 0.7, cx - rx * 1.30, cy + ry * 1.8, cx - rx * 1.22, cy + ry * 2.35);
    g.lineTo(cx + rx * 1.22, cy + ry * 2.35);
    g.bezierCurveTo(cx + rx * 1.30, cy + ry * 1.8, cx + rx * 1.40, cy + ry * 0.7, cx + rx * 1.06, cy - ry * 0.25);
    g.closePath();
    g.fillStyle = h.base; g.fill();
  }

  function drawHead(g, cx, cy, rx, ry, S, st, openness) {
    const skin = st.skin, h = st.hairColors, pr = st.prop;

    // ears
    [-1, 1].forEach(s => {
      g.fillStyle = skin.shade; g.strokeStyle = skin.line; g.lineWidth = 1.6;
      g.beginPath(); g.ellipse(cx + s * rx * 0.99, cy + ry * 0.10, rx * pr.ear, ry * 0.19, 0, 0, Math.PI * 2);
      g.fill(); g.stroke();
    });

    // face: cheekbones, then a jaw that is rounded (slim) or squarer (broad)
    const c1y = 0.58 + 0.14 * pr.jaw, c2x = 0.52 + 0.26 * pr.jaw;
    g.beginPath();
    g.moveTo(cx, cy - ry);
    g.bezierCurveTo(cx + rx * 0.80, cy - ry, cx + rx, cy - ry * 0.50, cx + rx, cy + ry * 0.04);
    g.bezierCurveTo(cx + rx, cy + ry * c1y, cx + rx * c2x, cy + ry, cx, cy + ry);
    g.bezierCurveTo(cx - rx * c2x, cy + ry, cx - rx, cy + ry * c1y, cx - rx, cy + ry * 0.04);
    g.bezierCurveTo(cx - rx, cy - ry * 0.50, cx - rx * 0.80, cy - ry, cx, cy - ry);
    g.closePath();
    g.fillStyle = skin.base; g.fill();
    g.strokeStyle = skin.line; g.lineWidth = 2; g.stroke();

    // cheek/jaw shade (one soft crescent on the viewer's right = light from the left)
    g.fillStyle = skin.shade;
    g.beginPath();
    g.moveTo(cx + rx * 0.98, cy + ry * 0.10);
    g.bezierCurveTo(cx + rx * 0.96, cy + ry * 0.60, cx + rx * 0.50, cy + ry * 0.98, cx + rx * 0.02, cy + ry * 0.99);
    g.bezierCurveTo(cx + rx * 0.62, cy + ry * 0.80, cx + rx * 0.86, cy + ry * 0.50, cx + rx * 0.98, cy + ry * 0.10);
    g.closePath(); g.fill();

    // hair: short crop with a soft side parting
    const top = cy - ry * 1.13, hl = cy - ry * 0.50;
    g.beginPath();
    g.moveTo(cx - rx * 1.04, cy + ry * 0.05);
    g.bezierCurveTo(cx - rx * 1.16, cy - ry * 0.95, cx - rx * 0.55, top, cx, top);
    g.bezierCurveTo(cx + rx * 0.55, top, cx + rx * 1.16, cy - ry * 0.95, cx + rx * 1.04, cy + ry * 0.05);
    g.bezierCurveTo(cx + rx * 1.02, cy - ry * 0.28, cx + rx * 0.82, hl - ry * 0.02, cx + rx * 0.42, hl - ry * 0.05);
    g.bezierCurveTo(cx + rx * 0.20, hl - ry * 0.16, cx - rx * 0.02, hl - ry * 0.16, cx - rx * 0.24, hl - ry * 0.04);
    g.bezierCurveTo(cx - rx * 0.62, hl + ry * 0.02, cx - rx * 0.98, cy - ry * 0.25, cx - rx * 1.04, cy + ry * 0.05);
    g.closePath();
    g.fillStyle = h.base; g.fill();
    g.strokeStyle = h.shine; g.lineWidth = 1.6; g.lineCap = "round";
    [[-0.55, -0.80, 0.05, -0.98], [-0.10, -0.92, 0.50, -0.90], [0.30, -0.78, 0.78, -0.62]].forEach(([x0, y0, x1, y1]) => {
      g.beginPath();
      g.moveTo(cx + rx * x0, cy + ry * y0);
      g.quadraticCurveTo(cx + rx * (x0 + x1) / 2, cy + ry * (Math.min(y0, y1) - 0.10), cx + rx * x1, cy + ry * y1);
      g.stroke();
    });

    // eyebrows: full and natural; heavier and flatter on the broad build
    const browY = cy - ry * 0.30;
    g.strokeStyle = h.base; g.lineWidth = Math.max(3, ry * pr.brow); g.lineCap = "round";
    [-1, 1].forEach(s => {
      g.beginPath();
      g.moveTo(cx + s * rx * 0.13, browY + ry * 0.015);
      g.quadraticCurveTo(cx + s * rx * 0.40, browY - ry * pr.browArch, cx + s * rx * 0.66, browY + ry * 0.03);
      g.stroke();
    });

    // eyes
    const eyeY = cy - ry * 0.10, ew = rx * 0.235;
    const eh = Math.max(0.6, ry * pr.lid * openness);
    const irisR = ry * 0.062;
    [-1, 1].forEach(s => {
      const ex = cx + s * rx * 0.38;
      g.fillStyle = "#f4eee6";
      g.beginPath(); g.ellipse(ex, eyeY, ew, eh + 0.4, 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = skin.iris;
      g.beginPath(); g.ellipse(ex, eyeY, irisR, Math.min(irisR, eh + 0.3), 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#0b0706";
      g.beginPath(); g.ellipse(ex, eyeY, irisR * 0.5, Math.min(irisR * 0.5, eh * 0.8 + 0.2), 0, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "#1a100c"; g.lineWidth = 2.0; g.lineCap = "round";
      g.beginPath(); g.moveTo(ex - ew * 1.05, eyeY + eh * 0.1);
      g.quadraticCurveTo(ex, eyeY - eh * 2.1 - 0.6, ex + ew * 1.05, eyeY + eh * 0.1); g.stroke();
      g.strokeStyle = skin.line; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(ex - ew * 0.8, eyeY + eh + 1.2);
      g.quadraticCurveTo(ex, eyeY + eh * 2.0 + 1.8, ex + ew * 0.8, eyeY + eh + 1.2); g.stroke();
    });

    // nose: bridge shadow and nostril wings, kept subtle
    g.strokeStyle = skin.line; g.lineCap = "round"; g.lineWidth = 1.8;
    g.beginPath(); g.moveTo(cx - rx * 0.06, cy + ry * 0.08);
    g.quadraticCurveTo(cx - rx * 0.13, cy + ry * 0.22, cx - rx * 0.14, cy + ry * 0.31); g.stroke();
    g.beginPath(); g.moveTo(cx - rx * 0.18, cy + ry * 0.33);
    g.quadraticCurveTo(cx, cy + ry * 0.42, cx + rx * 0.18, cy + ry * 0.33); g.stroke();

    // mouth: calm and closed. thinner, lower-contrast lips on the broad build
    const my = cy + ry * 0.60, mw = rx * 0.30, L = pr.lip;
    const lipColor = mix(skin.lip, skin.base, pr.lipTone);
    g.fillStyle = lipColor;
    g.beginPath();
    g.moveTo(cx - mw, my);
    g.quadraticCurveTo(cx - mw * 0.45, my - ry * 0.055 * L, cx, my - ry * 0.02 * L);
    g.quadraticCurveTo(cx + mw * 0.45, my - ry * 0.055 * L, cx + mw, my);
    g.quadraticCurveTo(cx, my + ry * 0.03 * L, cx - mw, my);
    g.closePath(); g.fill();
    g.beginPath();
    g.moveTo(cx - mw * 0.9, my + ry * 0.01);
    g.quadraticCurveTo(cx, my + ry * 0.17 * L, cx + mw * 0.9, my + ry * 0.01);
    g.quadraticCurveTo(cx, my + ry * 0.045 * L, cx - mw * 0.9, my + ry * 0.01);
    g.closePath(); g.fill();
    g.strokeStyle = skin.line; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(cx - mw, my - ry * 0.005);
    g.quadraticCurveTo(cx, my + ry * 0.04, cx + mw, my - ry * 0.005); g.stroke();
  }

  // Samples a polyline every few pixels and gives each sample a width that tapers from w0 to w1.
  function taperSamples(pts, w0, w1) {
    const seg = [0];
    for (let i = 1; i < pts.length; i++) seg.push(seg[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const total = seg[seg.length - 1] || 1;
    const n = Math.max(5, Math.min(9, Math.ceil(total / 5)));   // a few samples is plenty for a smooth taper
    const out = [];
    for (let k = 0; k <= n; k++) {
      const d = total * k / n;
      let i = 1; while (i < seg.length - 1 && seg[i] < d) i++;
      const t = seg[i] === seg[i - 1] ? 0 : (d - seg[i - 1]) / (seg[i] - seg[i - 1]);
      const u = d / total;
      out.push([pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t, w0 + (w1 - w0) * u]);
    }
    return out;
  }
  function strokeTaper(g, samples, color, extra) {
    g.strokeStyle = color; g.lineCap = "round"; g.lineJoin = "round";
    for (let i = 0; i < samples.length - 1; i++) {
      g.lineWidth = (samples[i][2] + samples[i + 1][2]) / 2 + extra;
      g.beginPath(); g.moveTo(samples[i][0], samples[i][1]); g.lineTo(samples[i + 1][0], samples[i + 1][1]); g.stroke();
    }
  }

  // joints: 21 [x, y, z] screen points. unit: finger width in px.
  // Palm + thumb pad form one silhouette; fingers taper toward the tip and overlap by depth.
  function drawHand(g, joints, unit, skin) {
    const line = mix(skin.line, skin.base, 0.18);       // soft outline, not a hard black rim
    const o = Math.max(1.0, unit * 0.11);
    const pw = unit * 1.08;
    const palmIdx = [0, 1, 5, 9, 13, 17];
    const palmPath = () => {
      g.beginPath();
      palmIdx.forEach((idx, k) => { if (k === 0) g.moveTo(joints[idx][0], joints[idx][1]); else g.lineTo(joints[idx][0], joints[idx][1]); });
      g.closePath();
    };
    g.lineJoin = "round"; g.lineCap = "round";

    // 1) palm and thumb-pad outline
    palmPath(); g.strokeStyle = line; g.lineWidth = pw + o * 2; g.stroke();
    strokeSeg(g, joints[0], joints[2], line, unit * 1.45 + o * 2, "round");
    // 2) palm and thumb-pad fill
    palmPath(); g.fillStyle = skin.base; g.fill();
    palmPath(); g.strokeStyle = skin.base; g.lineWidth = pw; g.stroke();
    strokeSeg(g, joints[0], joints[2], skin.base, unit * 1.45, "round");

    // 3) fingers, farthest from the camera first so nearer ones overlap correctly
    const order = HAND_FINGERS.map(fg => ({ fg, z: fg.chain.reduce((s, i) => s + joints[i][2], 0) / fg.chain.length }))
      .sort((a, b) => b.z - a.z);
    order.forEach(({ fg }) => {
      const pts = fg.chain.map(i => [joints[i][0], joints[i][1]]);
      const isThumb = fg.name === "thumb";
      const w0 = unit * fg.w * (isThumb ? 1.30 : 1.0), w1 = unit * fg.w * (isThumb ? 0.90 : 0.76);
      const samples = taperSamples(pts, w0, w1);
      strokeTaper(g, samples, line, o * 2);
      strokeTaper(g, samples, skin.base, 0);
      disc(g, pts[0], w0 * 0.5 + o * 0.9, skin.base);      // hides the round cap (and its outline ring) where the finger meets the palm
    });
  }

  /**
   * Main AvatarRenderer Class
   */
  class AvatarRenderer {
    constructor(canvas, options) {
      this.canvas = typeof canvas === "string" ? document.getElementById(canvas) : canvas;
      this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
      this.options = Object.assign({
        signs: {},
        speed: 1.0,
        fps: 30,          // raised from 25 to 30
        useIK: true,
        yUnstretch: 9 / 16, // compensates 16/9 training aspect; set to 1.0 to disable
        style: {},        // { skin, hair, hairColor, outfit, garment, trousers, build } - see presets above
        idleMotion: true, // light blinking only; set false for a perfectly still face
        glossStripEl: null,
        captionEl: null,
        statusEl: null,
        onFrame: null,
        onSignStart: null,
        onComplete: null
      }, options || {});

      this.signs = {};
      // Smooth the recorded landmarks once. (index.html passes `signs` here and never calls
      // setSigns() afterwards, so without this the animation played back raw, jittery data.)
      this.setSigns(this.options.signs || {});
      this.speed = this.options.speed;
      this.fps = this.options.fps;
      this.useIK = this.options.useIK;
      this._style = this._resolveStyle(this.options.style);
      this._lastFrame = REST_FRAME;

      // Playback state
      this.timeline = [];
      this.metadata = [];
      this.glossItems = [];
      this.currentFrameIdx = 0;
      this.isPlaying = false;
      this.isPaused = false;
      this.animTimer = null;
      this.activeSequence = [];

      if (this.canvas) {
        this.renderFrame(REST_FRAME);
      }
    }

    _resolveStyle(s) {
      const st = Object.assign({}, DEFAULT_STYLE, s || {});
      return {
        skin: SKINS[st.skin] || SKINS[DEFAULT_STYLE.skin],
        hairColors: HAIRS[st.hairColor] || HAIRS[DEFAULT_STYLE.hairColor],
        hairStyle: st.hair === "long" ? "long" : "short",
        outfit: st.outfit === "shirt" ? "shirt" : "kurta",
        garment: GARMENTS[st.garment] || GARMENTS[DEFAULT_STYLE.garment],
        trousers: TROUSERS[st.trousers] || TROUSERS[DEFAULT_STYLE.trousers],
        prop: BUILDS[st.build] || BUILDS[DEFAULT_STYLE.build]
      };
    }

    // Change the look at any time, e.g. avatar.setStyle({ skin: "deep", garment: "ecru" })
    setStyle(partial) {
      this.options.style = Object.assign({}, this.options.style || {}, partial || {});
      this._style = this._resolveStyle(this.options.style);
      if (this.canvas) this.renderFrame(this._lastFrame || REST_FRAME);
    }

    _clock() {
      return (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
    }

    // 1 = eyes open; dips briefly every ~4 seconds (only advances while frames are being drawn)
    _blinkOpenness() {
      if (this.options.idleMotion === false) return 1;
      const t = this._clock() % 3.9;
      return t < 0.16 ? 1 - 0.92 * Math.sin(Math.PI * t / 0.16) : 1;
    }

    setSigns(signs) {
      // Smooth each sign sequence once on load
      this.signs = {};
      if (!signs) return;
      for (const [id, seq] of Object.entries(signs)) {
        this.signs[id] = Array.isArray(seq) ? smoothSequence(seq.map(f => new Float32Array(f))) : seq;
      }
    }

    setSpeed(speed) {
      this.speed = Math.max(0.2, Math.min(3.0, speed));
    }

    // Build timeline frames from sequence of sign IDs
    buildTimeline(sequence) {
      const frames = [];
      const metadata = [];
      const glossItems = [];
      const restFrames = Math.max(4, Math.round(6 / this.speed));
      const transFrames = Math.max(4, Math.round(8 / this.speed));

      // Nosign hold length: max(14, round(18/speed)) frames
      const nosignHold = Math.max(14, Math.round(18 / this.speed));

      // 1. Intro rest
      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);
        metadata.push({ signId: "REST", signIndex: -1, isTrans: false, progress: 0, gloss: "Ready", status: "rest" });
      }

      let prevLast = REST_FRAME;
      const nosignWords = []; // collect English words with no recorded sign

      // 2. Sequential signs
      sequence.forEach((item, seqIdx) => {
        // Accept plain string IDs (legacy) or plan-item dicts {sign_id, gloss, kind}
        let rawId, gloss, kind, conceptEnglish;
        if (typeof item === "string") {
          rawId = item.toLowerCase().trim();
          gloss = rawId.toUpperCase();
          kind = "sign";
          conceptEnglish = gloss;
        } else {
          rawId = (item.sign_id || item.concept || "").toLowerCase().trim();
          gloss = item.gloss || rawId.toUpperCase();
          kind = item.kind || "sign";
          conceptEnglish = item.gloss || gloss;
        }

        const signSeq = this.signs[rawId];

        // Treat marker, nosign, and any sign with no data as a rest hold
        const isMissingSign = !signSeq || !signSeq.length;
        if (kind === "marker" || kind === "nosign" || (kind === "sign" && isMissingSign)) {
          const status = kind === "marker" ? "marker" : "nosign";
          if (kind === "nosign" || (kind === "sign" && isMissingSign)) {
            nosignWords.push(conceptEnglish);
          }
          const holdCount = nosignHold;
          glossItems.push({
            id: rawId, label: gloss, index: seqIdx,
            startFrame: frames.length, endFrame: frames.length + holdCount - 1, status
          });
          for (let f = 0; f < holdCount; f++) {
            frames.push(REST_FRAME);
            metadata.push({
              signId: rawId, signIndex: seqIdx, isTrans: false,
              progress: f / holdCount, gloss, status
            });
          }
          prevLast = REST_FRAME;
          return;
        }

        // Ease-in-out LERP transition from previous pose to first frame
        const firstFrame = signSeq[0];
        for (let t = 0; t < transFrames; t++) {
          const alpha = easeInOut(t / transFrames);
          frames.push(lerpFrame(prevLast, firstFrame, alpha));
          metadata.push({
            signId: rawId, signIndex: seqIdx, isTrans: true,
            progress: 0, gloss: `-> ${gloss}`, status: "native"
          });
        }

        const startFrame = frames.length;
        const totalRaw = signSeq.length;
        const signFrameCount = Math.max(12, Math.round(totalRaw / this.speed));
        for (let f = 0; f < signFrameCount; f++) {
          const progress = f / Math.max(1, signFrameCount - 1);
          const rawIdx = Math.min(totalRaw - 1, Math.floor(progress * (totalRaw - 1)));
          frames.push(signSeq[rawIdx]);
          metadata.push({ signId: rawId, signIndex: seqIdx, isTrans: false, progress, gloss, status: "native" });
        }
        const endFrame = frames.length - 1;
        glossItems.push({ id: rawId, label: gloss, index: seqIdx, startFrame, endFrame, status: "native" });
        prevLast = signSeq[totalRaw - 1];
      });

      // 3. Outro transition to rest pose
      for (let t = 0; t < transFrames; t++) {
        const alpha = easeInOut(t / transFrames);
        frames.push(lerpFrame(prevLast, REST_FRAME, alpha));
        metadata.push({ signId: "REST", signIndex: -1, isTrans: true, progress: 1, gloss: "-> Rest", status: "rest" });
      }

      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);
        metadata.push({ signId: "REST", signIndex: -1, isTrans: false, progress: 1, gloss: "Done", status: "rest" });
      }

      return { frames, metadata, glossItems, nosignWords };
    }

    // Playback control
    play(sequence, options) {
      if (!sequence || !sequence.length) return;
      if (options) {
        if (options.speed) this.setSpeed(options.speed);
        if (options.onComplete) this.options.onComplete = options.onComplete;
      }
      this.activeSequence = sequence.slice();
      this.stop();

      const compiled = this.buildTimeline(sequence);
      this.timeline = compiled.frames;
      this.metadata = compiled.metadata;
      this.glossItems = compiled.glossItems;
      this.nosignWords = compiled.nosignWords || [];
      this.currentFrameIdx = 0;
      this.isPlaying = true;
      this.isPaused = false;

      this.renderGlossStrip();
      this._renderNosignNote();
      this.tick();
    }

    pause() {
      if (this.isPlaying && !this.isPaused) {
        this.isPaused = true;
        if (this.animTimer) { clearTimeout(this.animTimer); this.animTimer = null; }
      }
    }

    resume() {
      if (this.isPlaying && this.isPaused) {
        this.isPaused = false;
        this.tick();
      }
    }

    replay() {
      if (this.activeSequence.length) { this.play(this.activeSequence); }
    }

    stop() {
      this.isPlaying = false;
      this.isPaused = false;
      if (this.animTimer) { clearTimeout(this.animTimer); this.animTimer = null; }
      this.currentFrameIdx = 0;
    }

    tick() {
      if (!this.isPlaying || this.isPaused) return;

      if (this.currentFrameIdx >= this.timeline.length) {
        this.isPlaying = false;
        this.renderFrame(REST_FRAME);
        this.updateCaption("Ready");
        this.highlightGloss(-1);
        if (this.options.onComplete) this.options.onComplete();
        return;
      }

      const frame = this.timeline[this.currentFrameIdx];
      const meta = this.metadata[this.currentFrameIdx];

      this.renderFrame(frame);
      this.updateCaption(meta.gloss + (meta.status === "nosign" ? " (no sign recorded)" : ""));
      this.highlightGloss(meta.signIndex, meta.progress);

      if (this.options.onFrame) { this.options.onFrame(this.currentFrameIdx, meta); }

      this.currentFrameIdx++;
      const frameInterval = Math.round(1000 / this.fps);
      this.animTimer = setTimeout(() => this.tick(), frameInterval);
    }

    // Highlighting synced gloss strip
    renderGlossStrip() {
      const container = this.options.glossStripEl || document.getElementById("gloss-strip");
      if (!container) return;
      container.innerHTML = "";
      this.glossItems.forEach((item, idx) => {
        const pill = document.createElement("span");
        pill.id = `gloss-pill-${idx}`;
        pill.className = `gloss-pill ${item.status}`;
        pill.textContent = item.label;
        pill.title = item.status === "nosign" ? "No recorded sign yet" : item.label;
        pill.onclick = () => { this.currentFrameIdx = item.startFrame; };
        container.appendChild(pill);
      });
    }

    _renderNosignNote() {
      // Show a one-line note for words without a recorded sign
      const el = document.getElementById("nosign-note");
      if (!el) return;
      const words = this.nosignWords || [];
      if (words.length > 0) {
        el.textContent = "No recorded sign yet for: " + words.join(", ");
        el.style.display = "";
      } else {
        el.textContent = "";
        el.style.display = "none";
      }
    }

    highlightGloss(activeIdx, progress) {
      this.glossItems.forEach((_, idx) => {
        const pill = document.getElementById(`gloss-pill-${idx}`);
        if (!pill) return;
        if (idx === activeIdx) {
          pill.classList.add("active");
          pill.classList.remove("passed");
          if (progress !== undefined) {
            pill.style.setProperty("--progress", `${Math.round(progress * 100)}%`);
          }
        } else if (idx < activeIdx) {
          pill.classList.remove("active");
          pill.classList.add("passed");
        } else {
          pill.classList.remove("active", "passed");
        }
      });
    }

    updateCaption(text) {
      const el = this.options.captionEl || document.getElementById("caption");
      if (el) el.textContent = text || "";
    }

    // Draw the signer: body, head and arms first, then the hands on top so they are never hidden.
    renderFrame(f) {
      if (!this.ctx || !this.canvas) return;
      this._lastFrame = f;
      const g = this.ctx;
      const W = this.canvas.width;
      const H = this.canvas.height;
      // Framing was fitted to the measured extents of all recorded signs so hanging hands and
      // raised hands stay in view: scale 0.26*H, neck line at 32% of the height.
      const S = H * 0.26;
      const ox = W * 0.50;
      const oy = H * 0.32;
      const st = this._style;

      // yUnstretch: compensate 16/9 aspect ratio from training
      const U = (this.options && this.options.yUnstretch !== undefined) ? this.options.yUnstretch : (9 / 16);

      g.clearRect(0, 0, W, H);
      g.lineCap = "round";
      g.lineJoin = "round";

      // P(i) applies y-unstretch to pose landmarks
      const P = i => [f[126 + i * 3], f[126 + i * 3 + 1] * U, f[126 + i * 3 + 2]];
      const ok = i => f[126 + i * 3] !== 0 || f[126 + i * 3 + 1] !== 0;
      const X = p => [ox + p[0] * S, oy + p[1] * S];

      if (ok(11) && ok(12)) {
        const p11 = P(11), p12 = P(12);
        const x11 = X(p11), x12 = X(p12);
        const xNeck = X([(p11[0] + p12[0]) * 0.5, (p11[1] + p12[1]) * 0.5, (p11[2] + p12[2]) * 0.5]);
        const xNose = X(ok(0) ? P(0) : [0, -1.0 * U, 0]);
        const head = [xNeck[0] * 0.2 + xNose[0] * 0.8, xNeck[1] * 0.2 + xNose[1] * 0.8];
        const rx = S * 0.215, ry = S * 0.285;

        drawBackHair(g, head[0], head[1], rx, ry, S, st);
        drawTorso(g, x11, x12, xNeck, S, st, H);
        drawNeckAndCollar(g, xNeck, head[0], head[1], rx, ry, S, st);
        drawHead(g, head[0], head[1], rx, ry, S, st, this._blinkOpenness());

        // Arms: recorded elbows when present, otherwise two-bone IK.
        const p15 = ok(15) ? P(15) : [0.48, 2.75 * U, 0];
        const p16 = ok(16) ? P(16) : [-0.48, 2.75 * U, 0];
        let p13, p14;
        if (ok(13)) p13 = P(13);
        else p13 = this.useIK ? solveTwoBoneIK(p11, p15, null, true, 0.85, 0.8) : [p11[0] + 0.08, (p11[1] + p15[1]) * 0.5, 0];
        if (ok(14)) p14 = P(14);
        else p14 = this.useIK ? solveTwoBoneIK(p12, p16, null, false, 0.85, 0.8) : [p12[0] - 0.08, (p12[1] + p16[1]) * 0.5, 0];

        const skin = st.skin, gar = st.garment;
        const drawArm = (ps, pe, pw) => {
          const xs = X(ps), xe = X(pe), xw = X(pw);
          const xc = lerp2(xe, xw, 0.58);                 // where the sleeve ends on the forearm
          const wUp = S * st.prop.wUp, wFore = S * st.prop.wFore, wSkin = S * st.prop.wSkin, ol = 1.8;
          // bare forearm
          limb(g, xe, xw, wSkin, skin.base, skin.line, ol, "round");
          // sleeve: outlines first, then fills, so no seam shows at the elbow
          strokeSeg(g, xs, xe, gar.line, wUp + ol * 2, "round");
          strokeSeg(g, xe, xc, gar.line, wFore + ol * 2, "butt");
          strokeSeg(g, xs, xe, gar.base, wUp, "round");
          strokeSeg(g, xe, xc, gar.base, wFore, "butt");
          // crisp cuff edge
          const dx = xw[0] - xe[0], dy = xw[1] - xe[1], d = Math.hypot(dx, dy) || 1;
          const nx = -dy / d, ny = dx / d, hw = wFore / 2 + ol;
          strokeSeg(g, [xc[0] - nx * hw, xc[1] - ny * hw], [xc[0] + nx * hw, xc[1] + ny * hw], gar.line, 2.4, "butt");
        };
        drawArm(p11, p13, p15);
        drawArm(p12, p14, p16);
      }

      // Hands: left block [0,63) belongs to pose wrist 15; right block [63,126) to wrist 16.
      [[0, 15], [63, 16]].forEach(([offset, wristPoseIdx]) => {
        if (!ok(wristPoseIdx)) return;
        let any = false;
        for (let i = offset; i < offset + 63; i++) { if (f[i] !== 0) { any = true; break; } }
        // The wrist is tracked but no hand landmarks were: draw a relaxed hand instead of leaving
        // a handless stump. (Usually a missed detection in the recording, not an absent hand.)
        const src = any ? f : REST_FRAME;

        const wP = P(wristPoseIdx);  // already unscaled by P()
        // wrist-to-middle-fingertip distance in hand-block space, with U applied to y
        const dd = Math.hypot(src[offset + 36], src[offset + 37] * U) || 1;
        const hs = clamp(0.5 / dd, 0.25, 1.2);  // hand scale

        const joints = [];
        for (let i = 0; i < 21; i++) {
          joints.push([
            ox + (wP[0] + src[offset + i * 3] * hs) * S,
            oy + (wP[1] + src[offset + i * 3 + 1] * U * hs) * S,
            src[offset + i * 3 + 2]
          ]);
        }
        drawHand(g, joints, S * 0.5 * 0.124, st.skin);
      });
    }
  }

  // Export to global scope
  global.AvatarRenderer = AvatarRenderer;
  global.createAvatarRenderer = function (canvas, options) {
    return new AvatarRenderer(canvas, options);
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      AvatarRenderer,
      REST_FRAME,
      solveTwoBoneIK,
      easeInOut,
      lerpFrame,
      smoothSequence,
      SKINS, HAIRS, GARMENTS, TROUSERS, BUILDS, DEFAULT_STYLE
    };
  }

})(typeof window !== "undefined" ? window : global);