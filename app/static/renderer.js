/**
 * ClarifySign Decoupled Avatar Renderer (HTML5 Canvas 2D)
 *
 * Custom illustrated avatar: blue/yellow pagri (turban) with polka dots, sunglasses,
 * curled moustache, yellow kurta with a blue inner shirt and blue detailing.
 *
 * Hands are drawn as natural, human hands: human palm proportions, a consistent size
 * (so they do not pulse while fingers open and close), a smooth palm shape, tapered
 * fingers with knuckle creases and soft palm shading.
 *
 * Hand and pose smoothing preserve detected landmarks across missing frames.
 * Playback logic, timeline, IK, exports and sign data format are retained.
 * ZERO dependencies on NLU or planner modules.
 */

(function (global) {
  "use strict";

  // Hand look. Recorded hand landmarks are vertically stretched about 1.5x more than a real
  // hand, so hand Y is scaled by HAND_ASPECT on top of the body unstretch. HAND_PALM is the
  // target wrist->knuckle length of the drawn hand (in shoulder-width units): hands are
  // resized to it per frame so their size stays natural and steady.
  const HAND_ASPECT = 0.68;
  const HAND_PALM = 0.27;

  const HAND_PALM_CONNECTIONS = [
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]
  ];

  const HAND_FINGERS = [
    { name: "thumb", chain: [1, 2, 3, 4], w: 1.20 },
    { name: "index", chain: [5, 6, 7, 8], w: 1.00 },
    { name: "middle", chain: [9, 10, 11, 12], w: 1.04 },
    { name: "ring", chain: [13, 14, 15, 16], w: 0.95 },
    { name: "pinky", chain: [17, 18, 19, 20], w: 0.80 }
  ];

  const SKINS = {
    light: { base: "#e2b790", shade: "#c99a75", line: "#8a5a3d", lip: "#b87565", iris: "#3b2a20" },
    medium: { base: "#c68e64", shade: "#aa734c", line: "#6f4228", lip: "#9a5646", iris: "#2e1f17" },
    deep: { base: "#8f5c3c", shade: "#744729", line: "#44270f", lip: "#6c382c", iris: "#1f140e" }
  };

  const HAIRS = {
    black: { base: "#1b1512", shine: "#4a3a32" },
    brown: { base: "#2f2019", shine: "#5c4337" }
  };

  const GARMENTS = {
    indigo: { base: "#4a5f9e", shade: "#3a4b82", line: "#232f58", trim: "#7184bd" },
    ecru: { base: "#e7dfcf", shade: "#cdc3ae", line: "#8f8570", trim: "#b8ac94" },
    clay: { base: "#b86a50", shade: "#9a523b", line: "#65311f", trim: "#d79479" },
    sage: { base: "#7d9983", shade: "#667f6c", line: "#415648", trim: "#a4bba9" },
    yellow: { base: "#f2c400", shade: "#d9a900", line: "#26334b", trim: "#168bd2" }
  };

  const TROUSERS = {
    charcoal: { base: "#2f3541", line: "#1b1f27" },
    navy: { base: "#2a3552", line: "#171e33" },
    khaki: { base: "#8c7d5e", line: "#5d513a" }
  };

  const DEFAULT_STYLE = {
    skin: "medium",
    hair: "short",
    hairColor: "black",
    outfit: "kurta",
    garment: "yellow",
    trousers: "charcoal",
    build: "broad"
  };

  function easeInOut(t) {
    t = Math.max(0, Math.min(1, t));
    return t * t * (3 - 2 * t);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  // Interpolation must not shrink a detected hand toward zero when the
  // neighbouring frame contains a missing-hand block.
  function lerpFrame(f1, f2, alpha) {
    const out = new Float32Array(225);

    for (let i = 0; i < 225; i++) {
      out[i] = (1 - alpha) * f1[i] + alpha * f2[i];
    }

    const blockPresent = (frame, start, end) => {
      for (let i = start; i < end; i++) {
        if (frame[i] !== 0) return true;
      }
      return false;
    };

    [[0, 63], [63, 126]].forEach(([start, end]) => {
      const aPresent = blockPresent(f1, start, end);
      const bPresent = blockPresent(f2, start, end);

      if (aPresent && !bPresent) {
        out.set(f1.subarray(start, end), start);
      } else if (!aPresent && bPresent) {
        out.set(f2.subarray(start, end), start);
      } else if (!aPresent && !bPresent) {
        out.fill(0, start, end);
      }
    });

    // Keep valid pose geometry instead of interpolating it toward zero.
    const posePresent = frame => {
      const a = 126 + 11 * 3;
      const b = 126 + 12 * 3;

      return (frame[a] !== 0 || frame[a + 1] !== 0) &&
        (frame[b] !== 0 || frame[b + 1] !== 0);
    };

    const aPose = posePresent(f1);
    const bPose = posePresent(f2);

    if (aPose && !bPose) {
      out.set(f1.subarray(126, 225), 126);
    } else if (!aPose && bPose) {
      out.set(f2.subarray(126, 225), 126);
    } else if (!aPose && !bPose) {
      out.fill(0, 126, 225);
    }

    return out;
  }

  // Smooth only across frames where the relevant body part was detected.
  // This prevents hands and shoulders from shrinking at detection boundaries.
  function smoothSequence(seq) {
    if (!seq || seq.length < 3) return seq;

    const blockPresent = (frame, a, b) => {
      for (let i = a; i < b; i++) {
        if (frame[i] !== 0) return true;
      }
      return false;
    };

    const posePresent = frame =>
      (frame[126 + 11 * 3] !== 0 || frame[126 + 11 * 3 + 1] !== 0) &&
      (frame[126 + 12 * 3] !== 0 || frame[126 + 12 * 3 + 1] !== 0);

    const out = [];

    for (let i = 0; i < seq.length; i++) {
      const candidates = [
        seq[Math.max(0, i - 1)],
        seq[i],
        seq[Math.min(seq.length - 1, i + 1)]
      ];

      const curr = seq[i];
      const sm = new Float32Array(curr);

      const blocks = [
        { start: 0, end: 63, valid: f => blockPresent(f, 0, 63) },
        { start: 63, end: 126, valid: f => blockPresent(f, 63, 126) },
        { start: 126, end: 225, valid: posePresent }
      ];

      blocks.forEach(({ start, end, valid }) => {
        if (!valid(curr)) {
          sm.fill(0, start, end);
          return;
        }

        const usable = candidates.filter(valid);

        for (let d = start; d < end; d++) {
          sm[d] = usable.reduce((sum, frame) => sum + frame[d], 0) / usable.length;
        }
      });

      out.push(sm);
    }

    return out;
  }

  // Canonical Rest Frame (225 dimensions).
  function buildRestFrame() {
    const f = new Float32Array(225);

    const setPose = (idx, x, y, z) => {
      const base = 126 + idx * 3;
      f[base] = x;
      f[base + 1] = y;
      f[base + 2] = z;
    };

    setPose(0, 0.0, -1.0, -0.3);
    setPose(7, 0.35, -1.0, 0.0);
    setPose(8, -0.35, -1.0, 0.0);
    setPose(9, 0.12, -0.85, -0.2);
    setPose(10, -0.12, -0.85, -0.2);
    setPose(11, 0.5, 0.0, 0.0);
    setPose(12, -0.5, 0.0, 0.0);
    setPose(13, 0.56, 1.30, 0.15);
    setPose(14, -0.56, 1.30, 0.15);
    setPose(15, 0.40, 2.30, 0.0);
    setPose(16, -0.40, 2.30, 0.0);
    setPose(23, 0.28, 2.50, 0.0);
    setPose(24, -0.28, 2.50, 0.0);

    const HAND_REST = [
      [0.00, 0.00, 0.00],
      [0.14, 0.12, -0.05], [0.26, 0.26, -0.06], [0.31, 0.38, -0.06], [0.33, 0.50, -0.06],
      [0.15, 0.42, -0.02], [0.16, 0.62, -0.02], [0.16, 0.76, -0.02], [0.15, 0.88, -0.02],
      [0.04, 0.44, 0.00], [0.04, 0.66, 0.00], [0.04, 0.80, 0.00], [0.04, 0.92, 0.00],
      [-0.07, 0.43, 0.02], [-0.07, 0.63, 0.02], [-0.07, 0.76, 0.02], [-0.07, 0.86, 0.02],
      [-0.17, 0.40, 0.04], [-0.18, 0.56, 0.04], [-0.18, 0.67, 0.04], [-0.18, 0.76, 0.04]
    ];

    [false, true].forEach(isRight => {
      const side = isRight ? 1.0 : -1.0;

      HAND_REST.forEach(([x, y, z], idx) => {
        const base = (isRight ? 63 : 0) + idx * 3;
        f[base] = side * x;
        // hand Y is scaled by HAND_ASPECT when drawn, so pre-compensate to keep the rest hand's shape
        f[base + 1] = y / HAND_ASPECT;
        f[base + 2] = z;
      });
    });

    return f;
  }

  const REST_FRAME = buildRestFrame();

  // Two-bone analytical inverse kinematics for arms.
  function solveTwoBoneIK(shoulder, wrist, elbowHint, isLeft, l1, l2) {
    l1 = l1 !== undefined ? l1 : 0.85;
    l2 = l2 !== undefined ? l2 : 0.80;

    const dx = wrist[0] - shoulder[0];
    const dy = wrist[1] - shoulder[1];
    const d = Math.hypot(dx, dy);

    if (d < 1e-4) {
      return [
        shoulder[0] + (isLeft ? l1 : -l1),
        shoulder[1],
        shoulder[2]
      ];
    }

    const ux = dx / d;
    const uy = dy / d;
    const normSign = isLeft ? 1.0 : -1.0;

    let vx = -uy * normSign;
    let vy = ux * normSign;

    if (elbowHint) {
      const ehx = elbowHint[0] - shoulder[0];
      const ehy = elbowHint[1] - shoulder[1];

      if (ehx * vx + ehy * vy < 0) {
        vx = -vx;
        vy = -vy;
      }
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

  // Canvas drawing helpers.
  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function mix(a, b, t) {
    const A = hexToRgb(a);
    const B = hexToRgb(b);

    return "#" + [0, 1, 2].map(i =>
      Math.round(A[i] + (B[i] - A[i]) * t).toString(16).padStart(2, "0")
    ).join("");
  }

  function strokeSeg(g, a, b, color, width, cap) {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.lineCap = cap || "round";
    g.lineJoin = "round";
    g.beginPath();
    g.moveTo(a[0], a[1]);
    g.lineTo(b[0], b[1]);
    g.stroke();
  }

  function disc(g, p, r, fill) {
    g.fillStyle = fill;
    g.beginPath();
    g.arc(p[0], p[1], r, 0, Math.PI * 2);
    g.fill();
  }

  function limb(g, a, b, width, fill, line, outline, cap) {
    strokeSeg(g, a, b, line, width + outline * 2, cap);
    strokeSeg(g, a, b, fill, width, cap);
  }

  function lerp2(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t
    ];
  }

  const BUILDS = {
    broad: {
      cap: 0.115, rise: 0.045, neckTop: 0.46, neckBase: 0.58,
      wUp: 0.215, wFore: 0.186, wSkin: 0.112,
      jaw: 1.0, brow: 0.095, browArch: 0.05,
      lip: 0.62, lipTone: 0.35, ear: 0.145, lid: 0.060
    },
    slim: {
      cap: 0.075, rise: 0.075, neckTop: 0.36, neckBase: 0.46,
      wUp: 0.190, wFore: 0.162, wSkin: 0.105,
      jaw: 0.0, brow: 0.075, browArch: 0.085,
      lip: 1.0, lipTone: 0.0, ear: 0.13, lid: 0.068
    }
  };

  function drawTorso(g, x11, x12, xNeck, S, st, canvasH) {
    const gar = st.garment;
    const tr = st.trousers;
    const pr = st.prop;

    const L = x12;
    const R = x11;
    const cap = S * pr.cap;
    const neckHalf = S * (pr.neckBase * 0.215 * 0.9);
    const hemY = xNeck[1] + S * 1.62;
    const bottomY = canvasH + 8;

    const bl = [
      xNeck[0] + (L[0] - xNeck[0]) * 0.93 - S * 0.03,
      hemY
    ];

    const br = [
      xNeck[0] + (R[0] - xNeck[0]) * 0.93 + S * 0.03,
      hemY
    ];

    // Trousers.
    g.beginPath();
    g.moveTo(bl[0] + S * 0.03, hemY - S * 0.08);
    g.lineTo(br[0] - S * 0.03, hemY - S * 0.08);
    g.lineTo(br[0] - S * 0.07, bottomY);
    g.lineTo(bl[0] + S * 0.07, bottomY);
    g.closePath();
    g.fillStyle = tr.base;
    g.fill();
    g.strokeStyle = tr.line;
    g.lineWidth = 2;
    g.lineJoin = "round";
    g.stroke();

    strokeSeg(g, [xNeck[0], hemY], [xNeck[0], bottomY], tr.line, 1.6);

    // Kurta body.
    g.beginPath();
    g.moveTo(L[0] - cap, L[1] + S * 0.05);
    g.quadraticCurveTo(
      L[0] - cap * 0.2,
      L[1] - S * pr.rise,
      xNeck[0] - neckHalf,
      xNeck[1] - S * 0.02
    );
    g.lineTo(xNeck[0] + neckHalf, xNeck[1] - S * 0.02);
    g.quadraticCurveTo(
      R[0] + cap * 0.2,
      R[1] - S * pr.rise,
      R[0] + cap,
      R[1] + S * 0.05
    );
    g.lineTo(br[0], br[1]);
    g.quadraticCurveTo(xNeck[0], hemY + S * 0.045, bl[0], bl[1]);
    g.closePath();
    g.fillStyle = gar.base;
    g.fill();
    g.strokeStyle = gar.line;
    g.lineWidth = 2;
    g.lineJoin = "round";
    g.stroke();

    // Side shading.
    [[L, bl, 1], [R, br, -1]].forEach(([sh, hem, dir]) => {
      g.fillStyle = gar.shade;
      g.beginPath();
      g.moveTo(sh[0] - dir * cap, sh[1] + S * 0.06);
      g.lineTo(hem[0], hem[1]);
      g.lineTo(hem[0] + dir * S * 0.13, hem[1]);
      g.quadraticCurveTo(
        sh[0] + dir * S * 0.10,
        sh[1] + S * 0.9,
        sh[0] + dir * S * 0.12,
        sh[1] + S * 0.12
      );
      g.closePath();
      g.fill();
    });

    // Kurta detailing (as in the reference illustration): curved yoke seams from the
    // neck to the underarm, soft chest creases, and a stitched trim line above the hem.
    [[L, 1], [R, -1]].forEach(([sh, inward]) => {
      const neckX = xNeck[0] - inward * neckHalf * 2.1;
      g.strokeStyle = gar.line; g.lineWidth = 1.6; g.lineCap = "round"; g.globalAlpha = 0.65;
      g.beginPath();
      g.moveTo(neckX, xNeck[1] + S * 0.10);
      g.quadraticCurveTo(sh[0] + inward * S * 0.01, sh[1] + S * 0.24, sh[0] + inward * S * 0.075, sh[1] + S * 0.70);
      g.stroke();
      g.lineWidth = 1.2; g.globalAlpha = 0.4;
      g.beginPath();
      g.moveTo(sh[0] + inward * S * 0.20, sh[1] + S * 0.46);
      g.quadraticCurveTo(sh[0] + inward * S * 0.27, sh[1] + S * 0.70, sh[0] + inward * S * 0.20, sh[1] + S * 0.98);
      g.stroke();
      g.globalAlpha = 1;
    });
    g.save();
    g.setLineDash([S * 0.028, S * 0.02]);
    g.strokeStyle = gar.trim; g.lineWidth = 1.5; g.lineCap = "butt";
    g.beginPath();
    g.moveTo(bl[0] + S * 0.05, bl[1] - S * 0.13);
    g.quadraticCurveTo(xNeck[0], hemY - S * 0.08, br[0] - S * 0.05, br[1] - S * 0.13);
    g.stroke();
    g.restore();

    // Hem and side vents.
    g.strokeStyle = gar.trim;
    g.lineWidth = 1.6;
    g.lineCap = "round";
    g.beginPath();
    g.moveTo(bl[0] + S * 0.04, bl[1] - S * 0.07);
    g.quadraticCurveTo(xNeck[0], hemY - S * 0.02, br[0] - S * 0.04, br[1] - S * 0.07);
    g.stroke();

    strokeSeg(g,
      [bl[0] + S * 0.01, bl[1] - S * 0.16],
      [bl[0] + S * 0.01, bl[1] + S * 0.01],
      gar.line, 1.6
    );

    strokeSeg(g,
      [br[0] - S * 0.01, br[1] - S * 0.16],
      [br[0] - S * 0.01, br[1] + S * 0.01],
      gar.line, 1.6
    );
  }

  function drawNeckAndCollar(g, xNeck, cx, cy, rx, ry, S, st) {
    const skin = st.skin;
    const gar = st.garment;
    const pr = st.prop;

    const nw1 = rx * pr.neckTop;
    const nw2 = rx * pr.neckBase;

    g.beginPath();
    g.moveTo(cx - nw1, cy + ry * 0.55);
    g.lineTo(cx + nw1, cy + ry * 0.55);
    g.lineTo(xNeck[0] + nw2, xNeck[1] + S * 0.01);
    g.lineTo(xNeck[0] - nw2, xNeck[1] + S * 0.01);
    g.closePath();
    g.fillStyle = skin.shade;
    g.fill();
    g.strokeStyle = skin.line;
    g.lineWidth = 1.6;
    g.stroke();

    if (st.outfit === "shirt") {
      g.fillStyle = skin.shade;
      g.beginPath();
      g.moveTo(xNeck[0] - nw2, xNeck[1] - S * 0.02);
      g.lineTo(xNeck[0], xNeck[1] + S * 0.17);
      g.lineTo(xNeck[0] + nw2, xNeck[1] - S * 0.02);
      g.closePath();
      g.fill();

      [-1, 1].forEach(side => {
        g.fillStyle = gar.base;
        g.strokeStyle = gar.line;
        g.lineWidth = 1.6;
        g.beginPath();
        g.moveTo(xNeck[0] + side * nw2 * 0.92, xNeck[1] - S * 0.045);
        g.lineTo(xNeck[0] + side * nw2 * 1.85, xNeck[1] + S * 0.10);
        g.lineTo(xNeck[0] + side * S * 0.012, xNeck[1] + S * 0.185);
        g.closePath();
        g.fill();
        g.stroke();
      });

      strokeSeg(g,
        [xNeck[0], xNeck[1] + S * 0.185],
        [xNeck[0], xNeck[1] + S * 0.62],
        gar.trim, 2
      );

      [0.30, 0.62, 0.94].forEach(t =>
        disc(g, [xNeck[0], xNeck[1] + S * (0.19 + 0.43 * t)], 2.1, gar.trim)
      );
    } else {
      // Kurta with a V neckline: light-blue inner shirt, striped collar band,
      // lighter yellow facings with blue stitched piping, and a button placket.
      const nx = xNeck[0], ny = xNeck[1];
      const ink = gar.line;
      const inner = "#6fb7ee", innerShade = "#4f98d3";
      const facing = mix(gar.base, "#ffffff", 0.12);
      const topW = nw2 * 1.32, outW = nw2 * 2.45, vTip = ny + S * 0.80;

      // blue inner shirt, visible in the V
      g.beginPath();
      g.moveTo(nx - topW, ny - S * 0.03);
      g.lineTo(nx, vTip);
      g.lineTo(nx + topW, ny - S * 0.03);
      g.closePath();
      g.fillStyle = inner; g.fill();
      g.strokeStyle = ink; g.lineWidth = 2; g.lineJoin = "round"; g.stroke();
      // soft shade on one side of the shirt
      g.beginPath();
      g.moveTo(nx + topW * 0.1, ny + S * 0.10);
      g.lineTo(nx, vTip - S * 0.02);
      g.lineTo(nx + topW * 0.85, ny - S * 0.02);
      g.closePath();
      g.globalAlpha = 0.45; g.fillStyle = innerShade; g.fill(); g.globalAlpha = 1;

      // yellow facings (the kurta's folded neckline edges)
      [-1, 1].forEach(sd => {
        g.beginPath();
        g.moveTo(nx + sd * topW, ny - S * 0.03);
        g.lineTo(nx + sd * outW, ny + S * 0.05);
        g.quadraticCurveTo(nx + sd * outW * 0.62, ny + S * 0.52, nx + sd * S * 0.014, vTip + S * 0.05);
        g.lineTo(nx, vTip);
        g.closePath();
        g.fillStyle = facing; g.fill();
        g.strokeStyle = ink; g.lineWidth = 2; g.stroke();
        // stitched blue piping along the inner edge
        g.save();
        g.setLineDash([S * 0.022, S * 0.016]);
        g.strokeStyle = gar.trim; g.lineWidth = 1.7; g.lineCap = "butt";
        g.beginPath();
        g.moveTo(nx + sd * (topW + S * 0.022), ny + S * 0.015);
        g.lineTo(nx + sd * S * 0.03, vTip - S * 0.035);
        g.stroke();
        g.restore();
      });

      // centre placket and buttons on the blue shirt
      strokeSeg(g, [nx, ny + S * 0.085], [nx, vTip - S * 0.01], ink, 1.2);
      [0.20, 0.38, 0.56].forEach(t => {
        const by = ny + S * (0.085 + (0.80 - 0.085) * t);
        disc(g, [nx, by], 2.9, ink);
        disc(g, [nx, by], 1.9, "#ffffff");
      });

      // collar band around the neck, with small blue/white stripes
      const cw = nw2 * 1.22;
      const lowY = x => ny + S * 0.095 * (1 - Math.pow((x - nx) / cw, 2)) - S * 0.02;   // lower edge (parabola)
      const upY = x => ny - S * 0.075 + S * 0.022 * (1 - Math.pow((x - nx) / cw, 2));   // upper edge
      g.beginPath();
      g.moveTo(nx - cw, lowY(nx - cw));
      for (let i = 1; i <= 16; i++) { const x = nx - cw + (2 * cw * i) / 16; g.lineTo(x, lowY(x)); }
      for (let i = 16; i >= 0; i--) { const x = nx - cw * 0.92 + (2 * cw * 0.92 * i) / 16; g.lineTo(x, upY(x)); }
      g.closePath();
      g.fillStyle = inner; g.fill();
      g.strokeStyle = ink; g.lineWidth = 1.8; g.stroke();
      const nTick = 9;
      for (let i = 0; i < nTick; i++) {
        const t = (i + 0.5) / nTick, xl = nx - cw + 2 * cw * t, xu = nx - cw * 0.92 + 2 * cw * 0.92 * t;
        strokeSeg(g, [xu, upY(xu) + 1], [xl, lowY(xl) - 1], i % 2 ? "#ffffff" : "#2d6fb8", Math.max(2, S * 0.011), "butt");
      }
    }
  }

  function drawBackHair(g, cx, cy, rx, ry, S, st) {
    if (st.hairStyle !== "long") return;

    const h = st.hairColors;

    g.beginPath();
    g.moveTo(cx - rx * 1.06, cy - ry * 0.25);
    g.bezierCurveTo(
      cx - rx * 1.40, cy + ry * 0.7,
      cx - rx * 1.30, cy + ry * 1.8,
      cx - rx * 1.22, cy + ry * 2.35
    );
    g.lineTo(cx + rx * 1.22, cy + ry * 2.35);
    g.bezierCurveTo(
      cx + rx * 1.30, cy + ry * 1.8,
      cx + rx * 1.40, cy + ry * 0.7,
      cx + rx * 1.06, cy - ry * 0.25
    );
    g.closePath();
    g.fillStyle = h.base;
    g.fill();
  }

  // ---------------------------------------------------------------------------
  // Pagri: big bulbous wrap, alternating yellow/blue bands, a thick yellow twist
  // crossing a blue twist, dense white polka dots, sticker-style white edge.
  // ---------------------------------------------------------------------------
  function drawTurban(g, cx, cy, rx, ry, S) {
    const ink = "#17243b";
    const yellow = "#f5c62e", yellowShade = "#d9a50e", yellowHi = "#ffe27a";
    const blue = "#3a8ee0", blueShade = "#2a6fc2";
    const white = "#fff8e9";

    // keep the dome clear of the top edge of the canvas (squash it a little, pivoting at the brow)
    const pivot = cy - ry * 0.45;
    g.save();
    g.translate(0, pivot); g.scale(1, 0.88); g.translate(0, -pivot);

    const Q = a => [cx + a[0] * rx, cy + a[1] * ry];
    const cubic = (p, k) => {
      const u = 1 - k;
      return [
        u * u * u * p[0][0] + 3 * u * u * k * p[1][0] + 3 * u * k * k * p[2][0] + k * k * k * p[3][0],
        u * u * u * p[0][1] + 3 * u * u * k * p[1][1] + 3 * u * k * k * p[2][1] + k * k * k * p[3][1]
      ];
    };

    // Silhouette: wider than the face, tall dome, forehead edge arching up in the middle.
    const sil = () => {
      g.beginPath();
      g.moveTo(cx - rx * 1.14, cy - ry * 0.20);
      g.bezierCurveTo(cx - rx * 1.66, cy - ry * 0.55, cx - rx * 1.70, cy - ry * 1.32, cx - rx * 1.08, cy - ry * 1.74);
      g.bezierCurveTo(cx - rx * 0.72, cy - ry * 2.04, cx + rx * 0.38, cy - ry * 2.14, cx + rx * 0.88, cy - ry * 1.84);
      g.bezierCurveTo(cx + rx * 1.58, cy - ry * 1.54, cx + rx * 1.74, cy - ry * 0.70, cx + rx * 1.14, cy - ry * 0.20);
      g.bezierCurveTo(cx + rx * 0.80, cy - ry * 0.48, cx + rx * 0.40, cy - ry * 0.58, cx, cy - ry * 0.58);
      g.bezierCurveTo(cx - rx * 0.40, cy - ry * 0.58, cx - rx * 0.80, cy - ry * 0.48, cx - rx * 1.14, cy - ry * 0.20);
      g.closePath();
    };

    // white sticker edge, then the blue base
    sil(); g.lineJoin = "round";
    g.strokeStyle = white; g.lineWidth = Math.max(6, S * 0.034); g.stroke();
    sil(); g.fillStyle = blue; g.fill();

    g.save();
    sil(); g.clip();

    const dotR = Math.max(1.3, S * 0.0072);

    // one wrapped band: outline, colour, shade line, highlight, polka dots
    const band = (pts, wFrac, color, shade, dotColor, hi) => {
      const p = pts.map(Q), w = ry * wFrac;
      const path = (dy) => { g.beginPath(); g.moveTo(p[0][0], p[0][1] + dy); g.bezierCurveTo(p[1][0], p[1][1] + dy, p[2][0], p[2][1] + dy, p[3][0], p[3][1] + dy); };
      g.lineCap = "round";
      path(0); g.strokeStyle = ink; g.lineWidth = w + Math.max(4, S * 0.022); g.stroke();
      path(0); g.strokeStyle = color; g.lineWidth = w; g.stroke();
      g.globalAlpha = 0.6; path(w * 0.30); g.strokeStyle = shade; g.lineWidth = w * 0.22; g.stroke(); g.globalAlpha = 1;
      if (hi) { g.globalAlpha = 0.85; path(-w * 0.26); g.strokeStyle = hi; g.lineWidth = Math.max(1.6, w * 0.10); g.stroke(); g.globalAlpha = 1; }
      for (let k = 0.03, n = 0; k < 0.98; k += 0.042, n++) {
        const q = cubic(p, k), off = ((n % 3) - 1) * w * 0.24;
        disc(g, [q[0], q[1] + off], dotR * (n % 2 ? 0.85 : 1.1), dotColor);
      }
    };

    // back to front
    band([[-1.95, -0.30], [-1.15, -1.05], [-0.30, -1.00], [0.75, -0.42]], 0.64, yellow, yellowShade, "#fff3b8", null);   // lower-left yellow
    band([[0.25, -0.42], [0.90, -0.56], [1.50, -0.60], [2.05, -0.46]], 0.52, yellow, yellowShade, "#fff3b8", null);   // lower-right yellow
    band([[-1.75, -1.58], [-1.05, -2.18], [0.00, -2.24], [0.90, -1.72]], 0.50, yellow, yellowShade, "#fff3b8", null);   // upper yellow
    band([[-1.95, -0.98], [-1.25, -1.58], [-0.40, -1.58], [0.55, -1.12]], 0.46, blue, blueShade, white, null);           // blue band, left
    band([[1.95, -0.88], [1.15, -1.22], [0.30, -1.38], [-0.55, -1.92]], 0.62, blue, blueShade, white, null);           // blue twist, right to left
    band([[-1.55, -0.50], [-0.60, -0.72], [0.20, -1.18], [1.30, -1.98]], 0.74, yellow, yellowShade, "#fff3b8", yellowHi); // thick yellow twist, in front

    // roundness: soft shade down the right side
    g.globalAlpha = 0.16; g.fillStyle = ink;
    g.beginPath(); g.ellipse(cx + rx * 1.75, cy - ry * 0.95, rx * 0.75, ry * 1.35, 0, 0, Math.PI * 2); g.fill();
    g.globalAlpha = 1;
    g.restore();

    sil(); g.strokeStyle = ink; g.lineWidth = Math.max(2.6, S * 0.014); g.lineJoin = "round"; g.stroke();
    g.restore();
  }

  // Cartoon avatar face and turban.
  function drawHead(g, cx, cy, rx, ry, S, st, openness) {
    const skin = st.skin;
    const ink = "#17243b";

    // Ears.
    [-1, 1].forEach(side => {
      g.beginPath();
      g.ellipse(cx + side * rx * 0.96, cy + ry * 0.16,
        rx * 0.15, ry * 0.22, 0, 0, Math.PI * 2);
      g.fillStyle = skin.base;
      g.strokeStyle = ink;
      g.lineWidth = Math.max(1.5, S * 0.012);
      g.fill();
      g.stroke();

      g.beginPath();
      g.ellipse(cx + side * rx * 0.98, cy + ry * 0.16,
        rx * 0.065, ry * 0.12, 0, 0, Math.PI * 2);
      g.strokeStyle = skin.line;
      g.lineWidth = Math.max(1, S * 0.006);
      g.stroke();
    });

    // Rounded face.
    g.beginPath();
    g.moveTo(cx, cy - ry * 0.98);
    g.bezierCurveTo(cx + rx * 0.88, cy - ry * 1.02,
      cx + rx * 1.02, cy - ry * 0.35,
      cx + rx * 0.91, cy + ry * 0.48);
    g.bezierCurveTo(cx + rx * 0.82, cy + ry * 0.95,
      cx + rx * 0.42, cy + ry * 1.10,
      cx, cy + ry * 1.08);
    g.bezierCurveTo(cx - rx * 0.42, cy + ry * 1.10,
      cx - rx * 0.82, cy + ry * 0.95,
      cx - rx * 0.91, cy + ry * 0.48);
    g.bezierCurveTo(cx - rx * 1.02, cy - ry * 0.35,
      cx - rx * 0.88, cy - ry * 1.02,
      cx, cy - ry * 0.98);
    g.closePath();
    g.fillStyle = skin.base;
    g.fill();
    g.strokeStyle = ink;
    g.lineWidth = Math.max(2, S * 0.012);
    g.stroke();

    // Pagri (turban) sits over the top of the face.
    drawTurban(g, cx, cy, rx, ry, S);

    // Sunglasses.
    const glassesY = cy - ry * 0.12;

    [-1, 1].forEach(side => {
      const gx = cx + side * rx * 0.43;

      g.beginPath();
      g.moveTo(gx - rx * 0.36, glassesY - ry * 0.20);
      g.quadraticCurveTo(gx, glassesY - ry * 0.32,
        gx + rx * 0.34, glassesY - ry * 0.18);
      g.lineTo(gx + rx * 0.28, glassesY + ry * 0.22);
      g.quadraticCurveTo(gx, glassesY + ry * 0.39,
        gx - rx * 0.28, glassesY + ry * 0.22);
      g.closePath();
      g.fillStyle = "#080d18";
      g.fill();
      g.strokeStyle = ink;
      g.lineWidth = Math.max(2, S * 0.012);
      g.stroke();
    });

    g.beginPath();
    g.moveTo(cx - rx * 0.12, glassesY - ry * 0.12);
    g.quadraticCurveTo(cx, glassesY - ry * 0.20,
      cx + rx * 0.12, glassesY - ry * 0.12);
    g.strokeStyle = ink;
    g.lineWidth = Math.max(3, S * 0.018);
    g.stroke();

    strokeSeg(g,
      [cx - rx * 0.78, glassesY - ry * 0.10],
      [cx - rx * 1.03, glassesY - ry * 0.20],
      ink, Math.max(2, S * 0.012)
    );

    strokeSeg(g,
      [cx + rx * 0.78, glassesY - ry * 0.10],
      [cx + rx * 1.03, glassesY - ry * 0.20],
      ink, Math.max(2, S * 0.012)
    );

    // Nose.
    g.beginPath();
    g.moveTo(cx, cy + ry * 0.12);
    g.quadraticCurveTo(cx - rx * 0.12, cy + ry * 0.34,
      cx, cy + ry * 0.38);
    g.quadraticCurveTo(cx + rx * 0.12, cy + ry * 0.34,
      cx, cy + ry * 0.12);
    g.strokeStyle = skin.line;
    g.lineWidth = Math.max(1.8, S * 0.010);
    g.stroke();

    // Curled moustache.
    g.beginPath();
    g.moveTo(cx, cy + ry * 0.48);
    g.bezierCurveTo(cx - rx * 0.22, cy + ry * 0.27,
      cx - rx * 0.45, cy + ry * 0.34,
      cx - rx * 0.62, cy + ry * 0.48);
    g.bezierCurveTo(cx - rx * 0.86, cy + ry * 0.72,
      cx - rx * 1.03, cy + ry * 0.58,
      cx - rx * 0.96, cy + ry * 0.42);
    g.bezierCurveTo(cx - rx * 0.89, cy + ry * 0.78,
      cx - rx * 0.55, cy + ry * 0.88,
      cx - rx * 0.20, cy + ry * 0.66);
    g.quadraticCurveTo(cx, cy + ry * 0.54,
      cx, cy + ry * 0.48);
    g.bezierCurveTo(cx, cy + ry * 0.48,
      cx + rx * 0.22, cy + ry * 0.27,
      cx + rx * 0.62, cy + ry * 0.48);
    g.bezierCurveTo(cx + rx * 0.86, cy + ry * 0.72,
      cx + rx * 1.03, cy + ry * 0.58,
      cx + rx * 0.96, cy + ry * 0.42);
    g.bezierCurveTo(cx + rx * 0.89, cy + ry * 0.78,
      cx + rx * 0.55, cy + ry * 0.88,
      cx + rx * 0.20, cy + ry * 0.66);
    g.quadraticCurveTo(cx, cy + ry * 0.54,
      cx, cy + ry * 0.48);
    g.closePath();
    g.fillStyle = "#111827";
    g.fill();
    g.strokeStyle = ink;
    g.lineWidth = Math.max(1.5, S * 0.009);
    g.stroke();
  }

  // Samples a polyline and tapers its width from w0 to w1.
  function taperSamples(pts, w0, w1) {
    const seg = [0];

    for (let i = 1; i < pts.length; i++) {
      seg.push(seg[i - 1] +
        Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }

    const total = seg[seg.length - 1] || 1;
    const n = Math.max(5, Math.min(9, Math.ceil(total / 5)));
    const out = [];

    for (let k = 0; k <= n; k++) {
      const d = total * k / n;
      let i = 1;

      while (i < seg.length - 1 && seg[i] < d) i++;

      const t = seg[i] === seg[i - 1]
        ? 0
        : (d - seg[i - 1]) / (seg[i] - seg[i - 1]);

      const u = d / total;

      out.push([
        pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t,
        pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t,
        w0 + (w1 - w0) * u
      ]);
    }

    return out;
  }

  function strokeTaper(g, samples, color, extra) {
    g.strokeStyle = color;
    g.lineCap = "round";
    g.lineJoin = "round";

    for (let i = 0; i < samples.length - 1; i++) {
      g.lineWidth = (samples[i][2] + samples[i + 1][2]) / 2 + extra;
      g.beginPath();
      g.moveTo(samples[i][0], samples[i][1]);
      g.lineTo(samples[i + 1][0], samples[i + 1][1]);
      g.stroke();
    }
  }

  // Draw a natural, human hand: smooth rounded palm with soft shading, a thick thumb pad,
  // tapered depth-ordered fingers with rounded tips, and knuckle creases.
  function drawHand(g, joints, unit, skin) {
    const line = mix(skin.line, skin.base, 0.10);
    const o = Math.max(1.1, unit * 0.13);
    const pw = unit * 1.18;
    const shade = mix(skin.base, skin.shade, 0.65);
    const crease = mix(skin.line, skin.base, 0.35);
    const rgba = (hex, a) => {
      const c = hexToRgb(hex);
      return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
    };

    g.lineJoin = "round";
    g.lineCap = "round";

    // Palm: rounded outline through the palm landmarks (quadratic curves via the midpoints).
    const pts = [0, 1, 5, 9, 13, 17].map(i => joints[i]);
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const palmPath = () => {
      const n = pts.length;
      const m0 = mid(pts[n - 1], pts[0]);
      g.beginPath();
      g.moveTo(m0[0], m0[1]);
      for (let k = 0; k < n; k++) {
        const m = mid(pts[k], pts[(k + 1) % n]);
        g.quadraticCurveTo(pts[k][0], pts[k][1], m[0], m[1]);
      }
      g.closePath();
    };

    // outline, thumb pad outline
    palmPath();
    g.strokeStyle = line;
    g.lineWidth = pw + o * 2;
    g.stroke();
    strokeSeg(g, joints[0], joints[2], line, unit * 1.55 + o * 2, "round");

    // skin fill
    palmPath();
    g.fillStyle = skin.base;
    g.fill();
    palmPath();
    g.strokeStyle = skin.base;
    g.lineWidth = pw;
    g.stroke();
    strokeSeg(g, joints[0], joints[2], skin.base, unit * 1.55, "round");

    // soft shading toward the wrist (gives the palm some volume)
    g.save();
    palmPath();
    g.clip();
    const gr = g.createLinearGradient(joints[0][0], joints[0][1], joints[9][0], joints[9][1]);
    gr.addColorStop(0, rgba(shade, 0.55));
    gr.addColorStop(0.65, rgba(shade, 0.0));
    g.fillStyle = gr;
    g.fillRect(-5000, -5000, 10000, 10000);
    g.restore();

    // fingers, farthest from the camera first
    const order = HAND_FINGERS.map(fg => ({
      fg,
      z: fg.chain.reduce((sum, i) => sum + joints[i][2], 0) / fg.chain.length
    })).sort((a, b) => b.z - a.z);

    order.forEach(({ fg }) => {
      const chain = fg.chain.map(i => [joints[i][0], joints[i][1]]);
      const isThumb = fg.name === "thumb";

      const w0 = unit * fg.w * (isThumb ? 1.28 : 1.0);
      const w1 = unit * fg.w * (isThumb ? 0.88 : 0.74);

      const samples = taperSamples(chain, w0, w1);

      strokeTaper(g, samples, line, o * 2);
      strokeTaper(g, samples, skin.base, 0);

      // base of the finger blends into the palm
      disc(g, chain[0], w0 * 0.5 + o * 0.5, skin.base);

      // knuckle creases across the finger at the two inner joints
      g.save();
      g.globalAlpha = 0.5;
      [1, 2].forEach(k => {
        const a = chain[k - 1], b = chain[k], c = chain[k + 1];
        const dx = c[0] - a[0], dy = c[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        const half = (w0 + (w1 - w0) * (k / 3)) * 0.30;
        const nx = -dy / len, ny = dx / len;
        strokeSeg(g,
          [b[0] - nx * half, b[1] - ny * half],
          [b[0] + nx * half, b[1] + ny * half],
          crease, Math.max(1, unit * 0.075), "round");
      });
      g.restore();
    });
  }

  /**
   * Main AvatarRenderer Class
   */
  class AvatarRenderer {
    constructor(canvas, options) {
      this.canvas = typeof canvas === "string"
        ? document.getElementById(canvas)
        : canvas;

      this.ctx = this.canvas ? this.canvas.getContext("2d") : null;

      this.options = Object.assign({
        signs: {},
        speed: 1.0,
        fps: 30,
        useIK: true,
        yUnstretch: 9 / 16,
        handAutoSize: true,   // resize hands per frame to a natural, steady size (false = fixed 0.8 scale)
        style: {},
        idleMotion: true,
        glossStripEl: null,
        captionEl: null,
        statusEl: null,
        onFrame: null,
        onSignStart: null,
        onComplete: null
      }, options || {});

      this.signs = {};
      this.setSigns(this.options.signs || {});

      this.speed = this.options.speed;
      this.fps = this.options.fps;
      this.useIK = this.options.useIK;
      this._style = this._resolveStyle(this.options.style);
      this._lastFrame = REST_FRAME;
      this._handScale = { 0: null, 63: null };

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

    setStyle(partial) {
      this.options.style = Object.assign({}, this.options.style || {}, partial || {});
      this._style = this._resolveStyle(this.options.style);

      if (this.canvas) {
        this.renderFrame(this._lastFrame || REST_FRAME);
      }
    }

    _clock() {
      return (typeof performance !== "undefined"
        ? performance.now()
        : Date.now()) / 1000;
    }

    _blinkOpenness() {
      if (this.options.idleMotion === false) return 1;

      const t = this._clock() % 3.9;

      return t < 0.16
        ? 1 - 0.92 * Math.sin(Math.PI * t / 0.16)
        : 1;
    }

    setSigns(signs) {
      this.signs = {};

      if (!signs) return;

      for (const [id, seq] of Object.entries(signs)) {
        this.signs[id] = Array.isArray(seq)
          ? smoothSequence(seq.map(f => new Float32Array(f)))
          : seq;
      }
    }

    setSpeed(speed) {
      this.speed = Math.max(0.2, Math.min(3.0, speed));
    }

    // Build timeline frames from sequence of sign IDs.
    buildTimeline(sequence) {
      const frames = [];
      const metadata = [];
      const glossItems = [];

      const restFrames = Math.max(4, Math.round(6 / this.speed));
      const transFrames = Math.max(4, Math.round(8 / this.speed));
      const nosignHold = Math.max(14, Math.round(18 / this.speed));

      // Intro rest.
      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);
        metadata.push({
          signId: "REST",
          signIndex: -1,
          isTrans: false,
          progress: 0,
          gloss: "Ready",
          status: "rest"
        });
      }

      let prevLast = REST_FRAME;
      const nosignWords = [];

      // Sequential signs.
      sequence.forEach((item, seqIdx) => {
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
        const isMissingSign = !signSeq || !signSeq.length;

        if (kind === "marker" || kind === "nosign" ||
          (kind === "sign" && isMissingSign)) {
          const status = kind === "marker" ? "marker" : "nosign";

          if (kind === "nosign" || (kind === "sign" && isMissingSign)) {
            nosignWords.push(conceptEnglish);
          }

          const holdCount = nosignHold;

          glossItems.push({
            id: rawId,
            label: gloss,
            index: seqIdx,
            startFrame: frames.length,
            endFrame: frames.length + holdCount - 1,
            status
          });

          for (let f = 0; f < holdCount; f++) {
            frames.push(REST_FRAME);

            metadata.push({
              signId: rawId,
              signIndex: seqIdx,
              isTrans: false,
              progress: f / holdCount,
              gloss,
              status
            });
          }

          prevLast = REST_FRAME;
          return;
        }

        // Ease-in-out transition.
        const firstFrame = signSeq[0];

        for (let t = 0; t < transFrames; t++) {
          const alpha = easeInOut(t / transFrames);

          frames.push(lerpFrame(prevLast, firstFrame, alpha));

          metadata.push({
            signId: rawId,
            signIndex: seqIdx,
            isTrans: true,
            progress: 0,
            gloss: `-> ${gloss}`,
            status: "native"
          });
        }

        const startFrame = frames.length;
        const totalRaw = signSeq.length;
        const signFrameCount = Math.max(12, Math.round(totalRaw / this.speed));

        for (let f = 0; f < signFrameCount; f++) {
          const progress = f / Math.max(1, signFrameCount - 1);
          const rawIdx = Math.min(totalRaw - 1, Math.floor(progress * (totalRaw - 1)));

          frames.push(signSeq[rawIdx]);

          metadata.push({
            signId: rawId,
            signIndex: seqIdx,
            isTrans: false,
            progress,
            gloss,
            status: "native"
          });
        }

        const endFrame = frames.length - 1;

        glossItems.push({
          id: rawId,
          label: gloss,
          index: seqIdx,
          startFrame,
          endFrame,
          status: "native"
        });

        prevLast = signSeq[totalRaw - 1];
      });

      // Outro transition.
      for (let t = 0; t < transFrames; t++) {
        const alpha = easeInOut(t / transFrames);

        frames.push(lerpFrame(prevLast, REST_FRAME, alpha));

        metadata.push({
          signId: "REST",
          signIndex: -1,
          isTrans: true,
          progress: 1,
          gloss: "-> Rest",
          status: "rest"
        });
      }

      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);

        metadata.push({
          signId: "REST",
          signIndex: -1,
          isTrans: false,
          progress: 1,
          gloss: "Done",
          status: "rest"
        });
      }

      return { frames, metadata, glossItems, nosignWords };
    }

    // Playback control.
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

        if (this.animTimer) {
          clearTimeout(this.animTimer);
          this.animTimer = null;
        }
      }
    }

    resume() {
      if (this.isPlaying && this.isPaused) {
        this.isPaused = false;
        this.tick();
      }
    }

    replay() {
      if (this.activeSequence.length) {
        this.play(this.activeSequence);
      }
    }

    stop() {
      this.isPlaying = false;
      this.isPaused = false;

      if (this.animTimer) {
        clearTimeout(this.animTimer);
        this.animTimer = null;
      }

      this.currentFrameIdx = 0;
    }

    tick() {
      if (!this.isPlaying || this.isPaused) return;

      if (this.currentFrameIdx >= this.timeline.length) {
        this.isPlaying = false;
        this.renderFrame(REST_FRAME);
        this.updateCaption("Ready");
        this.highlightGloss(-1);

        if (this.options.onComplete) {
          this.options.onComplete();
        }

        return;
      }

      const frame = this.timeline[this.currentFrameIdx];
      const meta = this.metadata[this.currentFrameIdx];

      this.renderFrame(frame);

      this.updateCaption(
        meta.gloss +
        (meta.status === "nosign" ? " (no sign recorded)" : "")
      );

      this.highlightGloss(meta.signIndex, meta.progress);

      if (this.options.onFrame) {
        this.options.onFrame(this.currentFrameIdx, meta);
      }

      this.currentFrameIdx++;

      const frameInterval = Math.round(1000 / this.fps);

      this.animTimer = setTimeout(() => this.tick(), frameInterval);
    }

    // Highlighting synced gloss strip.
    renderGlossStrip() {
      const container =
        this.options.glossStripEl ||
        document.getElementById("gloss-strip");

      if (!container) return;

      container.innerHTML = "";

      this.glossItems.forEach((item, idx) => {
        const pill = document.createElement("span");

        pill.id = `gloss-pill-${idx}`;
        pill.className = `gloss-pill ${item.status}`;
        pill.textContent = item.label;
        pill.title = item.status === "nosign"
          ? "No recorded sign yet"
          : item.label;

        pill.onclick = () => {
          this.currentFrameIdx = item.startFrame;
        };

        container.appendChild(pill);
      });
    }

    _renderNosignNote() {
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
            pill.style.setProperty(
              "--progress",
              `${Math.round(progress * 100)}%`
            );
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
      const el =
        this.options.captionEl ||
        document.getElementById("caption");

      if (el) {
        el.textContent = text || "";
      }
    }

    // Draw the signer: body, head and arms first, then hands on top.
    renderFrame(f) {
      if (!this.ctx || !this.canvas) return;

      this._lastFrame = f;

      const g = this.ctx;
      const W = this.canvas.width;
      const H = this.canvas.height;

      const S = H * 0.26;
      const ox = W * 0.50;
      const oy = H * 0.32;
      const st = this._style;

      // Training coordinates use a 16:9 vertical scaling convention.
      const U = this.options.yUnstretch !== undefined
        ? this.options.yUnstretch
        : (9 / 16);

      g.clearRect(0, 0, W, H);
      g.lineCap = "round";
      g.lineJoin = "round";

      // Normalize the shoulder width, keeping the body scale stable.
      let k = 1;
      let mx = 0;
      let my = 0;

      {
        const ax = f[126 + 11 * 3];
        const ay = f[126 + 11 * 3 + 1] * U;

        const bx = f[126 + 12 * 3];
        const by = f[126 + 12 * 3 + 1] * U;

        const half = Math.hypot(ax - bx, ay - by) / 2;

        if (half > 1e-4) {
          k = clamp(0.5 / half, 0.5, 2.0);
          mx = (ax + bx) / 2;
          my = (ay + by) / 2;
        }
      }

      // Convert normalized pose coordinates to renderer coordinates.
      const P = i => {
        const x = f[126 + i * 3];
        const y = f[126 + i * 3 + 1] * U;
        const z = f[126 + i * 3 + 2];

        return [
          mx + (x - mx) * k,
          my + (y - my) * k,
          z
        ];
      };

      const ok = i =>
        f[126 + i * 3] !== 0 ||
        f[126 + i * 3 + 1] !== 0;

      const X = p => [
        ox + p[0] * S,
        oy + p[1] * S
      ];

      // BODY AND HEAD.
      if (ok(11) && ok(12)) {
        const p11 = P(11);
        const p12 = P(12);

        const x11 = X(p11);
        const x12 = X(p12);

        const xNeck = X([
          (p11[0] + p12[0]) * 0.5,
          (p11[1] + p12[1]) * 0.5,
          (p11[2] + p12[2]) * 0.5
        ]);

        const xNose = X(
          ok(0) ? P(0) : [0, -1.0 * U, 0]
        );

        // IMPORTANT:
        // Nose movement only adjusts the horizontal head position.
        // Head height remains fixed relative to the shoulder centre.
        // This prevents animated nose coordinates from stretching the neck.
        const headOffsetX = clamp(
          (xNose[0] - xNeck[0]) * 0.35,
          -S * 0.10,
          S * 0.10
        );

        const head = [
          xNeck[0] + headOffsetX,
          xNeck[1] - S * 0.45
        ];

        const rx = S * 0.215;
        const ry = S * 0.285;

        drawBackHair(
          g,
          head[0],
          head[1],
          rx,
          ry,
          S,
          st
        );

        drawTorso(
          g,
          x11,
          x12,
          xNeck,
          S,
          st,
          H
        );

        drawNeckAndCollar(
          g,
          xNeck,
          head[0],
          head[1],
          rx,
          ry,
          S,
          st
        );

        // Enlarge the illustrated head independently of the body landmarks.
        // The wider silhouette brings the turban/face closer to the reference,
        // while the slight upward shift preserves clearance above the collar.
        drawHead(
          g,
          head[0],
          head[1] - S * 0.025,
          rx * 1.32,
          ry * 1.20,
          S,
          st,
          this._blinkOpenness()
        );

        // ARM POSITIONS.
        const p15 = ok(15)
          ? P(15)
          : [0.48, 2.75 * U, 0];

        const p16 = ok(16)
          ? P(16)
          : [-0.48, 2.75 * U, 0];

        let p13;
        let p14;

        if (ok(13)) {
          p13 = P(13);
        } else {
          p13 = this.useIK
            ? solveTwoBoneIK(
              p11,
              p15,
              null,
              true,
              0.85,
              0.8
            )
            : [
              p11[0] + 0.08,
              (p11[1] + p15[1]) * 0.5,
              0
            ];
        }

        if (ok(14)) {
          p14 = P(14);
        } else {
          p14 = this.useIK
            ? solveTwoBoneIK(
              p12,
              p16,
              null,
              false,
              0.85,
              0.8
            )
            : [
              p12[0] - 0.08,
              (p12[1] + p16[1]) * 0.5,
              0
            ];
        }

        const skin = st.skin;
        const gar = st.garment;

        // Draw sleeves, forearms and cuffs.
        const drawArm = (ps, pe, pw) => {
          const xs = X(ps);
          const xe = X(pe);
          const xw = X(pw);

          const xc = lerp2(xe, xw, 0.58);

          const wUp = S * st.prop.wUp;
          const wFore = S * st.prop.wFore;
          const wSkin = S * st.prop.wSkin;

          const ol = 1.8;

          // Bare forearm.
          limb(
            g,
            xe,
            xw,
            wSkin,
            skin.base,
            skin.line,
            ol,
            "round"
          );

          // Upper sleeve outline.
          strokeSeg(
            g,
            xs,
            xe,
            gar.line,
            wUp + ol * 2,
            "round"
          );

          // Forearm sleeve outline.
          strokeSeg(
            g,
            xe,
            xc,
            gar.line,
            wFore + ol * 2,
            "butt"
          );

          // Sleeve fills.
          strokeSeg(
            g,
            xs,
            xe,
            gar.base,
            wUp,
            "round"
          );

          strokeSeg(
            g,
            xe,
            xc,
            gar.base,
            wFore,
            "butt"
          );

          // Cuff edge.
          const dx = xw[0] - xe[0];
          const dy = xw[1] - xe[1];

          const d = Math.hypot(dx, dy) || 1;

          const nx = -dy / d;
          const ny = dx / d;

          const hw = wFore / 2 + ol;

          strokeSeg(
            g,
            [
              xc[0] - nx * hw,
              xc[1] - ny * hw
            ],
            [
              xc[0] + nx * hw,
              xc[1] + ny * hw
            ],
            gar.line,
            2.4,
            "butt"
          );

          // Blue piping just inside the cuff edge.
          const ux = dx / d, uy = dy / d, back = S * 0.045;
          strokeSeg(g,
            [xc[0] - ux * back - nx * (hw - 2), xc[1] - uy * back - ny * (hw - 2)],
            [xc[0] - ux * back + nx * (hw - 2), xc[1] - uy * back + ny * (hw - 2)],
            gar.trim, 2, "butt"
          );
        };

        drawArm(p11, p13, p15);
        drawArm(p12, p14, p16);
      }

      // HANDS.
      // Left hand: 0-62, attached to wrist 15.
      // Right hand: 63-125, attached to wrist 16.
      [[0, 15], [63, 16]].forEach(
        ([offset, wristPoseIdx]) => {
          if (!ok(wristPoseIdx)) return;

          let any = false;

          for (
            let i = offset;
            i < offset + 63;
            i++
          ) {
            if (f[i] !== 0) {
              any = true;
              break;
            }
          }

          // If no hand was detected, use the canonical relaxed hand.
          // This does not modify the actual sign sequence.
          const src = any ? f : REST_FRAME;

          const wP = P(wristPoseIdx);

          // Landmark offsets from the wrist, in shoulder-width units, with a human
          // palm aspect (HAND_ASPECT) on top of the body's vertical unstretch.
          const uh = U * HAND_ASPECT;
          const rel = [];

          for (let i = 0; i < 21; i++) {
            rel.push([
              src[offset + i * 3],
              src[offset + i * 3 + 1] * uh,
              src[offset + i * 3 + 2]
            ]);
          }

          // Natural, steady hand size: measure the palm (wrist -> knuckles) and resize the
          // hand around the wrist to HAND_PALM. The scale is eased frame to frame so the
          // hand never pulses as the fingers open and close.
          let sc = 0.8;

          if (this.options.handAutoSize !== false) {
            let palm = 0;
            [5, 9, 13, 17].forEach(j => { palm += Math.hypot(rel[j][0], rel[j][1]); });
            palm /= 4;

            const target = palm > 1e-4 ? clamp(HAND_PALM / palm, 0.6, 1.6) : 1;
            const prev = this._handScale[offset];

            sc = prev === null || prev === undefined
              ? target
              : prev + (target - prev) * 0.35;

            this._handScale[offset] = sc;
          }

          const joints = [];

          for (let i = 0; i < 21; i++) {
            joints.push([
              ox + (wP[0] + rel[i][0] * sc) * S,
              oy + (wP[1] + rel[i][1] * sc) * S,
              rel[i][2]
            ]);
          }

          drawHand(
            g,
            joints,
            S * 0.047,
            st.skin
          );
        }
      );
    }
  }

  // Export to global scope.
  global.AvatarRenderer = AvatarRenderer;

  global.createAvatarRenderer = function (canvas, options) {
    return new AvatarRenderer(canvas, options);
  };

  // Node.js exports for tests.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      AvatarRenderer,
      REST_FRAME,
      solveTwoBoneIK,
      easeInOut,
      lerpFrame,
      smoothSequence,
      SKINS,
      HAIRS,
      GARMENTS,
      TROUSERS,
      BUILDS,
      DEFAULT_STYLE
    };
  }

})(typeof window !== "undefined" ? window : global);