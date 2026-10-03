/**
 * ClarifySign Decoupled Avatar Renderer (HTML5 Canvas 2D)
 *
 * Requirements:
 * 1. Proper articulated avatar:
 *    - Torso polygon, head oval, neck, face features (eyes, mouth, brows).
 *    - Bones with rounded thick strokes (lineCap="round", lineJoin="round").
 *    - Hands with articulated per-finger bones using real connection tables.
 *    - NO dot clouds.
 * 2. Two-Bone IK for arms (shoulder -> IK elbow -> wrist).
 * 3. Timeline with ease-in-out LERP between clips and rest pose.
 * 4. Controls: configurable speed, pause, resume, replay.
 * 5. Per-sign highlighting synced to a gloss/caption strip.
 * 6. Fallback for missing signs: fingerspell or visible unknown sign marker (NEVER silent skip).
 *
 * ZERO dependencies on NLU or planner modules.
 */

(function (global) {
  "use strict";

  // Real MediaPipe Connections
  const HAND_PALM_CONNECTIONS = [
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]
  ];
  const HAND_FINGERS = [
    { name: "thumb", color: "#e17055", segs: [[1, 2], [2, 3], [3, 4]] },
    { name: "index", color: "#0984e3", segs: [[5, 6], [6, 7], [7, 8]] },
    { name: "middle", color: "#00b894", segs: [[9, 10], [10, 11], [11, 12]] },
    { name: "ring", color: "#6c5ce7", segs: [[13, 14], [14, 15], [15, 16]] },
    { name: "pinky", color: "#fd79a8", segs: [[17, 18], [18, 19], [19, 20]] }
  ];

  // Smoothstep ease-in-out
  function easeInOut(t) {
    t = Math.max(0, Math.min(1, t));
    return t * t * (3 - 2 * t);
  }

  // Linear interpolation of 225-dim float arrays with easing
  function lerpFrame(f1, f2, alpha) {
    const out = new Float32Array(225);
    for (let i = 0; i < 225; i++) {
      out[i] = (1 - alpha) * f1[i] + alpha * f2[i];
    }
    return out;
  }

  // Canonical Rest Frame (225 dims)
  function buildRestFrame() {
    const f = new Float32Array(225);
    // Pose (indices 126..224)
    // Left shoulder 11 (+0.5, 0), Right shoulder 12 (-0.5, 0)
    const setPose = (idx, x, y, z) => {
      const base = 126 + idx * 3;
      f[base] = x; f[base + 1] = y; f[base + 2] = z;
    };
    setPose(0, 0.0, -1.0, -0.3);    // Nose
    setPose(7, 0.35, -1.0, 0.0);    // Left ear
    setPose(8, -0.35, -1.0, 0.0);   // Right ear
    setPose(9, 0.12, -0.85, -0.2);  // Mouth left
    setPose(10, -0.12, -0.85, -0.2);// Mouth right
    setPose(11, 0.5, 0.0, 0.0);     // Left shoulder
    setPose(12, -0.5, 0.0, 0.0);    // Right shoulder
    setPose(13, 0.58, 1.45, 0.15);  // Left elbow
    setPose(14, -0.58, 1.45, 0.15); // Right elbow
    setPose(15, 0.48, 2.75, 0.0);   // Left wrist (relaxed low)
    setPose(16, -0.48, 2.75, 0.0);  // Right wrist (relaxed low)
    setPose(23, 0.28, 2.50, 0.0);   // Left hip
    setPose(24, -0.28, 2.50, 0.0);  // Right hip

    // Hands: relaxed curl
    const setHand = (isRight, idx, x, y, z) => {
      const base = (isRight ? 63 : 0) + idx * 3;
      f[base] = x; f[base + 1] = y; f[base + 2] = z;
    };
    [false, true].forEach(isRight => {
      const sign = isRight ? 1.0 : -1.0;
      setHand(isRight, 0, 0, 0, 0);
      setHand(isRight, 4, sign * 0.35, 0.40, -0.15);
      setHand(isRight, 8, sign * 0.10, 0.85, -0.15);
      setHand(isRight, 12, 0.0, 0.90, -0.15);
      setHand(isRight, 16, -sign * 0.10, 0.82, -0.15);
      setHand(isRight, 20, -sign * 0.20, 0.70, -0.15);
    });
    return f;
  }

  const REST_FRAME = buildRestFrame();

  // Visible Unknown Sign Marker Sequence (48 frames)
  function buildUnknownSignSequence() {
    const seq = [];
    const rest = REST_FRAME;
    for (let i = 0; i < 48; i++) {
      const t = i / 47;
      let s;
      if (t < 0.3) s = 0.5 - 0.5 * Math.cos(Math.PI * (t / 0.3));
      else if (t < 0.7) s = 1.0 + 0.05 * Math.sin(Math.PI * 2 * (t - 0.3) / 0.4);
      else s = 0.5 + 0.5 * Math.cos(Math.PI * ((t - 0.7) / 0.3));

      const fr = new Float32Array(rest);
      // Lift shoulders slightly
      fr[126 + 11 * 3 + 1] -= 0.12 * s;
      fr[126 + 12 * 3 + 1] -= 0.12 * s;
      // Head tilt
      fr[126 + 0 * 3] += 0.06 * s;
      // Lift wrists to chest level (y ~ 0.75) and spread outward
      fr[126 + 15 * 3] = (1 - s) * rest[126 + 15 * 3] + s * 0.65;
      fr[126 + 15 * 3 + 1] = (1 - s) * rest[126 + 15 * 3 + 1] + s * 0.75;
      fr[126 + 16 * 3] = (1 - s) * rest[126 + 16 * 3] + s * (-0.65);
      fr[126 + 16 * 3 + 1] = (1 - s) * rest[126 + 16 * 3 + 1] + s * 0.75;

      // Elbows bent
      fr[126 + 13 * 3] = (1 - s) * rest[126 + 13 * 3] + s * 0.72;
      fr[126 + 13 * 3 + 1] = (1 - s) * rest[126 + 13 * 3 + 1] + s * 1.05;
      fr[126 + 14 * 3] = (1 - s) * rest[126 + 14 * 3] + s * (-0.72);
      fr[126 + 14 * 3 + 1] = (1 - s) * rest[126 + 14 * 3 + 1] + s * 1.05;

      // Open hands
      for (let j = 0; j < 21; j++) {
        fr[j * 3 + 1] = Math.min(1.0, 0.4 + j * 0.03 * s);
        fr[63 + j * 3 + 1] = Math.min(1.0, 0.4 + j * 0.03 * s);
      }
      seq.push(fr);
    }
    return seq;
  }

  const UNKNOWN_SEQUENCE = buildUnknownSignSequence();

  // Two-Bone Analytical Inverse Kinematics for arms
  function solveTwoBoneIK(shoulder, wrist, elbowHint, isLeft, l1, l2) {
    l1 = l1 || 1.45;
    l2 = l2 || 1.35;
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
      const clampedCosA = Math.max(-1, Math.min(1, cosA));
      const sinA = Math.sqrt(Math.max(0, 1 - clampedCosA * clampedCosA));
      ex = shoulder[0] + l1 * (clampedCosA * ux + sinA * vx);
      ey = shoulder[1] + l1 * (clampedCosA * uy + sinA * vy);
    }
    const ez = (shoulder[2] + wrist[2]) * 0.5;
    return [ex, ey, ez];
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
        fps: 25,
        useIK: true,
        glossStripEl: null,
        captionEl: null,
        statusEl: null,
        onFrame: null,
        onSignStart: null,
        onComplete: null
      }, options || {});

      this.signs = this.options.signs || {};
      this.speed = this.options.speed;
      this.fps = this.options.fps;
      this.useIK = this.options.useIK;

      // Playback state
      this.timeline = [];
      this.metadata = [];
      this.glossItems = [];
      this.currentFrameIdx = 0;
      this.isPlaying = false;
      this.isPaused = false;
      this.animTimer = null;
      this.activeSequence = [];

      // Color scheme
      this.colors = {
        torso: "#415482",
        torsoStroke: "#28375a",
        spine: "#28375a",
        neck: "#e0b6a0",
        head: "#eecab2",
        headStroke: "#c39b82",
        eyes: "#2d3441",
        armL: "#4e6eaf",
        armR: "#3e5c9b",
        joint: "#233050",
        palm: "#e6c0a8",
        palmStroke: "#c39b82"
      };

      if (this.canvas) {
        this.renderFrame(REST_FRAME);
      }
    }

    setSigns(signs) {
      this.signs = signs || {};
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

      // 1. Intro rest
      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);
        metadata.push({ signId: "REST", signIndex: -1, isTrans: false, progress: 0, gloss: "Ready", status: "rest" });
      }

      let prevLast = REST_FRAME;

      // 2. Sequential signs
      sequence.forEach((item, seqIdx) => {
        // Accept plain string IDs (legacy) or plan-item dicts {sign_id, gloss, kind}
        let rawId, gloss, kind;
        if (typeof item === "string") {
          rawId = item.toLowerCase().trim();
          gloss = rawId.toUpperCase();
          kind = "sign";
        } else {
          rawId = (item.sign_id || item.concept || "").toLowerCase().trim();
          gloss = item.gloss || rawId.toUpperCase();
          kind = item.kind || "sign";
        }

        let signSeq = this.signs[rawId];
        let status = "native";

        if (kind === "marker") {
          // Rest-pose hold ~0.4s = ~10 frames at 25fps, with gloss pill
          const holdCount = Math.max(6, Math.round(10 / this.speed));
          glossItems.push({ id: rawId, label: gloss, index: seqIdx,
            startFrame: frames.length, endFrame: frames.length + holdCount - 1, status: "marker" });
          for (let f = 0; f < holdCount; f++) {
            frames.push(REST_FRAME);
            metadata.push({ signId: rawId, signIndex: seqIdx, isTrans: false,
              progress: f / holdCount, gloss, status: "marker" });
          }
          prevLast = REST_FRAME;
          return;
        }

        if (!signSeq || !signSeq.length) {
          status = kind === "nosign" ? "nosign" : "unknown";
          signSeq = UNKNOWN_SEQUENCE;
        }

        // Ease-in-out LERP transition from previous pose to first frame
        const firstFrame = signSeq[0];
        for (let t = 0; t < transFrames; t++) {
          const alpha = easeInOut(t / transFrames);
          frames.push(lerpFrame(prevLast, firstFrame, alpha));
          metadata.push({ signId: rawId, signIndex: seqIdx, isTrans: true,
            progress: 0, gloss: `→ ${gloss}`, status });
        }

        const startFrame = frames.length;
        const totalRaw = signSeq.length;
        const signFrameCount = Math.max(12, Math.round(totalRaw / this.speed));
        for (let f = 0; f < signFrameCount; f++) {
          const progress = f / Math.max(1, signFrameCount - 1);
          const rawIdx = Math.min(totalRaw - 1, Math.floor(progress * (totalRaw - 1)));
          frames.push(signSeq[rawIdx]);
          metadata.push({ signId: rawId, signIndex: seqIdx, isTrans: false, progress, gloss, status });
        }
        const endFrame = frames.length - 1;
        glossItems.push({ id: rawId, label: gloss, index: seqIdx, startFrame, endFrame, status });
        prevLast = signSeq[totalRaw - 1];
      });

      // 3. Outro transition to rest pose
      for (let t = 0; t < transFrames; t++) {
        const alpha = easeInOut(t / transFrames);
        frames.push(lerpFrame(prevLast, REST_FRAME, alpha));
        metadata.push({ signId: "REST", signIndex: -1, isTrans: true, progress: 1, gloss: "→ Rest", status: "rest" });
      }

      for (let i = 0; i < restFrames; i++) {
        frames.push(REST_FRAME);
        metadata.push({ signId: "REST", signIndex: -1, isTrans: false, progress: 1, gloss: "Done", status: "rest" });
      }

      return { frames, metadata, glossItems };
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
      this.currentFrameIdx = 0;
      this.isPlaying = true;
      this.isPaused = false;

      this.renderGlossStrip();
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
        if (this.options.onComplete) this.options.onComplete();
        return;
      }

      const frame = this.timeline[this.currentFrameIdx];
      const meta = this.metadata[this.currentFrameIdx];

      this.renderFrame(frame);
      this.updateCaption(meta.gloss + (meta.status === "unknown" ? " (unknown sign)" : ""));
      this.highlightGloss(meta.signIndex, meta.progress);

      if (this.options.onFrame) {
        this.options.onFrame(this.currentFrameIdx, meta);
      }

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
        pill.title = item.status === "unknown" ? "Unknown Sign (Marker Active)" : item.label;
        pill.onclick = () => {
          this.currentFrameIdx = item.startFrame;
        };
        container.appendChild(pill);
      });
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

    // Draw Proper Articulated Avatar (No dot clouds)
    renderFrame(f) {
      if (!this.ctx || !this.canvas) return;
      const g = this.ctx;
      const W = this.canvas.width;
      const H = this.canvas.height;
      const S = H * 0.28;
      const ox = W * 0.50;
      const oy = H * 0.38;

      g.clearRect(0, 0, W, H);
      g.lineCap = "round";
      g.lineJoin = "round";

      const P = i => [f[126 + i * 3], f[126 + i * 3 + 1], f[126 + i * 3 + 2]];
      const ok = i => f[126 + i * 3] !== 0 || f[126 + i * 3 + 1] !== 0;
      const X = p => [ox + p[0] * S, oy + p[1] * S];

      const seg = (a, b, color, width) => {
        g.strokeStyle = color;
        g.lineWidth = width;
        g.beginPath();
        g.moveTo(a[0], a[1]);
        g.lineTo(b[0], b[1]);
        g.stroke();
      };

      const hasPose = ok(11) && ok(12);
      if (hasPose) {
        const p11 = P(11), p12 = P(12);
        const p23 = ok(23) ? P(23) : [0.28, 2.5, 0];
        const p24 = ok(24) ? P(24) : [-0.28, 2.5, 0];

        const x11 = X(p11), x12 = X(p12), x23 = X(p23), x24 = X(p24);
        const neckMid = [(p11[0] + p12[0]) * 0.5, (p11[1] + p12[1]) * 0.5, (p11[2] + p12[2]) * 0.5];
        const xNeck = X(neckMid);

        // 1. Torso Fill & Outline
        g.fillStyle = this.colors.torso;
        g.beginPath();
        g.moveTo(x11[0], x11[1]);
        g.lineTo(x12[0], x12[1]);
        g.lineTo(x24[0], x24[1]);
        g.lineTo(x23[0], x23[1]);
        g.closePath();
        g.fill();

        seg(x11, x12, this.colors.torsoStroke, 6);
        seg(x12, x24, this.colors.torsoStroke, 6);
        seg(x24, x23, this.colors.torsoStroke, 6);
        seg(x23, x11, this.colors.torsoStroke, 6);

        // Spine
        const hipMid = [(x23[0] + x24[0]) * 0.5, (x23[1] + x24[1]) * 0.5];
        seg(xNeck, hipMid, this.colors.spine, 3);

        // 2. Head & Neck
        const noseP = ok(0) ? P(0) : [0, -1.0, 0];
        const xNose = X(noseP);
        const headCenter = [xNeck[0] * 0.2 + xNose[0] * 0.8, xNeck[1] * 0.2 + xNose[1] * 0.8];

        // Neck
        seg(xNeck, headCenter, this.colors.neck, 14);

        // Head Oval
        const rx = S * 0.28, ry = S * 0.35;
        g.fillStyle = this.colors.head;
        g.strokeStyle = this.colors.headStroke;
        g.lineWidth = 4;
        g.beginPath();
        g.ellipse(headCenter[0], headCenter[1], rx, ry, 0, 0, Math.PI * 2);
        g.fill();
        g.stroke();

        // Eyes
        const eyeOffset = rx * 0.38;
        const eyeY = headCenter[1] - ry * 0.08;
        const eyeR = Math.max(3, S * 0.035);
        g.fillStyle = this.colors.eyes;
        g.beginPath();
        g.arc(headCenter[0] - eyeOffset, eyeY, eyeR, 0, Math.PI * 2);
        g.arc(headCenter[0] + eyeOffset, eyeY, eyeR, 0, Math.PI * 2);
        g.fill();

        // Eyebrows
        seg([headCenter[0] - eyeOffset - eyeR * 1.5, eyeY - eyeR * 1.8],
            [headCenter[0] - eyeOffset + eyeR * 1.5, eyeY - eyeR * 2.0], this.colors.eyes, 3);
        seg([headCenter[0] + eyeOffset - eyeR * 1.5, eyeY - eyeR * 2.0],
            [headCenter[0] + eyeOffset + eyeR * 1.5, eyeY - eyeR * 1.8], this.colors.eyes, 3);

        // Smile
        const mouthY = headCenter[1] + ry * 0.45;
        seg([headCenter[0] - rx * 0.2, mouthY], [headCenter[0] + rx * 0.2, mouthY], this.colors.headStroke, 3);

        // 3. Arms with Two-Bone IK
        // Left Arm (pose 11 -> 13 -> 15)
        const p15 = ok(15) ? P(15) : [0.48, 2.75, 0];
        let p13 = ok(13) ? P(13) : null;
        if (this.useIK && p15) {
          p13 = solveTwoBoneIK(p11, p15, p13, true);
        }
        const x13 = X(p13), x15 = X(p15);
        seg(x11, x13, this.colors.armL, 12);
        seg(x13, x15, this.colors.armL, 10);
        g.fillStyle = this.colors.joint;
        g.beginPath(); g.arc(x13[0], x13[1], 6, 0, Math.PI * 2); g.fill();

        // Right Arm (pose 12 -> 14 -> 16)
        const p16 = ok(16) ? P(16) : [-0.48, 2.75, 0];
        let p14 = ok(14) ? P(14) : null;
        if (this.useIK && p16) {
          p14 = solveTwoBoneIK(p12, p16, p14, false);
        }
        const x14 = X(p14), x16 = X(p16);
        seg(x12, x14, this.colors.armR, 12);
        seg(x14, x16, this.colors.armR, 10);
        g.fillStyle = this.colors.joint;
        g.beginPath(); g.arc(x14[0], x14[1], 6, 0, Math.PI * 2); g.fill();
      }

      // 4. Hands with Per-Finger Articulated Bones
      const handScale = S * 0.40;
      [[0, 15], [63, 16]].forEach(([offset, wristPoseIdx]) => {
        const blk = f.slice(offset, offset + 63);
        const hasHand = blk.some(v => v !== 0);
        if (!hasHand || !ok(wristPoseIdx)) return;

        const wP = P(wristPoseIdx);
        const joints = [];
        for (let i = 0; i < 21; i++) {
          const jx = ox + (wP[0] + blk[i * 3] * 0.45) * S;
          const jy = oy + (wP[1] + blk[i * 3 + 1] * 0.45) * S;
          joints.push([jx, jy]);
        }

        // Palm Plate Fill
        g.fillStyle = this.colors.palm;
        g.beginPath();
        [0, 1, 5, 9, 13, 17].forEach((idx, k) => {
          if (k === 0) g.moveTo(joints[idx][0], joints[idx][1]);
          else g.lineTo(joints[idx][0], joints[idx][1]);
        });
        g.closePath();
        g.fill();

        // Palm strokes
        HAND_PALM_CONNECTIONS.forEach(([a, b]) => {
          seg(joints[a], joints[b], this.colors.palmStroke, 4);
        });

        // Per-Finger Bones
        HAND_FINGERS.forEach(finger => {
          finger.segs.forEach(([a, b]) => {
            seg(joints[a], joints[b], finger.color, 5);
            g.fillStyle = finger.color;
            g.beginPath();
            g.arc(joints[b][0], joints[b][1], 3, 0, Math.PI * 2);
            g.fill();
          });
        });
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
      lerpFrame
    };
  }

})(typeof window !== "undefined" ? window : global);
