#!/usr/bin/env python3
"""ClarifySign live data capture tool.

Records dataset clips identical to what LandmarkExtractor receives at inference.
Saves data_own/<word>/r<round>_<n>.npz with:
  raw_landmarks: (T, 225) float32 un-trimmed feature sequence
  features:      (48, 225) float32 sequence after prepare_sequence()
  word:          sign_id string
  round:         int (1, 2, or 3)
  rep:           int (1..5 for words, 1..30 for _idle)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarifysign_ml.features import LandmarkExtractor, prepare_sequence, FEATURE_DIM, SEQ_LEN


IDLE_PROMPTS = [
    # 30 reps guide for idle classes
    "Hands down, sitting comfortably at rest",
    "Hands down, resting hands in lap",
    "Hands down, looking at screen",
    "Hands down, natural breathing / slight shifting",
    "Hands down, sitting still",
    "Hands down, resting arms on desk",
    "Hands down, relaxing shoulders",
    "Hands down, resting at keyboard",
    "Hands raised but completely still (resting near chest)",
    "Hands raised still (holding mug or resting on chin)",
    "Hands raised still (chin resting on hand)",
    "Hands raised still (open hands resting upright)",
    "Hands raised still (clasping hands in front)",
    "Hands raised still (crossed arms)",
    "Hands raised still (one hand on cheek)",
    "Random movement: scratching nose / touching face",
    "Random movement: adjusting glasses or hair",
    "Random movement: rubbing eye / forehead",
    "Random movement: casually waving hand",
    "Random movement: reaching for mouse / clicking",
    "Random movement: typing on keyboard",
    "Random movement: stretching arms",
    "Random movement: turning head left and right",
    "Random movement: taking a drink / picking up cup",
    "Empty frame: step out of view or point camera at empty wall",
    "Empty frame: empty chair / background",
    "Empty frame: no person in frame",
    "Empty frame: empty room",
    "Empty frame: completely clear view",
    "Empty frame: final empty background rep",
]


def draw_hud(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    status: str,
    status_color: tuple[int, int, int],
    progress_text: str,
    sub_progress: str,
    timer_text: str = "",
    detected_info: dict | None = None,
) -> np.ndarray:
    """Draw a clean, responsive HUD on the preview frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Top overlay bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (15, 25, 25), -1)
    # Bottom overlay bar
    cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Header text
    cv2.putText(out, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(out, sub_progress, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 180, 180), 1, cv2.LINE_AA)

    # Status / Timer pill on top right
    cv2.putText(out, status, (w - 280, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
    if timer_text:
        cv2.putText(out, timer_text, (w - 280, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)

    # Footer status
    cv2.putText(out, progress_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Detection indicators on bottom right
    if detected_info:
        pose_ok = detected_info.get("pose", False)
        lh_ok = detected_info.get("lh", False)
        rh_ok = detected_info.get("rh", False)
        hud_lms = f"Pose:{'OK' if pose_ok else '--'}  LH:{'OK' if lh_ok else '--'}  RH:{'OK' if rh_ok else '--'}"
        color = (0, 255, 0) if (lh_ok or rh_ok) else ((0, 200, 255) if pose_ok else (120, 120, 120))
        cv2.putText(out, hud_lms, (w - 300, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return out


def record_clip(
    cap: cv2.VideoCapture,
    extractor: LandmarkExtractor,
    word: str,
    round_id: int,
    rep_id: int,
    out_path: Path,
    countdown_s: float,
    duration_s: float,
    progress_str: str,
    total_reps_for_word: int,
    idle_note: str = "",
) -> bool:
    """Record a single clip with 1s countdown and 3s capture. Returns False if user aborted."""
    subtitle = f"Word: '{word}' | Round {round_id}/3 (Rep {rep_id}/{total_reps_for_word})"
    if idle_note:
        sub_progress = f"Prompt: {idle_note}"
    else:
        sub_progress = "Sign naturally. Start after countdown, finish before time ends."

    # --- Phase 1: Countdown ---
    t_start = time.time()
    while True:
        elapsed = time.time() - t_start
        remaining = countdown_s - elapsed
        if remaining <= 0:
            break

        ok, raw_frame = cap.read()
        if not ok:
            print("\n[ERROR] Failed to read frame from camera.")
            return False

        # Live detection for HUD feedback during countdown
        feat = extractor(raw_frame)
        det_info = getattr(extractor, "last", {})

        # Display preview is mirrored for natural human feedback, but raw_frame is unmirrored
        preview = cv2.flip(raw_frame, 1)
        preview = draw_hud(
            preview,
            title=f"GET READY TO SIGN: {word.upper()}",
            subtitle=subtitle,
            status="READY...",
            status_color=(0, 215, 255),
            progress_text=progress_str,
            sub_progress=sub_progress,
            timer_text=f"Starting in {remaining:.1f}s",
            detected_info=det_info,
        )
        cv2.imshow("ClarifySign Recording Tool", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == 32:  # Space to skip countdown
            break

    # --- Phase 2: Recording (3.0 seconds) ---
    t_record_start = time.time()
    raw_features: list[np.ndarray] = []

    while True:
        elapsed = time.time() - t_record_start
        remaining = duration_s - elapsed
        if remaining <= 0:
            break

        ok, raw_frame = cap.read()
        if not ok:
            print("\n[ERROR] Failed to read frame from camera.")
            return False

        # Extract features using exact Python LandmarkExtractor on UNMIRRORED frame
        feat = extractor(raw_frame)
        raw_features.append(feat)
        det_info = getattr(extractor, "last", {})

        preview = cv2.flip(raw_frame, 1)
        preview = draw_hud(
            preview,
            title=f"RECORDING: {word.upper()}",
            subtitle=subtitle,
            status="● RECORDING",
            status_color=(0, 0, 255),
            progress_text=progress_str,
            sub_progress=sub_progress,
            timer_text=f"{remaining:.1f}s left",
            detected_info=det_info,
        )
        cv2.imshow("ClarifySign Recording Tool", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    if not raw_features:
        print(f"\n[WARNING] No frames captured for {word} r{round_id}_{rep_id}. Retrying.")
        return True

    # Stack raw feature vectors (T, 225)
    raw_arr = np.stack(raw_features).astype(np.float32)

    # Apply exact repository prepare_sequence
    prep_seq = prepare_sequence(raw_arr).astype(np.float32)

    # Save npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        raw_landmarks=raw_arr,
        features=prep_seq,
        word=str(word),
        round=int(round_id),
        rep=int(rep_id),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ClarifySign Live Clip Recording Tool")
    parser.add_argument("--vocab", type=str, default="vocab.txt", help="Path to vocab.txt")
    parser.add_argument("--idle", action="store_true", help="Include _idle class (3 rounds x 30 reps)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--out-dir", type=str, default="data_own", help="Output directory (default: data_own)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--reps-per-round", type=int, default=5, help="Reps per round for words (default: 5)")
    parser.add_argument("--idle-reps-per-round", type=int, default=30, help="Reps per round for _idle (default: 30)")
    parser.add_argument("--countdown", type=float, default=1.0, help="Countdown duration in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds (default: 3.0)")
    parser.add_argument("--task-path", type=str, default="models/holistic_landmarker.task", help="Path to holistic task")
    args = parser.parse_args()

    vocab_file = Path(args.vocab)
    if not vocab_file.is_file():
        print(f"[ERROR] Vocab file not found: {vocab_file}")
        return 1

    words = [line.strip() for line in vocab_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not words:
        print(f"[ERROR] Vocab file {vocab_file} is empty.")
        return 1

    if args.idle and "_idle" not in words:
        words.append("_idle")

    # Plan schedule: for each round (1..3), cycle through all words
    # Each word gets reps-per-round (or idle-reps-per-round)
    total_clips = 0
    tasks: list[tuple[str, int, int, Path, str]] = []  # (word, round_id, rep_id, path, idle_prompt)

    out_root = Path(args.out_dir)

    for r in range(1, args.rounds + 1):
        for w in words:
            reps = args.idle_reps_per_round if w == "_idle" else args.reps_per_round
            for rep in range(1, reps + 1):
                clip_path = out_root / w / f"r{r}_{rep}.npz"
                idle_prompt = ""
                if w == "_idle":
                    idx = (rep - 1) % len(IDLE_PROMPTS)
                    idle_prompt = IDLE_PROMPTS[idx]
                tasks.append((w, r, rep, clip_path, idle_prompt))
                total_clips += 1

    existing_count = sum(1 for _, _, _, p, _ in tasks if p.exists())
    print(f"\n================ CLARIFYSIGN CAPTURE PROTOCOL ================")
    print(f"Total vocabulary classes: {len(words)} (words: {len(words) - (1 if '_idle' in words else 0)}, idle: {'yes' if '_idle' in words else 'no'})")
    print(f"Total planned clips:     {total_clips} (existing: {existing_count}, remaining: {total_clips - existing_count})")
    print(f"Rounds:                  {args.rounds}")
    print(f"Word reps per round:     {args.reps_per_round} (total 15 reps/word)")
    if args.idle:
        print(f"Idle reps per round:     {args.idle_reps_per_round} (total 90 clips)")
    print(f"Clip timing:             {args.countdown:.1f}s countdown + {args.duration:.1f}s recording")
    est_minutes = ((total_clips - existing_count) * (args.countdown + args.duration + 0.3)) / 60.0
    print(f"Estimated time left:     ~{est_minutes:.1f} minutes")
    print(f"Output directory:        {out_root.resolve()}")
    print("Controls in window:      'q' = save & quit (resumable) | SPACE = skip countdown")
    print("==============================================================\n")

    if existing_count == total_clips:
        print("All clips already recorded! Ready for Phase 3 training.")
        return 0

    # Initialize Camera
    print(f"Opening camera index {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(
            f"\n[ERROR] Could not open camera {args.camera_index}.\n"
            "On macOS, please ensure Camera permission is granted for Terminal / iTerm / IDE:\n"
            "  System Settings -> Privacy & Security -> Camera -> Enable for your terminal application.\n"
            "If using an external camera, you can specify --camera-index 1."
        )
        return 1

    # Camera warmup
    for _ in range(15):
        cap.read()

    # Initialize LandmarkExtractor
    print("Initializing LandmarkExtractor (Holistic + Hand Fallback on CPU)...")
    task_path = str(ROOT / args.task_path) if not os.path.isabs(args.task_path) else args.task_path
    try:
        extractor = LandmarkExtractor(task_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load LandmarkExtractor with model at {task_path}: {exc}")
        cap.release()
        return 1

    completed = existing_count
    try:
        for w, r, rep, path, idle_prompt in tasks:
            if path.exists():
                continue

            total_reps_for_w = args.idle_reps_per_round if w == "_idle" else args.reps_per_round
            pct = (completed / total_clips) * 100.0
            prog_str = f"Progress: [{completed + 1}/{total_clips}] ({pct:.1f}%) | Press 'q' to pause/quit"
            print(f"\n[{completed + 1}/{total_clips}] {w} - Round {r}/3, Rep {rep}/{total_reps_for_w} ...")

            ok = record_clip(
                cap=cap,
                extractor=extractor,
                word=w,
                round_id=r,
                rep_id=rep,
                out_path=path,
                countdown_s=args.countdown,
                duration_s=args.duration,
                progress_str=prog_str,
                total_reps_for_word=total_reps_for_w,
                idle_note=idle_prompt,
            )
            if not ok:
                print("\nRecording paused/cancelled by user. Run the same command to resume.")
                break

            completed += 1
            print(f"  ✓ Saved to {path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    print(f"\nSession finished. Total completed: {completed}/{total_clips} clips.")
    if completed == total_clips:
        print("ALL CLIPS RECORDED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
