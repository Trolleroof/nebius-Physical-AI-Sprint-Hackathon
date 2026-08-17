#!/usr/bin/env python
"""Generate synthetic SO-101 episodes in the exact format the simulator will emit.

Per episode:
  epNNN.npz  with "observation.state" (T,6) float32 rad and "action" (T,6) float32 rad
  epNNN.mp4  640x480 RGB, T frames, 10 Hz

This exists only to prove the training pipeline before real data lands.
"""

import argparse
from pathlib import Path

import numpy as np

FPS = 10
W, H = 640, 480
JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
# Rough plausible ranges (radians) for the SO-101 arm.
LOW = np.array([-1.8, -1.4, -1.4, -1.4, -2.6, -0.1], dtype=np.float32)
HIGH = np.array([1.8, 1.4, 1.4, 1.4, 2.6, 1.2], dtype=np.float32)


def smooth_trajectory(rng: np.random.Generator, T: int) -> np.ndarray:
    """Random-but-smooth (T,6) trajectory built from a few low-frequency sinusoids."""
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    traj = np.zeros((T, 6), dtype=np.float32)
    for k in range(1, 4):
        amp = rng.uniform(-1.0, 1.0, size=(1, 6)).astype(np.float32) / k
        phase = rng.uniform(0.0, 2 * np.pi, size=(1, 6)).astype(np.float32)
        traj += amp * np.sin(2 * np.pi * k * 0.5 * t + phase)
    traj /= np.abs(traj).max(axis=0, keepdims=True) + 1e-6
    mid = (LOW + HIGH) / 2.0
    half = (HIGH - LOW) / 2.0 * 0.7
    return (mid + traj * half).astype(np.float32)


def render_video(path: Path, T: int, rng: np.random.Generator) -> None:
    import cv2

    bg = rng.integers(30, 200, size=3).tolist()
    sq = [255 - c for c in bg]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    side = 90
    for i in range(T):
        f = i / max(T - 1, 1)
        # BGR for cv2; the solid colour + moving square keeps the video non-degenerate.
        frame = np.full((H, W, 3), bg[::-1], dtype=np.uint8)
        x = int(20 + f * (W - side - 40))
        y = int(H / 2 - 120 + 200 * np.sin(2 * np.pi * f))
        y = int(np.clip(y, 0, H - side))
        cv2.rectangle(frame, (x, y), (x + side, y + side), sq[::-1], -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_raw", help="output directory")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        T = int(rng.integers(130, 151))
        state = smooth_trajectory(rng, T)
        # `action` is the COMMANDED target: the controller leads the measured state.
        # Model that as the state one step ahead plus a small command offset.
        lead = np.vstack([state[1:], state[-1:]])
        action = lead + rng.normal(0.0, 0.01, size=lead.shape).astype(np.float32)
        action = np.clip(action, LOW, HIGH).astype(np.float32)

        stem = out / f"ep{ep:03d}"
        np.savez(
            stem.with_suffix(".npz"),
            **{"observation.state": state, "action": action},
        )
        render_video(stem.with_suffix(".mp4"), T, rng)
        print(f"wrote {stem}.npz / .mp4  T={T}")


if __name__ == "__main__":
    main()
