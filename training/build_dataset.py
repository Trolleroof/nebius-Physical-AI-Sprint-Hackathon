#!/usr/bin/env python
"""Convert a directory of (epNNN.npz + epNNN.mp4) pairs into a LeRobotDataset.

Verified against lerobot 0.4.4:
  lerobot.datasets.lerobot_dataset.LeRobotDataset.create(repo_id, fps, features, root=...)
  ds.add_frame({...,"task": str}) -> ds.save_episode() -> ds.finalize()

Produced features:
  observation.images.wrist : video, (480, 640, 3), uint8 HWC frames
  observation.state        : float32, (6,)  measured joint positions [rad]
  action                   : float32, (6,)  COMMANDED joint targets  [rad]
  task                     : language string, per frame
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

FPS = 10
JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
IMG_KEY = "observation.images.wrist"
HEIGHT, WIDTH = 480, 640

FEATURES = {
    IMG_KEY: {
        "dtype": "video",
        "shape": (HEIGHT, WIDTH, 3),
        # names[2] == "channels" is what makes lerobot transpose to (C,H,W) for the policy.
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
}


def read_video_frames(path: Path) -> np.ndarray:
    """Decode an mp4 into a (T, H, W, 3) uint8 RGB array."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[0] != HEIGHT or bgr.shape[1] != WIDTH:
            bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with epNNN.npz + epNNN.mp4")
    ap.add_argument("--repo-id", default="local/so101_pick_place")
    ap.add_argument("--root", required=True, help="output dataset dir")
    ap.add_argument("--task", default="Put the block in the tray.")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_dir = Path(args.raw_dir)
    npz_paths = sorted(raw_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"no .npz files in {raw_dir}")

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=FEATURES,
        root=root,
        robot_type="so101_follower",
        use_videos=True,
    )

    total = 0
    for npz_path in npz_paths:
        mp4_path = npz_path.with_suffix(".mp4")
        if not mp4_path.exists():
            raise SystemExit(f"missing video for {npz_path.name}: {mp4_path}")

        d = np.load(npz_path)
        state = np.asarray(d["observation.state"], dtype=np.float32)
        action = np.asarray(d["action"], dtype=np.float32)
        frames = read_video_frames(mp4_path)

        if state.ndim != 2 or state.shape[1] != 6:
            raise SystemExit(f"{npz_path.name}: observation.state must be (T,6), got {state.shape}")
        if action.shape != state.shape:
            raise SystemExit(
                f"{npz_path.name}: action {action.shape} != observation.state {state.shape}"
            )

        T = min(len(state), len(action), len(frames))
        if T != len(state) or T != len(frames):
            print(
                f"  {npz_path.name}: length mismatch "
                f"(state={len(state)}, action={len(action)}, video={len(frames)}) -> truncating to {T}"
            )

        for i in range(T):
            ds.add_frame(
                {
                    IMG_KEY: frames[i],
                    "observation.state": state[i],
                    "action": action[i],
                    "task": args.task,
                }
            )
        ds.save_episode()
        total += T
        print(f"saved episode from {npz_path.name}: {T} frames")

    ds.finalize()
    print(f"\ndone: {len(npz_paths)} episodes, {total} frames -> {root}")


if __name__ == "__main__":
    main()
