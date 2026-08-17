# SO-101 ACT training pipeline

Verified end-to-end on Apple Silicon (MPS) with **lerobot 0.4.4 / torch 2.10.0**.

```bash
export L=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot
cd /Users/nikhi/nebius-Physical-AI-Sprint-Hackathon/training
```

## Input format (fixed, produced by the simulator)

Per episode, in one directory:

- `epNNN.npz`
  - `observation.state` — float32 `(T, 6)`, **measured** joint positions, radians,
    order `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`
  - `action` — float32 `(T, 6)`, **commanded** joint targets, same order/units
- `epNNN.mp4` — 640x480 RGB wrist camera, `T` frames, 10 Hz

`action` must be the commanded target, not a copy of the measured state. ACT learns
state -> future commanded chunk; if action == state the policy learns the identity map
and does nothing useful on the robot.

## 1. Build the LeRobot dataset from real data

```bash
$L/bin/python build_dataset.py \
  --raw-dir data/real_raw \
  --root data/lerobot/so101_pick_place \
  --repo-id local/so101_pick_place \
  --task "Put the block in the tray." \
  --fps 10 \
  --overwrite
```

Sanity-check that it loads back:

```bash
$L/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/so101_pick_place', root='data/lerobot/so101_pick_place')
print(ds); print({k: getattr(v,'shape',v) for k,v in ds[0].items()})
"
```

## 2. Real training run (15000 steps, batch 8, chunk 100, save every 5000)

```bash
$L/bin/lerobot-train \
  --dataset.repo_id=local/so101_pick_place \
  --dataset.root=data/lerobot/so101_pick_place \
  --policy.type=act \
  --policy.device=mps \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.push_to_hub=false \
  --output_dir=outputs/act_so101 \
  --job_name=act_so101 \
  --batch_size=8 \
  --steps=15000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false
```

Checkpoints land in `outputs/act_so101/checkpoints/{005000,010000,015000,last}/pretrained_model/`.
Resume an interrupted run with `--resume=true` (keep the same `--output_dir`).

## 3. Synthetic smoke test (no real data needed)

```bash
$L/bin/python make_fake_episodes.py --out data/fake_raw --episodes 6
$L/bin/python build_dataset.py --raw-dir data/fake_raw \
  --root data/lerobot/so101_pick_place_fake \
  --repo-id local/so101_pick_place_fake --overwrite
$L/bin/lerobot-train \
  --dataset.repo_id=local/so101_pick_place_fake \
  --dataset.root=data/lerobot/so101_pick_place_fake \
  --policy.type=act --policy.device=mps \
  --policy.chunk_size=20 --policy.n_action_steps=20 \
  --policy.push_to_hub=false \
  --output_dir=outputs/smoke --job_name=act_smoke \
  --batch_size=2 --steps=200 --save_freq=100 --log_freq=25 \
  --num_workers=0 --wandb.enable=false
```

Confirmed result: loss 30.30 (step 25) -> 4.93 (step 200), finite throughout,
checkpoints written at 100 and 200.

## Notes / gotchas

- `--output_dir` must not already exist unless you pass `--resume=true`.
- `chunk_size` and `n_action_steps` must be set together; `n_action_steps` may not
  exceed `chunk_size`.
- Chunk 100 at 10 Hz = 10 s of lookahead, more than half a ~14 s episode. If real
  episodes come in much shorter than 140 frames, drop chunk to ~50.
- `num_workers>0` plus video decoding can be flaky on macOS; if dataloader workers
  crash, fall back to `--num_workers=0`.
- The homebrew ffmpeg and PyAV ship duplicate `libavdevice`; the `objc[...] Class
  AVFFrameReceiver is implemented in both ...` warning at import is harmless.
- Video encoding uses libsvtav1 and prints a wall of `Svt[info]` lines. Harmless.
