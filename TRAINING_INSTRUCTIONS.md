# ACT training instructions — M5 Pro Mac (Bhargav)

These instructions are written for a coding agent to follow exactly. The goal:
train the ACT policy on 60 MuJoCo-collected demonstrations, on this Mac, inside
~45 minutes of wall clock. Everything needed is in this repo; the demonstration
data regenerates deterministically (same seed = byte-identical episodes), so
nothing large needs to be transferred.

Context in one line: `sim/mujoco/` contains a verified local pipeline
(100% grasp success over 60 episodes on Windows) that records demos in exactly
the format `training/build_dataset.py` consumes; `sim/mujoco/run.sh` chains
collect → build dataset → train.

## Step 0 — prerequisites (5 min, one-time)

1. `git pull` on `main` at the repo root. Confirm these exist:
   `sim/mujoco/scene.xml`, `sim/mujoco/collect.py`, `sim/mujoco/run.sh`,
   `sim/mujoco/robotstudio_so101/so101.xml`, `training/build_dataset.py`.
2. The pipeline expects the existing conda env at
   `/opt/homebrew/Caskroom/miniforge/base/envs/lerobot` (override with
   `L=/path/to/env`). Verify: `conda env list` and
   `$L/bin/python -c "import torch, lerobot; print('ok')"`.
3. Install the three sim/render packages into THAT env (they are small):
   ```bash
   /opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/pip install mujoco imageio imageio-ffmpeg
   ```
4. Do NOT set `MUJOCO_GL` on macOS — offscreen rendering works by default and
   the Linux values (egl/osmesa) break it.
5. MuJoCo rendering must run on the main thread (default; just do not move
   collection into worker threads).

## Step 1 — collect the 60 episodes (~5 min)

```bash
COLLECT_ONLY=1 bash sim/mujoco/run.sh
```

Expected output: 60 `[OK ]` lines and `final success rate = 100.0%` (or very
near), files in `data/sim_raw/ep000..ep059` (.npz + .mp4 pairs). Seed 0 is the
default, matching the verified Windows run exactly.

If the success rate is below 90%: STOP and report the printed failure lines —
do not proceed to training on a degraded dataset. (This has not happened in
any run so far; 60/60 on Windows.)

## Step 2 — the 200-step timing gate (2 min, do NOT skip)

Before committing to the full run, measure this Mac's real training speed:

```bash
STEPS=200 bash sim/mujoco/run.sh
```

(Collection already done in step 1 re-runs in ~5 min — acceptable; or run the
build+train stages manually per run.sh if you prefer.) Read the `steps/s` or
per-step timing from the lerobot-train log and compute
`affordable_steps = 2400 seconds × measured_steps_per_second`.

| Measured speed | Action |
|---|---|
| ≥ 4 steps/s | run `STEPS=10000` |
| 2–4 steps/s | run `STEPS=8000` (the default) |
| 1–2 steps/s | run `STEPS=5000` |
| < 1 step/s | STOP — report the number back to Pranav's session before burning the budget; the fallback is re-collecting at 224×224 (supported via `collect.py --render-size 224x224`, gives ~2–3× speedup, but then `training/build_dataset.py`'s declared video shape must be checked — it declares (480,640,3) and resizes) |

## Step 3 — full training run

```bash
STEPS=<chosen> bash sim/mujoco/run.sh
```

This re-collects (5 min, deterministic, harmless), rebuilds the dataset, and
trains. Flags already baked into run.sh — do not change them:
`chunk_size=30 n_action_steps=15` (a 10 Hz policy must re-plan every 1.5 s;
the old chunk_size=100 plan would be blind for 10 s), `use_amp=false` (MPS),
`scheduler_decay_steps=$STEPS` (LR must decay within a short run),
`image_transforms.enable=true`, `batch_size=8`, `num_workers=2`.

Known acceptable warning: if your lerobot build rejects `use_amp`,
`scheduler_decay_steps`, or `image_transforms.enable`, drop ONLY the rejected
flag (they are each on their own line) and note which one.

Sanity checks during training:
- Loss must fall and stay finite. If NaN: stop, report.
- After the dataset build, check `data/lerobot/so101_pick_place/meta/stats.json`:
  no `action` or `observation.state` std below 1e-3. If one is: report before
  training (normalization would explode).

## Step 4 — verify the policy by ROLLOUT, not by loss (5 min)

Loss numbers do not predict task success. After training, run the policy
closed-loop in the same MuJoCo env and count successes. A minimal eval loop
(policy in, env steps, success() check from `sim/mujoco/env.py`) over 20
episodes with fixed seeds 1000–1019. Report `X/20`.

Expected honest range for this setup (60 scripted demos, wrist cam only,
5–10k steps): anywhere from 30% to 80% in-distribution is normal. Do not
panic-tune below that; report the number.

## Step 5 — deliverables to report back

1. Measured steps/s and the STEPS you chose.
2. Final train loss + the rollout success X/20.
3. Checkpoint path: `outputs/act_so101/checkpoints/last/pretrained_model`.
4. Copy the checkpoint somewhere safe (it is gitignored).

## Hard warnings (do not violate)

- NEVER train on a dataset where `action` equals `observation.state` — the
  collector asserts this; if you regenerate data any other way, keep the
  assert.
- The wrist camera in the recordings is the policy's only visual input; do not
  swap camera names in the dataset config.
- When this policy later runs on the REAL arm: the model outputs radians;
  LeRobot's SO-101 driver defaults to degrees (and gripper 0–100). A unit
  adapter is REQUIRED before any real deployment, validated by replaying one
  recorded episode on hardware first, with `--robot.max_relative_target` set
  small. That step is out of scope for this document — coordinate with
  Pranav's session before touching the real arm.
