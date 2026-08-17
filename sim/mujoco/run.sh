#!/usr/bin/env bash
# SO-101 sim pipeline on the Mac: collect -> build_dataset -> lerobot-train.
#
# Everything (mujoco, cv2, torch, lerobot) is expected in the single conda env
#   /opt/homebrew/Caskroom/miniforge/base/envs/lerobot
# Override with:  L=/path/to/env ./sim/mujoco/run.sh
#
#   ./sim/mujoco/run.sh                 # 60 episodes, then build + train
#   EPISODES=120 ./sim/mujoco/run.sh
#   COLLECT_ONLY=1 ./sim/mujoco/run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
echo "repo root: $PWD"

L="${L:-/opt/homebrew/Caskroom/miniforge/base/envs/lerobot}"
PY="$L/bin/python"
[ -x "$PY" ] || { echo "no python at $PY -- set L=/path/to/env" >&2; exit 1; }

EPISODES="${EPISODES:-60}"
SEED="${SEED:-0}"
RAW_DIR="${RAW_DIR:-data/sim_raw}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot/so101_pick_place}"
REPO_ID="${REPO_ID:-local/so101_pick_place}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/act_so101}"
SCENE="${SCENE:-sim/mujoco/scene.xml}"
GRASP="${GRASP:-friction}"          # friction | weld
STEPS="${STEPS:-8000}"
DEVICE="${DEVICE:-mps}"
TASK="${TASK:-Put the orange cube in the tray.}"

# NOTE: expanded below as ${EXTRA[@]+"${EXTRA[@]}"}, not "${EXTRA[@]}".  macOS
# ships bash 3.2, where expanding an EMPTY array under `set -u` aborts with
# "EXTRA[@]: unbound variable" -- so the plain form breaks every default run.
EXTRA=()
[ "${WORLD_VIDEO:-0}" = "1" ]   && EXTRA+=(--world-video)
[ "${KEEP_FAILURES:-0}" = "1" ] && EXTRA+=(--keep-failures)

# wrist_roll's servo is broken on the physical arm and the joint is taped at
# -pi/2.  This gate proves it cannot move before we spend an hour training on
# data that might roll it.  `set -e` aborts the run if it fails.
echo; echo "== 0/3 wrist lock gate =="
"$PY" sim/mujoco/check_wrist_lock.py --scene "$SCENE"

echo; echo "== 1/3 collect =="
"$PY" sim/mujoco/collect.py \
  --episodes "$EPISODES" \
  --out      "$RAW_DIR" \
  --seed     "$SEED" \
  --scene    "$SCENE" \
  --grasp    "$GRASP" \
  ${EXTRA[@]+"${EXTRA[@]}"}

# Re-check against what was actually recorded, not just what the code intends.
"$PY" sim/mujoco/check_wrist_lock.py --scene "$SCENE" --episodes "$RAW_DIR"

if [ "${COLLECT_ONLY:-0}" = "1" ]; then
  echo; echo "COLLECT_ONLY=1: stopping after collection."
  exit 0
fi

echo; echo "== 2/3 build_dataset =="
"$PY" training/build_dataset.py \
  --raw-dir "$RAW_DIR" \
  --root    "$DATASET_ROOT" \
  --repo-id "$REPO_ID" \
  --task    "$TASK" \
  --fps     10 \
  --overwrite

# NOTE: --policy.use_amp / --policy.scheduler_decay_steps /
# --dataset.image_transforms.enable are accepted by lerobot 0.4.x; if your build
# rejects one of them, drop that single flag.
echo; echo "== 3/3 lerobot-train =="
"$L/bin/lerobot-train" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device="$DEVICE" \
  --policy.chunk_size=30 \
  --policy.n_action_steps=15 \
  --policy.use_amp=false \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_so101 \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq="$STEPS" \
  --log_freq=100 \
  --num_workers=2 \
  --wandb.enable=false

echo; echo "done. checkpoints -> $OUTPUT_DIR/checkpoints/last/pretrained_model"
