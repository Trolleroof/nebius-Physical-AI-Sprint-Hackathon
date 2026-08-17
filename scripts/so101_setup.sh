#!/usr/bin/env bash
# SO-101 bring-up helper — Hackathon Guide 1 (macOS)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/hardware/so101.env"

echo "== SO-101 setup check =="
echo "Repo: $ROOT"
echo

# --- conda / lerobot ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install miniconda/miniforge first."
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

python - <<'PY'
import lerobot, torch
print(f"lerobot {lerobot.__version__}  torch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

echo
echo "== USB serial ports =="
PORTS=(/dev/tty.usbmodem*)
if [[ ! -e "${PORTS[0]}" ]]; then
  echo "No /dev/tty.usbmodem* ports found."
  echo "  • Plug in one arm at a time with a data-capable USB-C cable"
  echo "  • Power on the arm supply"
  echo "  • Re-run: bash scripts/so101_setup.sh"
else
  ls -l /dev/tty.usbmodem*
fi

echo
echo "== Calibration profiles =="
CAL_DIR="$HOME/.cache/huggingface/lerobot/calibration"
if [[ -d "$CAL_DIR" ]] && ls "$CAL_DIR"/*.json >/dev/null 2>&1; then
  ls -la "$CAL_DIR"
else
  echo "None yet — run calibration after ports are set (Step 5)."
fi

if [[ -f "$ENV_FILE" ]]; then
  echo
  echo "== Loaded $ENV_FILE =="
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo "LEADER=$LEADER"
  echo "FOLLOWER=$FOLLOWER"
  echo "LEADER_ID=${LEADER_ID:-my_leader_arm}"
  echo "FOLLOWER_ID=${FOLLOWER_ID:-my_follower_arm}"
else
  echo
  echo "Tip: cp hardware/so101.env.example hardware/so101.env and set LEADER/FOLLOWER."
fi

echo
echo "== Next commands (Guide 1) =="
cat <<'EOF'
# 1) Find ports (interactive unplug test):
lerobot-find-port

# 2) Identify leader vs follower (passive, no motion):
python replay/check_so101.py --port "$LEADER" --type leader
python replay/check_so101.py --port "$FOLLOWER" --type follower

# 3) Calibrate each arm (~20 min total):
lerobot-calibrate --robot.type=so101_follower \
  --robot.port="$FOLLOWER" --robot.id=my_follower_arm
lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port="$LEADER" --teleop.id=my_leader_arm

# 4) Back up calibration immediately:
bash scripts/backup_so101_calibration.sh

# 5) First teleop:
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$FOLLOWER" --robot.id=my_follower_arm \
  --teleop.type=so101_leader --teleop.port="$LEADER" --teleop.id=my_leader_arm
EOF
