#!/usr/bin/env bash
# Back up LeRobot SO-101 calibration profiles (Guide 1 checkpoint).
set -euo pipefail

SRC="$HOME/.cache/huggingface/lerobot/calibration"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/hardware/calibration_backup/$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$SRC" ]] || ! ls "$SRC"/*.json >/dev/null 2>&1; then
  echo "No calibration files in $SRC — calibrate first."
  exit 1
fi

mkdir -p "$DEST"
cp -a "$SRC/." "$DEST/"
echo "Backed up to $DEST"
ls -la "$DEST"
