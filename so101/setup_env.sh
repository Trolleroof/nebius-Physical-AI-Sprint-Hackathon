#!/usr/bin/env bash
# Build the lerobot environment this repo's SO-101 scripts run in (Guide 1, Step 1).
#
# uv rather than conda: conda is not installed on this Mac (the `conda` shell
# function in ~/.zshrc is a leftover pointing at nothing), and uv builds the
# same env in about a minute. ffmpeg comes from Homebrew instead of conda-forge
# -- lerobot shells out to it for dataset video encoding, so it only has to be
# on PATH, not in the venv.
#
#     ./setup_env.sh          # create or update ~/lerobot-env
#
# Re-running is safe; uv reuses what is already installed.

set -euo pipefail

ENV_DIR="${LEROBOT_ENV:-$HOME/lerobot-env}"
# The version the hackathon guides are tested against. Flags move between
# lerobot releases, so pin it rather than tracking latest mid-hackathon.
LEROBOT_VERSION="0.4.4"

command -v uv >/dev/null || {
    echo "uv is not installed: https://docs.astral.sh/uv/" >&2
    exit 1
}

command -v ffmpeg >/dev/null || {
    echo "ffmpeg is not on PATH. Dataset recording needs it: brew install ffmpeg" >&2
    exit 1
}

uv venv "$ENV_DIR" --python 3.10
# opencv-python in the same breath: anything touching a camera needs it and the
# error you get without it is unhelpful.
VIRTUAL_ENV="$ENV_DIR" uv pip install "lerobot==$LEROBOT_VERSION" opencv-python

echo
echo "done. activate with:"
echo "    source $ENV_DIR/bin/activate"
echo "then check it with:"
echo "    python $(dirname "$0")/preflight.py"
