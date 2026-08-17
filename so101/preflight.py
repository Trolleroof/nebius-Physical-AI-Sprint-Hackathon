#!/usr/bin/env python
"""Check everything Guide 1 needs before an arm moves, in one pass.

Every line is either OK, WARN (fine until you need that step), or FAIL (the
next command will not work). Nothing here opens a serial port or commands
motion -- run check_so101.py for that.

    python preflight.py
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration"
# The version the guides are written against. A different one is usable, but
# flags drift between releases, so say so rather than let it surprise you later.
EXPECTED_LEROBOT = "0.4.4"

failed = False


def report(status: str, label: str, detail: str = "") -> None:
    global failed
    if status == "FAIL":
        failed = True
    print(f"  {status:<4}  {label:<22} {detail}")


def check_python() -> None:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    inside_venv = sys.prefix != sys.base_prefix
    if not inside_venv:
        report("FAIL", "python", f"{version}, but not in a venv -- source ~/lerobot-env/bin/activate")
    elif (v.major, v.minor) != (3, 10):
        report("WARN", "python", f"{version} in {sys.prefix} (guides use 3.10)")
    else:
        report("OK", "python", f"{version} in {sys.prefix}")


def check_lerobot() -> None:
    try:
        import lerobot
    except ImportError:
        report("FAIL", "lerobot", "not installed -- run ./setup_env.sh")
        return
    version = getattr(lerobot, "__version__", "unknown")
    status = "OK" if version == EXPECTED_LEROBOT else "WARN"
    detail = version if status == "OK" else f"{version} (guides tested against {EXPECTED_LEROBOT})"
    report(status, "lerobot", detail)


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        report("FAIL", "torch", "not installed -- run ./setup_env.sh")
        return
    if torch.backends.mps.is_available():
        report("OK", "torch", f"{torch.__version__}, MPS available")
    else:
        # Not fatal: everything but training still runs on CPU.
        report("WARN", "torch", f"{torch.__version__}, no MPS -- training will crawl")


def check_opencv() -> None:
    try:
        import cv2
    except ImportError:
        report("FAIL", "opencv", "not installed -- cameras will fail obscurely")
        return
    report("OK", "opencv", cv2.__version__)


def check_ffmpeg() -> None:
    path = shutil.which("ffmpeg")
    if not path:
        report("WARN", "ffmpeg", "not on PATH -- needed to record datasets (brew install ffmpeg)")
        return
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    # "ffmpeg version 8.1 Copyright (c) ..." -> "8.1"
    version = out.split()[2] if out else "?"
    report("OK", "ffmpeg", f"{version} at {path}")


def check_ports() -> None:
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if ports:
        report("OK", "serial ports", ", ".join(ports))
    else:
        report("WARN", "serial ports", "none -- plug an arm in, or swap a charge-only cable (./usbcheck.py)")
    # Port names are not stable across replugs, so an exported name from an
    # earlier session is a real trap: it silently points at nothing.
    for var in ("LEADER", "FOLLOWER"):
        value = os.environ.get(var)
        if value is None:
            report("WARN", f"${var}", "unset -- see RUNBOOK step 3")
        elif value not in ports:
            report("FAIL", f"${var}", f"{value} is not a live port -- the hub renumbered it")
        else:
            report("OK", f"${var}", value)


def check_calibration() -> None:
    profiles = sorted(CALIB_DIR.glob("*/*.json")) if CALIB_DIR.is_dir() else []
    if not profiles:
        report("WARN", "calibration", f"no profiles in {CALIB_DIR} -- run lerobot-calibrate (step 5)")
        return
    names = ", ".join(f"{p.parent.name}/{p.stem}" for p in profiles)
    report("OK", "calibration", names)


def check_camera() -> None:
    """macOS gates camera access per application, and a denied camera returns
    black frames with no error at all -- so probe it here instead of finding
    out from an unusable dataset."""
    try:
        import cv2
    except ImportError:
        return
    capture = cv2.VideoCapture(0)
    opened = capture.isOpened()
    grabbed = bool(capture.read()[0]) if opened else False
    capture.release()
    if grabbed:
        report("OK", "camera", "index 0 returns frames")
    else:
        report("WARN", "camera", "no frames -- grant your terminal System Settings > Privacy & Security > Camera")


def main() -> None:
    print("SO-101 preflight\n")
    for check in (
        check_python,
        check_lerobot,
        check_torch,
        check_opencv,
        check_ffmpeg,
        check_ports,
        check_calibration,
        check_camera,
    ):
        check()
    print("\nFAIL means the next command will not work; WARN is fine until you need that step.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
