# SO-101 on this Mac — Guide 1 runbook

Environment is already built. Everything below runs in the `lerobot` conda env.

| Piece | Value |
| ----- | ----- |
| Env | conda env `lerobot` at `/opt/anaconda3/envs/lerobot` |
| Python | 3.10.20 |
| lerobot | 0.4.4 (the version the hackathon guides are tested against) |
| torch | 2.10.0, MPS available (no CUDA on Mac, and none needed) |
| ffmpeg | 9.0.1 via conda-forge (used for dataset video encoding) |
| opencv | 5.0.0 with Cocoa GUI support |

```bash
conda activate lerobot
```

**Install the feetech extra, not plain lerobot.** `pip install lerobot` does not pull
`scservo_sdk`, so every SO-101 command dies with `ModuleNotFoundError: No module
named 'scservo_sdk'` the moment it opens the servo bus -- including
`lerobot-calibrate`. The base install looks fine until then, because the robot
classes import without it.

```bash
pip install "lerobot[feetech]==0.4.4"
```

## Step 2 — find the serial ports

Plug in one arm at a time.

```bash
ls /dev/tty.usbmodem*
```

One port per plugged-in arm. Nothing printed = charge-only cable or a loose plug;
swap the cable before debugging anything else. `lerobot-find-port` also works but
on macOS it dumps ~150 pty devices around the answer, so the `ls` is easier to read.

Port names are **not** stable across reboots and replugs — re-check them at the
start of every session. You do not need `sudo`, `chmod`, or udev rules on macOS.

## Step 3 — tell the two arms apart

Unplug test: list ports with both connected, unplug one, list again, note which
name vanished. Or the passive wiggle test:

```bash
python check_so101.py --port /dev/tty.usbmodemXXXXXXXX
```

Reads all six joints for 15 s with torque disabled and commands no motion, so you
can move joints by hand. Good output: numbers repeatable at rest, none parked near
±180, and moving one joint changes only its own column, consistently.

Then pin them for the session:

```bash
export LEADER=/dev/tty.usbmodemAAAAAAAA
export FOLLOWER=/dev/tty.usbmodemBBBBBBBB
```

## Step 4 — servo IDs (DIY kits only)

Pre-assembled arms ship with IDs set — skip. From a kit, one servo connected at a
time, or you write the same ID to several at once:

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER
```

## Step 5 — calibrate each arm

In the manual's calibration start pose. The `--id` you pick here is the `--id`
every later command must use, because profiles are keyed by it.

```bash
lerobot-calibrate --robot.type=so101_follower \
    --robot.port=$FOLLOWER --robot.id=my_follower_arm

lerobot-calibrate --teleop.type=so101_leader \
    --teleop.port=$LEADER --teleop.id=my_leader_arm
```

Then back the profiles up immediately — recalibrating mid-demo costs you the demo:

```bash
ls ~/.cache/huggingface/lerobot/calibration/
cp -R ~/.cache/huggingface/lerobot/calibration ~/lerobot-calibration-backup
```

## Step 6 — pair them (teleoperate)

```bash
lerobot-teleoperate \
    --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=my_follower_arm \
    --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=my_leader_arm
```

Hand near the power switch on the first run, move the leader slowly. A joint
travelling the wrong way is a calibration or servo-ID mismatch, not a hardware
fault. The arm stiffening the moment the script connects is expected — `connect()`
enables torque.

## Step 7 — record a dataset

```bash
lerobot-find-cameras opencv
lerobot-record --help    # camera and dataset flags
```

**Grant your terminal camera access first**: System Settings → Privacy & Security →
Camera. Without it capture returns black frames and reports no error at all.

Episodes land in `~/.cache/huggingface/lerobot/<your-repo-id>/` as video + parquet,
which is what `lerobot-train` consumes (Guide 3).

## Version notes for this install

- CLI types are `so101_follower` / `so101_leader` as the guide says.
- If you import classes in your own code, the 0.4.4 module paths are
  `lerobot.robots.so_follower` and `lerobot.teleoperators.so_leader` (unified
  SO-100/SO-101 modules), not `lerobot.robots.so101_follower`.
- Flags shift slightly between lerobot releases — `<command> --help` is the truth.
