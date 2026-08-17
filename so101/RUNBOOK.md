# SO-101 on this Mac — Guide 1 runbook

Environment is built. Everything below runs in the `~/lerobot-env` venv.

| Piece | Value |
| ----- | ----- |
| Env | uv venv at `~/lerobot-env` |
| Python | 3.10.20 |
| lerobot | 0.4.4 (the version the hackathon guides are tested against) |
| torch | 2.10.0, MPS available (no CUDA on Mac, and none needed) |
| ffmpeg | 8.1 from Homebrew (lerobot shells out to it for dataset video encoding) |
| opencv | 4.12.0 |

uv, not conda: there is no conda on this Mac — the `conda` shell function in
`~/.zshrc` is a leftover that points at nothing, so every `conda` command dies
with `permission denied`. Ignore it.

```bash
source ~/lerobot-env/bin/activate
```

## Step 1 — rebuild the environment (only if it is missing)

```bash
./setup_env.sh
```

Idempotent, takes about a minute. Needs `uv` and `brew install ffmpeg`.

## Step 1b — check everything at once

```bash
python preflight.py
```

Python, lerobot, torch/MPS, opencv, ffmpeg, live serial ports, `$LEADER` /
`$FOLLOWER` still pointing at real ports, calibration profiles, and whether the
camera actually returns frames. FAIL means the next command will not work; WARN
is fine until you reach that step. Run it at the start of every session — it
catches the renumbered port and the denied camera before they cost you an hour.

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

When no port appears, `./usbcheck.py` says whether the arm is even enumerating —
it decodes the USB bus by interface class, so it separates "bad cable, nothing on
the bus" from "device is there but no serial driver attached". `./usbcheck.py
--watch` prints arrivals live while you replug.

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
Camera. Without it capture returns black frames and reports no error at all —
right now this Mac has *not* granted it, and `lerobot-find-cameras opencv` finds
zero cameras with `not authorized to capture video`. `preflight.py` checks the
same thing in one line.

Episodes land in `~/.cache/huggingface/lerobot/<your-repo-id>/` as video + parquet,
which is what `lerobot-train` consumes (Guide 3).

## Version notes for this install

- CLI types are `so101_follower` / `so101_leader` as the guide says.
- If you import classes in your own code, the 0.4.4 module paths are
  `lerobot.robots.so_follower` and `lerobot.teleoperators.so_leader` (unified
  SO-100/SO-101 modules), not `lerobot.robots.so101_follower`.
- Flags shift slightly between lerobot releases — `<command> --help` is the truth.
- The `lerobot-*` commands live in `~/lerobot-env/bin` and are on PATH once the
  venv is activated; without activating, call them by full path.

## What is not done yet

Steps 3–7 need hardware on the desk. As of this setup nothing is plugged in:
no `/dev/tty.usbmodem*`, no calibration profiles, and the terminal has no camera
permission. Everything that can be done without an arm is done.
