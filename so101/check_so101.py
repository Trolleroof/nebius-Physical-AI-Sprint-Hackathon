#!/usr/bin/env python
"""Passive SO-101 joint reader. Reads for 15 seconds and commands no motion.

Use it to tell the leader and follower apart (Guide 1, Step 3) and to sanity
check wiring before anything moves. Torque is left disabled the whole time, so
you can move joints by hand while it prints.

Sanity bar while you wiggle each joint by hand:
  * at rest the numbers are repeatable
  * none of them sit near +/- 180
  * moving one joint moves only its own number, in a consistent direction

    python check_so101.py --port /dev/tty.usbmodemXXXXXXXX
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Same servo table the SO101Follower builds, but with no calibration attached,
# so this runs before either arm has ever been calibrated.
MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.DEGREES),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodem58FA0836661")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    print(f"connecting to {args.port} ...")
    bus.connect()
    try:
        bus.disable_torque()
        print("torque disabled: move the joints by hand\n")
        print("  ".join(f"{name:>14}" for name in MOTORS))

        deadline = time.perf_counter() + args.seconds
        period = 1.0 / args.hz
        while time.perf_counter() < deadline:
            positions = bus.sync_read("Present_Position", normalize=False)
            print("  ".join(f"{positions[name]:>14.0f}" for name in MOTORS))
            time.sleep(period)
    finally:
        bus.disconnect()
        print("\ndisconnected")


if __name__ == "__main__":
    main()
