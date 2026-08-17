#!/usr/bin/env python3
"""Passive SO-101 joint read — no motion commands (Hackathon Guide 1, Step 3).

Connects to one arm and prints joint positions for ~15 seconds. Wiggle each joint
by hand and confirm:
  - at rest, readings are repeatable
  - no value sits near +180 or -180
  - moving one joint changes only that joint's number, in a consistent direction

Torque is enabled on connect (normal LeRobot behavior); this script still sends
no goal positions.
"""

from __future__ import annotations

import argparse
import sys
import time

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _connect_follower(port: str, arm_id: str):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=arm_id, use_degrees=True))
    robot.connect(calibrate=False)
    return robot, robot.get_observation


def _connect_leader(port: str, arm_id: str):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    teleop = SO101Leader(SO101LeaderConfig(port=port, id=arm_id, use_degrees=True))
    teleop.connect(calibrate=False)
    return teleop, teleop.get_action


def _format_line(values: dict[str, float]) -> str:
    parts = []
    for joint in JOINTS:
        key = f"{joint}.pos"
        val = values.get(key)
        if val is None:
            parts.append(f"{joint:>13}=   n/a")
            continue
        flag = "!" if abs(val) > 170 else " "
        parts.append(f"{joint:>13}={val:7.2f}{flag}")
    return "  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodemXXXXXXXX")
    parser.add_argument(
        "--type",
        choices=("follower", "leader"),
        default="follower",
        help="Arm role (either type can identify a port)",
    )
    parser.add_argument("--id", default="probe_arm", help="LeRobot calibration id (probe only)")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to stream readings")
    parser.add_argument("--hz", type=float, default=5.0, help="Print rate")
    args = parser.parse_args()

    print(f"Connecting to {args.type} on {args.port} (no motion commands)...")
    print("Keep a hand near the power switch. Move joints gently while numbers print.\n")

    connect = _connect_follower if args.type == "follower" else _connect_leader
    device, read_fn = connect(args.port, args.id)

    period = 1.0 / max(args.hz, 0.1)
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            values = read_fn()
            print(_format_line(values), flush=True)
            sleep_for = period - (time.monotonic() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        device.disconnect()
        print("Disconnected.")

    print(
        "\nSanity: '!' marks values near ±180. One joint moving should change one column only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
