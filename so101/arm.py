#!/usr/bin/env python3
"""Programmatic control of the SO-101 follower arm.

Importable as a library and usable from the shell. Every motion is interpolated
rather than commanded as a jump, because a Goal_Position write far from the
present position makes the arm snap there as fast as the servo can manage.

Safety, in the order it matters:

  * --dry-run exercises the entire path with sends suppressed. Run it whenever
    the code or the target changed. It is the cheapest way to find out that a
    pose is wrong.
  * max_relative_target clamps every command to a bounded step from the present
    position, so a bad number degrades into a small move instead of a lunge.
  * The rest pose is read at connect time, and disconnect ramps back to it
    before releasing torque. Cutting torque while the arm holds a raised pose
    drops the arm.
  * Keep a hand near the power switch the first time any new motion runs.

    ./arm.py state
    ./arm.py home --dry-run
    ./arm.py move --joint shoulder_pan --to 20 --dry-run
    ./arm.py pose --joints shoulder_pan=10,elbow_flex=-20 --seconds 3
    ./arm.py gripper --to 80

Joint angles are degrees. The gripper is 0-100 percent (0 closed, 100 open),
which is how lerobot normalises that motor regardless of use_degrees.
"""

import argparse
import sys
import time

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"
ALL_JOINTS = ARM_JOINTS + [GRIPPER]

# Per-command ceiling on how far any single joint may be asked to move from
# where it currently is. Interpolation keeps steps far below this; it exists to
# catch a bad target, not to shape normal motion. It must stay above the
# interactive lead limit in keyboard_control, or it silently throttles driving
# instead of acting as the backstop it is meant to be.
MAX_STEP_DEG = 25.0
DEFAULT_HZ = 50


class Arm:
    def __init__(self, port: str, arm_id: str = "my_follower_arm", dry_run: bool = False):
        self.port = port
        self.arm_id = arm_id
        self.dry_run = dry_run
        self.robot = None
        self.rest_pose = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> "Arm":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(
            port=self.port,
            id=self.arm_id,
            use_degrees=True,
            max_relative_target=MAX_STEP_DEG,
        ))
        self.robot.connect()
        # Whatever pose the arm is in now is a pose it can hold safely, which
        # makes it the right place to return to before torque is released.
        self.rest_pose = self.read()
        mode = " (DRY RUN -- no motion will be sent)" if self.dry_run else ""
        print(f"connected to {self.arm_id} on {self.port}{mode}", file=sys.stderr)
        print(f"  per-command clamp: {self.robot.config.max_relative_target} deg",
              file=sys.stderr)
        return self

    @staticmethod
    def quiet_clamp_warnings() -> None:
        """Stop lerobot logging a multi-line warning on every clamped command.

        The clamp firing is normal during interactive driving -- it is the
        backstop doing its job. Logged at 40 Hz it scrolls the status line off
        the screen and hides everything that matters.
        """
        import logging

        class _DropClampWarning(logging.Filter):
            def filter(self, record):
                return "had to be clamped" not in record.getMessage()

        logging.getLogger().addFilter(_DropClampWarning())
        for handler in logging.getLogger().handlers:
            handler.addFilter(_DropClampWarning())

    def disconnect(self, ramp_home: bool = True) -> None:
        if self.robot is None:
            return
        try:
            if ramp_home and self.rest_pose and not self.dry_run:
                print("ramping back to rest pose before releasing torque", file=sys.stderr)
                self.move_to(self.rest_pose, seconds=2.0)
        finally:
            self.robot.disconnect()
            self.robot = None
            print("disconnected", file=sys.stderr)

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    # -- state -------------------------------------------------------------

    def read(self) -> dict:
        """Current position of every joint. Degrees, except gripper (percent)."""
        obs = self.robot.get_observation()
        return {name: float(obs[f"{name}.pos"]) for name in ALL_JOINTS}

    # -- motion ------------------------------------------------------------

    def move_to(self, targets: dict, seconds: float = 2.0, hz: int = DEFAULT_HZ) -> dict:
        """Interpolate from the present pose to `targets` over `seconds`.

        `targets` may name any subset of joints; the rest are held where they
        are. Returns the pose actually reached (or the intended one in dry run).
        """
        unknown = set(targets) - set(ALL_JOINTS)
        if unknown:
            raise ValueError(f"unknown joints: {sorted(unknown)}")

        start = self.read()
        goal = {**start, **{k: float(v) for k, v in targets.items()}}

        moving = {k: (start[k], goal[k]) for k in goal if abs(goal[k] - start[k]) > 0.05}
        if not moving:
            print("already there", file=sys.stderr)
            return start

        print("  " + "  ".join(f"{k}: {a:+.1f} -> {b:+.1f}" for k, (a, b) in moving.items()),
              file=sys.stderr)

        steps = max(2, int(seconds * hz))
        period = 1.0 / hz
        for step in range(1, steps + 1):
            # Smoothstep: zero velocity at both ends, so the arm eases in and
            # out instead of jerking at the start and slamming at the target.
            t = step / steps
            blend = t * t * (3 - 2 * t)
            frame = {f"{k}.pos": start[k] + (goal[k] - start[k]) * blend for k in goal}
            if not self.dry_run:
                self.robot.send_action(frame)
            time.sleep(period)

        return goal if self.dry_run else self.read()

    def send_frame(self, pose: dict) -> None:
        """Send one command frame with no interpolation.

        For interactive control, where the caller is already producing a smooth
        stream of small targets. Do not use this to jump to a distant pose --
        max_relative_target will clamp it, but the arm will still lurch toward
        it a step at a time on every call. move_to is the safe way to travel.
        """
        if not self.dry_run:
            self.robot.send_action({f"{k}.pos": v for k, v in pose.items()})

    def set_gripper(self, percent: float, seconds: float = 1.0) -> dict:
        return self.move_to({GRIPPER: max(0.0, min(100.0, percent))}, seconds=seconds)

    def home(self, seconds: float = 3.0) -> dict:
        """A neutral, low pose that is safe to sit in."""
        return self.move_to(
            {"shoulder_pan": 0.0, "shoulder_lift": -40.0, "elbow_flex": 60.0,
             "wrist_flex": -20.0, "wrist_roll": 0.0},
            seconds=seconds,
        )


# -- CLI -------------------------------------------------------------------

def resolve_port(given: str | None) -> str:
    import glob
    if given:
        return given
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not ports:
        sys.exit("no /dev/tty.usbmodem* port. Is the arm powered and plugged in? (./usbcheck.py)")
    return ports[0]


def print_state(pose: dict) -> None:
    for name in ARM_JOINTS:
        print(f"  {name:<15} {pose[name]:+8.2f} deg", flush=True)
    print(f"  {GRIPPER:<15} {pose[GRIPPER]:+8.2f} %", flush=True)


def main() -> None:
    # Shared flags live on a parent parser so they are accepted both before and
    # after the subcommand -- "arm.py home --dry-run" is what people actually type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", default=None, help="default: first usbmodem port")
    common.add_argument("--id", default="my_follower_arm", help="lerobot calibration id")
    common.add_argument("--dry-run", action="store_true", help="run everything, send nothing")
    common.add_argument("--seconds", type=float, default=2.5, help="duration of the move")

    parser = argparse.ArgumentParser(description=__doc__, parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state", parents=[common], help="print current joint positions")
    sub.add_parser("home", parents=[common], help="ramp to a neutral pose")

    p = sub.add_parser("move", parents=[common], help="move one joint")
    p.add_argument("--joint", required=True, choices=ALL_JOINTS)
    p.add_argument("--to", type=float, required=True)

    p = sub.add_parser("pose", parents=[common], help="move several joints at once")
    p.add_argument("--joints", required=True, help="e.g. shoulder_pan=10,elbow_flex=-20")

    p = sub.add_parser("gripper", parents=[common], help="open or close the gripper")
    p.add_argument("--to", type=float, required=True, help="0 closed .. 100 open")

    args = parser.parse_args()
    arm = Arm(resolve_port(args.port), args.id, dry_run=args.dry_run)

    with arm:
        if args.cmd == "state":
            print_state(arm.read())
            return  # nothing moved, so no ramp needed on the way out

        if args.cmd == "home":
            arm.home(seconds=args.seconds)
        elif args.cmd == "move":
            arm.move_to({args.joint: args.to}, seconds=args.seconds)
        elif args.cmd == "pose":
            targets = {}
            for pair in args.joints.split(","):
                name, _, value = pair.partition("=")
                targets[name.strip()] = float(value)
            arm.move_to(targets, seconds=args.seconds)
        elif args.cmd == "gripper":
            arm.set_gripper(args.to, seconds=args.seconds)

        print_state(arm.read())


if __name__ == "__main__":
    main()
