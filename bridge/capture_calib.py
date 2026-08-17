#!/usr/bin/env python3
"""Collect the pixel <-> table correspondences the homography needs.

The workflow, once hardware is on the desk:

  1. Torque comes off, so you move the arm by hand.
  2. Touch the gripper tip to a spot on the table.
  3. Click that same spot in the camera window.
  4. Press SPACE. It reads the joints, runs FK for the table XY, and pairs that
     with the pixel you clicked.
  5. Repeat 8-12 times, spread across the whole workspace, then press Q.

Output feeds straight into pixel_to_joints.py calibrate.

    ./capture_calib.py verify  --port $FOLLOWER --id my_follower_arm
    ./capture_calib.py capture --port $FOLLOWER --id my_follower_arm --camera 0
    ./capture_calib.py capture --simulate --camera 0      # no arm needed

RUN VERIFY FIRST. lerobot's calibrated degrees and the URDF's joint zeros are
two separate conventions, and nothing guarantees they agree on sign or offset.
verify prints joints and the FK position live so you can confirm that moving a
joint moves the predicted position the way it should. If it does not, the
correspondences you collect will be quietly wrong and the homography will fit
beautifully to garbage.
"""

import argparse
import json
import sys
import time

import numpy as np

from pixel_to_joints import ARM_JOINTS, ArmSolver


def connect_arm(port: str, arm_id: str):
    """Connect and drop torque so the arm can be posed by hand.

    Torque is safe to cut here specifically because the arm has never been given
    a commanded pose to hold -- the house rule about never cutting torque under
    load is about a loaded arm, which this is not.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=arm_id, use_degrees=True))
    robot.connect()
    robot.bus.disable_torque()
    print("connected, torque disabled -- move the arm by hand", file=sys.stderr)
    return robot


def read_joints(robot) -> np.ndarray:
    """Joint positions in degrees, ordered to match the URDF chain."""
    obs = robot.get_observation()
    return np.array([obs[f"{name}.pos"] for name in ARM_JOINTS], dtype=float)


def cmd_verify(args) -> None:
    arm = ArmSolver(args.urdf)
    robot = None if args.simulate else connect_arm(args.port, args.id)
    print(f"{'joints (deg)':<44}  {'FK xyz (m)':<26}")
    try:
        while True:
            if robot is None:
                phase = time.time() % 6.0
                joints = np.array([15 * np.sin(phase), -45, 60, -20, 0])
            else:
                joints = read_joints(robot)
            xyz = arm.fk(joints)[:3, 3]
            print(f"  {str(np.round(joints, 1)):<42}  {np.round(xyz, 4)}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if robot is not None:
            robot.disconnect()


def cmd_capture(args) -> None:
    import cv2

    arm = ArmSolver(args.urdf)
    robot = None if args.simulate else connect_arm(args.port, args.id)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"camera {args.camera} would not open. On macOS grant your terminal\n"
            "camera access: System Settings > Privacy & Security > Camera,\n"
            "then restart the terminal."
        )

    clicked = {"point": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["point"] = (float(x), float(y))

    window = "calibration -- click gripper tip, SPACE to record, Q to finish"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    pairs = []
    print("click the gripper tip, then press SPACE. Q when done.", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("dropped frame", file=sys.stderr)
                continue

            display = frame.copy()
            for i, pair in enumerate(pairs):
                px, py = int(pair["pixel"][0]), int(pair["pixel"][1])
                cv2.circle(display, (px, py), 7, (0, 200, 0), 2)
                cv2.putText(display, str(i + 1), (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
            if clicked["point"]:
                px, py = int(clicked["point"][0]), int(clicked["point"][1])
                cv2.drawMarker(display, (px, py), (0, 165, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(display, f"captured: {len(pairs)}  (need 4+, want 8-12)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if clicked["point"] is None:
                    print("click the gripper tip first", file=sys.stderr)
                    continue
                if robot is None:
                    # Spread simulated points so the fit is not degenerate.
                    joints = np.array([-30 + 12 * len(pairs), -45, 60, -20, 0], dtype=float)
                else:
                    joints = read_joints(robot)
                xyz = arm.fk(joints)[:3, 3]
                pairs.append({
                    "pixel": list(clicked["point"]),
                    "table": [float(xyz[0]), float(xyz[1])],
                    "z": float(xyz[2]),
                    "joints_deg": [float(v) for v in joints],
                })
                print(f"  [{len(pairs)}] pixel={clicked['point']} "
                      f"table=({xyz[0]:+.4f},{xyz[1]:+.4f}) z={xyz[2]:.4f}", file=sys.stderr)
                clicked["point"] = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot is not None:
            robot.disconnect()

    if len(pairs) < 4:
        sys.exit(f"only {len(pairs)} points -- need at least 4 for a homography")

    heights = [p["z"] for p in pairs]
    spread = (max(heights) - min(heights)) * 1000
    if spread > 15:
        print(f"WARNING: touch heights vary by {spread:.0f} mm. A homography assumes "
              "every point is on one plane; keep the tip on the table.", file=sys.stderr)

    with open(args.out, "w") as handle:
        json.dump(pairs, handle, indent=2)
    print(f"\nwrote {len(pairs)} correspondences to {args.out}")
    print(f"next:  ./pixel_to_joints.py calibrate --points {args.out} --out table_calib.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("verify", "capture"):
        p = sub.add_parser(name)
        p.add_argument("--port", help="e.g. /dev/tty.usbmodemXXXX")
        p.add_argument("--id", default="my_follower_arm", help="lerobot calibration id")
        p.add_argument("--simulate", action="store_true", help="no arm; fake joint readings")
        if name == "capture":
            p.add_argument("--camera", type=int, default=0)
            p.add_argument("--out", default="calib.json")
        p.set_defaults(func=cmd_verify if name == "verify" else cmd_capture)

    args = parser.parse_args()
    if args.urdf is None:
        from pixel_to_joints import DEFAULT_URDF
        args.urdf = DEFAULT_URDF
    if not args.simulate and not args.port:
        sys.exit("--port is required unless you pass --simulate")
    args.func(args)


if __name__ == "__main__":
    main()
