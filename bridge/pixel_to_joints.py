#!/usr/bin/env python3
"""The bridge: ER 2 pixel coordinates -> SO-101 joint targets.

Two stages, deliberately kept separate so each can be checked on its own.

  1. Pixel -> table.  Objects sit on a flat table, so a planar homography maps
     image pixels to table coordinates in the arm's base frame. Four or more
     correspondences fit it; more is better. This is the part that needs the
     real camera bolted in its real position.

  2. Table -> joints.  Inverse kinematics against the SO-101 URDF.

Two things about lerobot's RobotKinematics that are not obvious and cost real
time if you find them the hard way:

  * forward_kinematics and inverse_kinematics both speak DEGREES. They convert
    internally, so passing radians silently gives you near-zero angles and every
    target lands in the same place.
  * inverse_kinematics runs ONE solver step, not a converged solve. placo's
    solver is differential, so a single call barely moves off the seed. You have
    to iterate it, feeding each result back as the next seed. solve_ik does that.

    ./pixel_to_joints.py selftest
    ./pixel_to_joints.py calibrate --points calib.json --out table_calib.json
    ./pixel_to_joints.py solve --calib table_calib.json --pixel 130 350 --z 0.03
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "..", "so101", "urdf", "so101_new_calib.urdf")

# The five arm joints. The gripper is commanded separately -- it is not part of
# the kinematic chain to the tool frame.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class ArmSolver:
    """FK/IK for the SO-101, with the iteration lerobot's wrapper leaves to you."""

    def __init__(self, urdf_path: str = DEFAULT_URDF, frame: str = "gripper_frame_link"):
        from lerobot.model.kinematics import RobotKinematics

        # placo only resolves the URDF's relative mesh paths when the URDF path
        # itself is absolute. A relative path raises "Mesh ... could not be found"
        # even when the meshes are sitting right there.
        self.kin = RobotKinematics(
            os.path.abspath(urdf_path), target_frame_name=frame, joint_names=ARM_JOINTS
        )

    def fk(self, joints_deg) -> np.ndarray:
        return self.kin.forward_kinematics(np.asarray(joints_deg, dtype=float))

    def solve_ik(self, target_xyz, seed_deg=None, iters: int = 60,
                 tol_mm: float = 0.5, orientation_weight: float = 0.0):
        """Position-only IK by default: we care where the gripper is, and letting
        the solver choose the wrist orientation keeps far more targets reachable.

        Returns (joints_deg, error_mm, iterations).
        """
        target = np.eye(4)
        target[:3, 3] = np.asarray(target_xyz, dtype=float)

        q = np.zeros(len(ARM_JOINTS)) if seed_deg is None else np.array(seed_deg, float)
        err = float("inf")
        for step in range(iters):
            q = self.kin.inverse_kinematics(
                q, target, position_weight=1.0, orientation_weight=orientation_weight
            )
            err = float(np.linalg.norm(self.fk(q)[:3, 3] - target[:3, 3])) * 1000
            if err < tol_mm:
                return q, err, step + 1
        return q, err, iters


class TableCalibration:
    """Planar homography: image pixels <-> table coordinates in the base frame.

    Valid only for points ON the table plane. Anything lifted off it needs a
    real camera model, not a homography.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)

    @classmethod
    def fit(cls, pixels, table_xy):
        import cv2

        pixels = np.asarray(pixels, dtype=np.float32)
        table_xy = np.asarray(table_xy, dtype=np.float32)
        if len(pixels) < 4:
            sys.exit(f"need at least 4 correspondences, got {len(pixels)}")
        matrix, mask = cv2.findHomography(pixels, table_xy, method=cv2.RANSAC)
        if matrix is None:
            sys.exit("homography fit failed -- check that the points are not collinear")
        inliers = int(mask.sum()) if mask is not None else len(pixels)
        print(f"fitted on {inliers}/{len(pixels)} inlier correspondences", file=sys.stderr)
        return cls(matrix)

    def pixel_to_table(self, px: float, py: float):
        point = np.array([px, py, 1.0])
        mapped = self.matrix @ point
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def residuals_mm(self, pixels, table_xy):
        out = []
        for (px, py), (tx, ty) in zip(pixels, table_xy):
            mx, my = self.pixel_to_table(px, py)
            out.append(float(np.hypot(mx - tx, my - ty)) * 1000)
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump({"homography": self.matrix.tolist()}, handle, indent=2)
        print(f"wrote {path}", file=sys.stderr)

    @classmethod
    def load(cls, path: str):
        with open(path) as handle:
            return cls(json.load(handle)["homography"])


def cmd_selftest(args) -> None:
    """Prove the whole chain without hardware.

    Build a synthetic camera looking at the table, use the arm's own FK to
    generate ground-truth table points, project them to pixels, fit the
    homography on those, then run the full pipeline backwards and check we
    recover the position we started from.
    """
    arm = ArmSolver(args.urdf)

    # Ground truth: poses the arm can actually reach, on a plane.
    configs = [[-30, -40, 60, -20, 0], [-15, -55, 75, -20, 0], [0, -40, 60, -20, 0],
               [15, -55, 75, -20, 0], [30, -40, 60, -20, 0], [0, -60, 80, -20, 0],
               [-20, -50, 70, -20, 0], [20, -50, 70, -20, 0]]
    table_pts, heights = [], []
    for cfg in configs:
        xyz = arm.fk(cfg)[:3, 3]
        table_pts.append([xyz[0], xyz[1]])
        heights.append(xyz[2])
    table_pts = np.array(table_pts)
    plane_z = float(np.mean(heights))
    print(f"generated {len(table_pts)} reachable points, mean z = {plane_z*1000:.1f} mm")
    print(f"  x range {table_pts[:,0].min():.3f} .. {table_pts[:,0].max():.3f} m")
    print(f"  y range {table_pts[:,1].min():.3f} .. {table_pts[:,1].max():.3f} m")

    # Synthetic camera: a linear map from table metres to a 640x480 image, which
    # is exactly the planar relationship a real overhead camera has.
    def project(x, y):
        return (320 + (y - table_pts[:, 1].mean()) * 1400,
                240 - (x - table_pts[:, 0].mean()) * 1400)

    pixels = np.array([project(x, y) for x, y in table_pts])

    calib = TableCalibration.fit(pixels, table_pts)
    residuals = calib.residuals_mm(pixels, table_pts)
    print(f"homography residuals: max {max(residuals):.4f} mm, mean {np.mean(residuals):.4f} mm")

    print("\nfull chain, pixel -> table -> joints -> FK:")
    worst = 0.0
    for (px, py), truth in zip(pixels, table_pts):
        tx, ty = calib.pixel_to_table(px, py)
        q, err, iters = arm.solve_ik([tx, ty, plane_z])
        achieved = arm.fk(q)[:3, 3]
        total = float(np.linalg.norm(achieved[:2] - truth)) * 1000
        worst = max(worst, total)
        print(f"  px=({px:6.1f},{py:6.1f}) -> table=({tx:+.4f},{ty:+.4f}) "
              f"-> q={np.round(q,1)} ik_err={err:.2f}mm total={total:.2f}mm [{iters} it]")
    print(f"\nworst end-to-end error: {worst:.3f} mm")
    print("PASS" if worst < 2.0 else "FAIL -- investigate before trusting this on hardware")


def cmd_calibrate(args) -> None:
    """Fit from a JSON file of correspondences:

    [{"pixel": [x, y], "table": [X_metres, Y_metres]}, ...]

    Collect these by teleoperating the gripper to a spot, reading the joints,
    running FK to get the table XY, and noting the pixel the object appears at.
    """
    with open(args.points) as handle:
        pairs = json.load(handle)
    pixels = [p["pixel"] for p in pairs]
    table = [p["table"] for p in pairs]
    calib = TableCalibration.fit(pixels, table)
    residuals = calib.residuals_mm(pixels, table)
    print(f"residuals (mm): max {max(residuals):.2f}, mean {np.mean(residuals):.2f}")
    if max(residuals) > 10:
        print("WARNING: residuals over 10 mm. Points may be mismeasured, nearly "
              "collinear, or not all on the table plane.", file=sys.stderr)
    calib.save(args.out)


def cmd_solve(args) -> None:
    calib = TableCalibration.load(args.calib)
    arm = ArmSolver(args.urdf)
    tx, ty = calib.pixel_to_table(*args.pixel)
    q, err, iters = arm.solve_ik([tx, ty, args.z])
    print(f"pixel {tuple(args.pixel)} -> table ({tx:+.4f}, {ty:+.4f}) m")
    print("joints (deg):")
    for name, value in zip(ARM_JOINTS, q):
        print(f"  {name:<15} {value:+8.2f}")
    print(f"ik error: {err:.3f} mm in {iters} iterations")
    if err > 5:
        print("WARNING: IK did not converge -- target is likely out of reach.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(func=cmd_selftest)

    p = sub.add_parser("calibrate")
    p.add_argument("--points", required=True, help="JSON of pixel/table correspondences")
    p.add_argument("--out", default="table_calib.json")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("solve")
    p.add_argument("--calib", required=True)
    p.add_argument("--pixel", nargs=2, type=float, required=True, metavar=("X", "Y"))
    p.add_argument("--z", type=float, default=0.03, help="height above base frame, metres")
    p.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
