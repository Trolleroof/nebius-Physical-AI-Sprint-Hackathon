"""Measure compact, down-facing SO-101 poses without teleporting live PhysX.

    antioch run --no-stream src/probe_pose_grid.py
"""

from __future__ import annotations

import antioch


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    from trapezoid_block import GRASP_WIDTH

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    world.reset()
    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    grip = RigidPrim("/World/SO101/gripper")
    tcp_local = np.array([-0.0438 + GRASP_WIDTH / 2.0, 0.0, -0.085])

    def tcp_world() -> np.ndarray:
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return np.asarray(pos, dtype=float) + rotation @ tcp_local

    q_prev = np.array([0.0, 1.4, -0.8, -0.6, np.pi / 2.0, 0.85])
    rows = []
    for pitch in np.linspace(0.8, 1.7, 7):
        for elbow in np.linspace(-1.5, -0.3, 7):
            for wrist_pitch in np.linspace(-1.3, 0.3, 9):
                target = np.array([0.0, pitch, elbow, wrist_pitch, np.pi / 2.0, 0.85])
                for u in np.linspace(0.0, 1.0, 9)[1:]:
                    cmd = q_prev + u * (target - q_prev)
                    arm.apply_action(ArticulationAction(joint_positions=cmd.astype(np.float32)))
                    world.step(render=False)
                point = tcp_world()
                rows.append((target, point))
                q_prev = target

    for label, wanted in (("down", np.array([0.3722, 0.0, 0.020])),
                          ("lift", np.array([0.3722, 0.0, 0.110]))):
        ranked = sorted(rows, key=lambda row: float(np.linalg.norm(row[1] - wanted)))[:8]
        print(f"\n{label} candidates for {wanted.tolist()}")
        for q, point in ranked:
            print(
                f"  err_mm={np.linalg.norm(point - wanted) * 1000:6.2f} "
                f"tcp={np.round(point, 4).tolist()} q={np.round(q, 4).tolist()}"
            )


if __name__ == "__main__":
    main()
