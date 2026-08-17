"""Validate a smooth Jacobian Cartesian servo on the Antioch SO-101.

    antioch run --no-stream src/probe_cartesian_servo.py
"""

from __future__ import annotations

import antioch


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    from trapezoid_block import GRASP_WIDTH

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    world.reset()
    single = SingleArticulation(prim_path="/World/SO101", name="so101")
    single.initialize()
    tensor_arm = Articulation("/World/SO101")
    grip = RigidPrim("/World/SO101/gripper")
    grip_index = list(tensor_arm._link_names).index("gripper")
    tcp_local = np.array([-0.0438 + GRASP_WIDTH / 2.0, 0.0, -0.085])

    def pose():
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        offset = rotation @ tcp_local
        return np.asarray(pos, dtype=float) + offset, offset

    def servo(label: str, target: np.ndarray, ticks: int = 240):
        for _ in range(ticks):
            point, offset = pose()
            raw = tensor_arm.get_jacobian_matrices().numpy()[0, grip_index, :, 6:12]
            linear = raw[:3]
            angular = raw[3:]
            tcp_jacobian = linear + np.cross(angular.T, offset).T
            active = tcp_jacobian[:, :4]
            error = target - point
            damping = 0.025
            dq = active.T @ np.linalg.solve(
                active @ active.T + damping**2 * np.eye(3), error
            )
            dq = np.clip(dq, -0.015, 0.015)
            q = np.asarray(single.get_joint_positions(), dtype=float)
            q[:4] += dq
            q[:4] = np.clip(q[:4], [-1.9, -1.7, -1.65, -1.6], [1.9, 1.7, 1.65, 1.6])
            q[4] = np.pi / 2.0
            q[5] = 0.85
            single.apply_action(ArticulationAction(joint_positions=q.astype(np.float32)))
            world.step(render=False)
        reached, _ = pose()
        actual = np.asarray(single.get_joint_positions(), dtype=float)
        print(
            f"{label:10s} target={np.round(target, 4).tolist()} "
            f"reached={np.round(reached, 4).tolist()} "
            f"error_mm={np.linalg.norm(target - reached) * 1000:.2f} "
            f"q={np.round(actual, 5).tolist()}"
        )

    initial = np.array([0.34, 1.46, -0.60, -0.51, np.pi / 2.0, 0.85], dtype=np.float32)
    single.set_joint_positions(initial)
    single.apply_action(ArticulationAction(joint_positions=initial))
    for _ in range(120):
        world.step(render=False)

    servo("pick", np.array([0.3464, -0.1361, 0.022]))
    servo("pick_lift", np.array([0.3464, -0.1361, 0.110]))
    servo("place_lift", np.array([0.3395, 0.1744, 0.110]), ticks=360)
    servo("place", np.array([0.3395, 0.1744, 0.035]))


if __name__ == "__main__":
    main()
