"""Solve compact SO-101 tabletop poses against measured Antioch FK.

    antioch run --no-stream src/probe_numeric_ik.py

This is a trajectory-development probe, not the ACT rollout.  It deliberately
locks wrist roll at the physical 90 degree neutral and solves only the TCP
position with a finite-difference Jacobian from the loaded Antioch asset.
"""

from __future__ import annotations

import antioch


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation

    from trapezoid_block import GRASP_WIDTH

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    world.reset()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    gripper = RigidPrim("/World/SO101/gripper")
    tcp_local = np.array([-0.0438 + GRASP_WIDTH / 2.0, 0.0, -0.085])

    def tcp(q: np.ndarray) -> np.ndarray:
        arm.set_joint_positions(q.astype(np.float32))
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        world.step(render=False)
        pos, quat = (a.numpy()[0] for a in gripper.get_world_poses())
        w, x, y, z = quat
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return np.asarray(pos, dtype=float) + rotation @ tcp_local

    def solve(label: str, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.copy()
        active = (0, 1, 2, 3)
        eps = 0.005
        for iteration in range(80):
            here = tcp(q)
            error = target - here
            if np.linalg.norm(error) < 0.002:
                break
            jacobian = np.empty((3, len(active)))
            for column, joint in enumerate(active):
                shifted = q.copy()
                shifted[joint] += eps
                jacobian[:, column] = (tcp(shifted) - here) / eps
            damping = 0.008
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
            delta = np.clip(delta, -0.06, 0.06)
            # Backtrack against measured FK; large Newton steps readily jump
            # between the SO-101's folded and extended branches.
            best_q, best_error = q, np.linalg.norm(error)
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = q.copy()
                candidate[list(active)] += scale * delta
                candidate[:4] = np.clip(candidate[:4], -2.6, 2.6)
                candidate_error = np.linalg.norm(target - tcp(candidate))
                if candidate_error < best_error:
                    best_q, best_error = candidate, candidate_error
                    break
            q = best_q.copy()
            q[:4] = np.clip(q[:4], -2.6, 2.6)
        reached = tcp(q)
        print(
            f"{label:10s} target={np.round(target, 4).tolist()} "
            f"reached={np.round(reached, 4).tolist()} "
            f"error_mm={np.linalg.norm(target - reached) * 1000:.2f} "
            f"q={np.round(q, 5).tolist()}"
        )
        return q

    wrist_roll = np.pi / 2.0
    open_jaw = 0.85
    seed = np.array([0.0, 1.4, -0.8, -0.6, wrist_roll, open_jaw])
    block = np.array([0.3464, -0.1361, 0.018])
    tray = np.array([0.3395, 0.1744, 0.065])

    pick = solve("pick", block, seed)
    pick_lift = solve("pick_lift", block + np.array([0.0, 0.0, 0.10]), pick)
    place_lift = solve("place_lift", tray + np.array([0.0, 0.0, 0.10]), pick_lift)
    solve("place", tray, place_lift)


if __name__ == "__main__":
    main()
