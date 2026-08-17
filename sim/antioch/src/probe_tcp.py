"""Derive the tool frame from the grasp point that actually holds.

    antioch run --no-stream src/probe_tcp.py

The measured-good grasp centre at the pick pose is inverted through the gripper
transform to give TCP_LOCAL, then re-projected at the place rotation and the
lift pose so the scenario can aim at an arbitrary block.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
DOWN_PICK = (1.2785, -1.2550, 0.9550)
LIFT = (0.70, -0.85, 0.55)
ROLL = 1.5708
JAW_OPEN = 0.85
PICK_ROT, PLACE_ROT = 0.45, -0.45

# Centre of the 3/3 region from probe_grasp_rate, block centre in world.
GOOD_GRASP = (0.348, -0.114, 0.01524)


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    world.reset()
    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    grip = RigidPrim("/World/SO101/gripper")

    def frame():
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return pos, R

    def settle(rot, pose, jaw=JAW_OPEN, steps=250):
        q = np.array([rot, pose[0], pose[1], pose[2], ROLL, jaw], dtype=np.float32)
        arm.set_joint_positions(q)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        for _ in range(steps):
            arm.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=False)

    settle(PICK_ROT, DOWN_PICK)
    pos, R = frame()
    tcp_local = R.T @ (np.array(GOOD_GRASP) - pos)
    print(f"gripper pos at pick = {np.round(pos, 4).tolist()}")
    print(f"TCP_LOCAL = {np.round(tcp_local, 5).tolist()}")

    for label, rot, pose in (
        ("pick ", PICK_ROT, DOWN_PICK),
        ("place", PLACE_ROT, DOWN_PICK),
        ("lift@pick", PICK_ROT, LIFT),
        ("lift@place", PLACE_ROT, LIFT),
    ):
        settle(rot, pose)
        p, Rm = frame()
        print(f"{label:11s} rot={rot:+.2f}  tcp_world={np.round(p + Rm @ tcp_local, 4).tolist()}")


if __name__ == "__main__":
    main()
