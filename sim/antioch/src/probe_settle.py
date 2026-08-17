"""Does the arm hold a pose, do gains apply, and does the gripper collide at all?

    antioch run --no-stream src/probe_settle.py
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
DOWN_PICK = (1.2785, -1.2550, 0.9550, 0.0)
PICK_ROT = 0.45


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    from trapezoid_block import GRASP_WIDTH, HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    block = add_trapezoid_block(world, (0.35, -0.14, HEIGHT / 2.0))
    world.reset()
    block.bind()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    grip = RigidPrim("/World/SO101/gripper")
    controller = arm.get_articulation_controller()

    TCP_LOCAL = np.array([-0.0438 + GRASP_WIDTH / 2.0, 0.0, -0.085])

    def tcp_world():
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return pos + R @ TCP_LOCAL

    q = np.array([PICK_ROT, *DOWN_PICK, 0.85], dtype=np.float32)

    def trial(label, kp=None):
        if kp is not None:
            kps = np.array([kp] * 5 + [kp * 0.2], dtype=np.float32)
            controller.set_gains(kps=kps, kds=kps * 0.1)
        got = controller.get_gains()
        print(f"\n[{label}] gains readback kps={np.round(np.asarray(got[0]), 2).tolist()}")
        arm.set_joint_positions(q)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        for n in range(1, 1801):
            arm.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=False)
            if n in (60, 150, 300, 600, 1200, 1800):
                qa = np.asarray(arm.get_joint_positions(), dtype=float)
                print(
                    f"  step {n:5d}  tcp={np.round(tcp_world(), 4).tolist()}"
                    f"  q_err={np.round(qa - q, 3).tolist()}"
                )

    trial("asset defaults")
    trial("kp=200", 200.0)
    trial("kp=2000", 2000.0)

    # --- is there any gripper/block collision at all? ------------------------
    # Park the closed gripper low, drop the block onto it from above. If the
    # block passes through, collision between the two is not happening.
    print("\n[collision] dropping the block onto the closed gripper")
    q_shut = np.array([PICK_ROT, *DOWN_PICK, -0.17], dtype=np.float32)
    arm.set_joint_positions(q_shut)
    for _ in range(600):
        arm.apply_action(ArticulationAction(joint_positions=q_shut))
        world.step(render=False)
    tcp = tcp_world()
    block._rigid.set_world_pose(position=np.array([tcp[0], tcp[1], tcp[2] + 0.12], dtype=float))
    block._rigid.set_linear_velocity(np.zeros(3, dtype=float))
    for n in range(1, 601):
        arm.apply_action(ArticulationAction(joint_positions=q_shut))
        world.step(render=False)
        if n in (60, 150, 300, 600):
            bp = np.asarray(block.get_world_pose()[0], dtype=float)
            print(f"  step {n:4d}  block={np.round(bp, 4).tolist()}  gripper_tcp={np.round(tcp_world(), 4).tolist()}")


if __name__ == "__main__":
    main()
