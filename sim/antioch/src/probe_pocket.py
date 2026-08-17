"""Map the finger pocket empirically, then close on it.

    antioch run --no-stream src/probe_pocket.py

Free space is measured, not read off a bounding box: with the jaw open, a block
teleported into solid geometry gets pushed out, so "it stayed where I put it"
is the test for the pocket. Then close from inside the pocket and lift.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
PICK_ROT = 0.45
DOWN_PICK = (1.2785, -1.2550, 0.9550)
LIFT = (0.70, -0.85, 0.55)
ROLL = 1.5708
JAW_OPEN = 0.85
BASE_TO_CENTRE = 0.0152


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdPhysics, UsdShade

    from trapezoid_block import HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    block = add_trapezoid_block(world, (0.35, -0.14, HEIGHT))

    stage = get_current_stage()
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/grip")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.5)
    physics.CreateDynamicFrictionAttr(1.5)
    physics.CreateRestitutionAttr(0.0)
    for path in ("/World/SO101/gripper", "/World/SO101/jaw", block.prim_path):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    world.reset()
    block.bind()
    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()

    def key(pose, jaw):
        return np.array([PICK_ROT, pose[0], pose[1], pose[2], ROLL, jaw], dtype=np.float32)

    def hold(q, steps):
        for _ in range(steps):
            arm.apply_action(ArticulationAction(joint_positions=np.asarray(q, dtype=np.float32)))
            world.step(render=False)

    def place(x, y):
        block._rigid.set_world_pose(
            position=np.array([x, y, HEIGHT / 2.0 - BASE_TO_CENTRE + 0.0002])
        )
        block._rigid.set_linear_velocity(np.zeros(3))
        block._rigid.set_angular_velocity(np.zeros(3))

    q_open = key(DOWN_PICK, JAW_OPEN)
    X = 0.345

    print("--- free-space map (jaw open): does the block stay where it is put? ---")
    for y in np.arange(-0.185, -0.099, 0.005):
        place(0.0, 0.6)
        arm.set_joint_positions(q_open)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        hold(q_open, 180)
        place(X, float(y))
        hold(q_open, 120)
        p = np.asarray(block.get_world_pose()[0], dtype=float)
        drift = np.linalg.norm(p[:2] - np.array([X, y]))
        print(f"  y={y:+.3f}  settled={np.round(p,4).tolist()}  drift={drift*1000:6.1f} mm"
              f"  free={drift < 0.004 and abs(p[2]) < 0.003}")

    print("\n--- close and lift from inside the pocket ---")
    for y in np.arange(-0.175, -0.109, 0.005):
        for jaw_grip in (0.10, -0.05):
            place(0.0, 0.6)
            arm.set_joint_positions(q_open)
            arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
            hold(q_open, 180)
            place(X, float(y))
            hold(q_open, 120)
            before = np.asarray(block.get_world_pose()[0], dtype=float)
            if np.linalg.norm(before[:2] - np.array([X, y])) > 0.004:
                continue
            hold(key(DOWN_PICK, jaw_grip), 220)
            jaw_q = float(np.asarray(arm.get_joint_positions())[5])
            hold(key(LIFT, jaw_grip), 260)
            after = np.asarray(block.get_world_pose()[0], dtype=float)
            print(f"  y={y:+.3f} jaw={jaw_grip:+.2f}  jaw_settled={jaw_q:+.3f}"
                  f"  lifted_z={after[2]:+.4f}  HELD={after[2] > 0.05}")


if __name__ == "__main__":
    main()
