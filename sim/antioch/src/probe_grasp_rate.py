"""Success rate per candidate grasp cell -- repeatability, not a lucky trial.

    antioch run --no-stream src/probe_grasp_rate.py
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
REPEATS = 3


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

    def ramp(q_from, q_to, steps):
        """Smoothstep the command, the way the scenario keyframes do."""
        a = np.asarray(q_from, dtype=np.float64)
        b = np.asarray(q_to, dtype=np.float64)
        for t in range(1, steps + 1):
            u = t / steps
            cmd = a + (u * u * (3.0 - 2.0 * u)) * (b - a)
            arm.apply_action(ArticulationAction(joint_positions=cmd.astype(np.float32)))
            world.step(render=False)

    def place(x, y):
        block._rigid.set_world_pose(
            position=np.array([x, y, HEIGHT / 2.0 - BASE_TO_CENTRE + 0.0002])
        )
        block._rigid.set_linear_velocity(np.zeros(3))
        block._rigid.set_angular_velocity(np.zeros(3))

    q_open = key(DOWN_PICK, JAW_OPEN)

    def trial(x, y, jaw_grip):
        place(0.0, 0.6)
        arm.set_joint_positions(q_open)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        hold(q_open, 180)
        place(x, y)
        hold(q_open, 120)
        before = np.asarray(block.get_world_pose()[0], dtype=float)
        if np.linalg.norm(before[:2] - np.array([x, y])) > 0.004:
            return None
        ramp(key(DOWN_PICK, JAW_OPEN), key(DOWN_PICK, jaw_grip), 120)
        hold(key(DOWN_PICK, jaw_grip), 120)
        jaw_q = float(np.asarray(arm.get_joint_positions())[5])
        ramp(key(DOWN_PICK, jaw_grip), key(LIFT, jaw_grip), 300)
        hold(key(LIFT, jaw_grip), 180)
        after = np.asarray(block.get_world_pose()[0], dtype=float)
        return after[2] > 0.05, jaw_q

    best = []
    for x in (0.339, 0.345, 0.351):
        for y in (-0.118, -0.114, -0.110):
            for jaw_grip in (0.0, -0.10):
                outcomes, jaws = [], []
                for _ in range(REPEATS):
                    got = trial(x, y, jaw_grip)
                    if got is None:
                        outcomes.append(None)
                        continue
                    outcomes.append(got[0])
                    jaws.append(got[1])
                ok = sum(1 for o in outcomes if o)
                best.append((ok, x, y, jaw_grip))
                print(f"x={x:.3f} y={y:+.3f} jaw={jaw_grip:+.2f}  held {ok}/{REPEATS}"
                      f"  jaw_settled={[round(j,3) for j in jaws]}")

    best.sort(reverse=True)
    print("\nbest cells:", best[:5])


if __name__ == "__main__":
    main()
