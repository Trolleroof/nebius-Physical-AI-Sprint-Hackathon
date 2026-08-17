"""Grid-search the grasp point at wrist_roll = pi/2 and report what holds.

    antioch run --no-stream src/probe_grasp_search.py

Block placement accounts for the asset's origin being at its BASE, 15.2 mm
below the geometric centre.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
PICK_ROT = 0.45
DOWN_PICK = (1.2785, -1.2550, 0.9550)
LIFT = (0.70, -0.85, 0.55)
ROLL = 1.5708
JAW_OPEN = 0.85
BASE_TO_CENTRE = 0.0152  # origin sits this far below the box centre


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

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

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    def box(root_path):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for prim in Usd.PrimRange(get_prim_at_path(root_path)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lo = np.minimum(lo, np.array(rng.GetMin()))
            hi = np.maximum(hi, np.array(rng.GetMax()))
        return lo, hi

    def key(pose, jaw):
        return np.array([PICK_ROT, pose[0], pose[1], pose[2], ROLL, jaw], dtype=np.float32)

    def hold(q, steps):
        for _ in range(steps):
            arm.apply_action(ArticulationAction(joint_positions=np.asarray(q, dtype=np.float32)))
            world.step(render=False)

    def park(y_far=0.6):
        block._rigid.set_world_pose(position=np.array([0.0, y_far, 0.0]))
        block._rigid.set_linear_velocity(np.zeros(3))
        block._rigid.set_angular_velocity(np.zeros(3))

    # where the fixed finger and the open jaw actually are, at this pose
    park()
    q_open = key(DOWN_PICK, JAW_OPEN)
    arm.set_joint_positions(q_open)
    hold(q_open, 250)
    cache.Clear()
    jl, jh = box("/World/SO101/jaw")
    fl, fh = box("/World/SO101/gripper/visuals/wrist_roll_follower_so101_v1")
    print(f"open jaw      lo={np.round(jl,4).tolist()} hi={np.round(jh,4).tolist()}")
    print(f"fixed finger  lo={np.round(fl,4).tolist()} hi={np.round(fh,4).tolist()}")

    results = []
    for bx in (0.339, 0.345, 0.351):
        for by in (-0.136, -0.140, -0.144):
            for jaw_grip in (0.34, 0.30, 0.26):
                park()
                arm.set_joint_positions(q_open)
                arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
                hold(q_open, 200)

                block._rigid.set_world_pose(
                    position=np.array([bx, by, HEIGHT / 2.0 - BASE_TO_CENTRE + 0.0002])
                )
                block._rigid.set_linear_velocity(np.zeros(3))
                block._rigid.set_angular_velocity(np.zeros(3))
                hold(q_open, 90)
                before = np.asarray(block.get_world_pose()[0], dtype=float)

                hold(key(DOWN_PICK, jaw_grip), 200)
                jaw_q = float(np.asarray(arm.get_joint_positions())[5])
                hold(key(LIFT, jaw_grip), 260)
                after = np.asarray(block.get_world_pose()[0], dtype=float)

                held = after[2] > 0.05
                clean = abs(before[2]) < 0.003  # block actually sat on the table
                results.append((held, clean, bx, by, jaw_grip, after[2]))
                print(
                    f"x={bx:.3f} y={by:+.3f} jaw={jaw_grip:.2f}"
                    f"  before_z={before[2]:.4f} clean={clean}  stall={jaw_q - jaw_grip:+.3f}"
                    f"  lifted_z={after[2]:+.4f}  HELD={held}"
                )

    print("\nheld+clean:", [r[2:] for r in results if r[0] and r[1]])
    print("dirty starts:", sum(1 for r in results if not r[1]), "of", len(results))


if __name__ == "__main__":
    main()
