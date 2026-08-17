"""Why the grasp does not hold: drive limits, jaw stall, and a lift sweep.

    antioch run --no-stream src/probe_grasp.py

Aim is already solved (0.8 mm). This isolates the close-and-hold step: place
the block exactly where the fingers are, close the jaw to a range of targets,
lift, and report whether the block came with it.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"

LIFT = (0.70, -0.85, 0.55, 0.0)
DOWN_PICK = (1.2785, -1.2550, 0.9550, 0.0)
PICK_ROT = 0.45
JAW_OPEN = 0.85


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdPhysics, UsdShade

    from trapezoid_block import GRASP_WIDTH, HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    block = add_trapezoid_block(world, (0.35, -0.14, HEIGHT / 2.0))

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
    grip = RigidPrim("/World/SO101/gripper")

    print("dof names:", list(arm.dof_names))

    # --- what the drives can actually do -------------------------------------
    for joint in stage.Traverse():
        for kind in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(joint, kind)
            if not drive:
                continue
            def val(attr):
                a = attr()
                return round(float(a.Get()), 4) if a and a.Get() is not None else None
            print(
                f"drive {joint.GetPath().name:16s} {kind:7s} "
                f"stiffness={val(drive.GetStiffnessAttr)} damping={val(drive.GetDampingAttr)} "
                f"maxForce={val(drive.GetMaxForceAttr)} target={val(drive.GetTargetPositionAttr)}"
            )

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

    def key(rot, pose, jaw):
        return np.array([rot, pose[0], pose[1], pose[2], pose[3], jaw], dtype=np.float32)

    def hold(q, steps):
        for _ in range(steps):
            arm.apply_action(ArticulationAction(joint_positions=np.asarray(q, dtype=np.float32)))
            world.step(render=False)

    controller = arm.get_articulation_controller()

    def set_gains(arm_kp, jaw_kp):
        kps = np.array([arm_kp] * 5 + [jaw_kp], dtype=np.float32)
        kds = kps * 0.05
        controller.set_gains(kps=kps, kds=kds)

    # --- sweep gains and the closing target ----------------------------------
    for arm_kp, jaw_kp in ((None, None), (200.0, 40.0), (800.0, 150.0)):
      if arm_kp is not None:
        set_gains(arm_kp, jaw_kp)
      print(f"--- arm_kp={arm_kp} jaw_kp={jaw_kp} ---")
      for jaw_grip in (0.45, 0.34, 0.20):
          # reset: arm to the open pick pose, block teleported into the fingers
          q_open = key(PICK_ROT, DOWN_PICK, JAW_OPEN)
          arm.set_joint_positions(q_open)
          arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
          hold(q_open, 150)

          centre = tcp_world()
          block._rigid.set_world_pose(
              position=np.array([centre[0], centre[1], HEIGHT / 2.0], dtype=float)
          )
          hold(q_open, 60)

          start = np.asarray(block.get_world_pose()[0], dtype=float)
          tcp_at_pick = tcp_world()

          hold(key(PICK_ROT, DOWN_PICK, jaw_grip), 180)
          q_closed = float(np.asarray(arm.get_joint_positions())[5])
          after_close = np.asarray(block.get_world_pose()[0], dtype=float)

          hold(key(PICK_ROT, LIFT, jaw_grip), 240)
          lifted = np.asarray(block.get_world_pose()[0], dtype=float)

          print(
              f"jaw_grip={jaw_grip:+.2f}  tcp={np.round(tcp_at_pick, 4).tolist()}"
              f"  jaw_settled={q_closed:+.3f} (stall={q_closed - jaw_grip:+.3f})"
              f"  block_after_close={np.round(after_close, 4).tolist()}"
              f"  block_lifted_z={lifted[2]:.4f}"
              f"  HELD={lifted[2] > 0.06}"
          )


if __name__ == "__main__":
    main()
