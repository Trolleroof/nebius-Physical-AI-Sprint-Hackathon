"""Where are the pads, and where is the block relative to its own origin?

    antioch run --no-stream src/probe_grasp_geometry.py

Everything upstream is guessing about a ~15 mm offset. This prints world-space
collision boxes for the two pads and for the block so the grasp centre can be
computed instead of assumed.
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
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom, UsdPhysics

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

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    def boxes(root_path):
        out = []
        root = get_prim_at_path(root_path)
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            out.append((str(prim.GetPath()), np.array(rng.GetMin()), np.array(rng.GetMax())))
        return out

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

    # --- block: origin vs geometry, at rest on the table ---------------------
    for _ in range(400):
        world.step(render=False)
    cache.Clear()
    origin = np.asarray(block.get_world_pose()[0], dtype=float)
    print(f"block origin from get_world_pose = {np.round(origin, 4).tolist()}")
    for path, lo, hi in boxes(block.prim_path):
        print(f"  block box {path}\n    lo={np.round(lo,4).tolist()} hi={np.round(hi,4).tolist()}"
              f"\n    centre={np.round((lo+hi)/2,4).tolist()} size={np.round(hi-lo,4).tolist()}")
        print(f"    origin - box_centre = {np.round(origin - (lo+hi)/2, 4).tolist()}")

    # --- pads: where is the opening, in world, at the pick pose --------------
    for jaw in (0.85, 0.45, 0.34, -0.17):
        q = np.array([PICK_ROT, *DOWN_PICK, jaw], dtype=np.float32)
        arm.set_joint_positions(q)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        for _ in range(200):
            arm.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=False)
        cache.Clear()
        print(f"\njaw={jaw:+.2f}  tcp_formula={np.round(tcp_world(), 4).tolist()}")
        for root in ("/World/SO101/gripper", "/World/SO101/jaw"):
            for path, lo, hi in boxes(root):
                print(f"  {path.split('/')[-2]:32s} lo={np.round(lo,4).tolist()} hi={np.round(hi,4).tolist()}")


if __name__ == "__main__":
    main()
