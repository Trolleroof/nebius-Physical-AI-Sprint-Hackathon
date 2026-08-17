"""List compact-pose gripper collision bounds for grasp tuning."""

import antioch


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom, UsdPhysics

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    world.reset()
    arm = SingleArticulation("/World/SO101", name="so101")
    arm.initialize()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    for jaw in (0.85, 0.34, -0.17):
        q = np.array([0, 1.40, -0.80, -0.60, np.pi / 2, jaw], dtype=np.float32)
        arm.set_joint_positions(q)
        arm.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(120):
            world.step(render=False)
        cache.Clear()
        print(f"jaw={jaw:+.2f}")
        for root_path in ("/World/SO101/gripper", "/World/SO101/jaw"):
            root = get_prim_at_path(root_path)
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if box.IsEmpty():
                    continue
                lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
                print(f"{prim.GetPath()} lo={np.round(lo,4).tolist()} hi={np.round(hi,4).tolist()}")


if __name__ == "__main__":
    main()
