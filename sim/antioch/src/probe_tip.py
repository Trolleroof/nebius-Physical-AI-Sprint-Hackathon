"""Exact fingertip geometry in the gripper link's own frame."""

from __future__ import annotations

import antioch


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    world.reset()

    single = SingleArticulation(prim_path="/World/SO101", name="so101")
    single.initialize()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    gripper_prim = get_prim_at_path("/World/SO101/gripper")

    def settle(q, steps=100):
        q = np.asarray(q, dtype=np.float32)
        single.set_joint_positions(q)
        single.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps):
            world.step(render=False)

    for jaw in (-0.17, 0.3, 0.8, 1.4):
        settle([0, 1.0, -1.0, 0.7, 0, jaw])
        cache.Clear()
        print(f"\nJaw={jaw:+.2f}  (bounds relative to the gripper link frame, metres)")
        for path in ("/World/SO101/gripper", "/World/SO101/jaw"):
            r = cache.ComputeRelativeBound(get_prim_at_path(path), gripper_prim).ComputeAlignedRange()
            if r.IsEmpty():
                print(f"  {path:28s} EMPTY")
                continue
            lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
            print(f"  {path.split('/')[-1]:10s} lo={np.round(lo,4).tolist()}  hi={np.round(hi,4).tolist()}")


if __name__ == "__main__":
    main()
