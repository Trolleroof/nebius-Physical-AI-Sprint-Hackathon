"""Find the wrist roll and depth where the pads straddle the block.

    antioch run --no-stream src/probe_roll.py

The closing direction is measured, not assumed: the moving jaw's box centre at
open vs shut gives the closing axis, and its shut position gives the point the
pad actually converges on.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
PICK_ROT = 0.45
JAW_OPEN, JAW_SHUT = 0.85, -0.17


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom, UsdPhysics

    from trapezoid_block import HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    add_trapezoid_block(world, (0.35, -0.14, HEIGHT))
    world.reset()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    def box(root_path):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for prim in Usd.PrimRange(get_prim_at_path(root_path)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lo = np.minimum(lo, np.array(rng.GetMin()))
            hi = np.maximum(hi, np.array(rng.GetMax()))
        return lo, hi

    def settle(q):
        q = np.asarray(q, dtype=np.float32)
        arm.set_joint_positions(q)
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        for _ in range(200):
            arm.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=False)
        cache.Clear()

    # (pitch, elbow, wrist_pitch) candidates around the known-reachable depth
    poses = [
        ("head_down_pick", (1.2785, -1.2550, 0.9550)),
        ("deeper",         (1.3400, -1.3000, 1.0000)),
        ("shallower",      (1.2000, -1.2000, 0.9000)),
    ]

    for roll in (0.0, np.pi / 2.0):
        for name, (pitch, elbow, wpitch) in poses:
            settle([PICK_ROT, pitch, elbow, wpitch, roll, JAW_OPEN])
            jaw_open_lo, jaw_open_hi = box("/World/SO101/jaw")
            fix_lo, fix_hi = box("/World/SO101/gripper")
            settle([PICK_ROT, pitch, elbow, wpitch, roll, JAW_SHUT])
            jaw_shut_lo, jaw_shut_hi = box("/World/SO101/jaw")

            open_c = (jaw_open_lo + jaw_open_hi) / 2.0
            shut_c = (jaw_shut_lo + jaw_shut_hi) / 2.0
            axis = shut_c - open_c
            print(
                f"\nroll={roll:.3f} {name}"
                f"\n  jaw_open  lo={np.round(jaw_open_lo,4).tolist()} hi={np.round(jaw_open_hi,4).tolist()}"
                f"\n  jaw_shut  lo={np.round(jaw_shut_lo,4).tolist()} hi={np.round(jaw_shut_hi,4).tolist()}"
                f"\n  fixed     lo={np.round(fix_lo,4).tolist()} hi={np.round(fix_hi,4).tolist()}"
                f"\n  closing axis (shut-open centre) = {np.round(axis,4).tolist()}"
                f"  |horizontal|={np.linalg.norm(axis[:2]):.4f} vertical={axis[2]:+.4f}"
                f"\n  jaw lowest point: open={jaw_open_lo[2]:.4f} shut={jaw_shut_lo[2]:.4f}"
            )


if __name__ == "__main__":
    main()
