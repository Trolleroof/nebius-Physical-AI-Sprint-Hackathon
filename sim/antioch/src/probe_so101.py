"""Measure everything the pick-and-place needs, in Isaac, at runtime.

    antioch run --no-stream src/probe_so101.py

Nothing here is assumed from another simulator: the arm's base pose, the
gripper frame, where the fingertips actually are, and which way the jaw opens
are all measured off the live articulation.
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    world.reset()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    names = list(arm.dof_names)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    def settle(target, steps=90):
        target = np.asarray(target, dtype=np.float32)
        arm.set_joint_positions(target)
        arm.apply_action(ArticulationAction(joint_positions=target))
        for _ in range(steps):
            world.step(render=False)

    def bbox(path):
        r = cache.ComputeWorldBound(get_prim_at_path(path)).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        return lo, hi, (lo + hi) / 2

    base_pos, base_quat = arm.get_world_pose()
    print(f"base pose  pos={np.round(base_pos,4).tolist()} quat={np.round(base_quat,4).tolist()}")
    print(f"dof names  {names}")

    print("\n=== jaw travel: which direction opens?")
    for jaw in (-0.17, 0.0, 0.5, 1.0, 1.74):
        q = np.zeros(6, dtype=np.float32)
        q[5] = jaw
        settle(q, 60)
        cache.Clear()
        _, _, g_c = bbox("/World/SO101/gripper")
        _, _, j_c = bbox("/World/SO101/jaw")
        print(f"  Jaw={jaw:+.3f}  gripper_c={np.round(g_c,4).tolist()}  jaw_c={np.round(j_c,4).tolist()}  sep={np.linalg.norm(g_c-j_c)*1000:6.1f}mm")

    print("\n=== reach: FK of the gripper for sample configs (Antioch joint convention)")
    samples = {
        "zeros":       [0, 0, 0, 0, 0, 0],
        "pitch+0.6":   [0, 0.6, 0, 0, 0, 0],
        "pitch-0.6":   [0, -0.6, 0, 0, 0, 0],
        "elbow+0.6":   [0, 0, 0.6, 0, 0, 0],
        "elbow-0.6":   [0, 0, -0.6, 0, 0, 0],
        "rot+0.6":     [0.6, 0, 0, 0, 0, 0],
        "reach_fwd":   [0, 0.9, -0.9, 0.6, 0, 0.5],
        "reach_down":  [0, 1.2, -1.2, 0.9, 0, 0.5],
    }
    for label, q in samples.items():
        settle(np.array(q, dtype=np.float32), 90)
        cache.Clear()
        g_lo, g_hi, g_c = bbox("/World/SO101/gripper")
        j_lo, j_hi, j_c = bbox("/World/SO101/jaw")
        tip_z = min(g_lo[2], j_lo[2])
        print(f"  {label:11s} gripper_c={np.round(g_c,4).tolist()}  lowest_z={tip_z:+.4f}")

    print("\n=== achieved vs commanded (are the drives stiff enough?)")
    target = np.array([0, 0.9, -0.9, 0.6, 0, 0.5], dtype=np.float32)
    settle(target, 150)
    got = np.asarray(arm.get_joint_positions())
    for n, t, g in zip(names, target, got):
        print(f"  {n:12s} cmd={t:+.4f} got={g:+.4f} err={abs(t-g):.4f}")


if __name__ == "__main__":
    main()
