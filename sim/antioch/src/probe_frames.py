"""Pin down the tool frame and validate the Jacobian layout.

    antioch run --no-stream src/probe_frames.py
"""

from __future__ import annotations

import antioch

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"


def main() -> None:
    antioch.boot()

    import numpy as np
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom

    world = antioch.world()
    world.scene.add_ground_plane()
    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    world.reset()

    single = SingleArticulation(prim_path="/World/SO101", name="so101")
    single.initialize()
    arm = Articulation("/World/SO101")
    grip = RigidPrim("/World/SO101/gripper")
    links = list(arm._link_names)
    print("links:", links, "-> gripper idx", links.index("gripper"))

    # every purpose, or the collision prims come back as an empty range
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    def settle(q, steps=120):
        q = np.asarray(q, dtype=np.float32)
        single.set_joint_positions(q)
        single.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(steps):
            world.step(render=False)

    def pose():
        p, o = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = o
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return p, R

    def union_bbox(*paths):
        cache.Clear()
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for path in paths:
            r = cache.ComputeWorldBound(get_prim_at_path(path)).ComputeAlignedRange()
            if r.IsEmpty():
                continue
            lo = np.minimum(lo, np.array(r.GetMin()))
            hi = np.maximum(hi, np.array(r.GetMax()))
        return lo, hi

    print("\n=== tool frame: where are the fingertips in the gripper link frame?")
    for label, q in {
        "zeros":      [0, 0, 0, 0, 0, 0.5],
        "reach_down": [0, 1.2, -1.2, 0.9, 0, 0.5],
        "roll_+90":   [0, 1.2, -1.2, 0.9, np.pi / 2, 0.5],
        "roll_-90":   [0, 1.2, -1.2, 0.9, -np.pi / 2, 0.5],
        "pick_+90":   [0.45, 1.2785, -1.255, 0.955, np.pi / 2, 0.34],
        "pick_-90":   [0.45, 1.2785, -1.255, 0.955, -np.pi / 2, 0.34],
        "pitch_1.20": [0, 1.2785, -1.255, 1.20, np.pi / 2, 0.34],
        "pitch_1.45": [0, 1.2785, -1.255, 1.45, np.pi / 2, 0.34],
        "pitch_1.65": [0, 1.2785, -1.255, 1.65, np.pi / 2, 0.34],
        "compact_a":  [0, 1.28, -1.26, 0.0, np.pi / 2, 0.34],
        "compact_b":  [0, 1.45, -1.45, 0.0, np.pi / 2, 0.34],
        "compact_c":  [0, 1.55, -1.30, 0.0, np.pi / 2, 0.34],
        "compact_d":  [0, 1.55, -1.60, 0.0, np.pi / 2, 0.34],
        "compact_e":  [0, 1.35, -1.65, 0.0, np.pi / 2, 0.34],
        "compact_f":  [0, 1.60, -1.65, 0.0, np.pi / 2, 0.34],
        "folded_a":   [0, 1.40, -0.80, -0.60, np.pi / 2, 0.34],
        "folded_b":   [0, 1.50, -1.00, -0.50, np.pi / 2, 0.34],
        "folded_c":   [0, 1.60, -0.80, -0.80, np.pi / 2, 0.34],
    }.items():
        settle(q)
        p, R = pose()
        lo, hi = union_bbox("/World/SO101/gripper", "/World/SO101/jaw")
        corners = np.array([[cx, cy, cz] for cx in (lo[0], hi[0]) for cy in (lo[1], hi[1]) for cz in (lo[2], hi[2])])
        local = (R.T @ (corners - p).T).T
        print(f"  {label}: gripper pos {np.round(p,4).tolist()}")
        print(f"     union bbox world lo={np.round(lo,4).tolist()} hi={np.round(hi,4).tolist()}")
        print(f"     bbox in tool frame  lo={np.round(local.min(0),4).tolist()} hi={np.round(local.max(0),4).tolist()}")
        print(f"     tool x={np.round(R[:,0],3).tolist()} y={np.round(R[:,1],3).tolist()} z={np.round(R[:,2],3).tolist()}")

    print("\n=== jacobian layout: finite-difference check on the gripper link")
    base_q = np.array([0.0, 1.0, -1.0, 0.7, 0.0, 0.5], dtype=np.float32)
    settle(base_q)
    p0, _ = pose()
    J = arm.get_jacobian_matrices().numpy()
    gidx = links.index("gripper")
    print(f"  jacobian shape {J.shape}; using link index {gidx}")
    eps = 0.02
    for j in range(6):
        q = base_q.copy()
        q[j] += eps
        settle(q, 120)
        p1, _ = pose()
        fd = (p1 - p0) / eps
        col_no_off = J[0, gidx, :3, j]
        col_off6 = J[0, gidx, :3, 6 + j]
        e0 = float(np.linalg.norm(fd - col_no_off))
        e6 = float(np.linalg.norm(fd - col_off6))
        print(f"  dof {j}: fd={np.round(fd,3).tolist()}  |fd-J[:,{j}]|={e0:.3f}  |fd-J[:,{6+j}]|={e6:.3f}")
        settle(base_q, 60)


if __name__ == "__main__":
    main()
