"""Red trapezoid manipuland — SM_TrapezoidRed_01 from hackathon/GeometricBlocks 01.

Dimensions measured from the Antioch USD (v1.0.0), matching the physical block
on the desk: a square frustum with parallel ±Y grasp faces (30.48 mm) and X
faces that taper from 35.39 mm at the base to 25.57 mm at the top.

    antioch assets pull "hackathon/GeometricBlocks 01" -o antioch_assets/
"""

from __future__ import annotations

import numpy as np

BLOCK_ASSET = "hackathon/GeometricBlocks 01"
BLOCK_ASSET_VERSION = "1.0.0"
BLOCK_PRIM_NAME = "SM_TrapezoidRed_01"

# Metres — bounding box 35.39 × 30.48 × 30.48 mm
BASE_X = 0.03539
BASE_Y = 0.03048
TOP_X = 0.02557
TOP_Y = 0.03048
HEIGHT = 0.03048
MASS = 0.02
COLOR = np.array([0.578, 0.032, 0.023])  # M_BlockRed_01 linear RGB

GRASP_WIDTH = BASE_Y


class BlockHandle:
    """Thin wrapper so callers can use `.get_world_pose()[0]` like DynamicCuboid."""

    def __init__(self, prim_path: str):
        self.prim_path = prim_path
        self._rigid = None

    def bind(self) -> "BlockHandle":
        from isaacsim.core.prims import SingleRigidPrim

        self._rigid = SingleRigidPrim(prim_path=self.prim_path, name="block")
        self._rigid.initialize()
        return self

    def get_world_pose(self):
        pos, ori = self._rigid.get_world_pose()
        return np.asarray(pos, dtype=float), ori


def _frustum_points(base_x: float, base_y: float, top_x: float, top_y: float, height: float):
    bx, by = base_x / 2.0, base_y / 2.0
    tx, ty = top_x / 2.0, top_y / 2.0
    return [
        (-bx, -by, 0.0),
        (bx, -by, 0.0),
        (bx, by, 0.0),
        (-bx, by, 0.0),
        (-tx, -ty, height),
        (tx, -ty, height),
        (tx, ty, height),
        (-tx, ty, height),
    ]


def _add_mesh_block(world, position: tuple[float, float, float], *, prim_path: str, mass: float):
    """Procedural square frustum with mesh collision — offline-safe fallback."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage

    x, y, z_centre = position
    z_base = z_centre - HEIGHT / 2.0

    points = [
        Gf.Vec3f(px + x, py + y, pz + z_base)
        for px, py, pz in _frustum_points(BASE_X, BASE_Y, TOP_X, TOP_Y, HEIGHT)
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    face_vertex_indices = [
        0, 1, 2, 3,
        4, 7, 6, 5,
        0, 4, 5, 1,
        1, 5, 6, 2,
        2, 6, 7, 3,
        3, 7, 4, 0,
    ]

    stage = get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(float(COLOR[0]), float(COLOR[1]), float(COLOR[2]))])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.CreateApproximationAttr("convexHull")
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

    return BlockHandle(prim_path)


def _try_load_antioch_asset(world, position: tuple[float, float, float], *, prim_path: str):
    """Load the hackathon asset and isolate SM_TrapezoidRed_01."""
    import antioch
    from pxr import Gf, Usd, UsdGeom
    from isaacsim.core.utils.prims import get_prim_at_path
    from isaacsim.core.utils.stage import get_current_stage

    staging = f"{prim_path}_asset_root"
    antioch.load_asset(BLOCK_ASSET, prim_path=staging, version=BLOCK_ASSET_VERSION)
    stage = get_current_stage()

    src_path = None
    for prim in stage.Traverse():
        if prim.GetName() == BLOCK_PRIM_NAME:
            src_path = str(prim.GetPath())
            break
    if src_path is None:
        raise RuntimeError(f"{BLOCK_PRIM_NAME} not found under {BLOCK_ASSET}")

    x, y, z_centre = position
    xform = UsdGeom.Xformable(get_prim_at_path(src_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z_centre - HEIGHT / 2.0))

    # Hide sibling prims from the kit so only the red trapezoid is visible.
    root = stage.GetPrimAtPath(staging)
    for child in root.GetChildren():
        path = str(child.GetPath())
        if path != src_path:
            imageable = UsdGeom.Imageable(child)
            if imageable:
                imageable.MakeInvisible()

    return BlockHandle(src_path)


def add_trapezoid_block(
    world,
    position: tuple[float, float, float],
    *,
    prim_path: str = "/World/block",
    mass: float = MASS,
    prefer_asset: bool = True,
) -> BlockHandle:
    """Spawn the red trapezoid block and return a pose handle."""

    if prefer_asset:
        try:
            return _try_load_antioch_asset(world, position, prim_path=prim_path)
        except Exception:
            pass

    return _add_mesh_block(world, position, prim_path=prim_path, mass=mass)
