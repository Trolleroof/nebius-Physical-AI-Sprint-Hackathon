"""Simple collision geometry for the real wooden placement tray.

The photo shows a shallow rectangular tray. The handle cutout is visual-only
and is intentionally omitted: it does not affect pick/place contact or the
success gate.
"""

from __future__ import annotations

import numpy as np


def add_wooden_tray(
    world,
    *,
    prim_path: str = "/World/WoodenTray",
    center: tuple[float, float] = (-0.11, -0.13),
    outer_size: tuple[float, float] = (0.30, 0.18),
    floor_thickness: float = 0.006,
    wall_height: float = 0.030,
    wall_thickness: float = 0.012,
):
    """Add a fixed, open wooden tray and return its five rigid parts."""

    from isaacsim.core.api.objects import FixedCuboid

    length, width = outer_size
    x, y = center
    wood = np.array([0.48, 0.22, 0.07])
    dark_wood = np.array([0.30, 0.12, 0.035])
    parts = {}

    def add(name, size, position, color):
        parts[name] = world.scene.add(
            FixedCuboid(
                prim_path=f"{prim_path}/{name}",
                name=name,
                position=np.asarray(position, dtype=float),
                size=1.0,
                scale=np.asarray(size, dtype=float),
                color=color,
            )
        )

    add("floor", (length, width, floor_thickness), (x, y, floor_thickness / 2), dark_wood)
    wall_z = floor_thickness + wall_height / 2
    add("wall_x_min", (wall_thickness, width, wall_height), (x - (length - wall_thickness) / 2, y, wall_z), wood)
    add("wall_x_max", (wall_thickness, width, wall_height), (x + (length - wall_thickness) / 2, y, wall_z), wood)
    add("wall_y_min", (length - 2 * wall_thickness, wall_thickness, wall_height), (x, y - (width - wall_thickness) / 2, wall_z), wood)
    add("wall_y_max", (length - 2 * wall_thickness, wall_thickness, wall_height), (x, y + (width - wall_thickness) / 2, wall_z), wood)
    return parts
