#!/usr/bin/env python3
"""Self-checking test for sim/mujoco/pixel_world.py.

Run from the repo root:
    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/test_pixel_world.py

Checks
  1. project -> unproject round trip on >= 6 known world points is exact
     (< 1e-6 m) for the fixed `world` camera.
  2. unproject rejects rays that miss the plane.
  3. gemini_point_to_pixel round trip.
  4. VISUAL + programmatic proof: render the `world` camera at 640x480, draw
     crosshairs at the projected cube and tray centres, and assert the pixel
     under the cube crosshair is actually the orange cube.

Exits nonzero on the first failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import mujoco
import imageio.v2 as iio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pixel_world import project, unproject, gemini_point_to_pixel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")
OUT_PNG = os.path.join(HERE, "test_pixel_world.png")

W, H = 640, 480
CAM = "world"

_failures = []


def check(ok: bool, msg: str) -> None:
    if ok:
        print(f"PASS  {msg}")
    else:
        print(f"FAIL  {msg}")
        _failures.append(msg)


def draw_crosshair(img, u, v, rgb, arm=14, gap=3, thick=1):
    """Draw a crosshair centred on (u, v) with a hollow centre."""
    h, w = img.shape[:2]
    cu, cv = int(round(u)), int(round(v))
    for d in range(gap, arm + 1):
        for t in range(-thick, thick + 1):
            for py, px in ((cv + t, cu + d), (cv + t, cu - d),
                           (cv + d, cu + t), (cv - d, cu + t)):
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = rgb


def main() -> int:
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    tray_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    cube_pos = np.array(data.xpos[cube_bid], dtype=np.float64)
    tray_pos = np.array(data.xpos[tray_bid], dtype=np.float64)
    tray_center = tray_pos + np.array([0.0, 0.0, 0.006])  # tray_center site

    # ---- 1. round trip -------------------------------------------------
    # Cube centre, the four outer tray-floor corners (half extents 0.105 x
    # 0.07 from scene.xml) at the tray floor top z=0.006, the tray centre,
    # and two arbitrary points on z=0.02 inside the view.
    points = [
        ("cube body pos", cube_pos),
        ("tray corner -x -y", tray_pos + np.array([-0.105, -0.070, 0.006])),
        ("tray corner +x -y", tray_pos + np.array([+0.105, -0.070, 0.006])),
        ("tray corner +x +y", tray_pos + np.array([+0.105, +0.070, 0.006])),
        ("tray corner -x +y", tray_pos + np.array([-0.105, +0.070, 0.006])),
        ("tray center site", tray_center),
        ("arbitrary A z=0.02", np.array([0.25, -0.05, 0.02])),
        ("arbitrary B z=0.02", np.array([0.12, 0.10, 0.02])),
    ]

    worst = 0.0
    worst_name = ""
    for name, p in points:
        u, v = project(model, data, CAM, p, W, H)
        back = unproject(model, data, CAM, u, v, float(p[2]), W, H)
        err = float(np.linalg.norm(back[:2] - p[:2]))
        if err > worst:
            worst, worst_name = err, name
        check(
            err < 1e-6,
            f"round trip {name:<20} p=({p[0]:+.5f},{p[1]:+.5f},{p[2]:.5f}) "
            f"-> px=({u:8.3f},{v:8.3f}) -> err={err:.3e} m",
        )
        check(
            abs(back[2] - p[2]) < 1e-12,
            f"round trip {name:<20} lands exactly on plane z={p[2]:.5f}",
        )
    print(f"      worst round-trip xy error = {worst:.3e} m ({worst_name})")

    # ---- 2. degenerate rays --------------------------------------------
    # A pixel on the far horizon side: shoot at a plane above the camera eye
    # so the intersection is behind the camera -> must raise.
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    eye_z = float(data.cam_xpos[cam_id][2])
    try:
        unproject(model, data, CAM, W / 2.0, H - 1.0, eye_z + 1.0, W, H)
        check(False, "unproject raises when the plane is behind the ray")
    except ValueError:
        check(True, "unproject raises when the plane is behind the ray")

    try:
        project(model, data, CAM, np.array([2.0, -2.0, 0.55]), W, H)
        check(False, "project raises for a point behind the camera")
    except ValueError:
        check(True, "project raises for a point behind the camera")

    # ---- 3. gemini point conversion ------------------------------------
    for yx in ([500.0, 500.0], [0.0, 0.0], [250.0, 750.0], [1000.0, 1000.0]):
        u, v = gemini_point_to_pixel(yx, W, H)
        y_back = v / H * 1000.0
        x_back = u / W * 1000.0
        ok = abs(y_back - yx[0]) < 1e-9 and abs(x_back - yx[1]) < 1e-9
        check(ok, f"gemini [y,x]={yx} -> px=({u:.3f},{v:.3f}) -> back=[{y_back:.3f},{x_back:.3f}]")
    # y-first ordering must actually matter
    u_a, v_a = gemini_point_to_pixel([250.0, 750.0], W, H)
    check(
        abs(u_a - 0.75 * W) < 1e-9 and abs(v_a - 0.25 * H) < 1e-9,
        "gemini point is [y, x] (y first), not [x, y]",
    )

    # ---- 4. render + crosshairs + colour proof --------------------------
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=CAM)
        img = r.render().copy()

    u_cube, v_cube = project(model, data, CAM, cube_pos, W, H)
    u_tray, v_tray = project(model, data, CAM, tray_center, W, H)

    check(
        0 <= u_cube < W and 0 <= v_cube < H,
        f"cube projects inside the image at ({u_cube:.2f}, {v_cube:.2f})",
    )
    check(
        0 <= u_tray < W and 0 <= v_tray < H,
        f"tray centre projects inside the image at ({u_tray:.2f}, {v_tray:.2f})",
    )

    px = img[int(round(v_cube)), int(round(u_cube))].astype(int)
    R_, G_, B_ = int(px[0]), int(px[1]), int(px[2])
    check(
        R_ > 150 and R_ > G_ + 40 and R_ > B_ + 60,
        f"pixel under the cube crosshair is orange/red: RGB=({R_},{G_},{B_})",
    )

    # The tray crosshair must land on the tray floor (material wood_dark,
    # rgba 0.40 0.25 0.13), not on the surrounding ground plane (0.72 0.55
    # 0.36) nor on the cube.  Lighting scales all channels together, so match
    # on the channel ratios: wood_dark is G/R=0.625, B/R=0.325 while the
    # ground plane is G/R=0.764, B/R=0.500 and the cube is G/R=0.45.
    tpx = img[int(round(v_tray)), int(round(u_tray))].astype(int)
    tR, tG, tB = int(tpx[0]), int(tpx[1]), int(tpx[2])
    gr, br = tG / max(tR, 1), tB / max(tR, 1)
    check(
        tR > tG > tB and 0.55 < gr < 0.70 and 0.25 < br < 0.40,
        f"pixel under the tray crosshair is the wood_dark tray floor: "
        f"RGB=({tR},{tG},{tB}) G/R={gr:.3f} B/R={br:.3f}",
    )

    draw_crosshair(img, u_cube, v_cube, np.array([0, 255, 0], dtype=img.dtype))
    draw_crosshair(img, u_tray, v_tray, np.array([0, 128, 255], dtype=img.dtype))
    iio.imwrite(OUT_PNG, img)
    print(f"      wrote {OUT_PNG}")

    if _failures:
        print(f"\n{len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
