"""Render and validate the SO-101's physical-style wrist camera.

    antioch scenario run --scenario wrist_camera_probe --no-stream --verbose
"""

from __future__ import annotations

from pathlib import Path

import antioch

from wrist_camera import (
    WRIST_CAMERA_LOCAL_QUAT_WXYZ,
    WRIST_CAMERA_LOCAL_TRANSLATION,
    add_wrist_camera,
    capture_wrist_rgb,
    measure_frame,
)

logger = antioch.Logger("wrist_camera")


@antioch.scenario(tags=["smoke", "camera"], capture=False)
def wrist_camera_probe(run: antioch.ScenarioRun) -> None:
    """The workspace RGB is exposed, not blank/blown out, and sees the red block."""

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction

    from trapezoid_block import HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 700.0})
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")

    # This compact, down-facing working pose keeps the camera over the block;
    # it avoids the fully extended pose that is unlike the real tabletop setup.
    block = add_trapezoid_block(world, (0.3464, -0.1361, HEIGHT / 2.0))
    world.reset()
    block.bind()
    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    camera = add_wrist_camera()

    q = np.array([0.0, 1.35, -1.65, 0.30, np.pi, 0.85], dtype=np.float32)
    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(120):
        world.step(render=True)

    # Isaac's first viewport grab can predate the rendered scene; discard it.
    capture_wrist_rgb(camera)
    rgb = None
    for _ in range(4):
        world.step(render=True)
        candidate = capture_wrist_rgb(camera)
        if candidate is not None:
            rgb = candidate
    if rgb is None:
        raise RuntimeError("wrist viewport never returned RGB")

    metrics = measure_frame(rgb)
    logger.image("camera/wrist_rgb", rgb)
    logger.scalar("camera/mean", metrics.mean)
    logger.scalar("camera/std", metrics.std)
    logger.scalar("camera/red_pixels", metrics.red_pixels)
    out = Path("/tmp/so101_wrist_camera_probe.png")
    from PIL import Image

    Image.fromarray(rgb).save(out)
    run.add_artifact(out, name="so101_wrist_camera_probe.png")
    run.add_result("camera_local_translation_m", [round(float(v), 4) for v in WRIST_CAMERA_LOCAL_TRANSLATION])
    run.add_result("camera_local_quat_wxyz", [round(float(v), 4) for v in WRIST_CAMERA_LOCAL_QUAT_WXYZ])
    run.add_result("camera_resolution", list(camera.get_resolution()))
    run.add_result("frame_mean", round(metrics.mean, 2))
    run.add_result("frame_std", round(metrics.std, 2))


@antioch.scenario(tags=["camera"], capture=False)
def wrist_camera_grid(run: antioch.ScenarioRun) -> None:
    """Render 8 candidate wrist-camera mounts x 2 quaternion conventions.

    One run replaces the guess-queue-look loop that has burned the whole
    day: every candidate view lands in a single labelled contact sheet, and
    the winner is whichever tile shows the fingers AND the red block. The
    arm is held at the ACTUAL grasp pose from pick_place (wrist roll pi/2)
    — the earlier probe used a different roll, so its tuning did not
    transfer.
    """

    import numpy as np
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction

    from pick_place import DOWN, JAW_OPEN, RADIUS, WRIST_ROLL, _rot_for
    from trapezoid_block import HEIGHT, add_trapezoid_block

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 700.0})
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")

    azimuth = -0.3157  # pick_place's default block bearing
    block_x = RADIUS * float(np.cos(azimuth))
    block_y = RADIUS * float(np.sin(azimuth))
    add_trapezoid_block(world, (block_x, block_y, HEIGHT / 2.0))
    world.reset()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    q = np.array(
        [_rot_for(block_x, block_y), DOWN[0], DOWN[1], DOWN[2], WRIST_ROLL, JAW_OPEN],
        dtype=np.float32,
    )
    arm.set_joint_positions(q)
    for _ in range(120):
        arm.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=True)

    camera = add_wrist_camera(resolution=(320, 240))

    candidates = [
        ("nik_tuned", (0.0, 0.075, 0.02), (-0.4089, -0.6641, 0.5769, -0.2429)),
        ("pickplace", (0.0, 0.075, 0.02), (0.0, 0.0, -1.0, 0.0)),
        ("identity", (0.0, 0.075, 0.02), (1.0, 0.0, 0.0, 0.0)),
        ("xm50_top", (0.0, 0.10, 0.04), (0.9063, -0.4226, 0.0, 0.0)),
        ("xp50_top", (0.0, 0.10, 0.04), (0.9063, 0.4226, 0.0, 0.0)),
        ("xm90", (0.0, 0.075, 0.02), (0.7071, -0.7071, 0.0, 0.0)),
        ("xp90", (0.0, 0.075, 0.02), (0.7071, 0.7071, 0.0, 0.0)),
        ("under_id", (0.0, -0.075, 0.02), (1.0, 0.0, 0.0, 0.0)),
    ]

    from PIL import Image, ImageDraw

    tiles = []
    report = {}
    for label, translation, quat in candidates:
        for axes in ("usd", "world"):
            try:
                camera.set_local_pose(
                    np.array(translation), np.array(quat), camera_axes=axes
                )
            except Exception as exc:  # noqa: BLE001 — a bad convention name skips one tile
                report[f"{label}/{axes}"] = f"set_local_pose: {type(exc).__name__}"
                continue
            for _ in range(6):
                world.step(render=True)
            rgb = capture_wrist_rgb(camera)
            if rgb is None:
                report[f"{label}/{axes}"] = "no frame"
                continue
            metrics = measure_frame(rgb)
            report[f"{label}/{axes}"] = f"red={metrics.red_pixels} std={metrics.std:.0f}"
            tile = Image.fromarray(rgb)
            ImageDraw.Draw(tile).text((6, 6), f"{label} [{axes}]", fill=(255, 255, 0))
            tiles.append(np.asarray(tile))

    if not tiles:
        raise RuntimeError("no candidate produced a frame")
    cols = 4
    rows = -(-len(tiles) // cols)
    blank = np.zeros_like(tiles[0])
    grid = np.concatenate(
        [np.concatenate((tiles + [blank] * (rows * cols - len(tiles)))[r * cols:(r + 1) * cols], axis=1) for r in range(rows)],
        axis=0,
    )
    out = Path("/tmp/wrist_camera_grid.png")
    Image.fromarray(grid).save(out)
    run.add_artifact(out, name="wrist_camera_grid.png")
    for key_, value in report.items():
        run.add_result(key_.replace("/", "_"), value)
    run.check("every candidate rendered", len(tiles) == 16, detail=f"{len(tiles)}/16 tiles")
    run.add_result("red_pixels", metrics.red_pixels)
    run.add_result("red_fraction", round(metrics.red_fraction, 6))
    run.check("wrist RGB is nonblank and correctly exposed", metrics.usable, detail=str(metrics))
    run.check("red trapezoid is visible in wrist RGB", metrics.red_block_visible, detail=str(metrics))


if __name__ == "__main__":
    # Scenario runs exercise artifact upload and check reporting; this entry
    # point is deliberately not a second, divergent execution path.
    raise SystemExit("Run with: antioch scenario run --scenario wrist_camera_probe --no-stream --verbose")
