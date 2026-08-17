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
    run.add_result("red_pixels", metrics.red_pixels)
    run.add_result("red_fraction", round(metrics.red_fraction, 6))
    run.check("wrist RGB is nonblank and correctly exposed", metrics.usable, detail=str(metrics))
    run.check("red trapezoid is visible in wrist RGB", metrics.red_block_visible, detail=str(metrics))


if __name__ == "__main__":
    # Scenario runs exercise artifact upload and check reporting; this entry
    # point is deliberately not a second, divergent execution path.
    raise SystemExit("Run with: antioch scenario run --scenario wrist_camera_probe --no-stream --verbose")
