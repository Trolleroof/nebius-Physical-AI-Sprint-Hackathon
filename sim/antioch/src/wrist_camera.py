"""SO-101 wrist-camera setup and one honest RGB quality gate.

The local pose follows the physical mount: PCB vertical above the gripper,
lens looking forward with the fingers.  The transform is relative to the
``gripper`` link, not the world, so the observation moves with the arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gripper frame, measured in probe_frames.py: +X closes the jaws, -Z follows
# the fingers, and +Y is up. Aim the offset lens at the measured grasp centre
# so the jaws remain in-frame instead of looking parallel past them.
WRIST_CAMERA_LOCAL_TRANSLATION = np.array([0.0, 0.075, 0.020], dtype=np.float32)
WRIST_CAMERA_LOCAL_QUAT_WXYZ = np.array([-0.4089, -0.6641, 0.5769, -0.2429], dtype=np.float32)
WRIST_CAMERA_PATH = "/World/SO101/gripper/wrist_cam"
WRIST_CAMERA_RESOLUTION = (640, 480)


@dataclass(frozen=True)
class FrameMetrics:
    mean: float
    std: float
    clipped_fraction: float
    red_pixels: int
    red_fraction: float

    @property
    def usable(self) -> bool:
        return 8.0 <= self.mean <= 245.0 and self.std >= 6.0 and self.clipped_fraction < 0.80

    @property
    def red_block_visible(self) -> bool:
        # The 30 mm block occupies much more than this at the mounted camera's
        # working distance.  It rejects an empty workspace without demanding
        # segmentation or a new dependency.
        return self.red_pixels >= 80 and self.red_fraction >= 0.00025


def add_wrist_camera(*, resolution: tuple[int, int] = WRIST_CAMERA_RESOLUTION):
    """Create the gripper-parented camera and route Antioch capture to it."""

    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport

    camera = Camera(
        prim_path=WRIST_CAMERA_PATH,
        translation=WRIST_CAMERA_LOCAL_TRANSLATION,
        orientation=WRIST_CAMERA_LOCAL_QUAT_WXYZ,
        frequency=20,
        resolution=resolution,
    )
    camera.initialize()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Antioch did not provide an active viewport for the wrist camera")
    viewport.camera_path = camera.prim_path
    return camera


def capture_wrist_rgb(camera) -> np.ndarray | None:
    """Return the wrist sensor's uint8 RGB frame, if Isaac rendered one."""

    frame = camera.get_rgba()
    if frame is None or not getattr(frame, "size", 0):
        return None
    rgb = np.asarray(frame)[..., :3]
    return np.clip(rgb, 0, 255).astype(np.uint8, copy=False)


def measure_frame(rgb: np.ndarray) -> FrameMetrics:
    """Check exposure plus the distinctive red GeometricBlocks trapezoid."""

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWxRGB frame, got {rgb.shape}")
    pixels = rgb[..., :3].astype(np.int16, copy=False)
    red, green, blue = (pixels[..., i] for i in range(3))
    red_mask = (red >= 70) & (red >= green + 25) & (red >= blue + 25) & (red * 10 >= green * 16)
    red_pixels = int(red_mask.sum())
    return FrameMetrics(
        mean=float(pixels.mean()),
        std=float(pixels.std()),
        clipped_fraction=float((pixels >= 250).all(axis=2).mean()),
        red_pixels=red_pixels,
        red_fraction=red_pixels / float(red_mask.size),
    )
