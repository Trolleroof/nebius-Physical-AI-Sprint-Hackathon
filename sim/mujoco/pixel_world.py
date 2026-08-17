#!/usr/bin/env python3
"""Exact pixel <-> world conversion for fixed MuJoCo cameras.

Used to turn a 2D point produced by a vision model (Gemini ER 2 returns
[y, x] normalised to 0-1000) into an exact 3D world point on a known
horizontal plane, and to draw world points back onto a rendered frame.

Conventions (MuJoCo):
  * ``data.cam_xmat[cam_id]`` is row-major 3x3; its COLUMNS are the camera
    frame axes expressed in world coordinates.
  * The camera looks down its own -Z axis; +X is image-right, +Y is image-UP.
  * ``mujoco.Renderer`` returns an upright image, so pixel row 0 is the TOP
    and ``v_px`` grows DOWNWARD (opposite of the camera's +Y).
  * Vertical field of view ``model.cam_fovy`` is in degrees; the focal length
    in pixels is ``f = 0.5 * height / tan(0.5 * fovy)`` and is the same for
    both axes (square pixels), so the horizontal FOV follows the aspect ratio.
  * The principal point is the image centre, ``((width - 1) / 2,
    (height - 1) / 2)`` -- this matches MuJoCo's own camera-matrix recipe.

All functions are pure: they read the model/data pose but never mutate it.
"""

from __future__ import annotations

import numpy as np

import mujoco

__all__ = ["project", "unproject", "gemini_point_to_pixel", "camera_pose", "focal_px"]


def _cam_id(model, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"no camera named {cam_name!r} in this model")
    return cam_id


def camera_pose(model, data, cam_name: str):
    """Return ``(cam_id, eye_world[3], R[3, 3])`` for a named camera.

    ``R`` maps camera-frame vectors to world-frame vectors, so ``R.T`` maps
    world -> camera.  ``mj_forward`` is called when ``data.cam_xpos`` has not
    been populated yet (all-zero rotation matrix).
    """
    cam_id = _cam_id(model, cam_name)
    R = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    if not np.any(R):  # kinematics never ran -> derived quantities are stale
        mujoco.mj_forward(model, data)
        R = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    eye = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
    return cam_id, eye, R


def focal_px(model, cam_id: int, height: int) -> float:
    """Focal length in pixels for a camera rendered at ``height`` rows."""
    fovy = np.deg2rad(float(model.cam_fovy[cam_id]))
    return 0.5 * float(height) / np.tan(0.5 * fovy)


def project(model, data, cam_name: str, xyz_world, width: int, height: int):
    """World point -> (u_px, v_px) in an image rendered at ``width x height``.

    ``u_px`` grows right, ``v_px`` grows DOWN from row 0 at the top.  Raises
    ``ValueError`` if the point is behind the camera (or exactly in its plane),
    where the perspective divide is undefined.
    """
    cam_id, eye, R = camera_pose(model, data, cam_name)
    p_cam = R.T @ (np.asarray(xyz_world, dtype=np.float64).reshape(3) - eye)

    depth = -p_cam[2]  # camera looks along -Z, so points in front have z < 0
    if depth <= 0.0:
        raise ValueError(
            f"point {np.asarray(xyz_world).tolist()} is not in front of camera "
            f"{cam_name!r} (depth={depth:.6g} m)"
        )

    f = focal_px(model, cam_id, height)
    u = 0.5 * (width - 1) + f * p_cam[0] / depth
    v = 0.5 * (height - 1) - f * p_cam[1] / depth  # +Y is up, v is down
    return float(u), float(v)


def unproject(
    model, data, cam_name: str, u_px: float, v_px: float, plane_z: float,
    width: int, height: int,
) -> np.ndarray:
    """Pixel -> the 3D point where its viewing ray meets the plane z=plane_z.

    Raises ``ValueError`` if the ray is parallel to the plane or points away
    from it (the intersection would be behind the camera).
    """
    cam_id, eye, R = camera_pose(model, data, cam_name)
    f = focal_px(model, cam_id, height)

    # Ray direction in camera frame; -1 on Z because the camera looks along -Z,
    # and the Y component is negated because v grows downward.
    d_cam = np.array(
        [(float(u_px) - 0.5 * (width - 1)) / f,
         -(float(v_px) - 0.5 * (height - 1)) / f,
         -1.0],
        dtype=np.float64,
    )
    d_world = R @ d_cam

    if abs(d_world[2]) < 1e-12:
        raise ValueError(
            f"ray through pixel ({u_px}, {v_px}) is parallel to the plane z={plane_z}"
        )
    t = (float(plane_z) - eye[2]) / d_world[2]
    if t <= 0.0:
        raise ValueError(
            f"ray through pixel ({u_px}, {v_px}) points away from the plane "
            f"z={plane_z} (t={t:.6g})"
        )
    return eye + t * d_world


def gemini_point_to_pixel(point_yx_norm, width: int, height: int):
    """Gemini ER 2 point ``[y, x]`` normalised to 0-1000 -> (u_px, v_px).

    Note the y-first ordering.  The returned pixel is the continuous image
    coordinate of the point, i.e. 0 is the left/top edge of the image.
    """
    y_norm, x_norm = (float(c) for c in np.asarray(point_yx_norm).reshape(2))
    u = x_norm / 1000.0 * float(width)
    v = y_norm / 1000.0 * float(height)
    return u, v
