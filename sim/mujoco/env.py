#!/usr/bin/env python
"""Minimal MuJoCo environment wrapper for the SO-101 pick-and-place task.

No gym dependency: three methods (``reset`` / ``step`` / ``render``) plus a
``success`` predicate, driven directly by ``collect.py``.

Design notes
------------
* ONE ``mujoco.Renderer`` is created in ``__init__`` and reused for every
  episode.  Creating/destroying renderers per episode leaks GL contexts on
  Windows and is the single most common cause of "renderer is not available".
* The model's own ``<option timestep=...>`` is never modified.  A 10 Hz control
  tick is realised as ``round(0.1 / timestep)`` physics substeps.
* ``render()`` returns the renderer output verbatim.  MuJoCo's ``Renderer``
  already hands back an upright RGB image -- do NOT flip it vertically.
* Body/site names are resolved through fallback lists so the same code runs
  against the scratch scene (``robotstudio_so101/scene_box.xml``, body ``box``,
  no tray) and against the real task scene (``sim/mujoco/scene.xml``, body
  ``cube`` + body ``tray``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# Name resolution: first entry that exists in the loaded model wins.
# --------------------------------------------------------------------------
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"

# --------------------------------------------------------------------------
# wrist_roll is BROKEN on the physical arm.  The motor is dead and the joint is
# taped in place at -pi/2 (claws side-by-side, hand the right way up).  Sending
# it a goal position on hardware stalls a servo that cannot answer and works the
# tape loose, so it must never be commanded anywhere but its lock.
#
# This is enforced here, in clip_cmd, rather than in each caller: every command
# path -- the scripted expert in collect.py, a trained policy at eval, replay,
# preview -- funnels through clip_cmd, so pinning it at this depth means no
# caller can roll the wrist even by accident.  Index 4 of the 6-vector.
# --------------------------------------------------------------------------
WRIST_ROLL_INDEX = 4
WRIST_ROLL_LOCK = -math.pi / 2.0

TCP_SITE_CANDIDATES = ("gripperframe", "tcp", "grip_site", "attachment_site")
CUBE_BODY_CANDIDATES = ("cube", "box", "block", "orange_cube")
TRAY_BODY_CANDIDATES = ("tray", "bin", "basket")
WRIST_CAMERA_CANDIDATES = ("wrist_cam", "wrist", "handeye")
WORLD_CAMERA_CANDIDATES = ("world", "overhead", "front")

CONTROL_HZ = 10.0


def _name2id(model: mujoco.MjModel, objtype: int, names: Iterable[str]) -> int:
    for n in names:
        i = mujoco.mj_name2id(model, objtype, n)
        if i >= 0:
            return i
    return -1


@dataclass
class SpawnParams:
    """Everything needed to reproduce one episode's initial condition."""

    seed: int
    cube_azimuth: float
    cube_radius: float
    cube_yaw: float
    tray_azimuth: float
    tray_radius: float

    def as_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "cube_azimuth": float(self.cube_azimuth),
            "cube_radius": float(self.cube_radius),
            "cube_yaw": float(self.cube_yaw),
            "tray_azimuth": float(self.tray_azimuth),
            "tray_radius": float(self.tray_radius),
        }


class SO101Env:
    """SO-101 arm + cube (+ optional tray) driven by 6 position actuators."""

    def __init__(
        self,
        scene_path: str,
        render_size: Sequence[int] = (224, 224),
        *,
        home_qpos: Optional[Sequence[float]] = None,
        gripper_open: float = 0.90,
        gripper_grip: float = 0.00,
        spawn_radius: Sequence[float] = (0.18, 0.26),
        spawn_azimuth: Sequence[float] = (-0.14, 0.96),
        tray_radius: float = 0.240,
        tray_floor_offset: float = 0.006,
        success_xy_tol: float = 0.06,
        control_hz: float = CONTROL_HZ,
        world_render_size: Sequence[int] = (640, 480),
    ) -> None:
        self.scene_path = str(scene_path)
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # ---- indices -----------------------------------------------------
        self.arm_jids = []
        for name in ARM_JOINTS:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise RuntimeError(f"{self.scene_path}: missing arm joint '{name}'")
            self.arm_jids.append(j)
        gj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        if gj < 0:
            raise RuntimeError(f"{self.scene_path}: missing joint '{GRIPPER_JOINT}'")
        self.grip_jid = gj
        self.joint_ids = np.array(self.arm_jids + [gj], dtype=int)

        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dofadr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)
        self.arm_qadr = self.qadr[:5]
        self.arm_dofadr = self.dofadr[:5]
        self.jnt_range = np.array([self.model.jnt_range[j] for j in self.joint_ids], dtype=float)

        # Actuators are assumed to be one position servo per joint, in order.
        self.act_ids = []
        for j in self.joint_ids:
            aid = -1
            for a in range(self.model.nu):
                if (
                    self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and self.model.actuator_trnid[a, 0] == j
                ):
                    aid = a
                    break
            if aid < 0:
                raise RuntimeError(
                    f"{self.scene_path}: no actuator drives joint "
                    f"{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)}"
                )
            self.act_ids.append(aid)
        self.act_ids = np.array(self.act_ids, dtype=int)
        self.ctrl_range = self.model.actuator_ctrlrange[self.act_ids].copy()

        self.tcp_sid = _name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_CANDIDATES)
        if self.tcp_sid < 0:
            raise RuntimeError(f"{self.scene_path}: no TCP site among {TCP_SITE_CANDIDATES}")
        self.tcp_site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_sid)

        self.cube_bid = _name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_CANDIDATES)
        if self.cube_bid < 0:
            raise RuntimeError(f"{self.scene_path}: no cube body among {CUBE_BODY_CANDIDATES}")
        self.cube_body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_bid)
        self.cube_qadr = self._free_qposadr(self.cube_bid)
        self.cube_dofadr = self._free_dofadr(self.cube_bid)
        if self.cube_qadr is None:
            raise RuntimeError(f"{self.scene_path}: body '{self.cube_body_name}' has no freejoint")
        self.cube_half_xy, self.cube_half_z = self._cube_half_extents(self.cube_bid)

        self.tray_bid = _name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TRAY_BODY_CANDIDATES)
        self.tray_body_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.tray_bid)
            if self.tray_bid >= 0
            else None
        )
        self.tray_qadr = self._free_qposadr(self.tray_bid) if self.tray_bid >= 0 else None
        if self.tray_bid >= 0:
            p = self.model.body_pos[self.tray_bid]
            self.tray_home_azimuth = float(math.atan2(p[1], p[0]))
            self.tray_home_radius = float(math.hypot(p[0], p[1]))
        else:
            self.tray_home_azimuth, self.tray_home_radius = -math.pi / 4.0, 0.24

        self.cam_id = _name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_CANDIDATES)
        if self.cam_id < 0:
            raise RuntimeError(f"{self.scene_path}: no wrist camera among {WRIST_CAMERA_CANDIDATES}")
        self.camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_id)

        self.world_cam_id = _name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WORLD_CAMERA_CANDIDATES)
        self.world_camera_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.world_cam_id)
            if self.world_cam_id >= 0
            else None
        )

        # ---- task geometry ----------------------------------------------
        self.gripper_open = float(gripper_open)
        self.gripper_grip = float(gripper_grip)
        self.spawn_radius = (float(spawn_radius[0]), float(spawn_radius[1]))
        self.spawn_azimuth = (float(spawn_azimuth[0]), float(spawn_azimuth[1]))
        self.tray_radius_default = float(tray_radius)
        self.tray_floor_offset = float(tray_floor_offset)
        self.success_xy_tol = float(success_xy_tol)

        if home_qpos is None:
            home_qpos = DEFAULT_HOME_QPOS
        self.home_qpos = np.asarray(home_qpos, dtype=float).copy()
        if self.home_qpos.shape != (6,):
            raise ValueError("home_qpos must have 6 entries")
        # A caller-supplied home pose does not get to unpin the dead joint.
        self.home_qpos[WRIST_ROLL_INDEX] = WRIST_ROLL_LOCK

        # ---- timing ------------------------------------------------------
        self.control_hz = float(control_hz)
        self.timestep = float(self.model.opt.timestep)
        self.substeps = int(round((1.0 / self.control_hz) / self.timestep))
        if self.substeps < 1:
            raise RuntimeError(
                f"model timestep {self.timestep} is coarser than the {self.control_hz} Hz tick"
            )

        # ---- renderer (created exactly once) -----------------------------
        w, h = int(render_size[0]), int(render_size[1])
        self.render_width, self.render_height = w, h
        # Offscreen buffer must be at least as large as the requested image.
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), w)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), h)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        # Second (optional) renderer for --world-video.  Also created at most
        # once and reused for every episode.
        self._world_size = (int(world_render_size[0]), int(world_render_size[1]))
        self._world_renderer: Optional[mujoco.Renderer] = None

        # ---- episode bookkeeping ----------------------------------------
        self.spawn: Optional[SpawnParams] = None
        self._tray_center = np.zeros(3)
        self._ctrl = self.home_qpos.copy()
        # --grasp weld fallback state (see attach_cube/detach_cube)
        self._weld_rel: Optional[tuple] = None

    # ------------------------------------------------------------------ util
    def _free_qposadr(self, bid: int) -> Optional[int]:
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == bid and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[j])
        return None

    def _cube_half_extents(self, bid: int) -> tuple:
        """Half width across the jaw axis and half height, read off the geometry."""
        hx = hz = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            s = self.model.geom_size[g]
            t = self.model.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                hx, hz = float(max(s[0], s[1])), float(s[2])
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                hx = hz = float(s[0])
            elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                hx, hz = float(s[0]), float(s[1] + (s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0))
            if hx is not None:
                break
        if hx is None:  # mesh or nothing recognisable
            hx = hz = 0.020
        return hx, hz

    def _free_dofadr(self, bid: int) -> Optional[int]:
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == bid and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_dofadr[j])
        return None

    def measured_qpos(self) -> np.ndarray:
        """The 6 measured joint positions, float32, [rad]."""
        return self.data.qpos[self.qadr].astype(np.float32).copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_sid].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    @property
    def tray_center(self) -> np.ndarray:
        """World xyz of the drop target (physical tray body, or a virtual pose)."""
        if self.tray_bid >= 0:
            return self.data.xpos[self.tray_bid].copy()
        return self._tray_center.copy()

    @property
    def tray_floor_top(self) -> float:
        return float(self.tray_center[2] + self.tray_floor_offset)

    def clip_cmd(self, cmd: Sequence[float]) -> np.ndarray:
        c = np.asarray(cmd, dtype=float).reshape(6)
        c = np.clip(c, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        # The dead wrist_roll servo is overwritten, not clipped: whatever the
        # caller asked for, the joint only ever gets its lock.  See the
        # WRIST_ROLL_LOCK note at the top of this module.
        #
        # Stamped AFTER the clip on purpose.  Doing it before means the clip can
        # drag the lock off again if ctrl_range is ever wrong -- and it can be:
        # a zero-width ctrlrange in the MJCF compiles to ctrllimited=False, on
        # which MuJoCo reports the range as [0, 0], which would clip the lock to
        # 0 rad.  Stamping last keeps this layer independent of the model.
        c[WRIST_ROLL_INDEX] = WRIST_ROLL_LOCK
        return c

    # ----------------------------------------------------------------- reset
    def reset(
        self,
        seed: int = 0,
        cube_azimuth: Optional[float] = None,
        tray_azimuth: Optional[float] = None,
        *,
        cube_radius: Optional[float] = None,
        tray_radius: Optional[float] = None,
        settle_seconds: float = 0.5,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))

        if cube_azimuth is None:
            cube_azimuth = float(rng.uniform(*self.spawn_azimuth))
        if cube_radius is None:
            cube_radius = float(rng.uniform(*self.spawn_radius))
        if tray_azimuth is None:
            # Default: leave the tray where the scene author put it.
            tray_azimuth = self.tray_home_azimuth
        if tray_radius is None:
            tray_radius = self.tray_home_radius
        cube_yaw = float(rng.uniform(-math.pi, math.pi))

        self.spawn = SpawnParams(
            seed=int(seed),
            cube_azimuth=float(cube_azimuth),
            cube_radius=float(cube_radius),
            cube_yaw=cube_yaw,
            tray_azimuth=float(tray_azimuth),
            tray_radius=float(tray_radius),
        )

        self._weld_rel = None
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.home_qpos
        self.data.qpos[self.grip_qadr] = self.gripper_open
        self.data.ctrl[self.act_ids] = self.clip_cmd(
            np.concatenate([self.home_qpos[:5], [self.gripper_open]])
        )
        self._ctrl = self.data.ctrl[self.act_ids].copy()

        # cube
        cx = cube_radius * math.cos(cube_azimuth)
        cy = cube_radius * math.sin(cube_azimuth)
        q = self.cube_qadr
        self.data.qpos[q : q + 3] = [cx, cy, self.cube_half_z + 1e-3]
        self.data.qpos[q + 3 : q + 7] = [
            math.cos(cube_yaw / 2.0),
            0.0,
            0.0,
            math.sin(cube_yaw / 2.0),
        ]
        if self.cube_dofadr is not None:
            self.data.qvel[self.cube_dofadr : self.cube_dofadr + 6] = 0.0

        # tray
        tx = tray_radius * math.cos(tray_azimuth)
        ty = tray_radius * math.sin(tray_azimuth)
        if self.tray_bid >= 0:
            if self.tray_qadr is not None:
                t = self.tray_qadr
                tz = float(self.model.body_pos[self.tray_bid, 2])
                self.data.qpos[t : t + 3] = [tx, ty, max(tz, 0.0)]
                self.data.qpos[t + 3 : t + 7] = [1.0, 0.0, 0.0, 0.0]
            else:
                self.model.body_pos[self.tray_bid, 0] = tx
                self.model.body_pos[self.tray_bid, 1] = ty
        else:
            self._tray_center = np.array([tx, ty, 0.0])

        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(round(settle_seconds / self.timestep))):
            mujoco.mj_step(self.model, self.data)
        return self.measured_qpos()

    @property
    def grip_qadr(self) -> int:
        return int(self.qadr[5])

    # ------------------------------------------------------------------ step
    def step(self, qpos_cmd6: Sequence[float]) -> np.ndarray:
        """Apply one 10 Hz control tick; return the measured 6-vector."""
        cmd = self.clip_cmd(qpos_cmd6)
        self.data.ctrl[self.act_ids] = cmd
        self._ctrl = cmd
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if self._weld_rel is not None:
                self._apply_weld()
        return self.measured_qpos()

    # ------------------------------------------------ --grasp weld fallback
    def attach_cube(self) -> None:
        """FALLBACK grasp: freeze the cube rigidly to the tool frame.

        This is *not* physics -- it is direct qpos anchoring, used only when
        ``--grasp weld`` is passed because the friction grasp is not holding.
        The cube pose is re-imposed after every physics substep, so contacts
        can no longer pull it out of the jaws.
        """
        site_p = self.data.site_xpos[self.tcp_sid].copy()
        site_R = self.data.site_xmat[self.tcp_sid].reshape(3, 3).copy()
        q = self.cube_qadr
        cube_p = self.data.qpos[q : q + 3].copy()
        cube_quat = self.data.qpos[q + 3 : q + 7].copy()
        rel_p = site_R.T @ (cube_p - site_p)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, np.ascontiguousarray(site_R).reshape(9))
        inv_site = np.zeros(4)
        mujoco.mju_negQuat(inv_site, site_quat)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv_site, cube_quat)
        self._weld_rel = (rel_p, rel_quat)
        self._apply_weld()

    def detach_cube(self) -> None:
        self._weld_rel = None

    def _apply_weld(self) -> None:
        rel_p, rel_quat = self._weld_rel
        site_p = self.data.site_xpos[self.tcp_sid]
        site_R = self.data.site_xmat[self.tcp_sid].reshape(3, 3)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, np.ascontiguousarray(site_R).reshape(9))
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, site_quat, rel_quat)
        q = self.cube_qadr
        self.data.qpos[q : q + 3] = site_p + site_R @ rel_p
        self.data.qpos[q + 3 : q + 7] = new_quat
        if self.cube_dofadr is not None:
            self.data.qvel[self.cube_dofadr : self.cube_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- render
    def render(self, camera: Optional[str] = None) -> np.ndarray:
        """uint8 RGB (H, W, 3).  Already upright -- never flip this."""
        cam = self.camera_name if camera is None else camera
        if isinstance(cam, str) and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            cam = -1  # free camera fallback
        self.renderer.update_scene(self.data, camera=cam)
        return np.ascontiguousarray(self.renderer.render(), dtype=np.uint8)

    def render_world(self) -> np.ndarray:
        """Third-person debug frame (``--world-video``).  Lazily created once."""
        if self._world_renderer is None:
            w, h = self._world_size
            self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), w)
            self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), h)
            self._world_renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = self.world_camera_name if self.world_camera_name else -1
        self._world_renderer.update_scene(self.data, camera=cam)
        return np.ascontiguousarray(self._world_renderer.render(), dtype=np.uint8)

    # --------------------------------------------------------------- success
    def success(self) -> bool:
        """Cube inside the tray footprint and resting above the tray floor."""
        cube = self.cube_pos()
        tray = self.tray_center
        xy_ok = float(np.linalg.norm(cube[:2] - tray[:2])) < self.success_xy_tol
        z_ok = float(cube[2]) > self.tray_floor_top
        return bool(xy_ok and z_ok)

    def set_cube_yaw(self, yaw: float, settle_seconds: float = 0.0) -> None:
        """Re-spawn the cube at the same xy with a new heading, then re-settle."""
        q = self.cube_qadr
        self.data.qpos[q + 2] = self.cube_half_z + 1e-3
        self.data.qpos[q + 3 : q + 7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        if self.cube_dofadr is not None:
            self.data.qvel[self.cube_dofadr : self.cube_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(round(settle_seconds / self.timestep))):
            mujoco.mj_step(self.model, self.data)
        if self.spawn is not None:
            self.spawn.cube_yaw = float(yaw)

    def cube_yaw(self) -> float:
        """Cube heading about the world z axis [rad]."""
        R = self.data.xmat[self.cube_bid].reshape(3, 3)
        return float(math.atan2(R[1, 0], R[0, 0]))

    def close(self) -> None:
        for r in (self.renderer, self._world_renderer):
            try:
                if r is not None:
                    r.close()
            except Exception:
                pass

    def __enter__(self) -> "SO101Env":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# MODEL_NOTES.md sec.11 `pregrasp` arm pose -- a top-down posture over the middle
# of the spawn arc, jaws open.  Used as the rest pose and as the IK nullspace bias.
# wrist_roll is pinned at WRIST_ROLL_LOCK (-pi/2) -- the joint's motor is broken
# on the physical arm and it is taped at that angle, so this is the only pose the
# hand can be in.  The arm angles come from the scene's `pregrasp` keyframe, but
# qpos[4] is deliberately overridden (that keyframe was baked at +1.58437).
DEFAULT_HOME_QPOS = (0.0, 0.000381818, 0.473496, 1.17717, WRIST_ROLL_LOCK, 0.90)


if __name__ == "__main__":  # tiny smoke test
    import sys

    scene = sys.argv[1] if len(sys.argv) > 1 else "sim/mujoco/scene.xml"
    env = SO101Env(scene, render_size=(640, 480))
    print("scene        :", env.scene_path)
    print("tcp site     :", env.tcp_site_name)
    print("cube body    :", env.cube_body_name)
    print("tray body    :", env.tray_body_name)
    print("camera       :", env.camera_name)
    print("timestep     :", env.timestep, "-> substeps/tick:", env.substeps)
    print("cube half    :", env.cube_half_xy, env.cube_half_z)
    q = env.reset(seed=0)
    print("reset qpos   :", np.round(q, 4))
    print("tcp          :", np.round(env.tcp_pos(), 4))
    print("cube         :", np.round(env.cube_pos(), 4))
    print("tray         :", np.round(env.tray_center, 4))
    img = env.render()
    print("frame        :", img.shape, img.dtype, int(img.min()), int(img.max()))
    print("success      :", env.success())
    env.close()
