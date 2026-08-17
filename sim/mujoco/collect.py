#!/usr/bin/env python
"""Scripted SO-101 pick-and-place expert + episode recorder (MuJoCo).

Emits exactly the format ``training/build_dataset.py`` consumes:

    <out>/epNNN.npz   keys "observation.state" (T,6) float32 rad  [MEASURED]
                           "action"            (T,6) float32 rad  [COMMANDED]
    <out>/epNNN.mp4   wrist camera RGB, T frames, 10 fps

Run (from the repo root)::

    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/collect.py \\
        --episodes 40 --out data/sim_raw --seed 0

Design decisions that are load-bearing (each one cost real debugging time):

* The arm command is **frozen** for the whole GRASP phase.  Moving the arm
  while the jaws are closing rakes the cube out of the gripper.
* Every Cartesian move is a **smoothstep ramp**, never a step command.  A step
  command on a kp=998 position servo produces a torque spike that throws the
  cube.
* DART exploration noise is added to the **commanded** arm targets only, and is
  faded to zero across the grasp and release windows.  The gripper channel
  never gets noise.
* The recorded ``action`` is the *noisy command*, the recorded
  ``observation.state`` is the *measured* qpos.  They must differ, otherwise
  ACT learns the identity map (asserted at the end of every episode).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SO101Env  # noqa: E402

# ==========================================================================
# MEASURED SCENE CONSTANTS
# --------------------------------------------------------------------------
# Source: sim/mujoco/MODEL_NOTES.md (Agent A), every value measured by
# sim/mujoco/verify_model.py.  If the scene changes, re-run that verifier and
# update this block -- nothing else in this file hard-codes scene geometry.
# ==========================================================================
DEFAULT_SCENE = "sim/mujoco/scene.xml"

# -- HARD REQUIREMENT (visually verified against the physical robot):
#    wrist_roll is pinned at -pi/2 in EVERY phase and every waypoint.  At -pi/2
#    the claws sit side-by-side horizontally the same way up as the real arm; at
#    +pi/2 the hand is upside down, and at roll 0 / pi the claws stack
#    vertically.  None of those ever happen on the real robot.
#    Consequence: wrist_roll is NOT an IK degree of freedom -- the IK solves
#    over 4 joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) and the
#    jaw-opening axis is whatever that configuration produces.  The cube is
#    therefore spawned with its yaw aligned to the measured jaw axis instead.
#    NOTE: flipping the roll sign rotates the jaw axis by 180 deg, which moves
#    the open-jaw grasp centre to the other side of the tool.  Nothing below
#    hard-codes that: GRASP_JAW_OFFSET is expressed in the SITE frame and the
#    jaw direction is read back from live FK, so the sign takes care of itself.
#    (The scene's `pregrasp`/`pickup` keyframes were baked at +1.58 and their
#    cube placement no longer lines up -- they are never used for alignment.)
WRIST_ROLL_LOCK = -math.pi / 2.0
IK_DOF = (0, 1, 2, 3)   # indices into the 5 arm joints

# -- gripper (MODEL_NOTES sec.5).  LARGER value = MORE OPEN.
GRIPPER_OPEN = 0.90       # 84 mm tip gap, clears the 40 mm cube with margin
GRIPPER_GRIP = 0.00       # commanded past the 0.312 rad stall -> steady squeeze
GRIP_STALL = 0.312        # qpos[5] parks here while holding the cube

# -- tool frame (MODEL_NOTES sec.6).  Offsets are expressed in the *site local*
#    frame: +x = approach axis (points down at the grasp pose), +z = jaw axis.
#    Taken straight off the verified `pregrasp` keyframe:
#        cube_centre = site + R_site @ (GRASP_DEPTH_OFFSET, 0, GRASP_JAW_OFFSET)
GRASP_DEPTH_OFFSET = -0.0066   # site sits 6.6 mm *below* the cube centre
GRASP_JAW_OFFSET = 0.0209      # cube centre sits 21 mm along +jaw from the site
                               # (the moving jaw swings wide; closing re-centres
                               #  the cube onto the site to within 3 mm)

# -- workspace (MODEL_NOTES sec.7/8/9)
CUBE_RADIUS_BAND = (0.185, 0.255)              # m, inside the measured 0.16-0.28
CUBE_AZIMUTH_BAND = (math.radians(-8.0), math.radians(55.0))
# ^ the full kinematic band is +/-60 deg, but the tray lives at -45 deg and its
#   near corner reaches -21.8 deg, so the cube band is clipped to stay clear.
CUBE_YAW_JITTER = math.radians(12.0)   # spawn yaw noise around the jaw axis

# -- heights (MODEL_NOTES sec.9: tray floor top z=0.006, wall top z=0.036)
# The rest pose puts the tool at z~0.013, i.e. ON the table, so EVERY lateral
# move must happen at CARRY_SITE_Z.  A straight line from the rest pose to a
# hover pose sweeps the tool across the table and plows the cube away -- that
# single mistake accounted for every early failure.
CARRY_SITE_Z = 0.070           # site z while transferring.  Measured: with the
                               # approach axis held vertical the IK stays inside
                               # 1.5 mm everywhere in the spawn band up to
                               # z=0.072 and degrades fast above it (10 mm at
                               # z=0.090) as wrist_flex saturates.  Carried cube
                               # bottom sits at 0.057, 21 mm over the tray wall.
RELEASE_SITE_Z = 0.060         # cube centre ends at ~0.067, clear of the 0.036
                               # wall top + 0.02 cube half-extent
TRAY_FLOOR_TOP = 0.006
SUCCESS_XY_TOL = 0.06

# -- rest pose.  The scene's own neutral poses put the tool ON the table in the
#    middle of the spawn arc, so `reset()` could settle the arm straight into a
#    freshly spawned cube and shove it 5 cm before the episode even started.
#    The rest pose is therefore solved at carry height, in the gap between the
#    cube band (-8..+55 deg) and the tray (-45 deg).
HOME_SITE_RADIUS = 0.200
HOME_SITE_AZIMUTH = math.radians(-20.0)

# -- phase lengths in 10 Hz control ticks
T_RISE = 6
T_TRAVERSE = 12
T_DESCEND = 10
T_PRESETTLE = 3
T_GRASP_RAMP = 7       # 0.7 s of jaw closing, arm frozen
T_GRASP_SETTLE = 4     # 0.4 s of settling, arm frozen
T_LIFT = 12
T_TRANSFER = 20
T_LOWER = 8
T_RELEASE_RAMP = 5
T_RELEASE_SETTLE = 4
T_RETREAT = 14
T_FINAL_SETTLE = 6

# -- DART noise: sigma = 0.5 % of each arm joint's range, gripper channel never
NOISE_FRACTION = 0.005

# -- IK
IK_LAMBDA = 0.05
IK_POS_TOL = 0.0015    # 1.5 mm, well inside the required 3 mm
IK_ROT_TOL = 0.03
IK_MAX_ITERS = 160
IK_ROT_WEIGHT = 0.30
IK_NULLSPACE_GAIN = 0.02
IK_MAX_STEP = 0.20

# -- video
VIDEO_FPS = 10
DEFAULT_RENDER_SIZE = (640, 480)   # (W, H) -- matches build_dataset.py's
                                   # declared feature shape (480, 640, 3) so no
                                   # resampling happens downstream.


# ==========================================================================
# IK
# ==========================================================================
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])


class ArmIK:
    """Damped-least-squares IK on the TCP site.

    Primary task: site position via ``mujoco.mj_jacSite`` (jacp), damping
    ``IK_LAMBDA``, iterated to well under the required 3 mm.

    Stacked underneath at weight ``IK_ROT_WEIGHT``: keep the tool **approach
    axis** (the site's local +X) pointing straight down.  Without it the solver
    wanders into poses where the fixed jaw rakes the cube off the table.  Only
    the approach *direction* is constrained -- spin about it is set by
    wrist_roll, which is pinned at ``WRIST_ROLL_LOCK`` and excluded from the
    solve (see IK_DOF).

    A posture bias toward ``home`` is projected into whatever nullspace is left.
    """

    def __init__(self, env: SO101Env, home: Optional[np.ndarray] = None) -> None:
        self.model = env.model
        self.data = mujoco.MjData(env.model)
        self.sid = env.tcp_sid
        self.qadr = env.arm_qadr
        self.dofadr = env.arm_dofadr[list(IK_DOF)]
        self.lo = env.jnt_range[:5, 0].copy()
        self.hi = env.jnt_range[:5, 1].copy()
        self.home = np.asarray(env.home_qpos[:5] if home is None else home, dtype=float)
        self.ndof = len(IK_DOF)
        self.idx = list(IK_DOF)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float).copy()
        q[4] = WRIST_ROLL_LOCK
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return (
            self.data.site_xpos[self.sid].copy(),
            self.data.site_xmat[self.sid].reshape(3, 3).copy(),
        )

    @staticmethod
    def jaw_yaw_of(R: np.ndarray) -> float:
        """World heading of the jaw-opening axis (the site's local +Z)."""
        return float(math.atan2(R[1, 2], R[0, 2]))

    def solve(self, pos: np.ndarray, q_seed: np.ndarray) -> Tuple[np.ndarray, float]:
        q = np.clip(np.asarray(q_seed, dtype=float).copy(), self.lo, self.hi)
        q[4] = WRIST_ROLL_LOCK
        eyed = np.eye(self.ndof)
        eye6 = np.eye(6)
        for _ in range(IK_MAX_ITERS):
            p, R = self.fk(q)
            e_pos = pos - p
            # rotation-vector error that swings the approach axis onto -Z; its
            # component along the approach axis is identically zero, so nothing
            # here fights the pinned wrist_roll.
            e_rot = np.cross(R[:, 0], APPROACH_DOWN)
            if np.linalg.norm(e_pos) < IK_POS_TOL and np.linalg.norm(e_rot) < IK_ROT_TOL:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.sid)
            J = np.vstack(
                [self._jacp[:, self.dofadr], IK_ROT_WEIGHT * self._jacr[:, self.dofadr]]
            )
            e = np.concatenate([e_pos, IK_ROT_WEIGHT * e_rot])
            JJti = np.linalg.inv(J @ J.T + (IK_LAMBDA ** 2) * eye6)
            Jdls = J.T @ JJti
            dq = Jdls @ e
            dq += (eyed - Jdls @ J) @ (IK_NULLSPACE_GAIN * (self.home[self.idx] - q[self.idx]))
            n = float(np.linalg.norm(dq))
            if n > IK_MAX_STEP:
                dq *= IK_MAX_STEP / n
            q[self.idx] = q[self.idx] + dq
            q = np.clip(q, self.lo, self.hi)
            q[4] = WRIST_ROLL_LOCK
        p, _ = self.fk(q)
        return q, float(np.linalg.norm(pos - p))

    def grasp_pose(self, cube_xyz: np.ndarray, q_seed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Site target + joint solution + jaw yaw for grasping ``cube_xyz``.

        The open-jaw offset is expressed along the jaw axis, but the jaw axis is
        an *output* of the pinned-roll IK -- so iterate twice (it converges
        immediately; the axis barely moves over a 2 cm target shift).
        """
        jaw_yaw = math.atan2(cube_xyz[1], cube_xyz[0]) + math.pi / 2.0
        target = q = None
        for _ in range(3):
            target = grasp_site_target(cube_xyz, jaw_yaw)
            q, _err = self.solve(target, q_seed if q is None else q)
            _p, R = self.fk(q)
            new_yaw = self.jaw_yaw_of(R)
            if abs(math.atan2(math.sin(new_yaw - jaw_yaw), math.cos(new_yaw - jaw_yaw))) < 1e-3:
                jaw_yaw = new_yaw
                break
            jaw_yaw = new_yaw
        target = grasp_site_target(cube_xyz, jaw_yaw)
        return target, q, jaw_yaw


# ==========================================================================
# trajectory helpers
# ==========================================================================
def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def interp_waypoint(p_from: np.ndarray, p_to: np.ndarray, s: float, polar: bool) -> np.ndarray:
    """Interpolate two tool positions, either as a straight line or as an arc.

    Long lateral moves must be **polar** (interpolate radius + azimuth, not x/y).
    A straight chord between two points on the working arc dips toward the base,
    which both drags the arm through a badly conditioned pose and shakes the
    cube out of the friction grasp part-way through the transfer.
    """
    if not polar:
        return p_from + s * (p_to - p_from)
    r0, r1 = math.hypot(p_from[0], p_from[1]), math.hypot(p_to[0], p_to[1])
    a0, a1 = math.atan2(p_from[1], p_from[0]), math.atan2(p_to[1], p_to[0])
    da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))   # shortest way round
    r = r0 + s * (r1 - r0)
    a = a0 + s * da
    return np.array([r * math.cos(a), r * math.sin(a), p_from[2] + s * (p_to[2] - p_from[2])])


def grasp_site_target(cube_xyz: np.ndarray, jaw_yaw: float) -> np.ndarray:
    """Site position that puts the cube where the open jaws can swallow it.

    ``cube = site + depth*(-approach) + jaw_offset*jaw_axis`` with the approach
    axis pointing straight down, so inverting it is just two world offsets.
    """
    jaw = np.array([math.cos(jaw_yaw), math.sin(jaw_yaw), 0.0])
    cube_xyz = np.asarray(cube_xyz, dtype=float)
    return cube_xyz - np.array([0.0, 0.0, -GRASP_DEPTH_OFFSET]) - GRASP_JAW_OFFSET * jaw


# ==========================================================================
# episode
# ==========================================================================
@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    frames: List[np.ndarray]
    world_frames: List[np.ndarray]
    success: bool
    spawn: dict
    max_ik_err: float
    diag: dict


class Recorder:
    """Runs the scripted expert for one episode and buffers the tensors."""

    def __init__(self, env: SO101Env, ik: ArmIK, rng: np.random.Generator, args) -> None:
        self.env = env
        self.ik = ik
        self.rng = rng
        self.args = args
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.frames: List[np.ndarray] = []
        self.world_frames: List[np.ndarray] = []
        # sigma = 0.5 % of each joint's range.  Zero on wrist_roll (pinned at
        # WRIST_ROLL_LOCK -- it is not an IK dof and must not wander) and the
        # gripper channel never sees noise at all (it is not in this vector).
        rng_span = env.jnt_range[:5, 1] - env.jnt_range[:5, 0]
        self.noise_sigma = np.zeros(5)
        self.noise_sigma[list(IK_DOF)] = NOISE_FRACTION * rng_span[list(IK_DOF)]
        self.max_ik_err = 0.0

    def tick(self, arm_cmd: np.ndarray, grip_cmd: float, noise_scale: float) -> None:
        """One 10 Hz control tick: observe, command, step."""
        env = self.env
        # observe BEFORE acting -> (state_t, action_t) pairs
        self.states.append(env.measured_qpos())
        self.frames.append(env.render())
        if self.args.world_video:
            self.world_frames.append(env.render_world())

        noisy = np.asarray(arm_cmd, dtype=float).copy()
        if noise_scale > 0.0 and not self.args.no_noise:
            noisy = noisy + self.rng.normal(0.0, self.noise_sigma * noise_scale)
        cmd = np.concatenate([noisy, [float(grip_cmd)]])
        cmd = env.clip_cmd(cmd)
        self.actions.append(cmd.astype(np.float32))
        env.step(cmd)

    def ramp(
        self,
        p_from: np.ndarray,
        p_to: np.ndarray,
        q_seed: np.ndarray,
        n: int,
        grip_cmd: float,
        noise_scale: float = 1.0,
        noise_to: Optional[float] = None,
        polar: bool = False,
    ) -> np.ndarray:
        """Smoothstep ramp between two tool positions; returns the last
        noise-free arm command.

        ``noise_to`` linearly fades the DART noise across the phase, which is
        how the noise reaches exactly zero by the time the jaws start moving
        (a step change in noise amplitude is itself a disturbance).
        ``polar`` follows the working arc instead of a straight chord.
        """
        q = np.asarray(q_seed, dtype=float).copy()
        end = noise_scale if noise_to is None else noise_to
        for i in range(n):
            u = (i + 1) / float(n)
            q, err = self.ik.solve(interp_waypoint(p_from, p_to, smoothstep(u), polar), q)
            self.max_ik_err = max(self.max_ik_err, err)
            self.tick(q, grip_cmd, noise_scale + u * (end - noise_scale))
        return q

    def hold(self, arm_cmd: np.ndarray, grip_cmd: float, n: int, noise_scale: float = 0.0) -> None:
        for _ in range(n):
            self.tick(arm_cmd, grip_cmd, noise_scale)


def run_episode(env: SO101Env, ik: ArmIK, args, seed: int, cube_azimuth: float) -> Episode:
    rng = np.random.default_rng(seed)
    cube_radius = float(rng.uniform(*CUBE_RADIUS_BAND))
    env.reset(
        seed=seed,
        cube_azimuth=cube_azimuth,
        cube_radius=cube_radius,
        tray_azimuth=None,
        settle_seconds=0.5,
    )

    q_home = np.asarray(env.home_qpos[:5], dtype=float).copy()
    q_home[4] = WRIST_ROLL_LOCK

    # With wrist_roll pinned the jaw axis is an output of the arm pose, so the
    # cube is re-spawned with its yaw squared to that axis (+/- jitter) rather
    # than the other way round.
    cube = env.cube_pos()
    _t, _q, jaw_yaw = ik.grasp_pose(cube, q_home)
    env.set_cube_yaw(jaw_yaw + float(rng.uniform(-CUBE_YAW_JITTER, CUBE_YAW_JITTER)), settle_seconds=0.3)

    rec = Recorder(env, ik, rng, args)
    cube = env.cube_pos()
    tray = env.tray_center
    grasp_p, _q, jaw_yaw = ik.grasp_pose(cube, q_home)
    hover_p = np.array([grasp_p[0], grasp_p[1], CARRY_SITE_Z])
    over_tray_p = np.array([tray[0], tray[1], CARRY_SITE_Z])
    drop_p = np.array([tray[0], tray[1], RELEASE_SITE_Z])

    q = q_home.copy()
    start_p, _ = ik.fk(q)
    rise_p = np.array([start_p[0], start_p[1], CARRY_SITE_Z])

    # 1a. rise straight up out of the table plane BEFORE moving sideways ----
    q = rec.ramp(start_p, rise_p, q, T_RISE, GRIPPER_OPEN)
    # 1b. traverse to the hover pose at carry height ------------------------
    q = rec.ramp(rise_p, hover_p, q, T_TRAVERSE, GRIPPER_OPEN, polar=True)
    # 2. descend to grasp height -------------------------------------------
    q = rec.ramp(hover_p, grasp_p, q, T_DESCEND, GRIPPER_OPEN, noise_scale=1.0, noise_to=0.0)
    # 3. let the servos catch up before touching the cube -------------------
    rec.hold(q, GRIPPER_OPEN, T_PRESETTLE)
    # 4. GRASP: arm command FROZEN, jaws ramp closed over 0.7 s, settle 0.4 s
    for i in range(T_GRASP_RAMP):
        g = GRIPPER_OPEN + smoothstep((i + 1) / float(T_GRASP_RAMP)) * (GRIPPER_GRIP - GRIPPER_OPEN)
        rec.tick(q, g, 0.0)
    rec.hold(q, GRIPPER_GRIP, T_GRASP_SETTLE)
    diag = {
        # qpos[5] parks at GRIP_STALL while an object is between the jaws; if it
        # walks all the way down to GRIPPER_GRIP the jaws closed on nothing.
        "grasped": bool(env.measured_qpos()[5] > 0.5 * (GRIPPER_GRIP + GRIP_STALL)),
        "grip_after_close": float(env.measured_qpos()[5]),
        "cube_after_close": env.cube_pos().round(4).tolist(),
        "grasp_target": np.asarray(grasp_p).round(4).tolist(),
        "tcp_after_close": env.tcp_pos().round(4).tolist(),
    }
    if args.grasp == "weld":
        env.attach_cube()  # FALLBACK: kinematic anchoring, see env.attach_cube
    # 5. RAMPED lift (smoothstep, never a step command) ---------------------
    q = rec.ramp(grasp_p, hover_p, q, T_LIFT, GRIPPER_GRIP, noise_scale=0.0, noise_to=1.0)
    # 6. transfer over the tray ---------------------------------------------
    diag["cube_after_lift"] = env.cube_pos().round(4).tolist()
    q = rec.ramp(hover_p, over_tray_p, q, T_TRANSFER, GRIPPER_GRIP, polar=True)
    # 7. lower to release height --------------------------------------------
    q = rec.ramp(over_tray_p, drop_p, q, T_LOWER, GRIPPER_GRIP, noise_scale=1.0, noise_to=0.0)
    # 8. release -------------------------------------------------------------
    diag["cube_before_release"] = env.cube_pos().round(4).tolist()
    if args.grasp == "weld":
        env.detach_cube()
    for i in range(T_RELEASE_RAMP):
        g = GRIPPER_GRIP + smoothstep((i + 1) / float(T_RELEASE_RAMP)) * (GRIPPER_OPEN - GRIPPER_GRIP)
        rec.tick(q, g, 0.0)
    rec.hold(q, GRIPPER_OPEN, T_RELEASE_SETTLE)
    # 9. retreat --------------------------------------------------------------
    q = rec.ramp(drop_p, over_tray_p, q, T_RETREAT, GRIPPER_OPEN, noise_scale=0.0, noise_to=1.0)
    rec.hold(q, GRIPPER_OPEN, T_FINAL_SETTLE)

    states = np.stack(rec.states).astype(np.float32)
    actions = np.stack(rec.actions).astype(np.float32)
    spawn = env.spawn.as_dict()
    spawn["jaw_yaw"] = float(jaw_yaw)
    diag["cube_final"] = env.cube_pos().round(4).tolist()
    diag["tray_center"] = env.tray_center.round(4).tolist()
    return Episode(
        states=states,
        actions=actions,
        frames=rec.frames,
        world_frames=rec.world_frames,
        success=env.success(),
        spawn=spawn,
        max_ik_err=rec.max_ik_err,
        diag=diag,
    )


# ==========================================================================
# writing
# ==========================================================================
def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = VIDEO_FPS) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,   # safe for sizes not divisible by 16 (e.g. 224)
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
        quality=8,
    )
    try:
        for f in frames:
            writer.append_data(np.ascontiguousarray(f, dtype=np.uint8))
    finally:
        writer.close()


def write_episode(out: Path, idx: int, ep: Episode, world_dir: Optional[Path]) -> Tuple[Path, Path]:
    stem = out / f"ep{idx:03d}"
    npz_path = stem.with_suffix(".npz")
    mp4_path = stem.with_suffix(".mp4")
    np.savez(
        npz_path,
        **{"observation.state": ep.states, "action": ep.actions},
    )
    write_video(mp4_path, ep.frames)
    if world_dir is not None and ep.world_frames:
        world_dir.mkdir(parents=True, exist_ok=True)
        write_video(world_dir / f"ep{idx:03d}.mp4", ep.world_frames)
    return npz_path, mp4_path


# ==========================================================================
# main
# ==========================================================================
def parse_render_size(text: str) -> Tuple[int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("render size must be WxH, e.g. 640x480")
    return int(parts[0]), int(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, required=True, help="number of episodes to KEEP")
    ap.add_argument("--out", required=True, help="output directory for epNNN.npz / epNNN.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--keep-failures", action="store_true", help="record failed episodes too")
    ap.add_argument(
        "--grasp",
        choices=("friction", "weld"),
        default="friction",
        help="'friction' = real contact physics (default). "
        "'weld' = FALLBACK, kinematically anchors the cube to the tool at grasp.",
    )
    ap.add_argument("--world-video", action="store_true", help="also write <out>/world/epNNN.mp4")
    ap.add_argument("--render-size", type=parse_render_size, default=DEFAULT_RENDER_SIZE,
                    help="WxH of the wrist video (default 640x480, matching build_dataset.py)")
    ap.add_argument("--no-noise", action="store_true", help="disable DART noise (debugging only)")
    ap.add_argument("--verbose", action="store_true", help="print per-episode grasp diagnostics")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="attempt budget; default 0 = 4x --episodes")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world_dir = out / "world" if args.world_video else None
    max_attempts = args.max_attempts if args.max_attempts > 0 else max(4 * args.episodes, 8)

    # The constant block at the top of this file is the single source of truth
    # for task geometry; the env takes all of it as arguments.
    env = SO101Env(
        args.scene,
        render_size=args.render_size,
        gripper_open=GRIPPER_OPEN,
        gripper_grip=GRIPPER_GRIP,
        spawn_radius=CUBE_RADIUS_BAND,
        spawn_azimuth=CUBE_AZIMUTH_BAND,
        tray_floor_offset=TRAY_FLOOR_TOP,
        success_xy_tol=SUCCESS_XY_TOL,
    )
    ik = ArmIK(env)   # nullspace bias stays on the scene's neutral arm posture

    # Solve the elevated rest pose and make it the env's reset posture.
    home_site = np.array([
        HOME_SITE_RADIUS * math.cos(HOME_SITE_AZIMUTH),
        HOME_SITE_RADIUS * math.sin(HOME_SITE_AZIMUTH),
        CARRY_SITE_Z,
    ])
    q_rest, rest_err = ik.solve(home_site, np.asarray(env.home_qpos[:5], dtype=float))
    if rest_err > 0.003:
        raise SystemExit(f"rest pose unreachable ({rest_err * 1000:.1f} mm) -- check HOME_SITE_*")
    env.home_qpos = np.concatenate([q_rest, [GRIPPER_OPEN]])

    print(
        f"scene={env.scene_path} tcp={env.tcp_site_name} cube={env.cube_body_name} "
        f"tray={env.tray_body_name} cam={env.camera_name}\n"
        f"timestep={env.timestep}s -> {env.substeps} substeps per 10 Hz tick; "
        f"render={env.render_width}x{env.render_height}; grasp={args.grasp}"
    )

    lo, hi = CUBE_AZIMUTH_BAND
    failures: List[dict] = []
    manifest: List[dict] = []
    kept = 0
    attempts = 0

    while kept < args.episodes and attempts < max_attempts:
        # Binned spawns: episode i owns bin i of the azimuth band; retries stay
        # in the same bin so coverage of the arc stays uniform.
        b_lo = lo + (hi - lo) * kept / args.episodes
        b_hi = lo + (hi - lo) * (kept + 1) / args.episodes
        seed = args.seed * 100003 + attempts * 97 + kept
        az = float(np.random.default_rng(seed ^ 0x5EED).uniform(b_lo, b_hi))
        attempts += 1

        ep = run_episode(env, ik, args, seed, az)

        dstate = float(np.abs(ep.actions - ep.states).mean())
        assert dstate > 1e-4, (
            f"action == observation.state (mean |diff| = {dstate:.2e}); ACT would "
            "learn the identity map.  Check the DART noise / command path."
        )
        assert np.all(ep.actions[:, 4] == np.float32(WRIST_ROLL_LOCK)), (
            "wrist_roll left its lock in a commanded action; the hand would roll "
            "away from the orientation the physical arm has."
        )

        tag = "OK " if ep.success else "FAIL"
        if ep.success or args.keep_failures:
            npz_path, _ = write_episode(out, kept, ep, world_dir)
            manifest.append({
                "episode": kept, "file": npz_path.name, "frames": int(len(ep.states)),
                "success": bool(ep.success), "attempt": attempts, **ep.spawn,
            })
            kept += 1
        if not ep.success:
            failures.append({"attempt": attempts, **ep.spawn, "diag": ep.diag})
        if args.verbose:
            print("      " + json.dumps(ep.diag))

        rate = (attempts - len(failures)) / attempts
        print(
            f"[{tag}] attempt {attempts:3d} -> ep{kept - 1 if (ep.success or args.keep_failures) else -1:03d} "
            f"T={len(ep.states):3d} az={math.degrees(az):+6.1f}deg "
            f"|a-s|={dstate:.4f} ik_err={ep.max_ik_err * 1000:.1f}mm "
            f"running_success={rate:.1%}"
        )

    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    (out / "episodes.json").write_text(json.dumps(manifest, indent=2))
    env.close()

    n_ok = attempts - len(failures)
    print(
        f"\nwrote {kept} episode(s) to {out}\n"
        f"attempts={attempts}  successes={n_ok}  failures={len(failures)}  "
        f"final success rate = {n_ok / max(attempts, 1):.1%}"
    )
    if kept < args.episodes:
        print(f"WARNING: only {kept}/{args.episodes} episodes kept (attempt budget exhausted)")
        sys.exit(1)


if __name__ == "__main__":
    main()
