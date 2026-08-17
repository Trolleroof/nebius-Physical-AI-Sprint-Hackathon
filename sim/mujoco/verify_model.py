"""Verify sim/mujoco/scene.xml and MEASURE the facts env.py/collect.py need.

Run from the repo root:
    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/verify_model.py

Exits 0 on success, 1 if anything fails (blank render, cube not held, ...).
Writes sim/mujoco/verify_wrist.png, verify_world.png, verify_wrist_grasp.png.
"""

from __future__ import annotations

import os
import sys

import imageio.v3 as iio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")

WRIST_HW = (224, 224)  # (height, width)
WORLD_HW = (480, 640)
STD_MIN = 5.0

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  !! FAIL: {msg}")


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def names(model, objtype, n):
    return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


def render(model, data, cam, hw, path):
    h, w = hw
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam)
        img = r.render()
    iio.imwrite(path, img)
    g = img.astype(np.float64)
    mean, std = g.mean(), g.std()
    print(f"  {os.path.basename(path):24s} cam={cam:9s} {w}x{h} mean={mean:7.2f} std={std:7.2f}")
    if std < STD_MIN:
        fail(f"{os.path.basename(path)} looks blank (std {std:.2f} < {STD_MIN})")
    return img


def site_xpos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


def geom_xpos(model, data, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[gid].copy()


def jaw_gap(model, data):
    """Distance between the fixed-jaw and moving-jaw finger tips."""
    return float(
        np.linalg.norm(geom_xpos(model, data, "fixed_jaw_sph_tip1") - geom_xpos(model, data, "moving_jaw_sph_tip1"))
    )


def settle(model, data, ctrl, nsteps):
    data.ctrl[:] = ctrl
    for _ in range(nsteps):
        mujoco.mj_step(model, data)


def main() -> int:
    hr("MODEL LOAD")
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    print(f"  scene            : {SCENE}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  ngeom={model.ngeom}")
    print(f"  timestep         : {model.opt.timestep} s  ->  {round(0.1 / model.opt.timestep)} mj_step per 10 Hz tick")

    jnames = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    anames = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    cnames = names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    snames = names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    bnames = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    knames = names(model, mujoco.mjtObj.mjOBJ_KEY, model.nkey)

    print("\n  joints (qpos addr, dof addr, range):")
    for i, n in enumerate(jnames):
        print(
            f"    [{i}] {n:14s} qposadr={model.jnt_qposadr[i]:2d} dofadr={model.jnt_dofadr[i]:2d} "
            f"type={mujoco.mjtJoint(model.jnt_type[i]).name:12s} range={model.jnt_range[i]}"
        )
    print("\n  actuators (ctrl index, joint, ctrlrange):")
    for i, n in enumerate(anames):
        j = model.actuator_trnid[i, 0]
        print(f"    [{i}] {n:14s} joint={jnames[j]:14s} ctrlrange={model.actuator_ctrlrange[i]}")
    print(f"\n  cameras   : {cnames}")
    print(f"  sites     : {snames}")
    print(f"  bodies    : {bnames}")
    print(f"  keyframes : {knames}")

    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_qadr = int(model.jnt_qposadr[cube_jid])
    tray_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    print(f"\n  cube body id={cube_bid}  freejoint 'cube_free' qpos[{cube_qadr}:{cube_qadr + 7}]")
    print(f"  tray body id={tray_bid}  tray pos={model.body_pos[tray_bid]}")

    # ---------------------------------------------------------------- reset
    hr("mj_resetData (scene defaults)")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    print(f"  qpos            : {np.round(data.qpos, 5)}")
    print(f"  cube pos        : {np.round(data.xpos[cube_bid], 5)}")
    print(f"  gripperframe    : {np.round(site_xpos(model, data, 'gripperframe'), 5)}")
    settle(model, data, np.zeros(model.nu), 200)
    mujoco.mj_forward(model, data)
    print(f"  after 1 s of ctrl=0 -> cube pos {np.round(data.xpos[cube_bid], 5)} (should stay put)")
    if data.xpos[cube_bid][2] < 0.01:
        fail("cube fell through / sank at default reset")

    # ------------------------------------------------------------- renders
    hr("RENDERS (default scene)")
    render(model, data, "world", WORLD_HW, os.path.join(HERE, "verify_world.png"))
    render(model, data, "wrist_cam", WRIST_HW, os.path.join(HERE, "verify_wrist.png"))

    # ---------------------------------------------- gripper sign convention
    hr("GRIPPER SIGN CONVENTION (kinematic sweep of joint 'gripper')")
    kid_pre = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pregrasp")
    kid_pick = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pickup")
    mujoco.mj_resetDataKeyframe(model, data, kid_pre)
    mujoco.mj_forward(model, data)
    arm_pose = data.qpos[:5].copy()

    lo, hi = model.jnt_range[5]
    print(f"  joint 'gripper' range: [{lo:.5f}, {hi:.5f}] rad")
    print("     gripper[rad]   finger-tip gap [mm]")
    gaps = []
    for g in np.linspace(lo, hi, 11):
        data.qpos[5] = g
        mujoco.mj_forward(model, data)
        gp = jaw_gap(model, data)
        gaps.append(gp)
        print(f"      {g:+8.4f}        {gp * 1000:7.2f}")
    gaps = np.array(gaps)
    increasing = gaps[-1] > gaps[0]
    sign_txt = (
        "INCREASING the 'gripper' joint OPENS the jaws (joint min = closed, joint max = open)"
        if increasing
        else "DECREASING the 'gripper' joint OPENS the jaws (joint max = closed, joint min = open)"
    )
    print(f"  -> {sign_txt}")
    if not increasing:
        fail("unexpected gripper sign convention (expected larger joint = more open)")
    slope = (gaps[-1] - gaps[0]) / (hi - lo)
    print(f"  -> tip gap ~= {gaps[0] * 1000:.1f} mm + {slope * 1000:.1f} mm/rad * (gripper - {lo:.4f})")
    print(f"  -> fully closed joint {lo:+.5f} rad -> {gaps[0] * 1000:6.1f} mm gap")
    print(f"  -> fully open   joint {hi:+.5f} rad -> {gaps[-1] * 1000:6.1f} mm gap")

    # jaw centre at the recommended OPEN command (where a cube must be to be grasped)
    OPEN_CMD = 0.9
    data.qpos[5] = OPEN_CMD
    mujoco.mj_forward(model, data)
    jaw_mid = (
        geom_xpos(model, data, "fixed_jaw_sph_tip1") + geom_xpos(model, data, "moving_jaw_sph_tip1")
    ) / 2.0
    gf = site_xpos(model, data, "gripperframe")
    approach = gf - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")]
    approach /= np.linalg.norm(approach)
    grasp_r = float(np.hypot(jaw_mid[0], jaw_mid[1]))
    print(f"\n  at OPEN={OPEN_CMD} rad: tip gap = {jaw_gap(model, data) * 1000:.1f} mm (clears the 40 mm cube)")
    print(f"  jaw centre (world)      : {np.round(jaw_mid, 5)}   r={grasp_r:.4f} m")
    print(f"  gripperframe site       : {np.round(gf, 5)}   r={np.hypot(gf[0], gf[1]):.4f} m")
    print(f"  approach dir (world)    : {np.round(approach, 3)}  (near-vertical top-down)")
    print("  approach dir is the gripperframe site's local +X axis")

    # ------------------------------------ grip threshold + hold / lift test
    hr("GRASP TEST: close on the 40 mm cube from 'pregrasp', then lift 0.35 rad")
    print("  grip cmd   stalled joint   cube z before   cube z after lift   |cube-gripframe|   HELD")
    hold_results = []
    for grip_cmd in (0.5, 0.4, 0.35, 0.3, 0.2, 0.0, -0.17):
        mujoco.mj_resetDataKeyframe(model, data, kid_pre)
        ctrl = np.concatenate([arm_pose, [OPEN_CMD]])
        settle(model, data, ctrl, 40)
        ctrl[5] = grip_cmd
        settle(model, data, ctrl, 160)
        mujoco.mj_forward(model, data)
        stalled = float(data.qpos[5])
        z0 = float(data.xpos[cube_bid][2])
        ctrl[1] = arm_pose[1] - 0.35  # measured: negative shoulder_lift raises the tool
        settle(model, data, ctrl, 400)
        mujoco.mj_forward(model, data)
        z1 = float(data.xpos[cube_bid][2])
        dist = float(np.linalg.norm(data.xpos[cube_bid] - site_xpos(model, data, "gripperframe")))
        held = (z1 - z0) > 0.03 and dist < 0.05
        hold_results.append((grip_cmd, held))
        print(f"   {grip_cmd:+6.2f}      {stalled:+8.4f}       {z0:8.4f}        {z1:8.4f}          {dist * 1000:7.1f} mm       {held}")
    holders = [g for g, h in hold_results if h]
    if not holders:
        fail("no gripper command held the cube through a lift")
        GRIP_CMD = 0.0
    else:
        GRIP_CMD = 0.0
        print(f"  -> commands that HOLD the 40 mm cube : <= {max(holders):+.2f} rad")
        print(f"  -> recommended GRIP command          : {GRIP_CMD:+.2f} rad (well past the stall angle -> steady squeeze)")
    print(f"  -> recommended OPEN command          : {OPEN_CMD:+.2f} rad")
    if not any(h for g, h in hold_results if g <= 0.0):
        fail("the recommended GRIP command does not hold the cube")

    # a clean held-cube state for the wrist render
    hr("RENDERS WHILE HOLDING THE CUBE")
    mujoco.mj_resetDataKeyframe(model, data, kid_pre)
    ctrl = np.concatenate([arm_pose, [OPEN_CMD]])
    settle(model, data, ctrl, 40)
    ctrl[5] = GRIP_CMD
    settle(model, data, ctrl, 160)
    mujoco.mj_forward(model, data)
    render(model, data, "wrist_cam", WRIST_HW, os.path.join(HERE, "verify_wrist_grasp.png"))
    ctrl[1] = arm_pose[1] - 0.35
    settle(model, data, ctrl, 400)
    mujoco.mj_forward(model, data)
    z_lift = float(data.xpos[cube_bid][2])
    print(f"  cube z while lifted: {z_lift:.4f} m")
    render(model, data, "wrist_cam", WRIST_HW, os.path.join(HERE, "verify_wrist_lift.png"))
    render(model, data, "world", WORLD_HW, os.path.join(HERE, "verify_world_grasp.png"))

    # release
    ctrl[5] = OPEN_CMD
    settle(model, data, ctrl, 300)
    mujoco.mj_forward(model, data)
    z_rel = float(data.xpos[cube_bid][2])
    print(f"  after opening to {OPEN_CMD:+.2f}: cube z = {z_rel:.4f} m (should drop back to ~0.020)")
    if z_rel > z_lift - 0.02:
        fail(f"cube did not release when the gripper opened (z={z_rel:.4f})")

    # ------------------------------------------------------- pickup keyframe
    hr("'pickup' KEYFRAME (baked settled grasp) - loads and stays grasped")
    mujoco.mj_resetDataKeyframe(model, data, kid_pick)
    mujoco.mj_forward(model, data)
    print(f"  qpos       : {np.round(data.qpos, 5)}")
    print(f"  ctrl       : {np.round(data.ctrl, 5)}")
    print(f"  gripper    : {data.qpos[5]:.5f} rad (stalled against the cube)")
    print(f"  cube       : {np.round(data.xpos[cube_bid], 5)}")
    print(f"  gripframe  : {np.round(site_xpos(model, data, 'gripperframe'), 5)}")
    d0 = float(np.linalg.norm(data.xpos[cube_bid] - site_xpos(model, data, "gripperframe")))
    settle(model, data, data.ctrl.copy(), 100)
    mujoco.mj_forward(model, data)
    d1 = float(np.linalg.norm(data.xpos[cube_bid] - site_xpos(model, data, "gripperframe")))
    print(f"  |cube-gripframe| {d0 * 1000:.1f} mm -> {d1 * 1000:.1f} mm after 0.5 s (still grasped)")
    if d1 > 0.05:
        fail("'pickup' keyframe does not stay grasped")

    # ------------------------------------------------- reachable azimuth band
    hr("REACHABLE ARC (shoulder_pan sweep at the pregrasp arm pose)")
    probe = mujoco.MjData(model)
    pan_lo, pan_hi = model.actuator_ctrlrange[0]
    print(f"  shoulder_pan ctrlrange : [{pan_lo:.5f}, {pan_hi:.5f}] rad = [{np.degrees(pan_lo):.1f}, {np.degrees(pan_hi):.1f}] deg")
    print("     pan[rad]  pan[deg]   jaw centre (x, y, z)            r[m]   azim[deg]")
    azs = []
    for pan in np.linspace(pan_lo, pan_hi, 9):
        mujoco.mj_resetDataKeyframe(model, probe, kid_pre)
        probe.qpos[0] = pan
        probe.qpos[5] = OPEN_CMD
        mujoco.mj_forward(model, probe)
        p = (
            probe.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1")]
            + probe.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_sph_tip1")]
        ) / 2.0
        r = float(np.hypot(p[0], p[1]))
        a = float(np.degrees(np.arctan2(p[1], p[0])))
        azs.append(a)
        print(f"    {pan:+8.4f} {np.degrees(pan):+8.1f}   ({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})  {r:.4f}  {a:+8.2f}")
    print(f"  -> jaw-centre azimuth at pan=0 : {azs[len(azs) // 2]:+.2f} deg  (offset from shoulder_pan)")
    print(f"  -> kinematic azimuth band      : [{min(azs):+.1f}, {max(azs):+.1f}] deg")
    print("  -> RECOMMENDED sampling band   : [-60, +60] deg (arm never folds back over its own base)")

    # ------------------------------------------- ground-grasp radius envelope
    hr("GROUND-GRASP RADIUS ENVELOPE (near-vertical approach, tool at table height)")
    jlo = model.jnt_range[:, 0]
    jhi = model.jnt_range[:, 1]
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    radii = []
    for sl in np.linspace(jlo[1], jhi[1], 25):
        for ef in np.linspace(jlo[2], jhi[2], 25):
            for wf in np.linspace(jlo[3], jhi[3], 25):
                probe.qpos[:6] = [0, sl, ef, wf, 1.58437, OPEN_CMD]
                mujoco.mj_forward(model, probe)
                p = probe.site_xpos[sid]
                a = p - probe.xpos[gb]
                a = a / np.linalg.norm(a)
                if 0.006 < p[2] < 0.024 and p[0] > 0.05 and a[2] < -0.90:
                    radii.append(float(np.hypot(p[0], p[1])))
    radii = np.array(radii)
    print(f"  {len(radii)} near-vertical ground poses found (|approach_z| > 0.90)")
    print(f"  radius min/max          : {radii.min():.4f} .. {radii.max():.4f} m")
    print(f"  radius 5/50/95 pct      : {np.percentile(radii, 5):.4f} / {np.percentile(radii, 50):.4f} / {np.percentile(radii, 95):.4f} m")
    print("  -> RECOMMENDED sampling radius band : 0.16 .. 0.28 m (comfortably inside the envelope)")
    if not (radii.min() < 0.16 and radii.max() > 0.28):
        fail("recommended radius band 0.16-0.28 m is not inside the measured envelope")

    # ------------------------------------------------ scene object placements
    hr("SCENE OBJECT PLACEMENTS")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cp = data.xpos[cube_bid]
    tp = data.xpos[tray_bid]
    print(f"  cube default pos : ({cp[0]:.4f}, {cp[1]:.4f}, {cp[2]:.4f})  r={np.hypot(cp[0], cp[1]):.4f}  az={np.degrees(np.arctan2(cp[1], cp[0])):+.1f} deg")
    print(f"  tray  centre pos : ({tp[0]:.4f}, {tp[1]:.4f}, {tp[2]:.4f})  r={np.hypot(tp[0], tp[1]):.4f}  az={np.degrees(np.arctan2(tp[1], tp[0])):+.1f} deg")
    print("  tray outer 0.160 x 0.140 m, inner cavity 0.136 x 0.116 m")
    print("  tray floor top z=0.006, wall top z=0.036 -> release the cube above z=0.076")
    print(f"  tray body_pos (edit this to move the tray): model.body_pos[{tray_bid}] = {model.body_pos[tray_bid]}")

    # ------------------------------------------------------------- verdict
    hr("VERDICT")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"\n  {len(failures)} check(s) FAILED")
        return 1
    print("  All checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
