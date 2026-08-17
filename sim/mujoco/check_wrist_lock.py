#!/usr/bin/env python3
"""Guard: prove wrist_roll cannot move, before training on any collected data.

wrist_roll (joint index 4 of 6, the wrist immediately BEFORE the gripper) has a
broken servo on the physical arm and is taped in place at -pi/2.  Commanding it
stalls a dead motor and works the tape loose, so the sim must never produce an
action that rolls it -- a policy trained on such data is useless on the real
robot and actively harmful to it.

Three independent layers pin it.  This script checks all three, plus the
recorded episodes if you point it at them.  Run from the repo root:

    python sim/mujoco/check_wrist_lock.py
    python sim/mujoco/check_wrist_lock.py --episodes data/sim_raw

Exit status 0 and a final "WRIST LOCK OK" mean it is safe to train.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Tolerances are on the MEASURED joint, which sags a little under gravity and
# inertia even when the command never moves; the COMMAND is held exactly.
CMD_TOL = 1e-9          # rad, on anything commanded
MEAS_TOL = math.radians(1.0)   # rad, on anything measured in physics

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join("sim", "mujoco", "scene.xml"))
    ap.add_argument("--episodes", default=None,
                    help="optional directory of collected .npz episodes to audit")
    args = ap.parse_args()

    import mujoco

    from env import WRIST_ROLL_INDEX, WRIST_ROLL_LOCK, SO101Env

    lock = WRIST_ROLL_LOCK
    idx = WRIST_ROLL_INDEX
    print(f"wrist_roll lock = {lock:+.7f} rad ({math.degrees(lock):+.2f} deg), "
          f"joint index {idx}\n")

    # ---------------------------------------------------------------- layer 1
    print("layer 1 - model clamp (SO101Env bypassed entirely)")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_roll")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
    qadr = int(model.jnt_qposadr[jid])

    # A zero-width ctrlrange compiles to ctrllimited=False and silently disables
    # the clamp, so this flag -- not the range values -- is the real check.
    check("actuator ctrllimited is True", bool(model.actuator_ctrllimited[aid]),
          "zero-width ctrlrange would disable the clamp")
    lo, hi = model.actuator_ctrlrange[aid]
    check("ctrlrange brackets the lock tightly",
          lo <= lock <= hi and (hi - lo) < 1e-3,
          f"[{lo:+.7f}, {hi:+.7f}] width={hi - lo:.1e}")

    worst_key = 0.0
    for k in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, k)
        worst_key = max(worst_key, abs(float(data.qpos[qadr]) - lock))
    check(f"all {model.nkey} keyframes start at the lock", worst_key < 1e-4,
          f"worst {math.degrees(worst_key):.4f} deg")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    hostile = np.array(model.actuator_ctrlrange[:, 1], dtype=float)  # every joint slammed high
    worst_raw = 0.0
    for _ in range(2000):
        data.ctrl[:] = hostile
        mujoco.mj_step(model, data)
        worst_raw = max(worst_raw, abs(float(data.qpos[qadr]) - lock))
    check("2000 raw `data.ctrl[:] = max` steps cannot roll it", worst_raw < MEAS_TOL,
          f"worst {math.degrees(worst_raw):.4f} deg")

    # ---------------------------------------------------------------- layer 2
    print("\nlayer 2 - SO101Env.clip_cmd (policy / replay / eval path)")
    env = SO101Env(args.scene, render_size=(96, 96))
    check("caller-supplied home_qpos cannot unpin it",
          abs(float(SO101Env(args.scene, render_size=(64, 64),
                             home_qpos=(0, 0, 0, 0, 1.58437, 0.9)).home_qpos[idx]) - lock) < 1e-9)

    for bad in (np.nan, np.inf, -np.inf, -99.0, 1e9):
        c = np.asarray(env.home_qpos, dtype=float).copy()
        c[idx] = bad
        out = env.clip_cmd(c)
        if not abs(float(out[idx]) - lock) <= CMD_TOL:
            check(f"clip_cmd survives {bad}", False, f"got {out[idx]}")
            break
    else:
        check("clip_cmd survives NaN / +-inf / huge on channel 4", True)

    env.reset(seed=1)
    rng = np.random.default_rng(1)
    worst_cmd = worst_meas = 0.0
    for _ in range(200):
        c = np.asarray(env.home_qpos, dtype=float).copy()
        c[idx] = rng.uniform(-3.0, 3.0)         # anything a policy might emit
        c[:4] += rng.uniform(-0.2, 0.2, 4)      # real motion on the live joints
        measured = env.step(c)
        worst_cmd = max(worst_cmd, abs(float(env._ctrl[idx]) - lock))
        worst_meas = max(worst_meas, abs(float(measured[idx]) - lock))
    check("200 random wrist commands never reach the actuator", worst_cmd <= CMD_TOL,
          f"worst commanded {math.degrees(worst_cmd):.7f} deg")
    check("measured joint stays at the lock", worst_meas < MEAS_TOL,
          f"worst {math.degrees(worst_meas):.4f} deg (sim servo sag; the real joint is taped)")
    env.close()

    # ---------------------------------------------------------------- layer 3
    print("\nlayer 3 - scripted expert")
    import collect
    check("wrist_roll excluded from the IK degrees of freedom",
          idx not in collect.IK_DOF, f"IK_DOF={collect.IK_DOF}")
    check("expert imports the lock rather than redefining it",
          collect.WRIST_ROLL_LOCK is WRIST_ROLL_LOCK or
          abs(collect.WRIST_ROLL_LOCK - lock) < 1e-12)

    # ------------------------------------------------------------- recordings
    if args.episodes:
        print(f"\nrecorded episodes in {args.episodes}")
        files = sorted(glob.glob(os.path.join(args.episodes, "*.npz")))
        if not files:
            check("found episodes to audit", False, "no .npz files")
        else:
            worst_a = worst_s = 0.0
            for f in files:
                z = np.load(f)
                worst_a = max(worst_a, float(np.abs(z["action"][:, idx] - lock).max()))
                worst_s = max(worst_s, float(np.abs(z["observation.state"][:, idx] - lock).max()))
            check(f"action[:, {idx}] is exactly the lock in all {len(files)} episodes",
                  worst_a <= CMD_TOL, f"worst {math.degrees(worst_a):.7f} deg")
            check(f"observation.state[:, {idx}] stays at the lock",
                  worst_s < MEAS_TOL, f"worst {math.degrees(worst_s):.4f} deg")

    print()
    if failures:
        print(f"WRIST LOCK FAILED ({len(failures)}): " + "; ".join(failures))
        print("DO NOT TRAIN on this data -- the policy would learn to roll a dead joint.")
        return 1
    print("WRIST LOCK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
