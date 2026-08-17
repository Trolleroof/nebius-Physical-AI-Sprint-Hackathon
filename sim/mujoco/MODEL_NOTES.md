# MODEL_NOTES.md — measured contract for `sim/mujoco/scene.xml`

Every number below was **measured** by running `sim/mujoco/verify_model.py`
(`sim\mujoco_venv\Scripts\python.exe sim/mujoco/verify_model.py`, exits 0).
Re-run it after any scene change; it re-measures everything and fails loudly.

Robot = MuJoCo Menagerie / LeRobot SO-101 (`robotstudio_so101/so101.xml`).
**Do not retune the gripper contact parameters** (elliptic cone, `impratio=10`,
`condim=6 priority=1` pads) — they are already tuned upstream.

---

## 1. Load

```python
model = mujoco.MjModel.from_xml_path("sim/mujoco/scene.xml")
data  = mujoco.MjData(model)
```

| | |
|---|---|
| `nq` | **13** |
| `nv` | **12** |
| `nu` | **6** |
| `nbody` | 11 |
| `ngeom` | 55 |

The scene loads cleanly from `mj_resetData` (arm at all-zeros, cube resting on
the floor, tray static). No keyframe is required.

## 2. Physics timestep and the control tick

| | |
|---|---|
| `model.opt.timestep` | **0.005 s** (200 Hz) — **do not change it** |
| integrator | `implicitfast`, `cone="elliptic"`, `impratio=10` |
| **10 Hz control tick** | **20 × `mujoco.mj_step(model, data)`** |

```python
SIM_DT   = 0.005          # model.opt.timestep
CTRL_HZ  = 10
N_SUBSTEPS = 20           # round(1/CTRL_HZ / SIM_DT)
```

## 3. Joint order (`qpos`) — 13 values

`qpos` index == joint index for the 6 arm joints (all hinges, `qposadr == i`).

| i | joint name | `qposadr` | `dofadr` | joint range (rad) |
|---|---|---|---|---|
| 0 | `shoulder_pan`  | 0 | 0 | −1.91986 … +1.91986 |
| 1 | `shoulder_lift` | 1 | 1 | −1.74533 … +1.74533 |
| 2 | `elbow_flex`    | 2 | 2 | −1.69000 … +1.69000 |
| 3 | `wrist_flex`    | 3 | 3 | −1.658063 … +1.658063 |
| 4 | `wrist_roll`    | 4 | 4 | −2.7438473 … +2.7438473 |
| 5 | `gripper`       | 5 | 5 | −0.174533 … +1.7453292 |
| 6 | `cube_free` (freejoint) | **6** | 6 | — |

So:
* arm joint positions  = `data.qpos[0:6]`
* arm joint velocities = `data.qvel[0:6]`
* cube pose            = `data.qpos[6:13]` = `[x, y, z, qw, qx, qy, qz]`
* cube twist           = `data.qvel[6:12]`

## 4. Actuator order (`data.ctrl`) — 6 values

All are `<position>` actuators (`kp=998.22`, `kv=2.731`, `forcerange ±2.94 N·m`).
**Actuator index i drives joint index i** — `ctrl` and `qpos[0:6]` are aligned,
so a position command is directly a target joint angle in radians.

| i | actuator | ctrlrange (rad) |
|---|---|---|
| 0 | `shoulder_pan`  | −1.91986 … +1.91986 |
| 1 | `shoulder_lift` | −1.74533 … +1.74533 |
| 2 | `elbow_flex`    | −1.69000 … +1.69000 |
| 3 | `wrist_flex`    | −1.65806 … +1.65806 |
| 4 | `wrist_roll`    | −2.74385 … **+2.84121** (asymmetric — upstream quirk) |
| 5 | `gripper`       | −0.17453 … +1.74533 |

Always clip actions with `np.clip(a, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1])`.

## 5. Gripper — sign convention and the two magic numbers

**Measured**: the finger-tip gap grows monotonically with the `gripper` joint.

```
tip_gap_mm  ≈  4.1 + 67.3 * (gripper_rad - (-0.1745))
```

| gripper joint (rad) | finger-tip gap |
|---|---|
| −0.17453 (min) | 4.1 mm  ← **fully CLOSED** |
| +0.312         | 40 mm (stall angle on the 40 mm cube) |
| +0.90          | 84.3 mm |
| +1.74533 (max) | 133.4 mm ← **fully OPEN** |

> **LARGER `gripper` value = MORE OPEN. SMALLER = MORE CLOSED.**

Recommended commands (verified by a close → lift → release cycle):

```python
GRIPPER_OPEN =  0.90   # 84 mm tip gap, clears the 40 mm cube with margin
GRIPPER_GRIP =  0.00   # commanded well past the 0.312 rad stall -> steady squeeze
```

Measured hold behaviour (close from `pregrasp`, then lift `shoulder_lift` by −0.35 rad):

| grip command | joint stalls at | cube lifted? |
|---|---|---|
| +0.50 | 0.501 | no (never touches) |
| +0.40 | 0.401 | no |
| +0.35 | 0.351 | no |
| **+0.30** | 0.312 | **yes** |
| **+0.20 / 0.00 / −0.17** | 0.312 | **yes** |

So **any command ≤ +0.30 rad holds the 40 mm cube**; `0.0` is the safe default.
Opening back to `0.90` reliably releases it (cube fell from z=0.064 back to z=0.020).

`data.qpos[5]` stalls at **0.312 rad** while holding the cube — a useful
"object is between the jaws" signal (`qpos[5] > 0.25` with `ctrl[5] <= 0.0`).

## 6. Tool frame / IK site

| | |
|---|---|
| site name for `mj_jacSite` | **`gripperframe`** (also `baseframe` at the robot origin) |
| lookup | `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")` |
| world pos | `data.site_xpos[sid]`, orientation `data.site_xmat[sid].reshape(3,3)` |
| **approach axis** | the site's **local +X** axis. At the pickup pose it is `[-0.073, 0.121, -0.990]` in world — i.e. pointing straight **down** |

**Where to aim IK.** Once the jaws are actually closed on the cube, the cube
centre and `gripperframe` coincide to within **2.8 mm** (measured at the
`pickup` keyframe: cube `(0.20858, 0.02151, 0.01785)` vs `gripperframe`
`(0.21108, 0.02158, 0.01920)`). **So: drive `gripperframe` to the cube centre.**

The *open-jaw* midpoint is offset, because the moving jaw swings wide:

* at the `pregrasp` pose with `gripper = 0.9`:
  * `gripperframe`      = `(0.20756, 0.01181, 0.01346)`, r = **0.2079 m**
  * finger-tip midpoint = `(0.20883, 0.03267, 0.02036)`, r = **0.2114 m**
  * delta = `(+0.0013, +0.0209, +0.0069)` m, mostly sideways, ~22 mm total

Practical recipe: approach with `gripperframe` ~20 mm above the cube centre,
descend so `gripperframe` ≈ cube centre, then command `GRIPPER_GRIP`.
The `pregrasp` arm pose below is a good IK seed.

## 7. Reachable workspace (measured)

**Graspable radius of the canonical top-down pose: r = 0.2114 m** from the base
origin (0,0,0), with the tool at table height (z ≈ 0.020).

Full envelope, from a 25×25×25 sweep of `shoulder_lift × elbow_flex × wrist_flex`
keeping the approach axis within 25° of vertical and the tool at table height:

| | |
|---|---|
| radius min … max | **0.059 … 0.361 m** |
| radius 5 / 50 / 95 pct | 0.089 / 0.188 / 0.339 m |
| **recommended sampling band** | **r ∈ [0.16, 0.28] m** |

Azimuth (sweeping `shoulder_pan` at the pregrasp pose, grasp-centre bearing
measured from the +X axis, CCW positive):

| `shoulder_pan` (deg) | grasp-centre azimuth (deg) | r (m) |
|---|---|---|
| −110.0 | +108.6 | 0.157 |
| −55.0  | +55.3  | 0.192 |
| 0.0    | **+8.9** | 0.211 |
| +55.0  | −36.5  | 0.203 |
| +110.0 | −86.2  | 0.171 |

* Note the **sign flip**: positive `shoulder_pan` swings the tool to **negative**
  azimuth. Offset at `pan = 0` is **+8.9°**.
* kinematic azimuth band: **[−86.2°, +108.6°]**
* **recommended sampling band: azimuth ∈ [−60°, +60°]** — inside this band the
  arm never folds back over its own base and r stays ≈ 0.19–0.21 m.

## 8. Cube (the object to pick)

| | |
|---|---|
| body name | **`cube`** (body id 9) |
| freejoint name | **`cube_free`** |
| **qpos address** | **`qpos[6:13]`** = `[x, y, z, qw, qx, qy, qz]`; qvel `[6:12]` |
| geom name | `cube` |
| size | `0.02 0.02 0.02` half-extents → **40 mm cube** |
| mass | 0.03 kg |
| rgba | `1 0.45 0.05 1` (orange-red) |
| contacts | `condim=3`, `friction="1 0.03 0.003"`, `solref="0.01 1"` |
| default pose | `(0.20419, 0.05471, 0.020)` → r = 0.2114 m, azimuth **+15°** |
| resting z | **0.020** (half-extent on the floor plane) |

Randomise it like this:

```python
CUBE_QADR = 6                      # model.jnt_qposadr[jid of "cube_free"]
r  = rng.uniform(0.16, 0.28)
az = np.radians(rng.uniform(-60, 60))
data.qpos[CUBE_QADR:CUBE_QADR+3] = [r*np.cos(az), r*np.sin(az), 0.02]
data.qpos[CUBE_QADR+3:CUBE_QADR+7] = [1, 0, 0, 0]   # or a yaw-only quat
data.qvel[6:12] = 0
mujoco.mj_forward(model, data)
```

## 9. Tray (the place target)

| | |
|---|---|
| body name | **`tray`** (body id 10, static — no joint) |
| centre | `model.body_pos[tray_bid]` = `(0.16971, −0.16971, 0.0)` → r = 0.24 m, azimuth **−45°** |
| site at the tray floor centre | `tray_center` (local `0 0 0.006`) |
| outer footprint | 0.160 × 0.140 m |
| inner cavity | 0.136 × 0.116 m |
| floor thickness | 0.006 (floor top at **z = 0.006**) |
| wall height / thickness | 0.030 / 0.012 (wall top at **z = 0.036**) |
| geoms | `tray_floor`, `tray_wall_x_min`, `tray_wall_x_max`, `tray_wall_y_min`, `tray_wall_y_max` |
| colours | walls `0.55 0.36 0.20 1`, floor `0.40 0.25 0.13 1` |

Geometry mirrors `sim/antioch/src/wooden_tray.py` (outer size shrunk to
0.16 × 0.14 for the sim task).

**To move the tray at runtime** (it is a static body, so just write `body_pos`):

```python
tray_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
model.body_pos[tray_bid] = [x, y, 0.0]
mujoco.mj_forward(model, data)
```

Release the cube above **z = 0.076** (wall top 0.036 + cube half 0.02 + 20 mm
clearance) so it clears the walls and drops in.
**Success gate suggestion:** cube centre inside `|dx| < 0.058`, `|dy| < 0.048`
of the tray centre and `0.020 < z < 0.050`, with `|v| < 0.02 m/s`.

## 10. Cameras

| name | mount | recommended render size (H×W) |
|---|---|---|
| **`wrist_cam`** | on the gripper camera mount (from `so101.xml`) | **224 × 224** |
| **`world`** | fixed, `pos="0.85 -0.45 0.55"`, aimed at the workspace | **480 × 640** |

`<visual><global offwidth="640" offheight="480"/>` is set, so **640×480 is the
maximum offscreen buffer** — do not request anything larger without raising it.

```python
with mujoco.Renderer(model, height=224, width=224) as r:
    r.update_scene(data, camera="wrist_cam")
    wrist = r.render()               # uint8 HxWx3, RGB
```

Sanity floor from the verifier: a good frame has pixel `std > 5`
(observed: world ≈ 42, wrist ≈ 34 idle / ≈ 55 while holding the cube).

## 11. Keyframes

| id | name | what it is |
|---|---|---|
| 0 | `pregrasp` | arm at the top-down approach pose, jaws **open** (0.9) straddling the cube, cube on the ground at `(0.20883, 0.032665, 0.02)` |
| 1 | `pickup` | settled **closed grasp** on the cube (`gripper` stalled at 0.312, `ctrl[5]=0`); load it and keep stepping with its `ctrl` and it stays grasped (cube–tool distance 2.8 mm → 2.5 mm after 0.5 s) |
| 2 | `home` | neutral arm pose `[0, −1.5, 1.5, 0.8, 0, 1.0]`, cube/tray at scene defaults |

```python
kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pregrasp")
mujoco.mj_resetDataKeyframe(model, data, kid)
```

Approach arm pose (useful as an IK seed / scripted waypoint):

```python
ARM_APPROACH = [0.0, 0.000381818, 0.473496, 1.17717, 1.58437]   # + gripper
```

`shoulder_lift` **more negative raises the tool** (measured: −0.35 rad lifts the
grasp centre from z = 0.020 to z = 0.064).

## 12. Files

| file | owner | note |
|---|---|---|
| `sim/mujoco/scene.xml` | Agent A | the task world; includes `robotstudio_so101/so101.xml` and re-roots `meshdir` to `robotstudio_so101/assets` **after** the include |
| `sim/mujoco/verify_model.py` | Agent A | measures everything above; exit 0 = scene healthy |
| `sim/mujoco/MODEL_NOTES.md` | Agent A | this file |
| `sim/mujoco/verify_*.png` | generated | `verify_world`, `verify_wrist`, `verify_wrist_grasp`, `verify_wrist_lift`, `verify_world_grasp` |
| `sim/mujoco/robotstudio_so101/` | upstream | **do not edit** |
