# SO-101 MuJoCo episode collector

Scripted pick-and-place expert that produces LeRobot-ready episodes from
`sim/mujoco/scene.xml` (SO-101 arm + 40 mm orange cube + wooden tray).

| file | what it is |
|---|---|
| `env.py` | `SO101Env` — thin MuJoCo wrapper: `reset` / `step` / `render` / `success` |
| `collect.py` | the scripted expert + recorder + CLI |
| `run.ps1` | this Windows box: collect → `training/build_dataset.py` → `lerobot-train` |
| `run.sh` | the Mac (`/opt/homebrew/Caskroom/miniforge/base/envs/lerobot`), same three stages |
| `MODEL_NOTES.md`, `verify_model.py`, `scene.xml` | owned by the scene author — the measured contract this collector is calibrated against |

---

## Run it

Always from the **repo root**.

```powershell
# smoke test
sim\mujoco_venv\Scripts\python.exe sim\mujoco\collect.py --episodes 2 --out data\smoke --seed 0

# a real batch
sim\mujoco_venv\Scripts\python.exe sim\mujoco\collect.py --episodes 60 --out data\sim_raw --seed 0

# whole pipeline (collect + dataset + train)
powershell -File sim\mujoco\run.ps1 -Episodes 60
powershell -File sim\mujoco\run.ps1 -Episodes 60 -CollectOnly    # no torch/lerobot needed
```

```bash
# Mac
./sim/mujoco/run.sh                       # 60 episodes, build, train
EPISODES=120 SEED=1 ./sim/mujoco/run.sh
COLLECT_ONLY=1 ./sim/mujoco/run.sh
```

`env.py` also runs standalone as a one-line scene check:

```powershell
sim\mujoco_venv\Scripts\python.exe sim\mujoco\env.py [scene.xml]
```

### CLI

| flag | default | meaning |
|---|---|---|
| `--episodes N` | *required* | number of episodes to **keep** (failures are retried, not counted) |
| `--out DIR` | *required* | output directory |
| `--seed S` | `0` | master seed; every episode seed derives from it deterministically |
| `--scene PATH` | `sim/mujoco/scene.xml` | any scene with the SO-101 + a free-jointed cube |
| `--keep-failures` | off | write failed episodes too (still logged to `failures.json`) |
| `--grasp friction\|weld` | `friction` | `weld` is the emergency fallback, see below |
| `--world-video` | off | also write `<out>/world/epNNN.mp4` (third-person debug view) |
| `--render-size WxH` | `640x480` | wrist video size |
| `--max-attempts N` | `4 × episodes` | attempt budget before giving up |
| `--no-noise` | off | disable DART noise (debugging only — **never** for training data) |
| `--verbose` | off | print per-episode grasp diagnostics |

---

## Output format — the guarantees

Per episode, in `<out>/`:

```
epNNN.npz    "observation.state"  float32 (T, 6)   MEASURED joint positions [rad]
             "action"             float32 (T, 6)   COMMANDED joint targets  [rad]
epNNN.mp4    640x480 RGB wrist camera, exactly T frames, fps = 10, libx264/yuv420p
```

Joint order is the model's own: `[shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper]`, which is what `training/build_dataset.py`
labels `[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]`.

Guaranteed on every write:

* exactly those **two** npz keys, both `float32`, both `(T, 6)`, both finite;
* `len(video) == T` — the recorder appends one frame per control tick, so the
  two can't drift;
* `mean |action − observation.state| > 1e-4`, **asserted** before the episode is
  written. If `action` were a copy of the state, ACT would learn the identity
  map and do nothing on the robot. Observed values are ~0.04 rad.
* frame rate 10 Hz, matching `model.opt.timestep = 0.005 s` × 20 substeps per
  tick. The model timestep is never modified.
* `macro_block_size=1`, so non-multiple-of-16 sizes (e.g. `--render-size 224`)
  encode without ffmpeg silently padding the image.

Also written, for bookkeeping (both ignored by `build_dataset.py`, which globs
only `*.npz`):

* `<out>/episodes.json` — one record per kept episode: frame count, success
  flag, attempt number, spawn parameters.
* `<out>/failures.json` — spawn parameters **and grasp diagnostics** for every
  failed attempt, so a bad region of the workspace is one `jq` away.

### Why 640×480 and not 224×224

`training/build_dataset.py` declares the image feature as `(480, 640, 3)` and
resizes anything else with `cv2.INTER_AREA`. Rendering natively at 640×480 means
no resampling anywhere in the pipeline. `MODEL_NOTES.md` suggests 224×224 for
the wrist camera; if you want that, pass `--render-size 224x224` — it is
supported and produces valid episodes, they just get upscaled downstream.

---

## What the expert does

Damped-least-squares IK on the `gripperframe` site (`mujoco.mj_jacSite`,
λ = 0.05, converges to ≤ 1.5 mm everywhere in the spawn band), then a fixed
phase sequence at 10 Hz:

| # | phase | ticks | notes |
|---|---|---|---|
| 1a | rise | 6 | straight up to carry height **before** any lateral motion |
| 1b | traverse | 12 | across to above the cube, at carry height, **along the arc** |
| 2 | descend | 10 | down to the grasp pose |
| 3 | pre-settle | 3 | let the position servos catch up before touching the cube |
| 4 | **grasp** | 7 + 4 | jaws ramp closed over 0.7 s, settle 0.4 s — **arm command frozen** |
| 5 | lift | 12 | smoothstep back to carry height |
| 6 | transfer | 20 | smoothstep across to over the tray, **along the arc** |
| 7 | lower | 8 | down to release height |
| 8 | release | 5 + 4 | jaws ramp open, settle — arm command frozen |
| 9 | retreat | 14 + 6 | back up and settle |

**T = 111 frames ≈ 11.1 s per episode.**

Rules baked in, each of which cost real debugging time:

* **The arm command is frozen for the entire grasp and release phase.** Moving
  the arm while the jaws travel rakes the cube out of the gripper.
* **Every Cartesian move is a smoothstep ramp, never a step command.** A step
  target on a `kp = 998` position servo is a torque spike that throws the cube.
* **Long lateral moves follow the arc, not a chord.** Interpolating x/y between
  two points on the working arc dips the tool toward the base — a badly
  conditioned pose that shook the cube out of the grasp mid-transfer. Phases 1b
  and 6 interpolate radius and azimuth instead.
* **Nothing moves sideways at table height, and the arm never rests there
  either.** The scene's neutral poses put the tool at z ≈ 0.013, in the middle
  of the spawn arc: a straight line from there to a hover pose ploughs the cube
  off the table, and `reset()` could settle the arm straight into a freshly
  spawned cube and shove it 5 cm before the episode began. So the rest pose is
  re-solved at carry height in the gap between the cube band and the tray
  (`HOME_SITE_*`), and phase 1a rises before anything moves laterally. Between
  them these two rules account for every failure seen during development.
* **`wrist_roll` is pinned at −π/2 in every phase and every waypoint.** That is
  the orientation the physical arm has (claws side-by-side, hand the right way
  up). It is therefore *not* an IK degree of freedom: the solver uses four
  joints and the jaw-opening axis is an output. The cube is spawned with its yaw
  squared to that measured axis (± 12°) rather than the other way round.
* **The open-jaw grasp centre is offset from the tool frame.** The fixed jaw
  sits 20 mm off-axis while the moving jaw swings wide, so the site is aimed
  6.6 mm below and 21 mm back along the jaw axis from the cube centre; closing
  re-centres the cube onto the site to within 3 mm. Both offsets are expressed
  in the *site* frame and the jaw direction is read from live FK, so flipping
  the wrist_roll sign is handled automatically.

### DART noise

Gaussian noise with σ = 0.5 % of each joint's range (≈ 0.017 rad) is added to
the **commanded** arm targets every tick. It is faded linearly to exactly zero
across the descend and lower phases so it is already zero by the time the jaws
start moving (a step change in noise amplitude is itself a disturbance), stays
zero through the grasp and release windows, and fades back in over the lift and
retreat. It is **never** applied to the gripper channel, nor to `wrist_roll`
(which is pinned). The recorded `action` is the noisy command, the recorded
`observation.state` is the measured `qpos` — that difference is the signal ACT
trains on.

Two invariants are asserted before every episode is written:
`mean |action − state| > 1e-4`, and every commanded `wrist_roll` is exactly
`WRIST_ROLL_LOCK`.

### `--grasp weld` (fallback only)

If contact tuning ever stops holding the cube, `--grasp weld` kinematically
anchors the cube to the tool frame at the moment of grasp and releases it at the
moment of opening (`env.attach_cube()` / `env.detach_cube()`). This is **not
physics** — the cube pose is re-imposed after every substep. Use it only to
unblock data collection; the friction grasp is the real one and currently holds
100 % of the time.

---

## Tuning knobs

All of them are named constants at the top of `collect.py`, sourced from
`MODEL_NOTES.md`. Re-run `verify_model.py` after any scene change and update
that block; nothing else in the file hard-codes scene geometry.

| constant | value | tune when |
|---|---|---|
| `GRIPPER_OPEN` / `GRIPPER_GRIP` | `0.90` / `0.00` | jaws clip the cube on descent / grip slips. Larger = more open; anything ≤ +0.30 holds the 40 mm cube |
| `GRASP_DEPTH_OFFSET` / `GRASP_JAW_OFFSET` | `−0.0066` / `+0.0209` | the cube is knocked sideways or missed during the close |
| `CUBE_RADIUS_BAND` | `(0.185, 0.255)` | full measured envelope is 0.16–0.28 m |
| `CUBE_AZIMUTH_BAND` | `(−8°, +55°)` | kinematic band is ±60°, clipped here so the cube never spawns on the tray (whose near corner reaches −21.8°) |
| `HOME_SITE_RADIUS` / `HOME_SITE_AZIMUTH` | `0.200` / `−20°` | the rest pose is IK-solved to this point at `CARRY_SITE_Z`; keep it clear of both the cube band and the tray |
| `CARRY_SITE_Z` | `0.070` | IK stays ≤ 1.5 mm up to z = 0.072 and degrades fast above (10 mm at 0.090) as `wrist_flex` saturates — **do not raise blindly** |
| `RELEASE_SITE_Z` | `0.060` | cube centre lands at ≈ 0.067, clearing the 0.036 tray wall; lower it if the cube bounces out |
| `T_*` | see table above | episode length; `T = 111` at the defaults. `T_TRANSFER` is the one to raise if the cube slips mid-carry |
| `NOISE_FRACTION` | `0.005` | more noise = better ACT robustness, worse success rate |
| `IK_LAMBDA`, `IK_ROT_WEIGHT`, `IK_NULLSPACE_GAIN` | `0.05`, `0.30`, `0.02` | IK jitters or stalls short of the target |
| `WRIST_ROLL_LOCK` | `−π/2` | **hard requirement — do not change without re-checking a render** |

Spawns are **stratified**: episode *i* of *N* owns bin *i* of the azimuth band
and retries stay inside the same bin, so a batch covers the arc uniformly no
matter how many attempts each episode took.

---

## Measured status

Against `sim/mujoco/scene.xml`, `--grasp friction`, default settings:

| run | result |
|---|---|
| seeds 0 / 42 / 99, 20 episodes each | 60 / 60 (100 %) |
| max IK position error, whole band | 1.5 mm |
| `mean \|action − state\|` | 0.035 – 0.050 rad |
| wrist frame pixel std | 61 – 90 (health floor is 5) |
| episode length | 111 frames |

Roughly 5 s per episode to collect (single-threaded, one reused renderer).
