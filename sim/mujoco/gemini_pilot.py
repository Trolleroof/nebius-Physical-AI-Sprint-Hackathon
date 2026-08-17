#!/usr/bin/env python
"""Natural language -> Gemini ER 2 -> SO-101 pick-and-place, in MuJoCo.

One episode of the demo:

  1. ``env.reset()`` spawns the cube (and jitters the tray) somewhere on the
     reachable arc.
  2. The ``world`` camera is rendered at 640x480 and sent to
     ``gemini-robotics-er-2-preview`` together with the operator's plain
     English prompt, asking for exactly two labelled points: the object to
     pick and the container to place it into.
  3. Those two image points are un-projected onto known horizontal planes
     (cube centre height for the object, tray floor for the container) by
     ``pixel_world.unproject``, which turns them into 3D world coordinates.
  4. The scripted phase machine from ``collect.py`` is driven with **those**
     coordinates.  The simulator's own cube/tray poses are never read by the
     policy -- they are used ONLY for scoring afterwards.

Run (from the repo root)::

    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/gemini_pilot.py \\
        --episodes 3 --out sim/mujoco/pilot_out

    # offline: no API call, ground truth projected to pixels + gaussian noise
    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/gemini_pilot.py \\
        --episodes 2 --mock-gemini

Outputs per episode in ``--out``:

    epNNN_world.mp4        world camera, one frame per 10 Hz control tick
    epNNN_annotated.png    first world frame + Gemini's two crosshairs
    results.json           machine-readable scoring for every episode

Honesty rules baked into this file:

* The only inputs to the motion are the two un-projected points.  Ground truth
  is read after the fact, to report pointing error in mm and task success.
* DART exploration noise is off: this is a demo of perception, not a data
  collection run, so nothing is injected between the command and the servo.
* A perception failure (unparseable JSON, fewer than two points, a ray that
  misses the plane) is retried once with a stricter reminder and then recorded
  as a failed episode with a reason.  It never raises.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SO101Env  # noqa: E402
from collect import (  # noqa: E402
    ArmIK,
    CARRY_SITE_Z,
    CUBE_AZIMUTH_BAND,
    CUBE_RADIUS_BAND,
    CUBE_YAW_JITTER,
    DEFAULT_SCENE,
    GRIPPER_GRIP,
    GRIPPER_OPEN,
    GRIP_STALL,
    HOME_SITE_AZIMUTH,
    HOME_SITE_RADIUS,
    RELEASE_SITE_Z,
    Recorder,
    SUCCESS_XY_TOL,
    TRAY_FLOOR_TOP,
    T_DESCEND,
    T_FINAL_SETTLE,
    T_GRASP_RAMP,
    T_GRASP_SETTLE,
    T_LIFT,
    T_LOWER,
    T_PRESETTLE,
    T_RELEASE_RAMP,
    T_RELEASE_SETTLE,
    T_RETREAT,
    T_RISE,
    T_TRANSFER,
    T_TRAVERSE,
    VIDEO_FPS,
    WRIST_ROLL_LOCK,
    smoothstep,
    write_video,
)
import pixel_world  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# ==========================================================================
# constants
# ==========================================================================
ER2_MODEL = "gemini-robotics-er-2-preview"
WORLD_CAMERA = "world"
WORLD_SIZE = (640, 480)          # (W, H) -- what Gemini sees and what we record

# Un-projection planes.  A single image point only becomes a 3D point once you
# say which surface it lies on (MODEL_NOTES sec.9: tray floor top z = 0.006).
OBJECT_PLANE_Z = 0.020           # cube centre height (40 mm cube on the table)
CONTAINER_PLANE_Z = TRAY_FLOOR_TOP   # 0.006, the tray floor

DEFAULT_PROMPT = "Pick up the orange cube and place it in the wooden tray."
DEFAULT_OUT = "sim/mujoco/pilot_out"
# Sigma of the fake pointing error, in pixels.  The world camera views the
# table obliquely (eye 0.85,-0.45,0.55), so a pixel is worth ~2 mm on the table
# ONCE THE RAY IS INTERSECTED: 6 px measures out at ~15 mm of ground error and
# 8 px at ~22 mm, which is past what the friction grasp tolerates on a 40 mm
# cube (measured: 100 % of mock episodes succeed at 6 px, ~33 % at 8 px).  The
# default is therefore the honest edge of the working envelope; raise it with
# --mock-noise-px to stress-test the failure reporting.
MOCK_NOISE_PX = 6.0

# The tray is nudged around its scene azimuth (-45 deg) rather than swept over
# the whole arc: its footprint is 0.21 x 0.14 m, so a large swing would overlap
# the cube spawn band and the two objects would fight for the same table.
TRAY_AZIMUTH_JITTER = math.radians(6.0)

CONTAINER_WORDS = (
    "tray", "bin", "basket", "container", "bowl", "box", "crate", "dish",
    "plate", "receptacle", "target",
)

POINT_PROMPT = """You are the vision system of a robot arm working on the tabletop in this image.

The operator said: "{prompt}"

Point at exactly two things:
  1. the object the arm must pick up -- aim at the centre of its top face;
  2. the container the object must be placed into -- aim at the centre of its
     inside floor, not at its rim.

Answer with JSON and nothing else, in this exact form:
[{{"point": [y, x], "label": "object"}}, {{"point": [y, x], "label": "container"}}]

The points are in [y, x] order, normalised to 0-1000 over the image height and
width respectively.  Return exactly two entries: the first labelled "object",
the second labelled "container"."""

STRICTER_REMINDER = """

Your previous answer could not be used ({why}).  Reply with ONLY the raw JSON
array of exactly two objects -- no markdown fence, no explanation, no extra
keys.  Example of a valid answer:
[{"point": [412, 655], "label": "object"}, {"point": [733, 240], "label": "container"}]"""


# ==========================================================================
# API key / client
# ==========================================================================
def _load_root_env() -> None:
    """Read the repo-root .env into os.environ, without overriding real env.

    Mirrors ``backend/critic.py::_load_root_env`` so the pilot picks up
    GEMINI_API_KEY with zero setup while the key stays out of the repo.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_client():
    """Sync google-genai client on the Gemini Developer API (ER 2 lives there)."""
    from google import genai

    _load_root_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "No API key.  Put GEMINI_API_KEY=... in the repo-root .env "
            "(or the environment), or run with --mock-gemini."
        )
    return genai.Client(api_key=key)


# ==========================================================================
# perception
# ==========================================================================
def parse_json(text: str):
    """Fence-tolerant JSON extraction (same defensive shape as er2_client)."""
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    payload = fenced.group(1) if fenced else text
    start = min((i for i in (payload.find("["), payload.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found in response")
    try:
        return json.loads(payload[start:])
    except json.JSONDecodeError:
        end = max(payload.rfind("]"), payload.rfind("}"))
        if end <= start:
            raise ValueError("truncated JSON in response")
        return json.loads(payload[start : end + 1])


def _valid_point(item) -> Optional[Tuple[float, float]]:
    if not isinstance(item, dict):
        return None
    pt = item.get("point") or item.get("coordinates") or item.get("position")
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None
    try:
        y, x = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= y <= 1000.0 and 0.0 <= x <= 1000.0):
        return None
    return y, x


def select_points(items) -> Tuple[dict, dict]:
    """Pick the object point and the container point out of the model's answer.

    Labels are the primary signal (a tray/bin/basket word wins the container
    slot); when the labels say nothing useful, the declared order is trusted,
    which is exactly the order the prompt asks for.
    """
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise ValueError("response is not a JSON list")

    good = []
    for it in items:
        pt = _valid_point(it)
        if pt is not None:
            good.append({"point": [pt[0], pt[1]], "label": str(it.get("label") or "")})
    if len(good) < 2:
        raise ValueError(f"needed 2 usable points, got {len(good)}")

    def is_container(entry: dict) -> bool:
        lab = entry["label"].lower()
        return "container" in lab or any(w in lab for w in CONTAINER_WORDS)

    containers = [e for e in good if is_container(e)]
    others = [e for e in good if not is_container(e)]
    if containers and others:
        return others[0], containers[0]
    return good[0], good[1]


def ask_gemini(client, model: str, png_bytes: bytes, prompt: str,
               extra: str = "") -> Tuple[list, str, float]:
    """One pointing call.  Returns (parsed_items, raw_text, latency_ms)."""
    from google.genai import types

    text = POINT_PROMPT.format(prompt=prompt) + extra
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            text,
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return parse_json(response.text), (response.text or ""), latency_ms


def mock_points(env: SO101Env, rng: np.random.Generator, noise_px: float,
                width: int, height: int) -> Tuple[dict, dict, float]:
    """Offline stand-in for Gemini: ground truth projected to pixels + noise.

    This exercises every line of the real path except the network call, so the
    un-projection, the driving and the scoring can be trusted before a single
    token is spent.
    """
    t0 = time.perf_counter()
    cube = env.cube_pos()
    tray = env.tray_center
    floor = np.array([tray[0], tray[1], env.tray_floor_top])

    out = []
    for xyz, label in ((cube, "object"), (floor, "container")):
        u, v = pixel_world.project(env.model, env.data, WORLD_CAMERA, xyz, width, height)
        u += float(rng.normal(0.0, noise_px))
        v += float(rng.normal(0.0, noise_px))
        u = float(np.clip(u, 0.0, width - 1))
        v = float(np.clip(v, 0.0, height - 1))
        # Hand it back in Gemini's own [y, x] 0-1000 convention so the rest of
        # the pipeline cannot tell the difference.
        out.append({"point": [v / height * 1000.0, u / width * 1000.0], "label": label})
    return out[0], out[1], (time.perf_counter() - t0) * 1000.0


# ==========================================================================
# drawing
# ==========================================================================
def annotate(frame: np.ndarray, marks: Sequence[dict], path: Path) -> None:
    """Save the frame with one labelled crosshair per Gemini point."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    for m in marks:
        u, v = float(m["pixel"][0]), float(m["pixel"][1])
        color = m.get("color", (0, 255, 0))
        arm, gap, r = 18, 5, 9
        draw.line([(u - arm, v), (u - gap, v)], fill=color, width=2)
        draw.line([(u + gap, v), (u + arm, v)], fill=color, width=2)
        draw.line([(u, v - arm), (u, v - gap)], fill=color, width=2)
        draw.line([(u, v + gap), (u, v + arm)], fill=color, width=2)
        draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=2)
        label = m.get("label", "")
        tx, ty = u + arm + 4, v - 8
        # cheap outline so the text survives on top of a bright tabletop
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color)
    img.save(str(path))


# ==========================================================================
# driving
# ==========================================================================
class WorldRecorder(Recorder):
    """``collect.Recorder`` that films the world camera and injects no noise.

    Everything else -- the ramps, the frozen-arm grasp, the polar transfer --
    is inherited unchanged, so the pilot and the scripted expert execute the
    identical motion for identical targets.
    """

    def tick(self, arm_cmd: np.ndarray, grip_cmd: float, noise_scale: float) -> None:
        env = self.env
        self.states.append(env.measured_qpos())
        self.world_frames.append(env.render_world())
        cmd = env.clip_cmd(np.concatenate([np.asarray(arm_cmd, dtype=float), [float(grip_cmd)]]))
        self.actions.append(cmd.astype(np.float32))
        env.step(cmd)


def drive(env: SO101Env, ik: ArmIK, obj_xyz: np.ndarray, container_xyz: np.ndarray,
          rng: np.random.Generator) -> Tuple[WorldRecorder, dict]:
    """Run the pick-and-place phase machine against two *estimated* points.

    ``obj_xyz`` and ``container_xyz`` come from Gemini (or the mock).  Nothing
    in here touches ``env.cube_pos()`` or ``env.tray_center``.
    """
    args = SimpleNamespace(world_video=True, no_noise=True, grasp="friction")
    rec = WorldRecorder(env, ik, rng, args)

    q_home = np.asarray(env.home_qpos[:5], dtype=float).copy()
    q_home[4] = WRIST_ROLL_LOCK

    grasp_p, _q, jaw_yaw = ik.grasp_pose(np.asarray(obj_xyz, dtype=float), q_home)
    hover_p = np.array([grasp_p[0], grasp_p[1], CARRY_SITE_Z])
    over_p = np.array([container_xyz[0], container_xyz[1], CARRY_SITE_Z])
    drop_p = np.array([container_xyz[0], container_xyz[1], RELEASE_SITE_Z])

    q = q_home.copy()
    start_p, _ = ik.fk(q)
    rise_p = np.array([start_p[0], start_p[1], CARRY_SITE_Z])

    # 1. up out of the table plane, then across at carry height (polar arc)
    q = rec.ramp(start_p, rise_p, q, T_RISE, GRIPPER_OPEN, noise_scale=0.0)
    q = rec.ramp(rise_p, hover_p, q, T_TRAVERSE, GRIPPER_OPEN, noise_scale=0.0, polar=True)
    # 2. descend onto the estimated grasp pose
    q = rec.ramp(hover_p, grasp_p, q, T_DESCEND, GRIPPER_OPEN, noise_scale=0.0)
    rec.hold(q, GRIPPER_OPEN, T_PRESETTLE)
    # 3. GRASP -- arm command frozen while the jaws ramp closed
    for i in range(T_GRASP_RAMP):
        g = GRIPPER_OPEN + smoothstep((i + 1) / float(T_GRASP_RAMP)) * (GRIPPER_GRIP - GRIPPER_OPEN)
        rec.tick(q, g, 0.0)
    rec.hold(q, GRIPPER_GRIP, T_GRASP_SETTLE)
    diag = {
        "grasped": bool(env.measured_qpos()[5] > 0.5 * (GRIPPER_GRIP + GRIP_STALL)),
        "grip_after_close": round(float(env.measured_qpos()[5]), 4),
        "jaw_yaw": round(float(jaw_yaw), 4),
        "grasp_target": np.asarray(grasp_p).round(4).tolist(),
    }
    # 4. lift, transfer over the estimated container, lower, release, retreat
    q = rec.ramp(grasp_p, hover_p, q, T_LIFT, GRIPPER_GRIP, noise_scale=0.0)
    q = rec.ramp(hover_p, over_p, q, T_TRANSFER, GRIPPER_GRIP, noise_scale=0.0, polar=True)
    q = rec.ramp(over_p, drop_p, q, T_LOWER, GRIPPER_GRIP, noise_scale=0.0)
    for i in range(T_RELEASE_RAMP):
        g = GRIPPER_GRIP + smoothstep((i + 1) / float(T_RELEASE_RAMP)) * (GRIPPER_OPEN - GRIPPER_GRIP)
        rec.tick(q, g, 0.0)
    rec.hold(q, GRIPPER_OPEN, T_RELEASE_SETTLE)
    q = rec.ramp(drop_p, over_p, q, T_RETREAT, GRIPPER_OPEN, noise_scale=0.0)
    rec.hold(q, GRIPPER_OPEN, T_FINAL_SETTLE)

    diag["max_ik_err_mm"] = round(rec.max_ik_err * 1000.0, 2)
    return rec, diag


# ==========================================================================
# one episode
# ==========================================================================
def perceive(env: SO101Env, args, client, png: bytes, rng: np.random.Generator,
             width: int, height: int) -> Tuple[dict, dict, dict]:
    """Get two labelled image points, retrying once with a stricter reminder.

    Returns ``(object_entry, container_entry, meta)``.  Raises ValueError with
    a human-readable reason if both attempts fail.
    """
    if args.mock_gemini:
        obj, cont, latency = mock_points(env, rng, args.mock_noise_px, width, height)
        return obj, cont, {"latency_ms": round(latency, 1), "attempts": 1,
                           "model": "mock", "raw": None}

    extra, why, last = "", "", ""
    total_ms = 0.0
    for attempt in (1, 2):
        try:
            items, raw, latency = ask_gemini(client, args.model, png, args.prompt, extra)
            total_ms += latency
            obj, cont = select_points(items)
            return obj, cont, {"latency_ms": round(total_ms, 1), "attempts": attempt,
                               "model": args.model, "raw": raw[:2000]}
        except Exception as exc:   # network, parse, or too-few-points
            why = f"{type(exc).__name__}: {exc}"
            last = why
            extra = STRICTER_REMINDER.replace("{why}", why[:200])
            print(f"    perception attempt {attempt} failed -- {why}")
    raise ValueError(f"perception failed after 2 attempts ({last})")


def run_episode(env: SO101Env, ik: ArmIK, args, client, idx: int, seed: int,
                out: Path) -> dict:
    """Reset, perceive, drive, score.  Never raises."""
    W, H = WORLD_SIZE
    rng = np.random.default_rng(seed ^ 0xA5A5)
    stem = f"ep{idx:03d}"
    rec_out: dict = {
        "episode": idx, "seed": int(seed), "prompt": args.prompt,
        "mock": bool(args.mock_gemini), "success": False, "reason": None,
        "video": f"{stem}_world.mp4", "annotated": f"{stem}_annotated.png",
    }

    # ---- 1. scene randomisation (privileged by construction: this IS the
    #         world being set up, not the policy looking at it) -------------
    env.reset(
        seed=seed,
        tray_azimuth=env.tray_home_azimuth
        + float(rng.uniform(-TRAY_AZIMUTH_JITTER, TRAY_AZIMUTH_JITTER)),
        settle_seconds=0.5,
    )
    q_home = np.asarray(env.home_qpos[:5], dtype=float).copy()
    q_home[4] = WRIST_ROLL_LOCK
    # With wrist_roll pinned the jaw axis is an output of the arm pose, so the
    # cube is spawned square to it (+/- jitter) -- same convention as collect.py.
    _t, _q, jaw_yaw = ik.grasp_pose(env.cube_pos(), q_home)
    env.set_cube_yaw(jaw_yaw + float(rng.uniform(-CUBE_YAW_JITTER, CUBE_YAW_JITTER)),
                     settle_seconds=0.3)

    gt_cube = env.cube_pos()
    gt_tray = env.tray_center
    gt_container = np.array([gt_tray[0], gt_tray[1], env.tray_floor_top])
    rec_out["ground_truth"] = {
        "cube": gt_cube.round(4).tolist(),
        "tray_center": gt_tray.round(4).tolist(),
        "container_point": gt_container.round(4).tolist(),
    }

    # ---- 2. what the model sees ---------------------------------------------
    frame = env.render_world()
    png = _png_bytes(frame)

    try:
        obj_entry, cont_entry, meta = perceive(env, args, client, png, rng, W, H)
    except Exception as exc:
        rec_out["reason"] = str(exc)
        rec_out["gemini"] = {"latency_ms": None, "attempts": 2,
                             "model": "mock" if args.mock_gemini else args.model}
        annotate(frame, [], out / rec_out["annotated"])
        return rec_out
    rec_out["gemini"] = meta

    # ---- 3. pixels -> world --------------------------------------------------
    marks = []
    try:
        targets = {}
        for key, entry, plane_z, gt, color in (
            ("object", obj_entry, OBJECT_PLANE_Z, gt_cube, (255, 140, 0)),
            ("container", cont_entry, CONTAINER_PLANE_Z, gt_container, (0, 220, 255)),
        ):
            u, v = pixel_world.gemini_point_to_pixel(entry["point"], W, H)
            xyz = pixel_world.unproject(env.model, env.data, WORLD_CAMERA, u, v, plane_z, W, H)
            gu, gv = pixel_world.project(env.model, env.data, WORLD_CAMERA, gt, W, H)
            targets[key] = xyz
            rec_out[key] = {
                "label": entry["label"],
                "point_yx_norm": [round(float(c), 1) for c in entry["point"]],
                "pixel": [round(u, 1), round(v, 1)],
                "gt_pixel": [round(gu, 1), round(gv, 1)],
                "pixel_error_px": round(float(math.hypot(u - gu, v - gv)), 1),
                "world": [round(float(c), 4) for c in xyz],
                "gt_world": [round(float(c), 4) for c in gt],
                "error_mm": round(float(np.linalg.norm(xyz - gt)) * 1000.0, 1),
                "error_xy_mm": round(float(np.linalg.norm(xyz[:2] - gt[:2])) * 1000.0, 1),
                # Polar form of the estimate: a point that lands off the table
                # (radius far outside the ~0.16-0.28 m reachable band) explains
                # an episode where the IK clamped and the arm flailed.
                "radius_m": round(float(math.hypot(xyz[0], xyz[1])), 4),
                "azimuth_deg": round(math.degrees(math.atan2(xyz[1], xyz[0])), 1),
            }
            lab = entry["label"].strip()
            marks.append({"pixel": (u, v), "color": color,
                          "label": f"{key}: {lab}" if lab and lab.lower() != key else key})
    except Exception as exc:
        rec_out["reason"] = f"un-projection failed: {type(exc).__name__}: {exc}"
        annotate(frame, marks, out / rec_out["annotated"])
        return rec_out

    annotate(frame, marks, out / rec_out["annotated"])

    # ---- 4. drive on the estimates only -------------------------------------
    try:
        rec, diag = drive(env, ik, targets["object"], targets["container"], rng)
    except Exception as exc:
        rec_out["reason"] = f"execution failed: {type(exc).__name__}: {exc}"
        rec_out["traceback"] = traceback.format_exc(limit=4)
        return rec_out

    write_video(out / rec_out["video"], rec.world_frames, fps=VIDEO_FPS)

    # ---- 5. score (ground truth allowed from here down) ---------------------
    rec_out["success"] = bool(env.success())
    rec_out["diag"] = diag
    rec_out["ticks"] = int(len(rec.states))
    rec_out["cube_final"] = env.cube_pos().round(4).tolist()
    rec_out["cube_to_tray_xy_mm"] = round(
        float(np.linalg.norm(env.cube_pos()[:2] - env.tray_center[:2])) * 1000.0, 1)
    if not rec_out["success"]:
        rec_out["reason"] = "grasp failed" if not diag["grasped"] else "cube not in tray"
    return rec_out


def _png_bytes(frame: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ==========================================================================
# main
# ==========================================================================
def build_env(scene: str) -> Tuple[SO101Env, ArmIK]:
    """Env + IK wired to the same measured constants the expert uses."""
    env = SO101Env(
        scene,
        render_size=(224, 224),           # wrist cam: unused here, kept small
        world_render_size=WORLD_SIZE,
        gripper_open=GRIPPER_OPEN,
        gripper_grip=GRIPPER_GRIP,
        spawn_radius=CUBE_RADIUS_BAND,
        spawn_azimuth=CUBE_AZIMUTH_BAND,
        tray_floor_offset=TRAY_FLOOR_TOP,
        success_xy_tol=SUCCESS_XY_TOL,
    )
    if env.world_camera_name != WORLD_CAMERA:
        raise SystemExit(
            f"{scene}: needs a camera named '{WORLD_CAMERA}' "
            f"(found {env.world_camera_name!r}); pixel<->world maths is camera specific."
        )
    ik = ArmIK(env)
    # Elevated rest pose, exactly as collect.py solves it: the scene's neutral
    # poses park the tool ON the table in the middle of the spawn arc.
    home_site = np.array([
        HOME_SITE_RADIUS * math.cos(HOME_SITE_AZIMUTH),
        HOME_SITE_RADIUS * math.sin(HOME_SITE_AZIMUTH),
        CARRY_SITE_Z,
    ])
    q_rest, rest_err = ik.solve(home_site, np.asarray(env.home_qpos[:5], dtype=float))
    if rest_err > 0.003:
        raise SystemExit(f"rest pose unreachable ({rest_err * 1000:.1f} mm)")
    env.home_qpos = np.concatenate([q_rest, [GRIPPER_OPEN]])
    return env, ik


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="what the operator asks for, in plain English")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--model", default=ER2_MODEL, help="Gemini model id")
    ap.add_argument("--mock-gemini", action="store_true",
                    help="skip the API: ground truth projected to pixels + gaussian noise")
    ap.add_argument("--mock-noise-px", type=float, default=MOCK_NOISE_PX,
                    help=f"sigma of the mock pointing error in pixels (default {MOCK_NOISE_PX})")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env, ik = build_env(args.scene)
    client = None if args.mock_gemini else make_client()

    print(f"scene={env.scene_path} world_cam={env.world_camera_name} "
          f"image={WORLD_SIZE[0]}x{WORLD_SIZE[1]}")
    print(f"model={'mock' if args.mock_gemini else args.model}  prompt={args.prompt!r}")
    print(f"planes: object z={OBJECT_PLANE_Z}  container z={CONTAINER_PLANE_Z}\n")

    results: List[dict] = []
    for i in range(args.episodes):
        seed = args.seed * 1000 + i
        rec = run_episode(env, ik, args, client, i, seed, out)
        results.append(rec)

        tag = "OK  " if rec["success"] else "FAIL"
        if "object" in rec and "container" in rec:
            detail = (f"point_err obj={rec['object']['error_mm']:6.1f}mm "
                      f"cont={rec['container']['error_mm']:6.1f}mm "
                      f"({rec['object']['pixel_error_px']:.1f}px / "
                      f"{rec['container']['pixel_error_px']:.1f}px)")
        else:
            detail = "no usable points"
        lat = rec.get("gemini", {}).get("latency_ms")
        print(f"[{tag}] ep{i:03d} seed={seed} {detail} "
              f"gemini={lat if lat is None else f'{lat:.0f}'}ms"
              f"{'' if rec['reason'] is None else '  reason=' + rec['reason']}")

    scored = [r for r in results if "object" in r]
    n_ok = sum(1 for r in results if r["success"])
    summary = {
        "episodes": args.episodes,
        "successes": n_ok,
        "success_rate": n_ok / max(args.episodes, 1),
        "perception_failures": sum(1 for r in results if "object" not in r),
        "mean_object_error_mm": round(
            float(np.mean([r["object"]["error_mm"] for r in scored])), 1) if scored else None,
        "mean_container_error_mm": round(
            float(np.mean([r["container"]["error_mm"] for r in scored])), 1) if scored else None,
        "mean_latency_ms": round(float(np.mean(
            [r["gemini"]["latency_ms"] for r in results
             if r.get("gemini", {}).get("latency_ms") is not None])), 1)
        if any(r.get("gemini", {}).get("latency_ms") is not None for r in results) else None,
    }
    (out / "results.json").write_text(json.dumps(
        {"prompt": args.prompt,
         "model": "mock" if args.mock_gemini else args.model,
         "scene": env.scene_path,
         "image_size": list(WORLD_SIZE),
         "planes": {"object_z": OBJECT_PLANE_Z, "container_z": CONTAINER_PLANE_Z},
         "summary": summary,
         "episodes_detail": results},
        indent=2))
    env.close()

    print(f"\n{n_ok}/{args.episodes} succeeded ({summary['success_rate']:.0%})  "
          f"mean pointing error: object={summary['mean_object_error_mm']}mm "
          f"container={summary['mean_container_error_mm']}mm  "
          f"mean latency={summary['mean_latency_ms']}ms")
    print(f"wrote {out / 'results.json'} and {len(results)} episode(s) to {out}")


if __name__ == "__main__":
    main()
