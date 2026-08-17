"""SO-101 scripted pick and place, in joint space. No IK.

    antioch scenario run --scenario so101_pick_place

The expert is a list of joint keyframes interpolated with a smoothstep. Every
constant below was measured on the asset with the probes in this directory
rather than guessed, because four separate assumptions were wrong:

  * The jaw only closes horizontally at wrist_roll = pi/2 (probe_roll.py). At
    roll = 0 the closing axis is mostly vertical -- the jaw swings down from
    z=0.067 to z=0.006 and swats the top of a block sitting on the table, which
    is why the jaw always reached its commanded angle with nothing in it.
  * The grasp centre is TCP_LOCAL below, derived in probe_tcp.py by inverting a
    grasp point that actually holds. The previous value was ~35 mm out.
  * JAW_GRIP = 0.0 preloads; -0.10 over-squeezes and ejects the block on every
    one of nine trials (probe_grasp_rate.py).
  * The lift must be ramped. Stepping the command straight to the lift pose
    shakes the block out of the fingers; smoothstepping it holds 3/3.

The block is placed where the fingers land rather than the fingers being solved
onto the block, so the grasp is reachable by construction. Only the base
rotation is aimed, so the reachable set is an arc at RADIUS -- a block off that
radius fails the reach check loudly instead of silently missing.

ACT trains on what this records: wrist image + joint state -> next joint
targets. Joint space throughout.
"""

from __future__ import annotations

from pathlib import Path

import antioch

logger = antioch.Logger("pickplace")

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"

# Jaw angle -> gap, measured on the asset: -0.17 shut, +0.30 ~20 mm, +0.80 ~47 mm.
JAW_OPEN = 0.85
JAW_GRIP = 0.0  # the jaw stalls at ~0.28 on the 30.5 mm block: a real preload
GRIP_FRICTION = 1.5  # textured printed jaws; tune from matched real lifts

WRIST_ROLL = 1.5708  # the only roll at which the jaw closes horizontally

# Arm poses (Pitch, Elbow, Wrist_Pitch); the roll is fixed and the base
# rotation is aimed per block.
DOWN = (1.2785, -1.2550, 0.9550)
LIFT = (0.70, -0.85, 0.55)
RELEASE_FRACTION = 0.45  # release this far toward LIFT so the fingers clear the tray walls

# Grasp centre in the gripper link frame (probe_tcp.py).
TCP_LOCAL = (0.02098, 0.01385, -0.07704)

# Base rotation -> TCP azimuth, measured at rot = +-0.45: azimuth is linear in
# the base joint but not with unit slope, because reach and droop both vary.
AZIMUTH_AT_ZERO = 0.1016
AZIMUTH_PER_ROT = -0.9354
RADIUS = 0.3663  # TCP radius at the DOWN pose


def _rot_for(x, y):
    import numpy as np

    return float((AZIMUTH_AT_ZERO - np.arctan2(y, x)) / -AZIMUTH_PER_ROT)


def _smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@antioch.scenario(
    tags=["pickplace", "smoke"],
    sim=antioch.BootProfile(physics_dt=1.0 / 120.0, render_dt=1.0 / 60.0),
    cases=[
        antioch.case(
            # Sweep the arc the fixed reach can actually serve.
            grid={"block_azimuth": [-0.42, -0.32, -0.22, -0.12]},
            id="az{block_azimuth}",
            tags=["sweep"],
        )
    ],
)
def so101_pick_place(
    run: antioch.ScenarioRun,
    block_azimuth: float = antioch.param(
        -0.3157, ge=-0.9, le=0.9, description="Block bearing from the base, radians"
    ),
    tray_azimuth: float = antioch.param(
        0.5225, ge=-0.9, le=0.9, description="Tray bearing from the base, radians"
    ),
) -> None:
    """Pick a block off the table and place it in the wooden tray."""

    import numpy as np
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from pxr import UsdPhysics, UsdShade

    from trapezoid_block import HEIGHT, add_trapezoid_block
    from wooden_tray import add_wooden_tray

    block_x = RADIUS * float(np.cos(block_azimuth))
    block_y = RADIUS * float(np.sin(block_azimuth))
    tray_x = RADIUS * float(np.cos(tray_azimuth))
    tray_y = RADIUS * float(np.sin(tray_azimuth))
    pick_rot = _rot_for(block_x, block_y)
    place_rot = _rot_for(tray_x, tray_y)

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 700.0})

    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    add_wooden_tray(world, center=(tray_x, tray_y), outer_size=(0.16, 0.14))
    block = add_trapezoid_block(world, (block_x, block_y, HEIGHT / 2.0))

    stage = get_current_stage()
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/grip")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(GRIP_FRICTION)
    physics.CreateDynamicFrictionAttr(GRIP_FRICTION)
    physics.CreateRestitutionAttr(0.0)
    for path in ("/World/SO101/gripper", "/World/SO101/jaw", block.prim_path):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    world.reset()
    block.bind()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    grip = RigidPrim("/World/SO101/gripper")
    tcp_local = np.array(TCP_LOCAL)

    def tcp_world():
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return pos + R @ tcp_local

    set_camera_view(
        eye=[0.85, -0.45, 0.55],
        target=[0.5 * (block_x + tray_x), 0.5 * (block_y + tray_y), 0.05],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Wrist camera, parented to the gripper. This is the observation ACT trains
    # on. Gripper frame: fingers and lens point along -Z and +Y is up in this
    # pose, so mount above and roll the image plane 180 degrees to stay upright.
    cam = Camera(
        prim_path="/World/SO101/gripper/wrist_cam",
        translation=np.array([0.0, 0.075, 0.02]),
        orientation=np.array([0.0, 0.0, -1.0, 0.0]),
        frequency=20,
        resolution=(640, 480),
    )
    cam.initialize()
    shots = {}
    useful_frames = world_frames = 0

    def key(rot, pose, jaw):
        return np.array([rot, pose[0], pose[1], pose[2], WRIST_ROLL, jaw], dtype=np.float64)

    release_pose = tuple(
        d + RELEASE_FRACTION * (l - d) for d, l in zip(DOWN, LIFT)
    )

    keyframes = [
        ("start", key(pick_rot, LIFT, JAW_OPEN), 90),
        ("descend", key(pick_rot, DOWN, JAW_OPEN), 140),
        ("grasp", key(pick_rot, DOWN, JAW_GRIP), 140),
        ("lift", key(pick_rot, LIFT, JAW_GRIP), 300),
        ("transfer", key(place_rot, LIFT, JAW_GRIP), 220),
        ("lower", key(place_rot, release_pose, JAW_GRIP), 160),
        ("release", key(place_rot, release_pose, JAW_OPEN), 120),
        ("retreat", key(place_rot, LIFT, JAW_OPEN), 140),
    ]

    # Settle at the start pose before the demo begins, so the first keyframe is
    # not a slew from the asset's rest pose.
    q_prev = keyframes[0][1].copy()
    arm.set_joint_positions(q_prev.astype(np.float32))
    arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
    for _ in range(120):
        arm.apply_action(ArticulationAction(joint_positions=q_prev.astype(np.float32)))
        world.step(render=False)

    aim_error_mm = None
    states, actions = [], []

    for name, target, ticks in keyframes:
        for tick in range(ticks):
            cmd = q_prev + _smoothstep((tick + 1) / ticks) * (target - q_prev)
            states.append(np.asarray(arm.get_joint_positions(), dtype=np.float32).copy())
            actions.append(cmd.astype(np.float32).copy())
            arm.apply_action(ArticulationAction(joint_positions=cmd.astype(np.float32)))
            world.step(render=True)
        q_prev = target.copy()

        tcp = tcp_world()
        if name == "descend":
            aim_error_mm = float(np.linalg.norm(tcp[:2] - np.array([block_x, block_y])) * 1000.0)

        rgba = cam.get_rgba()
        if rgba is not None and rgba.size:
            rgb = np.rot90(np.asarray(rgba)[:, :, :3], 2).astype(np.uint8)
            shots[name] = rgb
            logger.image("camera/rgb", rgb)
            useful_frames += int(float(rgb.std()) > 5.0 and float((rgb >= 250).mean()) < 0.8)
        world_rgb = antioch.capture_viewport()
        if world_rgb is not None and world_rgb.size:
            logger.image("camera/world", np.asarray(world_rgb)[:, :, :3].astype(np.uint8))
            world_frames += 1

        bp = np.asarray(block.get_world_pose()[0], dtype=float)
        logger.scalar("block/z", float(bp[2]))
        logger.scalar("jaw/position", float(np.asarray(arm.get_joint_positions())[5]))
        print(
            f"  {name:9s} TCP={np.round(tcp, 4).tolist()}  block={np.round(bp, 3).tolist()}"
            f"  jaw={float(np.asarray(arm.get_joint_positions())[5]):+.3f}"
        )

    for _ in range(180):
        world.step(render=True)

    if shots:
        from PIL import Image

        order = [k for k in ("descend", "grasp", "lift", "lower") if k in shots]
        strip = np.concatenate([shots[k] for k in order], axis=1)
        out = Path("/tmp/pickplace_strip.png")
        Image.fromarray(strip).save(out)
        run.add_artifact(out, name="pickplace_strip.png")

    final = np.asarray(block.get_world_pose()[0], dtype=float)
    xy_err = float(np.linalg.norm(final[:2] - np.array([tray_x, tray_y])))
    moved = float(np.linalg.norm(final[:2] - np.array([block_x, block_y])))

    run.add_result("block_final", [round(float(v), 4) for v in final])
    run.add_result("place_xy_error_m", round(xy_err, 4))
    run.add_result("aim_error_mm", round(aim_error_mm or -1.0, 2))
    run.add_result("useful_camera_frames", useful_frames)
    run.add_result("world_camera_frames", world_frames)
    run.add_result("demo_steps", len(actions))

    run.check(
        "the fingers were aimed at the block",
        (aim_error_mm or 1e9) < 12.0,
        detail=f"{aim_error_mm:.1f} mm from the block centre" if aim_error_mm else "not measured",
    )
    run.check("the block moved from where it started", moved > 0.03, detail=f"moved {moved * 1000:.0f} mm")
    run.check("the block ended in the tray", xy_err < 0.06, detail=f"{xy_err * 1000:.0f} mm from the tray centre")
    run.check(
        "the wrist camera produced usable RGB",
        useful_frames >= 4,
        detail=f"{useful_frames}/{len(keyframes)} frames were non-uniform and not overexposed",
    )
    run.check(
        "the world camera produced RGB",
        world_frames >= 4,
        detail=f"{world_frames}/{len(keyframes)} world frames captured",
    )
