"""SO-101 scripted pick and place, in joint space. No IK.

    antioch scenario run --scenario so101_pick_place

The expert is a list of joint keyframes the arm is already known to reach,
interpolated with a smoothstep. The block is placed where the fingers land
rather than the fingers being solved onto the block -- which is why there is
no solver here and nothing to converge.

Pick and place are the SAME arm pose with a different Rotation angle, so both
are reachable by construction: measured, [_, 1.2, -1.2, 0.9, 0, _] puts the
fingertips on the table at radius 0.372 m, and +Rotation swings clockwise.

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
JAW_GRIP = 0.34  # inside contact on a 30 mm block, so the drive preloads
BLOCK = 0.030

# Arm poses (Pitch, Elbow, Wrist_Pitch, Wrist_Roll), tuned against measured TCP
# height. Pick and place need DIFFERENT depths despite being the same reach:
# gravity droop varies with Rotation, and at the untuned shared pose the fingers
# closed 14 mm above the block at pick and 10 mm below the tray floor at place.
LIFT = (0.70, -0.85, 0.55, 0.0)
DOWN_PICK = (1.2785, -1.2550, 0.9550, 0.0)
DOWN_PLACE = (1.1440, -1.1610, 0.8610, 0.0)
PICK_ROT = 0.45
PLACE_ROT = -0.45


def _key(rot, arm, jaw):
    return [rot, arm[0], arm[1], arm[2], arm[3], jaw]


def _smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


@antioch.scenario(
    tags=["pickplace", "smoke"],
    sim=antioch.BootProfile(physics_dt=1.0 / 120.0, render_dt=1.0 / 60.0),
    cases=[
        antioch.case(
            grid={"block_x": [0.31, 0.33, 0.35, 0.37], "block_y": [-0.16, -0.14, -0.12]},
            id="x{block_x}y{block_y}",
            tags=["sweep"],
        )
    ],
)
def so101_pick_place(
    run: antioch.ScenarioRun,
    block_x: float = antioch.param(0.3464, ge=0.10, le=0.50, description="Block x in metres"),
    block_y: float = antioch.param(-0.1361, ge=-0.40, le=0.40, description="Block y in metres"),
    tray_x: float = antioch.param(0.3395, ge=0.10, le=0.50, description="Tray centre x in metres"),
    tray_y: float = antioch.param(0.1744, ge=-0.40, le=0.40, description="Tray centre y in metres"),
) -> None:
    """Pick a block off the table and place it in the wooden tray."""

    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera

    from wooden_tray import add_wooden_tray

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 900.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 1800.0})

    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    add_wooden_tray(world, center=(tray_x, tray_y), outer_size=(0.16, 0.14))
    block = world.scene.add(
        DynamicCuboid(
            prim_path="/World/block",
            name="block",
            position=np.array([block_x, block_y, BLOCK / 2.0]),
            size=BLOCK,
            color=np.array([0.78, 0.14, 0.10]),
            mass=0.02,
        )
    )
    world.reset()

    arm = SingleArticulation(prim_path="/World/SO101", name="so101")
    arm.initialize()
    grip = RigidPrim("/World/SO101/gripper")

    # Grasp centre in the gripper link frame, measured off the asset: the jaw
    # closes along +X against a fixed finger at x=-0.0438, fingers run along -Z,
    # and the pads span z in [-0.115, -0.03].
    TCP_LOCAL = np.array([-0.0438 + BLOCK / 2.0, 0.0, -0.085])

    def tcp_world():
        pos, quat = (a.numpy()[0] for a in grip.get_world_poses())
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        return pos + R @ TCP_LOCAL

    set_camera_view(
        eye=[0.85, -0.45, 0.55],
        target=[0.5 * (block_x + tray_x), 0.5 * (block_y + tray_y), 0.05],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # Wrist camera, parented to the gripper. This is the observation ACT trains
    # on, and it doubles as the only reliable way to see what the fingers are
    # doing: capture_viewport() returns None headless, and a floating world
    # camera has to be aimed blind.
    # Gripper frame: +X is the jaw closing axis, fingers run along -Z, and the
    # body occupies y in [-0.028, 0.088] -- so sit above it and look down -Z.
    cam = Camera(
        prim_path="/World/SO101/gripper/wrist_cam",
        translation=np.array([0.0, 0.075, 0.02]),
        frequency=20,
        resolution=(640, 480),
    )
    cam.initialize()
    shots = {}

    keyframes = [
        ("start", _key(PICK_ROT, LIFT, JAW_OPEN), 90),
        ("descend", _key(PICK_ROT, DOWN_PICK, JAW_OPEN), 110),
        ("grasp", _key(PICK_ROT, DOWN_PICK, JAW_GRIP), 80),
        ("lift", _key(PICK_ROT, LIFT, JAW_GRIP), 110),
        ("transfer", _key(PLACE_ROT, LIFT, JAW_GRIP), 150),
        ("lower", _key(PLACE_ROT, DOWN_PLACE, JAW_GRIP), 120),
        ("release", _key(PLACE_ROT, DOWN_PLACE, JAW_OPEN), 80),
        ("retreat", _key(PLACE_ROT, LIFT, JAW_OPEN), 100),
    ]

    states, actions, frames = [], [], 0
    q_prev = np.array(keyframes[0][1], dtype=np.float64)
    arm.set_joint_positions(q_prev.astype(np.float32))

    for name, target, ticks in keyframes:
        target = np.array(target, dtype=np.float64)
        for tick in range(ticks):
            cmd = q_prev + _smoothstep((tick + 1) / ticks) * (target - q_prev)
            states.append(np.asarray(arm.get_joint_positions(), dtype=np.float32).copy())
            actions.append(cmd.astype(np.float32).copy())
            arm.apply_action(ArticulationAction(joint_positions=cmd.astype(np.float32)))
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is not None and rgba.size:
            rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            shots[name] = rgb
            logger.image("camera/rgb", rgb)
            frames += 1
        q_prev = target
        tcp = tcp_world()
        bp = np.asarray(block.get_world_pose()[0], dtype=float)
        logger.scalar("block/z", float(bp[2]))
        print(f"  {name:9s} TCP={np.round(tcp, 4).tolist()}  block={np.round(bp, 3).tolist()}")

    for _ in range(120):
        world.step(render=True)

    if shots:
        from PIL import Image

        strip = np.concatenate([shots[k] for k in ("descend", "grasp", "lift", "lower") if k in shots], axis=1)
        out = Path("/tmp/pickplace_strip.png")
        Image.fromarray(strip).save(out)
        run.add_artifact(out, name="pickplace_strip.png")

    final = np.asarray(block.get_world_pose()[0], dtype=float)
    xy_err = float(np.linalg.norm(final[:2] - np.array([tray_x, tray_y])))
    run.add_result("block_final", [round(float(v), 4) for v in final])
    run.add_result("place_xy_error_m", round(xy_err, 4))
    run.add_result("review_frames", frames)
    run.add_result("demo_steps", len(actions))
    run.check("the block moved from where it started", float(np.linalg.norm(final[:2] - np.array([block_x, block_y]))) > 0.03,
              detail=f"moved {np.linalg.norm(final[:2] - np.array([block_x, block_y])) * 1000:.0f} mm")
    run.check("the block ended in the tray", xy_err < 0.06, detail=f"{xy_err * 1000:.0f} mm from the tray centre")
