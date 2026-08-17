"""Smooth Cartesian-servo pick/place candidate for the physical SO-101 pose."""

from __future__ import annotations

from pathlib import Path

import antioch

logger = antioch.Logger("pickplace_servo")


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


@antioch.scenario(
    tags=["pickplace", "servo", "smoke"],
    sim=antioch.BootProfile(physics_dt=1.0 / 120.0, render_dt=1.0 / 60.0),
)
def so101_pick_place_servo(
    run: antioch.ScenarioRun,
    block_x: float = antioch.param(0.3464, ge=0.20, le=0.45),
    block_y: float = antioch.param(-0.1361, ge=-0.30, le=0.30),
    tray_x: float = antioch.param(0.3395, ge=0.20, le=0.45),
    tray_y: float = antioch.param(0.1744, ge=-0.30, le=0.30),
) -> None:
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from pxr import UsdPhysics, UsdShade

    from trapezoid_block import GRASP_WIDTH, HEIGHT, add_trapezoid_block
    from wooden_tray import add_wooden_tray
    from wrist_camera import measure_frame

    jaw_open, jaw_grip, wrist_roll = 0.85, 0.34, np.pi / 2.0
    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 300.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 700.0})
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    add_wooden_tray(world, center=(tray_x, tray_y), outer_size=(0.16, 0.14))
    block = add_trapezoid_block(world, (block_x, block_y, HEIGHT / 2.0))

    stage = get_current_stage()
    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/servo_grip")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(2.0)
    physics.CreateDynamicFrictionAttr(2.0)
    physics.CreateRestitutionAttr(0.0)
    for path in ("/World/SO101/gripper", "/World/SO101/jaw", block.prim_path):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    world.reset()
    block.bind()
    arm = SingleArticulation(prim_path="/World/SO101", name="servo_so101")
    arm.initialize()
    tensor_arm = Articulation("/World/SO101")
    grip = RigidPrim("/World/SO101/gripper")
    grip_index = list(tensor_arm._link_names).index("gripper")
    tcp_local = np.array([-0.0438 + GRASP_WIDTH / 2.0, 0.0, -0.085])

    def tcp_pose():
        pos, quat = (item.numpy()[0] for item in grip.get_world_poses())
        w, x, y, z = quat
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ])
        offset = rotation @ tcp_local
        return np.asarray(pos, dtype=float) + offset, offset

    camera = Camera(
        prim_path="/World/SO101/gripper/wrist_cam",
        translation=np.array([0.0, 0.075, 0.02]),
        orientation=np.array([0.0, 0.0, -1.0, 0.0]),
        frequency=20,
        resolution=(640, 480),
    )
    camera.initialize()
    set_camera_view(
        eye=[0.72, -0.48, 0.38],
        target=[0.34, 0.02, 0.055],
        camera_prim_path="/OmniverseKit_Persp",
    )

    initial = np.array([0.36, 1.415, -0.603, -1.459, wrist_roll, jaw_open], dtype=np.float32)
    arm.set_joint_positions(initial)
    arm.apply_action(ArticulationAction(joint_positions=initial))
    for _ in range(120):
        world.step(render=True)

    rest_z = float(block.get_world_pose()[0][2])
    peak_z = rest_z
    approach_z = HEIGHT / 2.0 + 0.095
    contact_z = HEIGHT / 2.0 + 0.005
    phases = [
        ("approach", np.array([block_x, block_y, approach_z]), jaw_open, 180),
        ("descend", np.array([block_x, block_y, contact_z]), jaw_open, 180),
        ("grasp", np.array([block_x, block_y, contact_z]), jaw_grip, 150),
        ("lift", np.array([block_x, block_y, approach_z]), jaw_grip, 210),
        ("transfer", np.array([tray_x, tray_y, approach_z]), jaw_grip, 360),
        ("lower", np.array([tray_x, tray_y, contact_z]), jaw_grip, 210),
        ("release", np.array([tray_x, tray_y, contact_z]), jaw_open, 150),
        ("retreat", np.array([tray_x, tray_y, approach_z]), jaw_open, 180),
    ]
    desired_start = tcp_pose()[0].copy()
    jaw_start = jaw_open
    shots = {}
    phase_errors = {}
    useful_frames = red_frames = 0

    for name, target, jaw_target, ticks in phases:
        for tick in range(ticks):
            blend = smoothstep((tick + 1) / ticks)
            desired = desired_start + blend * (target - desired_start)
            jaw = jaw_start + blend * (jaw_target - jaw_start)
            point, offset = tcp_pose()
            raw = tensor_arm.get_jacobian_matrices().numpy()[0, grip_index, :, 6:12]
            jacobian = raw[:3] + np.cross(raw[3:].T, offset).T
            active = jacobian[:, :4]
            error = desired - point
            damping = 0.025
            q = np.asarray(arm.get_joint_positions(), dtype=float)
            inverse = active.T @ np.linalg.inv(active @ active.T + damping**2 * np.eye(3))
            delta = inverse @ error
            target_angle = -np.arctan2(desired[1], desired[0])
            if name in {"descend", "grasp"}:
                reference = np.array([target_angle, 1.46, -0.60, -0.51])
            elif name in {"lower", "release"}:
                reference = np.array([target_angle, 1.19, -0.05, -1.29])
            else:
                reference = np.array([target_angle, 1.415, -0.603, -1.459])
            nullspace = np.eye(4) - inverse @ active
            delta += nullspace @ (0.08 * (reference - q[:4]))
            q[:4] += np.clip(delta, -0.015, 0.015)
            q[:4] = np.clip(q[:4], [-1.9, -1.7, -1.65, -1.6], [1.9, 1.7, 1.65, 1.6])
            q[4], q[5] = wrist_roll, jaw
            arm.apply_action(ArticulationAction(joint_positions=q.astype(np.float32)))
            world.step(render=True)
            peak_z = max(peak_z, float(block.get_world_pose()[0][2]))

        point, _ = tcp_pose()
        phase_errors[name] = float(np.linalg.norm(point - target))
        rgba = camera.get_rgba()
        if rgba is not None and rgba.size:
            rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            metrics = measure_frame(rgb)
            shots[name] = rgb
            logger.image("camera/wrist_rgb", rgb)
            useful_frames += int(metrics.usable)
            red_frames += int(metrics.red_block_visible)
        block_pos = np.asarray(block.get_world_pose()[0], dtype=float)
        print(
            f"{name:9s} tcp={np.round(point, 4).tolist()} "
            f"error={phase_errors[name] * 1000:.1f}mm block={np.round(block_pos, 4).tolist()} "
            f"q={np.round(arm.get_joint_positions(), 3).tolist()}"
        )
        desired_start, jaw_start = target, jaw_target

    for _ in range(120):
        world.step(render=True)
    final = np.asarray(block.get_world_pose()[0], dtype=float)
    moved = float(np.linalg.norm(final[:2] - np.array([block_x, block_y])))
    place_error = float(np.linalg.norm(final[:2] - np.array([tray_x, tray_y])))

    if shots:
        from PIL import Image
        order = [key for key in ("approach", "descend", "grasp", "lift", "lower") if key in shots]
        strip = np.concatenate([shots[key] for key in order], axis=1)
        output = Path("/tmp/servo_pickplace_strip.png")
        Image.fromarray(strip).save(output)
        run.add_artifact(output, name="servo_pickplace_strip.png")

    run.add_result("block_final", [round(float(value), 4) for value in final])
    run.add_result("peak_block_z_m", round(peak_z, 4))
    run.add_result("phase_tcp_error_m", {key: round(value, 4) for key, value in phase_errors.items()})
    run.add_result("useful_camera_frames", useful_frames)
    run.add_result("red_camera_frames", red_frames)
    run.check("the servo reached the block", phase_errors.get("grasp", 1.0) < 0.020,
              detail=f"grasp TCP error {phase_errors.get('grasp', 1.0) * 1000:.1f} mm")
    run.check("the block was lifted", peak_z > rest_z + 0.045,
              detail=f"block rose {(peak_z - rest_z) * 1000:.1f} mm")
    run.check("the block moved", moved > 0.03, detail=f"moved {moved * 1000:.1f} mm")
    run.check("the block ended in the tray", place_error < 0.06,
              detail=f"{place_error * 1000:.1f} mm from tray centre")
    run.check("the wrist camera returned usable RGB", useful_frames >= 4,
              detail=f"{useful_frames}/{len(shots)} usable phase frames")
    run.check("the wrist camera saw the red block", red_frames >= 2,
              detail=f"red block visible in {red_frames}/{len(shots)} phase frames")


if __name__ == "__main__":
    raise SystemExit("Run as an Antioch scenario: so101_pick_place_servo")
