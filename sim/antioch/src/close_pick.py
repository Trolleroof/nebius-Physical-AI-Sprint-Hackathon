"""Close, straight-down SO-101 pick matching the physical reference pose."""

from pathlib import Path

import antioch


@antioch.scenario(
    tags=["close-pick", "smoke"],
    sim=antioch.BootProfile(physics_dt=1.0 / 120.0, render_dt=1.0 / 60.0),
)
def so101_close_pick(
    run: antioch.ScenarioRun,
    block_distance: float = antioch.param(0.26, ge=0.24, le=0.30, description="Metres from base pivot"),
) -> None:
    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 700.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 1200.0})
    antioch.load_asset("so101_antioch", prim_path="/World/SO101", version="1.3.2")
    block = world.scene.add(
        DynamicCuboid(
            prim_path="/World/block",
            name="close_block",
            position=np.array([0.0, 0.60, 0.015]),
            size=0.03,
            color=np.array([0.85, 0.03, 0.02]),
            mass=0.02,
        )
    )
    world.reset()

    arm = SingleArticulation("/World/SO101", name="close_pick_arm")
    arm.initialize()
    gripper = RigidPrim("/World/SO101/gripper")
    # Measured by probe_tcp.py from a grasp cell that holds 3/3 trials.
    tcp_local = np.array([0.02098, 0.01385, -0.07704])
    finger_local = np.array([0.0, 0.0, -1.0])
    # !! WRIST_ROLL WARNING !!  The physical arm's wrist_roll servo is BROKEN --
    # the joint is taped at -pi/2 and must never be commanded off it.  The MuJoCo
    # pipeline (sim/mujoco/env.py WRIST_ROLL_LOCK) pins it at -pi/2, and there it
    # is verified that +pi/2 puts the hand UPSIDE DOWN.  The value below is
    # +pi/2, i.e. 180 deg from the physical lock.
    # This has NOT been flipped, because the Isaac `so101_antioch` asset may not
    # share MuJoCo's joint sign convention, and guessing wrong would command the
    # very 180 deg roll we are trying to prevent.  Confirm the sign against the
    # Isaac asset before running this scenario on hardware-facing data.
    wrist_roll, jaw_open, jaw_closed = np.pi / 2.0, 0.85, 0.20

    def pose():
        position, quaternion = (value.numpy()[0] for value in gripper.get_world_poses())
        w, x, y, z = quaternion
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ])
        return np.asarray(position) + rotation @ tcp_local, rotation @ finger_local

    # Bounded pose search: every trial remains inside the known-safe SO-101
    # posture range, so a bad IK step cannot poison live PhysX with NaNs.
    def solve(target, center=None):
        desired_axis = np.array([0.0, 0.0, -1.0])
        if center is None:
            grids = (np.linspace(0.65, 1.65, 7), np.linspace(-1.65, -0.35, 8),
                     np.linspace(-1.20, 1.20, 9))
        else:
            grids = tuple(np.linspace(value - 0.50, value + 0.50, 7) for value in center[1:4])
        best = None
        for shoulder in grids[0]:
            for elbow in grids[1]:
                for wrist in grids[2]:
                    q = np.array([0.0, shoulder, elbow, wrist, wrist_roll, jaw_open])
                    arm.set_joint_positions(q.astype(np.float32))
                    arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
                    world.step(render=False)
                    point, axis = pose()
                    if not np.isfinite(point).all() or not np.isfinite(axis).all():
                        continue
                    score = float(np.linalg.norm(target - point) + 0.12 * (1.0 - np.dot(axis, desired_axis)))
                    if best is None or score < best[0]:
                        best = (score, q.copy())
        if best is None:
            raise RuntimeError("no finite close-pick pose found")
        q = best[1]
        for step in (0.08, 0.025, 0.008):
            candidates = []
            for joint in (1, 2, 3):
                for direction in (-1.0, 1.0):
                    candidate = q.copy()
                    candidate[joint] += direction * step
                    candidates.append(candidate)
            candidates.append(q)
            for candidate in candidates:
                arm.set_joint_positions(candidate.astype(np.float32))
                arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
                world.step(render=False)
                point, axis = pose()
                score = float(np.linalg.norm(target - point) + 0.12 * (1.0 - np.dot(axis, desired_axis)))
                if np.isfinite(score) and score < best[0]:
                    best = (score, candidate.copy())
            q = best[1]
        arm.set_joint_positions(q.astype(np.float32))
        arm.set_joint_velocities(np.zeros(6, dtype=np.float32))
        world.step(render=False)
        point, axis = pose()
        if np.linalg.norm(target - point) > 0.015 or np.dot(axis, desired_axis) < 0.97:
            raise RuntimeError(
                f"close pose is not reachable: tcp={np.round(point, 4)}, axis={np.round(axis, 3)}"
            )
        return q

    grasp_target = np.array([block_distance, 0.0, 0.022])
    grasp = solve(grasp_target)
    above = solve(grasp_target + np.array([0.0, 0.0, 0.09]), grasp)

    # Real drives settle a few millimetres away from teleported FK. Measure the
    # loaded grasp pose once and stage the fixed pickup zone under that center.
    arm.set_joint_positions(grasp.astype(np.float32))
    for _ in range(120):
        arm.apply_action(ArticulationAction(joint_positions=grasp.astype(np.float32)))
        world.step(render=False)
    pickup_xy = pose()[0][:2]
    block.set_world_pose(position=np.r_[pickup_xy, 0.015])
    block.set_linear_velocity(np.zeros(3))
    block.set_angular_velocity(np.zeros(3))

    camera = Camera(
        prim_path="/World/SO101/gripper/wrist_cam",
        translation=np.array([0.0, 0.075, 0.020]),
        orientation=np.array([-0.4089, -0.6641, 0.5769, -0.2429]),
        frequency=20,
        resolution=(640, 480),
    )
    camera.initialize()
    set_camera_view(
        eye=[0.42, -0.40, 0.28],
        target=[0.15, 0.0, 0.09],
        camera_prim_path="/OmniverseKit_Persp",
    )

    phases = [
        ("approach", above, 140),
        ("descend", grasp, 140),
        ("close", np.r_[grasp[:5], jaw_closed], 120),
        ("lift", np.r_[above[:5], jaw_closed], 180),
    ]
    current = above.copy()
    arm.set_joint_positions(current.astype(np.float32))
    snapshots = []
    grasp_axis = None
    for name, target, ticks in phases:
        target = np.asarray(target, dtype=float)
        start = current.copy()
        for tick in range(ticks):
            u = (tick + 1) / ticks
            u = u * u * (3.0 - 2.0 * u)
            command = start + u * (target - start)
            arm.apply_action(ArticulationAction(joint_positions=command.astype(np.float32)))
            world.step(render=True)
        point, axis = pose()
        if name == "descend":
            grasp_axis = axis.copy()
        frame = antioch.capture_viewport()
        if frame is not None and frame.size:
            snapshots.append(np.asarray(frame)[..., :3].astype(np.uint8))
        print(f"{name:8s} tcp={np.round(point, 4).tolist()} axis={np.round(axis, 3).tolist()} "
              f"block={np.round(block.get_world_pose()[0], 4).tolist()}")
        current = target

    for _ in range(120):
        world.step(render=True)
    final = np.asarray(block.get_world_pose()[0], dtype=float)
    if snapshots:
        from PIL import Image
        output = Path("/tmp/close_pick_strip.png")
        Image.fromarray(np.concatenate(snapshots, axis=1)).save(output)
        run.add_artifact(output, name=output.name)

    verticality = float(np.dot(grasp_axis, np.array([0.0, 0.0, -1.0])))
    actual_distance = float(np.linalg.norm(pickup_xy))
    run.add_result("block_distance_m", round(actual_distance, 4))
    run.add_result("block_final", np.round(final, 4).tolist())
    run.add_result("grasp_verticality", round(verticality, 4))
    run.check("the block starts close to the arm", actual_distance <= 0.30,
              detail=f"{actual_distance * 100:.0f} cm from the base pivot")
    run.check("the fingers descend vertically", verticality > 0.985,
              detail=f"down-axis alignment {verticality:.3f}")
    run.check("the block lifts straight up", final[2] > 0.06,
              detail=f"final block height {final[2] * 1000:.0f} mm")
