"""SO-101 pick and place, solved against Isaac's own Jacobian.

    antioch scenario run --scenario so101_pick_place          # streams by default
    antioch scenario run --scenario so101_pick_place --no-stream

There is no precomputed trajectory. Each control tick reads the articulation's
Jacobian and current pose out of PhysX and takes one damped-least-squares step
toward the active Cartesian waypoint, so the arm self-corrects and nothing has
to be translated from another simulator.

Every constant below was measured off the live asset by src/probe_*.py rather
than assumed:

  * DOFs are ['Rotation','Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Jaw'].
  * The articulation is floating-base, so `get_jacobian_matrices()` returns 12
    columns and the joint columns are the LAST six. Using the first six drives
    the arm with the base's Jacobian and it barely moves.
  * The gripper is link index 5.
  * In the gripper link frame the fingers extend along -Z, the jaw closes along
    +X against a fixed finger whose inner face sits at x = -0.0438, and the pad
    region spans z in [-0.115, -0.03].
"""

from __future__ import annotations

import antioch

logger = antioch.Logger("pickplace")

ARM_ASSET, ARM_VERSION = "so101_antioch", "1.3.2"
GRIPPER_LINK = 5
N_DOF = 6

# Grasp centre in the gripper link frame: midway between the fixed finger's
# inner face and where the moving jaw meets a 30 mm block, at pad height.
BLOCK = 0.030
TCP_LOCAL = (-0.0438 + BLOCK / 2.0, 0.0, -0.085)

# Jaw angle -> gap, measured: -0.17 closed, +0.30 ~ 19.9 mm, +0.80 ~ 47.5 mm.
JAW_OPEN = 0.90
JAW_GRIP = 0.36  # just inside contact on a 30 mm block, so the drive preloads


def _quat_to_mat(q):
    import numpy as np

    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rot_error(R_des, R):
    """Axis-angle error taking R onto R_des."""
    import numpy as np

    E = R_des @ R.T
    angle = np.arccos(np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def _tool_down(yaw):
    """Fingers pointing at the table, jaw closing axis yawed by `yaw`."""
    import numpy as np

    c, s = np.cos(yaw), np.sin(yaw)
    # columns are the tool axes in world: +Z up means -Z (the fingers) points down
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@antioch.scenario(
    tags=["pickplace", "smoke"],
    sim=antioch.BootProfile(physics_dt=1.0 / 120.0, render_dt=1.0 / 60.0),
)
def so101_pick_place(
    run: antioch.ScenarioRun,
    pick_x: float = antioch.param(0.32, ge=0.15, le=0.45, description="Block pick x in metres"),
    pick_y: float = antioch.param(-0.06, ge=-0.35, le=0.35, description="Block pick y in metres"),
    place_x: float = antioch.param(0.30, ge=0.15, le=0.45, description="Tray centre x in metres"),
    place_y: float = antioch.param(0.16, ge=-0.35, le=0.35, description="Tray centre y in metres"),
    travel_z: float = antioch.param(0.14, ge=0.06, le=0.30, description="TCP height while traversing"),
) -> None:
    """Pick a block off the table and place it in the tray."""

    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.viewports import set_camera_view

    from wooden_tray import add_wooden_tray

    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 900.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 1800.0})

    antioch.load_asset(ARM_ASSET, prim_path="/World/SO101", version=ARM_VERSION)
    add_wooden_tray(world, center=(place_x, place_y), outer_size=(0.30, 0.18))
    block = world.scene.add(
        DynamicCuboid(
            prim_path="/World/block",
            name="block",
            position=np.array([pick_x, pick_y, BLOCK / 2.0]),
            size=BLOCK,
            color=np.array([0.78, 0.14, 0.10]),
            mass=0.02,
        )
    )
    world.reset()

    single = SingleArticulation(prim_path="/World/SO101", name="so101")
    single.initialize()
    arm = Articulation("/World/SO101")
    grip = RigidPrim("/World/SO101/gripper")

    tcp_local = np.array(TCP_LOCAL)
    set_camera_view(
        eye=[0.75, -0.55, 0.55],
        target=[0.5 * (pick_x + place_x), 0.5 * (pick_y + place_y), 0.05],
        camera_prim_path="/OmniverseKit_Persp",
    )

    def tcp_pose():
        p, q = (a.numpy()[0] for a in grip.get_world_poses())
        R = _quat_to_mat(q)
        return p + R @ tcp_local, R

    def ik_step(target_pos, target_R, jaw, gain=0.8, damping=0.04):
        """One resolved-rate step toward the target; returns the pose error."""
        q_now = np.asarray(single.get_joint_positions(), dtype=np.float64)
        pos, R = tcp_pose()

        err = np.zeros(6)
        err[:3] = target_pos - pos
        err[3:] = _rot_error(target_R, R)

        # joint columns are the LAST six of the floating-base jacobian
        J_full = arm.get_jacobian_matrices().numpy()[0, GRIPPER_LINK]
        J = np.array(J_full[:, -N_DOF:], dtype=np.float64)
        # shift the linear rows from the link origin out to the TCP
        r = R @ tcp_local
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        J[:3, :] = J[:3, :] - skew @ J[3:, :]

        dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(6), err * gain)
        q_cmd = q_now + np.clip(dq, -0.08, 0.08)
        q_cmd[5] = jaw
        single.apply_action(ArticulationAction(joint_positions=q_cmd.astype(np.float32)))
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    grasp_z = BLOCK / 2.0
    yaw = 0.0
    waypoints = [
        ("approach", (pick_x, pick_y, travel_z), JAW_OPEN, 110),
        ("descend", (pick_x, pick_y, grasp_z), JAW_OPEN, 110),
        ("grasp", (pick_x, pick_y, grasp_z), JAW_GRIP, 90),
        ("lift", (pick_x, pick_y, travel_z), JAW_GRIP, 110),
        ("transfer", (place_x, place_y, travel_z), JAW_GRIP, 160),
        ("lower", (place_x, place_y, grasp_z + 0.045), JAW_GRIP, 110),
        ("release", (place_x, place_y, grasp_z + 0.045), JAW_OPEN, 70),
        ("retreat", (place_x, place_y, travel_z), JAW_OPEN, 110),
    ]

    worst = {}
    frames = 0
    for name, target, jaw, ticks in waypoints:
        target = np.array(target, dtype=float)
        R_des = _tool_down(yaw)
        pe = re = float("nan")
        for tick in range(ticks):
            pe, re = ik_step(target, R_des, jaw)
            world.step(render=True)
            logger.scalar(f"error/{name}", pe)
            if tick % 25 == 0:
                frame = antioch.capture_viewport()
                if frame is not None:
                    rgb = np.asarray(frame)[:, :, :3]
                    if 10.0 <= float(rgb.mean()) <= 220.0:
                        logger.image("camera/rgb", rgb)
                        frames += 1
        worst[name] = round(pe, 4)
        block_p = block.get_world_pose()[0]
        print(f"  {name:9s} tcp_err={pe * 1000:6.1f}mm rot_err={re:5.3f}rad block={np.round(block_p, 3).tolist()}")

    for _ in range(120):
        world.step(render=True)

    final = np.asarray(block.get_world_pose()[0], dtype=float)
    run.add_result("block_final", [round(float(v), 4) for v in final])
    run.add_result("waypoint_tcp_error_m", worst)
    run.add_result("review_frames", frames)

    xy_err = float(np.linalg.norm(final[:2] - np.array([place_x, place_y])))
    run.add_result("place_xy_error_m", round(xy_err, 4))
    run.check(
        "the arm tracked every waypoint",
        max(worst.values()) < 0.02,
        detail=f"worst TCP error {max(worst.values()) * 1000:.1f} mm across {len(worst)} waypoints",
    )
    run.check("the block was lifted clear of the table", final[2] > 0.008, detail=f"block centre at {final[2]:.3f} m")
    run.check("the block ended in the tray", xy_err < 0.06, detail=f"{xy_err * 1000:.0f} mm from the tray centre")
