"""Render a quick world-camera video of the grasp-and-lift, for humans.

    sim\\mujoco_venv\\Scripts\\python.exe sim/mujoco/preview_video.py

Not part of the data pipeline — just visual proof the scene and grasp motion
look right before episodes are collected.
"""

import mujoco
import numpy as np
import imageio


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def main() -> None:
    m = mujoco.MjModel.from_xml_path("sim/mujoco/scene.xml")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640)

    mujoco.mj_resetDataKeyframe(m, d, 0)  # pregrasp: jaws open around the cube
    # User-verified orientation: at +pi/2 the hand is upside down vs the real
    # robot (camera bracket below). -pi/2 puts the bracket on top.
    d.qpos[4] = -1.5708
    d.ctrl[4] = -1.5708
    ctrl0 = d.ctrl.copy()

    # Flipping the roll moves the grasp centre (the jaws are offset from the
    # roll axis), so re-centre the cube under the flipped hand for this
    # preview. The real collector aims its IK at the cube instead.
    mujoco.mj_forward(m, d)
    site = d.site("gripperframe").xpos
    d.qpos[6] = site[0]
    d.qpos[7] = site[1]
    d.qpos[8] = 0.02

    frames = []
    steps_per_frame = 7  # ~30 fps at the model's 5 ms timestep

    def run(seconds, update):
        n = int(seconds / m.opt.timestep)
        for i in range(n):
            update(i / max(n - 1, 1))
            mujoco.mj_step(m, d)
            if i % steps_per_frame == 0:
                r.update_scene(d, camera="world")
                frames.append(r.render().copy())

    lift_target = ctrl0.copy()
    lift_target[1] -= 0.75  # shoulder_lift more negative raises the tool
    lift_target[2] -= 0.35

    run(0.8, lambda u: None)  # settle at pregrasp
    run(1.2, lambda u: d.ctrl.__setitem__(5, 0.9 + smoothstep(u) * (0.0 - 0.9)))  # ramp close
    run(0.6, lambda u: None)  # settle the grip
    run(1.8, lambda u: d.ctrl.__setitem__(
        slice(0, 5), ctrl0[:5] + smoothstep(u) * (lift_target[:5] - ctrl0[:5])
    ))  # ramped lift
    run(1.2, lambda u: None)  # hold

    cube_z = float(d.qpos[8])
    out = "sim/mujoco/preview_grasp.mp4"
    imageio.mimsave(out, frames, fps=30, macro_block_size=1)
    print(f"wrote {out} ({len(frames)} frames); final cube z = {cube_z:.3f} "
          f"({'HELD ALOFT' if cube_z > 0.05 else 'dropped/on ground'})")
    r.close()


if __name__ == "__main__":
    main()
