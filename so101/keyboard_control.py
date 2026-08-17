#!/usr/bin/env python3
"""Drive the SO-101 live from the keyboard. No second arm needed.

Holding a key nudges a joint; the terminal's own key auto-repeat gives you
continuous motion. The loop keeps a target pose and streams it at a fixed rate,
so motion stays smooth instead of arriving in jumps.

    ./keyboard_control.py --dry-run     # try the controls, send nothing
    ./keyboard_control.py

    CONTROLS
      a / d     shoulder_pan     -/+        [ / ]   gripper close / open
      w / s     shoulder_lift    +/-        - / =   step size down / up
      e / c     elbow_flex       +/-        h       ramp to home pose
      r / f     wrist_flex       +/-        SPACE   freeze target at present
      t / g     wrist_roll       +/-        q       quit (ramps to rest first)

Quitting with q ramps back to the pose the arm was in at connect before torque
is released, because cutting torque under a held pose drops the arm. Ctrl-C
does the same. Keep a hand near the power switch on the first run.
"""

import argparse
import glob
import select
import sys
import termios
import threading
import time
import tty

from arm import ALL_JOINTS, ARM_JOINTS, GRIPPER, Arm

# key -> (joint, direction)
BINDINGS = {
    "a": ("shoulder_pan", -1), "d": ("shoulder_pan", +1),
    "w": ("shoulder_lift", +1), "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1), "c": ("elbow_flex", -1),
    "r": ("wrist_flex", +1), "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1), "g": ("wrist_roll", -1),
    "[": (GRIPPER, -1), "]": (GRIPPER, +1),
}

RATE_HZ = 40
DEFAULT_STEP_DEG = 4.0      # per keypress; auto-repeat makes this continuous
MIN_STEP, MAX_STEP = 0.25, 20.0
# Never let the streamed target drift further than this ahead of where the arm
# actually is. Without it, holding a key while the arm is blocked winds the
# target far past reality, and the arm lunges when the obstruction clears.
#
# This is also the real speed governor, not the step size: once the target sits
# a full lead ahead, the arm moves as fast as the servo closes that gap, and a
# bigger step only gets you there sooner. Raising step without raising this
# does nothing.
MAX_LEAD_DEG = 15.0


class CameraView:
    """Live camera window, overlaid with the arm's joint state.

    Grabbing runs on a background thread so a slow camera read never stalls the
    control loop -- at 1080p a blocking read is tens of milliseconds, which
    would wreck a 40 Hz loop. Only the newest frame is kept; stale frames are
    worthless for looking at what the arm is doing right now.

    imshow stays on the main thread, because macOS will not accept UI calls
    from anywhere else.
    """

    def __init__(self, index: int, display_width: int = 960, display_hz: float = 20.0):
        self.index = index
        self.display_width = display_width
        self.min_period = 1.0 / display_hz
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.last_shown = 0.0
        self.window = "SO-101 view  (drive from the terminal, not this window)"

    def start(self) -> bool:
        import cv2

        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            print(f"camera {self.index} would not open -- continuing without video.\n"
                  "  On macOS grant your terminal camera access in\n"
                  "  System Settings > Privacy & Security > Camera, then restart it.",
                  file=sys.stderr)
            self.capture = None
            return False
        self.running = True
        threading.Thread(target=self._grab_loop, daemon=True).start()
        return True

    def _grab_loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def show(self, lines) -> None:
        if self.capture is None:
            return
        now = time.perf_counter()
        if now - self.last_shown < self.min_period:
            return
        self.last_shown = now

        import cv2

        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return

        height, width = frame.shape[:2]
        if width > self.display_width:
            scale = self.display_width / width
            frame = cv2.resize(frame, (self.display_width, int(height * scale)))

        # A translucent banner behind the text. An outline stroke was the first
        # attempt and it smears at this font size; a backing bar stays legible
        # over a bright, busy tabletop.
        if lines:
            banner_height = 14 + 24 * len(lines)
            panel = frame[0:banner_height].copy()
            panel[:] = (0, 0, 0)
            cv2.addWeighted(panel, 0.55, frame[0:banner_height], 0.45, 0, frame[0:banner_height])
            for row, text in enumerate(lines):
                cv2.putText(frame, text, (12, 26 + row * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 128), 1, cv2.LINE_AA)

        cv2.imshow(self.window, frame)
        cv2.waitKey(1)  # pumps the window's event loop; keys come from the terminal

    def stop(self) -> None:
        self.running = False
        time.sleep(0.05)
        if self.capture is not None:
            self.capture.release()
            import cv2
            cv2.destroyAllWindows()


class RawKeyboard:
    """cbreak mode: single keys arrive immediately, without Enter."""

    def __enter__(self):
        if not sys.stdin.isatty():
            sys.exit("keyboard_control needs a real terminal -- run it directly, not through a pipe.")
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def pressed(self) -> list:
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return keys


def resolve_port(given):
    if given:
        return given
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not ports:
        sys.exit("no /dev/tty.usbmodem* port. Is the arm powered and plugged in? (./usbcheck.py)")
    return ports[0]


def status_line(target, actual, step, dry) -> str:
    cells = [f"{n[:5]} {target[n]:+7.1f}" for n in ARM_JOINTS]
    cells.append(f"grip {target[GRIPPER]:5.1f}")
    drift = max(abs(target[n] - actual[n]) for n in ALL_JOINTS)
    tag = " DRY" if dry else ""
    return "  ".join(cells) + f" | step {step:.2f} lag {drift:4.1f}{tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default="my_follower_arm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_DEG)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--no-camera", action="store_true", help="skip the video window")
    args = parser.parse_args()

    Arm.quiet_clamp_warnings()   # the clamp fires constantly while driving; see arm.py
    arm = Arm(resolve_port(args.port), args.id, dry_run=args.dry_run)
    step = args.step
    view = None if args.no_camera else CameraView(args.camera)

    with arm, RawKeyboard() as keyboard:
        if view is not None and not view.start():
            view = None
        target = arm.read()
        period = 1.0 / RATE_HZ
        print(__doc__.split("CONTROLS")[1].rstrip(), file=sys.stderr)
        if view is not None:
            # The video window grabs focus the moment it opens, and keys typed
            # into it go nowhere. This costs everyone a few confused minutes.
            print("\n  >> CLICK THIS TERMINAL before typing -- the camera window takes focus. <<",
                  file=sys.stderr)
        if args.dry_run:
            print("  >> DRY RUN: keys respond but nothing is sent. Drop --dry-run to move. <<",
                  file=sys.stderr)
        print("  Press a key (no Enter). Hold it for continuous motion.\n", file=sys.stderr)

        quitting = False
        while not quitting:
            loop_start = time.perf_counter()
            actual = arm.read()

            for key in keyboard.pressed():
                if key in ("q", "\x03", "\x1b"):
                    quitting = True
                elif key == " ":
                    target = dict(actual)          # abandon accumulated lead
                elif key == "h":
                    arm.home()
                    target = arm.read()
                elif key == "-":
                    step = max(MIN_STEP, step / 1.5)
                elif key in ("=", "+"):
                    step = min(MAX_STEP, step * 1.5)
                elif key in BINDINGS:
                    joint, direction = BINDINGS[key]
                    target[joint] += direction * step

            # Clamp the target to stay near reality, per joint.
            for name in ALL_JOINTS:
                lead = target[name] - actual[name]
                if abs(lead) > MAX_LEAD_DEG:
                    target[name] = actual[name] + MAX_LEAD_DEG * (1 if lead > 0 else -1)
            target[GRIPPER] = max(0.0, min(100.0, target[GRIPPER]))

            arm.send_frame(target)
            print("\r" + status_line(target, actual, step, args.dry_run) + "  ",
                  end="", file=sys.stderr, flush=True)

            if view is not None:
                view.show([
                    " ".join(f"{n[:5]}{actual[n]:+7.1f}" for n in ARM_JOINTS),
                    f"gripper {actual[GRIPPER]:5.1f}%   step {step:.2f}deg"
                    + ("   DRY RUN" if args.dry_run else ""),
                ])

            time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))

        if view is not None:
            view.stop()
        print("\n", file=sys.stderr)


if __name__ == "__main__":
    main()
