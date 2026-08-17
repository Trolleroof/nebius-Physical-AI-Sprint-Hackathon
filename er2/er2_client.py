#!/usr/bin/env python3
"""Gemini Robotics-ER 2 client: ground a task in an image, in pixel coordinates.

This is the perception/reasoning half of the loop. It takes a camera frame plus
a task in plain language and returns either labelled points on the objects that
matter, or an ordered plan. Nothing here touches the arm, so it runs and demos
today with no hardware attached.

    export GEMINI_API_KEY=...

    ./er2_client.py point --image scene.jpg --query "the green cube and the tray"
    ./er2_client.py point --camera 0 --query "every vial in the rack" --annotate out.jpg
    ./er2_client.py plan  --image scene.jpg --task "Put the green cube in the tray."

ER 2 returns points as [y, x] normalized to 0-1000, which is not the [x, y] in
pixels that everything downstream wants, so the conversion happens here once
rather than at every call site.
"""

import argparse
import json
import os
import re
import sys
import tempfile

# ER 2 lives only on the Gemini Developer API. On Vertex it 404s in every region,
# because it is private preview on the Enterprise Agent Platform. So the backend
# and the model travel together: pick a backend, get the best model it can serve.
ER2_MODEL = "gemini-robotics-er-2-preview"
VERTEX_MODEL = "gemini-3.6-flash"


def pick_backend(requested: str):
    """Return (client, model, backend_name)."""
    from google import genai

    want_vertex = requested == "vertex" or (
        requested == "auto" and os.environ.get("GEMINI_USE_VERTEXAI") == "true"
    )
    if want_vertex:
        project = os.environ.get("GEMINI_GCP_PROJECT")
        location = os.environ.get("GEMINI_GCP_LOCATION", "global")
        if not project:
            sys.exit("GEMINI_USE_VERTEXAI is set but GEMINI_GCP_PROJECT is not.")
        client = genai.Client(vertexai=True, project=project, location=location)
        return client, os.environ.get("GEMINI_MODEL", VERTEX_MODEL), f"vertex:{location}"

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "No API key. Get one at https://aistudio.google.com/apikey then:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=key), os.environ.get("GEMINI_MODEL", ER2_MODEL), "gemini-api"

POINT_PROMPT = """Point to {query} in the image.
Point to no more than {limit} items. The label returned should be an
identifying name for the object detected.
The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...].
The points are in [y, x] format normalized to 0-1000."""

PLAN_PROMPT = """You are the high-level planner for a 6-DOF SO-101 robot arm with a
parallel gripper, working on the tabletop shown in the image.

Task: {task}

Return an ordered plan as JSON:
[{{"step": <int>, "action": "<one of: move_above, descend, grasp, lift, move_to, release, retreat>",
   "target": "<object name>", "point": [y, x], "why": "<short reason>"}}, ...]

The "point" is where on the image that step acts, in [y, x] format normalized to
0-1000. Only include a point for steps that act on a visible location. Keep the
plan under 12 steps and assume the gripper starts open and empty."""


def mime_for(path: str) -> str:
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def grab_frame(index: int) -> str:
    """Capture one frame from a USB camera and return the temp file path."""
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"Camera {index} would not open. On macOS this is almost always the\n"
            "per-app camera permission: System Settings > Privacy & Security >\n"
            "Camera, enable your terminal, then restart the terminal."
        )
    # First frames off a USB camera are often black while it auto-exposes.
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        sys.exit("Camera opened but returned no frame.")
    path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(path, frame)
    print(f"captured {frame.shape[1]}x{frame.shape[0]} -> {path}", file=sys.stderr)
    return path


def ask(client, model: str, image_path: str, prompt: str) -> str:
    """Inline the image bytes: generate_content works identically on both backends,
    whereas files.upload is Gemini-API-only."""
    from google.genai import types

    with open(image_path, "rb") as handle:
        data = handle.read()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime_for(image_path)),
            prompt,
        ],
    )
    return response.text


def parse_json(text: str):
    """Models wrap JSON in ``` fences often enough that this is worth doing once."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    payload = fenced.group(1) if fenced else text
    start = min((i for i in (payload.find("["), payload.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"no JSON found in response:\n{text}")
    try:
        return json.loads(payload[start:])
    except json.JSONDecodeError:
        # Trailing prose after the JSON is common; cut back to the last bracket.
        end = max(payload.rfind("]"), payload.rfind("}"))
        return json.loads(payload[start : end + 1])


def to_pixels(point, width: int, height: int):
    """ER 2 gives [y, x] normalized 0-1000. Everything downstream wants (x, y) px."""
    y_norm, x_norm = point
    return int(x_norm / 1000 * width), int(y_norm / 1000 * height)


def image_size(path: str):
    import cv2

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"could not read image: {path}")
    return img.shape[1], img.shape[0]


def annotate(path: str, items, out_path: str) -> None:
    import cv2

    img = cv2.imread(path)
    height, width = img.shape[:2]
    for item in items:
        if not item.get("point"):
            continue
        x, y = to_pixels(item["point"], width, height)
        cv2.circle(img, (x, y), 9, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        label = str(item.get("label") or item.get("target") or "")
        cv2.putText(img, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    print(f"annotated -> {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    for name in ("point", "plan"):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--image", help="path to an image file")
        src.add_argument("--camera", type=int, help="capture from this camera index")
        p.add_argument("--annotate", metavar="OUT.jpg", help="write a marked-up copy")
        p.add_argument("--backend", default="auto", choices=["auto", "vertex", "gemini-api"],
                       help="auto follows GEMINI_USE_VERTEXAI in the environment")
        if name == "point":
            p.add_argument("--query", required=True, help='e.g. "the green cube"')
            p.add_argument("--limit", type=int, default=10)
        else:
            p.add_argument("--task", required=True,
                           help='e.g. "Put the green cube in the tray."')

    args = parser.parse_args()
    client, model, backend = pick_backend(args.backend)
    print(f"# backend={backend} model={model}", file=sys.stderr)
    image_path = args.image or grab_frame(args.camera)

    if args.mode == "point":
        prompt = POINT_PROMPT.format(query=args.query, limit=args.limit)
    else:
        prompt = PLAN_PROMPT.format(task=args.task)

    raw = ask(client, model, image_path, prompt)
    items = parse_json(raw)
    if isinstance(items, dict):
        items = [items]

    width, height = image_size(image_path)
    print(f"# image {width}x{height}")
    for item in items:
        point = item.get("point")
        pixels = f"px=({to_pixels(point, width, height)})" if point else "px=(n/a)"
        if args.mode == "point":
            print(f"{item.get('label'):<28} norm={point} {pixels}")
        else:
            print(f"{item.get('step'):>2}. {item.get('action'):<12} "
                  f"{str(item.get('target')):<20} {pixels}  {item.get('why','')}")

    if args.annotate:
        annotate(image_path, items, args.annotate)


if __name__ == "__main__":
    main()
