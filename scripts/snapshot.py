"""Save the run currently in the backend as demo artifacts.

    backend/.venv/bin/python scripts/snapshot.py
    backend/.venv/bin/python scripts/snapshot.py --as-fixture

Run this after any run worth keeping. Without ``--as-fixture`` it archives
the run under artifacts/. With it, the run also *becomes* the replay the
dashboard falls back to — so by the end of the day the recorded story is a
real one and none of my invented numbers survive into the demo.

Plan section 31 wants the diagnosis, curriculum and evaluation saved as
separate artifacts; they are split out here so each can be shown on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
API = "http://localhost:8000"


def fetch(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=5) as response:
            return json.loads(response.read())
    except urllib.error.URLError as exc:
        sys.exit(f"Backend not reachable at {API} ({exc.reason}). Start it first.")


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-fixture",
        action="store_true",
        help="also install this run as the dashboard's recorded fallback",
    )
    args = parser.parse_args()

    history = fetch("/api/history")
    events = history.get("events", [])
    if not events:
        sys.exit("The backend has no events. Run the loop first, then snapshot it.")

    run_id = history["meta"].get("run_id") or "unknown"
    print(f"run {run_id}: {len(events)} events\n")

    write(ARTIFACTS / "demo" / f"run_{run_id}.json", history)

    by_type = {}
    for event in events:
        by_type.setdefault(event["type"], []).append(event)

    if diagnosis := by_type.get("diagnosis_ready"):
        write(ARTIFACTS / "diagnoses" / f"{run_id}.json", diagnosis[-1]["diagnosis"])

    if curriculum := by_type.get("curriculum_ready"):
        write(ARTIFACTS / "diagnoses" / f"{run_id}_curriculum.json", curriculum[-1])

    if metrics := by_type.get("metrics_ready"):
        write(ARTIFACTS / "evals" / f"{run_id}_metrics.json", metrics[-1]["metrics"])

    if args.as_fixture:
        # Keep meta.source == "fixture" so the DEMO DATA badge stays honest:
        # this is a recording of a real run, but it is still a recording.
        fixture = {
            "meta": {
                "source": "fixture",
                "run_id": run_id,
                "description": f"Recorded from live run {run_id}.",
                "warning": "Replay of a real run. Not live.",
                "event_count": len(events),
                "duration_s": round(events[-1]["ts"] - events[0]["ts"], 1),
            },
            "events": events,
        }
        print()
        write(ROOT / "fixtures" / "demo_run.json", fixture)
        write(ROOT / "frontend" / "public" / "fixtures" / "demo_run.json", fixture)

    missing = [
        name
        for name, present in (
            ("real failure video", any(e["type"] == "real_failed" for e in events)),
            ("critic diagnosis", "diagnosis_ready" in by_type),
            ("curriculum", "curriculum_ready" in by_type),
            ("targeted batch", "batch_complete" in by_type),
            ("v0/v1 metrics", "metrics_ready" in by_type),
            ("real success", any(e["type"] == "real_success" for e in events)),
        )
        if not present
    ]
    print("\nsection 31 demo assets:", "all present" if not missing else "MISSING " + ", ".join(missing))


if __name__ == "__main__":
    main()
