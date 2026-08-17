"""Generate the scripted demo run the dashboard is built against.

Run:  backend/.venv/bin/python fixtures/build_fixture.py

This file is the spec.  Every event the UI renders appears here at least
once, so the frontend can be built end to end before Antioch, the SO-101 or
the critic exist — and so section 21's replay fallback is the default code
path rather than an extra feature.

Timestamps are fixed rather than wall-clock so regenerating produces a clean
diff.  Gaps between events are realistic (a 4 s critic call, ~2 s per batch
scenario) so replay paces like the real thing.

The numbers here are ILLUSTRATIVE, taken from the plan's examples.  The
output carries ``meta.source = "fixture"`` and the dashboard must show a
DEMO DATA badge whenever it is running from this file, so placeholder
figures can never be mistaken for measurements (plan section 12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from events import (  # noqa: E402
    BatchComplete,
    BatchProgress,
    BatchStarted,
    CriticStarted,
    CurriculumReady,
    DeployStarted,
    DiagnosisReady,
    MetricsReady,
    PolicyLoaded,
    RealFailed,
    RealSuccess,
    RedeployStarted,
    SimPassed,
    SimProgress,
    SimStarted,
    VideoReady,
)
from schemas import (  # noqa: E402
    EstimatedCause,
    FailureDiagnosis,
    FailureMode,
    MetricRow,
    PolicyMetrics,
    SimChange,
    SimParameter,
    Stage,
)

RUN_ID = "demo-fixture-001"
T0 = 1_755_432_600.0  # fixed epoch; keeps regenerated fixtures diff-clean

#: The controlled out-of-distribution condition from plan section 8.
CONDITION = "block +6cm right of its trained position"

#: Which of the 30 targeted scenarios fail. Fixed, not random, so the batch
#: grid renders identically on every replay.
BATCH_FAILURES = {2, 6, 13, 21, 27}

events: list = []
clock = T0


def emit(event, gap: float = 0.0):
    """Append an event, advancing the fixture clock by ``gap`` seconds."""
    global clock
    clock += gap
    event.seq = len(events)
    event.ts = round(clock, 2)
    events.append(event)


# --- SIMULATE: ACT v0 trains and runs in Antioch -----------------------------

emit(SimStarted(run_id=RUN_ID, policy="v0", scenario="so101_pick_place", n_episodes=30))
for completed, successes in ((10, 9), (20, 18), (30, 27)):
    emit(SimProgress(run_id=RUN_ID, completed=completed, total=30, successes=successes), gap=13.0)

# --- VERIFY: the physics gate passes, so v0 is allowed to touch hardware ------

emit(SimPassed(run_id=RUN_ID, successes=27, total=30, threshold=0.8), gap=2.0)
emit(PolicyLoaded(run_id=RUN_ID, policy="v0", checkpoint="act_v0_step12000.safetensors"), gap=3.0)

# --- DEPLOY + EXPERIENCE: the real arm fails on the controlled condition -----

emit(DeployStarted(run_id=RUN_ID, policy="v0", condition=CONDITION), gap=6.0)
emit(
    VideoReady(
        run_id=RUN_ID,
        url="/artifacts/videos/v0_real_failure.mp4",
        poster_url="/artifacts/videos/v0_real_failure.jpg",
        duration_s=8.4,
    ),
    gap=11.0,
)
emit(RealFailed(run_id=RUN_ID, policy="v0", note="Gripper closed beside the block."), gap=0.5)

# --- DIAGNOSE: the critic reads the video ------------------------------------

emit(CriticStarted(run_id=RUN_ID, video_url="/artifacts/videos/v0_real_failure.mp4"), gap=2.0)

diagnosis = FailureDiagnosis(
    success=False,
    stage=Stage.GRASP,
    failure=FailureMode.FAILED_GRASP,
    confidence=0.88,
    estimated_causes=[
        EstimatedCause(cause="lateral_offset", confidence=0.69),
        EstimatedCause(cause="reach_error", confidence=0.24),
        EstimatedCause(cause="other", confidence=0.07),
    ],
    recommended_sim_changes=[
        SimChange(parameter=SimParameter.PICK_Y, min=-0.18, max=0.10),
        # Deliberately near-full-range: the critic asks to vary pick_x across
        # almost its entire legal span, which says nothing about where the
        # policy is weak. The mapper rejects it for the targeted rule and
        # flags the override, so the guardrail is visible on screen.
        SimChange(parameter=SimParameter.PICK_X, min=0.15, max=0.45),
    ],
    summary="The gripper closed beside the block rather than around it, catching only its near edge.",
)
emit(DiagnosisReady(run_id=RUN_ID, diagnosis=diagnosis, latency_ms=4200), gap=4.2)

# --- ADAPT: diagnosis becomes a clamped, legal simulation curriculum ---------

emit(
    CurriculumReady(
        run_id=RUN_ID,
        curriculum_id="curr-001",
        n_scenarios=30,
        baseline={"pick_y": -0.06, "pick_x": 0.32},
        changes=[
            SimChange(parameter=SimParameter.PICK_Y, min=-0.18, max=0.10),
            SimChange(parameter=SimParameter.PICK_X, min=0.24, max=0.40, clamped=True),
        ],
    ),
    gap=1.1,
)

emit(BatchStarted(run_id=RUN_ID, curriculum_id="curr-001", n_scenarios=30), gap=1.5)
for i in range(30):
    emit(
        BatchProgress(
            run_id=RUN_ID,
            index=i,
            total=30,
            success=i not in BATCH_FAILURES,
            thumbnail_url=f"/artifacts/demo/batch/{i:02d}.jpg",
        ),
        gap=2.0,
    )
emit(BatchComplete(run_id=RUN_ID, successes=30 - len(BATCH_FAILURES), total=30), gap=1.0)

# --- LEARN: v1 trained on the aggregated set, scored on the same held-out set -

emit(PolicyLoaded(run_id=RUN_ID, policy="v1", checkpoint="act_v1_step12000.safetensors"), gap=4.0)
emit(
    MetricsReady(
        run_id=RUN_ID,
        metrics=PolicyMetrics(
            rows=[
                MetricRow(label="Baseline sim", v0="90%", v1="93%"),
                MetricRow(label="Hard held-out set", v0="42%", v1="81%"),
                MetricRow(label="Real condition", v0="FAIL", v1="PASS"),
            ],
            held_out_scenarios=30,
            initial_demos=30,
            corrective_demos=20,
        ),
    ),
    gap=1.0,
)

# --- REDEPLOY: same physical condition, different outcome --------------------

emit(RedeployStarted(run_id=RUN_ID, policy="v1", condition=CONDITION), gap=3.0)
emit(
    VideoReady(
        run_id=RUN_ID,
        url="/artifacts/videos/v1_real_success.mp4",
        poster_url="/artifacts/videos/v1_real_success.jpg",
        duration_s=9.1,
    ),
    gap=11.0,
)
emit(RealSuccess(run_id=RUN_ID, policy="v1", note="Block placed in tray under the same condition."), gap=0.5)


payload = {
    "meta": {
        "source": "fixture",
        "run_id": RUN_ID,
        "description": "Scripted end-to-end run: v0 sim pass -> real fail -> diagnose -> adapt -> v1 -> real pass.",
        "warning": "ILLUSTRATIVE NUMBERS. The dashboard must show a DEMO DATA badge while replaying this.",
        "event_count": len(events),
        "duration_s": round(events[-1].ts - T0, 1),
    },
    "events": [json.loads(e.model_dump_json()) for e in events],
}

blob = json.dumps(payload, indent=2) + "\n"
root = Path(__file__).resolve().parent.parent

# Written twice on purpose: this file is the source of truth, and the copy
# under frontend/public is what the browser fetches. Regenerate rather than
# editing either by hand.
targets = [root / "fixtures" / "demo_run.json", root / "frontend" / "public" / "fixtures" / "demo_run.json"]
for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(blob)
    print(f"wrote {target.relative_to(root)}")
print(f"{len(events)} events, {payload['meta']['duration_s']}s of scripted time")
