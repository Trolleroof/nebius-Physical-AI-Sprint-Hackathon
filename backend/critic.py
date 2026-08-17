"""The embodied reasoning critic, behind one swappable interface.

Nothing downstream may import a vendor SDK. Everything talks to
``analyze(video_path, task) -> FailureDiagnosis`` so the model can be
replaced — including mid-demo, if the primary API dies — without touching
the mapper, the orchestrator or the dashboard (plan sections 9 and 26).

To plug in a real model, implement ``Critic.analyze`` and register it in
``get_critic``. The only hard requirements are that it accepts video (or
sampled frames) and can be constrained to emit the FailureDiagnosis schema.
Whatever it returns goes through ``parse_diagnosis``, which never raises, so
a badly behaved model degrades to "critic unsure" instead of killing the run.
"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

from schemas import FailureDiagnosis, parse_diagnosis

DEFAULT_TASK = "Put the green cube in the blue tray."

#: Handed to the model as the system instruction. Kept here rather than in the
#: implementation so swapping vendors does not silently change the prompt.
CRITIC_PROMPT = """You are analysing a video of a robot arm attempting a task.

Task: {task}

Report what happened using ONLY these vocabularies.

stage:   approach | grasp | lift | transport | place | complete | unknown
failure: missed_object | bad_alignment | failed_grasp | object_slip |
         premature_release | collision | unreachable_pose | placement_error |
         none | unknown

Rules:
- `stage` is where the attempt FAILED, or "complete" if it succeeded.
- Report the earliest stage that went wrong, not the last thing you saw.
- `confidence` is your belief in the diagnosis, 0.0 to 1.0. Be honest: a low
  number is more useful to us than a confident guess.
- `estimated_causes` are physical hypotheses (e.g. low_friction,
  grasp_offset), each with its own confidence.
- `recommended_sim_changes` may only name these parameters, which are the
  keyword arguments of the simulated task:
  pick_x, pick_y (block position, metres)
  place_x, place_y (tray position, metres)
  travel_z (height the gripper traverses at, metres)
  If the physical cause you suspect has no parameter here, say so in
  `estimated_causes` and leave `recommended_sim_changes` empty rather than
  naming a parameter that does not exist.
- `summary` is one plain sentence a non-expert can read.

Return JSON matching the FailureDiagnosis schema and nothing else."""


class Critic(Protocol):
    async def analyze(self, video_path: str, task: str = DEFAULT_TASK) -> FailureDiagnosis: ...


class MockCritic:
    """Stands in until a real video model is wired up.

    Returns the worked example from plan section 9 — including a mass range
    wider than the simulator allows, so the mapper's clamp is exercised on
    every run rather than only in the fixture.
    """

    latency_s = 2.5

    async def analyze(self, video_path: str, task: str = DEFAULT_TASK) -> FailureDiagnosis:
        await asyncio.sleep(self.latency_s)
        return parse_diagnosis(
            {
                "success": False,
                "stage": "grasp",
                "failure": "failed_grasp",
                "confidence": 0.88,
                "estimated_causes": [
                    {"cause": "lateral_offset", "confidence": 0.69},
                    {"cause": "reach_error", "confidence": 0.24},
                ],
                "recommended_sim_changes": [
                    {"parameter": "pick_y", "min": -0.18, "max": 0.10},
                    # Deliberately near-full-range: exercises the mapper's
                    # rejection of uninformative recommendations on every run.
                    {"parameter": "pick_x", "min": 0.15, "max": 0.45},
                ],
                "summary": (
                    "The gripper closed beside the block rather than around "
                    "it, catching only its near edge."
                ),
            }
        )


def get_critic() -> Critic:
    """Pick the critic implementation.

    Set CRITIC_BACKEND to something other than "mock" once a real one exists;
    add the branch here and nothing else in the codebase changes.
    """
    backend = os.environ.get("CRITIC_BACKEND", "mock").lower()
    if backend == "mock":
        return MockCritic()
    raise NotImplementedError(
        f"CRITIC_BACKEND={backend!r} has no implementation yet. "
        "Add it in backend/critic.py; it must satisfy the Critic protocol."
    )
