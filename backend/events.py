"""The backend -> dashboard contract: every event the UI can receive.

The dashboard is an event renderer (plan section 15).  Its only input is an
ordered list of these events, which means the same code path serves a live
SSE stream and a replayed fixture file — replay is the default mode, live is
the special case.  That is what makes section 21's fallback levels free.

Events carry ``seq`` so a reconnecting client can resume, and so a fixture
can be stepped through deterministically.

Events marked ADDED are not in section 15's list.  They cover states the
dashboard has to render anyway (a failed sim gate, the metrics table, an
error).  Flag them to the team before anyone codes against this.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from schemas import FailureDiagnosis, PolicyMetrics, SimChange


class LoopStage(str, Enum):
    """The seven stages on the always-visible rail across the top."""

    SIMULATE = "simulate"
    VERIFY = "verify"
    DEPLOY = "deploy"
    EXPERIENCE = "experience"
    DIAGNOSE = "diagnose"
    ADAPT = "adapt"
    LEARN = "learn"


ORDERED_LOOP_STAGES: tuple[LoopStage, ...] = tuple(LoopStage)


class BaseEvent(BaseModel):
    """Common envelope. Every event is (type, when, which run, what order)."""

    ts: float = Field(default_factory=time.time)
    run_id: str
    seq: int = 0


# --- simulation -------------------------------------------------------------


class SimStarted(BaseEvent):
    type: Literal["sim_started"] = "sim_started"
    policy: str  # "v0" | "v1"
    scenario: str
    n_episodes: int


class SimProgress(BaseEvent):  # ADDED
    type: Literal["sim_progress"] = "sim_progress"
    completed: int
    total: int
    successes: int


class SimPassed(BaseEvent):
    type: Literal["sim_passed"] = "sim_passed"
    successes: int
    total: int
    threshold: float  # fraction, e.g. 0.8


class SimFailed(BaseEvent):  # ADDED — the "no" branch of the sim gate
    type: Literal["sim_failed"] = "sim_failed"
    successes: int
    total: int
    threshold: float


# --- real deployment --------------------------------------------------------


class DeployStarted(BaseEvent):
    type: Literal["deploy_started"] = "deploy_started"
    policy: str
    condition: str  # the controlled OOD condition being tested (section 8)


class VideoReady(BaseEvent):
    type: Literal["video_ready"] = "video_ready"
    url: str
    duration_s: float | None = None
    poster_url: str | None = None


class RealFailed(BaseEvent):
    type: Literal["real_failed"] = "real_failed"
    policy: str
    note: str = ""


class RealSuccess(BaseEvent):
    type: Literal["real_success"] = "real_success"
    policy: str
    note: str = ""


# --- critic -----------------------------------------------------------------


class CriticStarted(BaseEvent):
    type: Literal["critic_started"] = "critic_started"
    video_url: str


class DiagnosisReady(BaseEvent):
    type: Literal["diagnosis_ready"] = "diagnosis_ready"
    diagnosis: FailureDiagnosis
    latency_ms: int | None = None


# --- adaptation -------------------------------------------------------------


class CurriculumReady(BaseEvent):
    type: Literal["curriculum_ready"] = "curriculum_ready"
    curriculum_id: str
    changes: list[SimChange]
    n_scenarios: int
    #: What the parameter was before adaptation, so the UI can show 0.32 -> 0.24-0.40.
    baseline: dict[str, float] = Field(default_factory=dict)
    #: Set when the diagnosis names a failure the scenario cannot reproduce.
    #: An empty curriculum with a stated reason is a finding about the
    #: simulator; an empty curriculum with no reason just looks broken.
    unmappable_reason: str | None = None


class BatchStarted(BaseEvent):
    type: Literal["batch_started"] = "batch_started"
    curriculum_id: str
    n_scenarios: int


class BatchProgress(BaseEvent):
    type: Literal["batch_progress"] = "batch_progress"
    index: int  # 0-based
    total: int
    success: bool
    thumbnail_url: str | None = None


class BatchComplete(BaseEvent):  # ADDED — so the grid knows it is done
    type: Literal["batch_complete"] = "batch_complete"
    successes: int
    total: int


# --- learning ---------------------------------------------------------------


class PolicyLoaded(BaseEvent):
    type: Literal["policy_loaded"] = "policy_loaded"
    policy: str
    checkpoint: str | None = None


class RedeployStarted(BaseEvent):
    type: Literal["redeploy_started"] = "redeploy_started"
    policy: str
    condition: str


class MetricsReady(BaseEvent):  # ADDED — the improvement panel needs a source
    type: Literal["metrics_ready"] = "metrics_ready"
    metrics: PolicyMetrics


class ErrorRaised(BaseEvent):  # ADDED — so a dead subsystem is visible, not silent
    type: Literal["error"] = "error"
    message: str
    where: str = ""


Event = Annotated[
    Union[
        SimStarted,
        SimProgress,
        SimPassed,
        SimFailed,
        DeployStarted,
        VideoReady,
        RealFailed,
        RealSuccess,
        CriticStarted,
        DiagnosisReady,
        CurriculumReady,
        BatchStarted,
        BatchProgress,
        BatchComplete,
        PolicyLoaded,
        RedeployStarted,
        MetricsReady,
        ErrorRaised,
    ],
    Field(discriminator="type"),
]


#: Which rail stage lights up when an event arrives.  The reducer reads this
#: table instead of hard-coding the mapping, so backend and UI cannot drift.
#: Events absent from this table update panels without moving the rail.
#:
#: ``policy_loaded`` is deliberately absent.  It fires both when v0 is loaded
#: before the first deployment and when v1 is loaded after retraining, so
#: putting it on the rail would flick the highlight to LEARN before the robot
#: has run once.  It updates the header's policy indicator instead; LEARN is
#: reached when ``metrics_ready`` proves the retraining actually helped.
LOOP_STAGE_FOR_EVENT: dict[str, LoopStage] = {
    "sim_started": LoopStage.SIMULATE,
    "sim_progress": LoopStage.SIMULATE,
    "sim_passed": LoopStage.VERIFY,
    "sim_failed": LoopStage.SIMULATE,
    "deploy_started": LoopStage.DEPLOY,
    "video_ready": LoopStage.EXPERIENCE,
    "real_failed": LoopStage.EXPERIENCE,
    "real_success": LoopStage.EXPERIENCE,
    "critic_started": LoopStage.DIAGNOSE,
    "diagnosis_ready": LoopStage.DIAGNOSE,
    "curriculum_ready": LoopStage.ADAPT,
    "batch_started": LoopStage.ADAPT,
    "batch_progress": LoopStage.ADAPT,
    "batch_complete": LoopStage.ADAPT,
    "metrics_ready": LoopStage.LEARN,
    "redeploy_started": LoopStage.DEPLOY,
}
