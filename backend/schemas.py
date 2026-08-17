"""The critic contract: what a video model is allowed to tell us.

Two rules govern this file.

1.  The vocabulary is closed.  Stages, failure modes and simulator
    parameters are fixed enums, so a video model cannot invent a category
    the mapper has never heard of.  These enums are also what you hand to
    the model as its JSON schema for constrained generation.

2.  Parsing is permissive; clamping is not.  ``parse_diagnosis`` must never
    raise on demo day, so unrecognised values degrade to ``unknown`` and
    unrecognised parameters are dropped.  Deciding what is *legal* to send
    to the simulator belongs to the mapper (plan section 10), not here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Stage(str, Enum):
    """Where in the task the rollout was when it went wrong."""

    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    COMPLETE = "complete"
    UNKNOWN = "unknown"


#: The stages that form the dashboard's progress checklist, in order.
#: ``complete`` and ``unknown`` are outcomes, not steps, so they are excluded.
ORDERED_STAGES: tuple[Stage, ...] = (
    Stage.APPROACH,
    Stage.GRASP,
    Stage.LIFT,
    Stage.TRANSPORT,
    Stage.PLACE,
)


class FailureMode(str, Enum):
    """What went wrong, in the critic's fixed vocabulary."""

    MISSED_OBJECT = "missed_object"
    BAD_ALIGNMENT = "bad_alignment"
    FAILED_GRASP = "failed_grasp"
    OBJECT_SLIP = "object_slip"
    PREMATURE_RELEASE = "premature_release"
    COLLISION = "collision"
    UNREACHABLE_POSE = "unreachable_pose"
    PLACEMENT_ERROR = "placement_error"
    NONE = "none"  # the rollout succeeded
    UNKNOWN = "unknown"


class SimParameter(str, Enum):
    """The only simulator knobs the mapper is permitted to turn.

    These are exactly the keyword arguments of ``so101_pick_place`` in
    sim/antioch/src/pick_place.py. That signature *is* the parameter schema:
    Antioch rejects an unknown ``--set`` key at dispatch, so a parameter
    invented here would surface as a hard failure rather than a silent no-op.

    The scenario is a scripted joint-space expert that reaches a fixed
    radius, so only the *bearing* to the block and tray can vary — there is
    no radial distance, friction, mass, yaw or injected noise. Adding one
    means adding a typed keyword argument to the scenario first.
    """

    BLOCK_AZIMUTH = "block_azimuth"
    TRAY_AZIMUTH = "tray_azimuth"


#: Hard limits per parameter, as ``(floor, ceiling, unit)``, taken from the
#: ``ge=``/``le=`` on each ``antioch.param`` in the scenario. The critic may
#: recommend anything; the mapper clamps into these before Antioch sees them.
PARAMETER_BOUNDS: dict[SimParameter, tuple[float, float, str]] = {
    SimParameter.BLOCK_AZIMUTH: (-0.9, 0.9, "rad"),
    SimParameter.TRAY_AZIMUTH: (-0.9, 0.9, "rad"),
}


#: Below this, the UI should render "critic unsure" instead of a percentage.
#: A diagnosis can still be used as a hypothesis; it just should not be
#: presented to a judge as though the model were certain.
CONFIDENCE_FLOOR = 0.35


class EstimatedCause(BaseModel):
    """One hypothesis for why the failure happened, with the critic's belief."""

    cause: str
    confidence: float = Field(ge=0.0, le=1.0)


class SimChange(BaseModel):
    """A requested range for one simulator parameter.

    Emitted by the critic as a *recommendation* and by the mapper as a
    *clamped* value.  ``clamped`` records whether the mapper had to pull the
    range in, which the dashboard can surface to show the guardrail working.
    """

    parameter: SimParameter
    min: float
    max: float
    clamped: bool = False


class FailureDiagnosis(BaseModel):
    """The critic's structured read of one real-world rollout.

    This is the only thing that crosses the boundary out of the video model.
    Everything downstream — mapper, curriculum, dashboard — reads this and
    nothing else.
    """

    success: bool
    stage: Stage
    failure: FailureMode
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_causes: list[EstimatedCause] = Field(default_factory=list)
    recommended_sim_changes: list[SimChange] = Field(default_factory=list)

    #: One plain sentence for the dashboard.  Presentation only — never parse it.
    summary: str = ""

    def stage_timeline(self) -> list[tuple[Stage, str]]:
        """Derive the per-stage checklist the failure panel renders.

        The critic reports a single failure stage; the checklist is a
        consequence of it, not a separate field.  Asking the model for both
        is asking it to contradict itself.

        Status is one of ``passed`` / ``failed`` / ``not_reached``.
        """
        if self.success:
            return [(stage, "passed") for stage in ORDERED_STAGES]

        if self.stage not in ORDERED_STAGES:
            # Unknown failure stage: report nothing rather than guess.
            return [(stage, "not_reached") for stage in ORDERED_STAGES]

        failed_at = ORDERED_STAGES.index(self.stage)
        timeline = []
        for i, stage in enumerate(ORDERED_STAGES):
            if i < failed_at:
                timeline.append((stage, "passed"))
            elif i == failed_at:
                timeline.append((stage, "failed"))
            else:
                timeline.append((stage, "not_reached"))
        return timeline


def parse_diagnosis(raw: dict[str, Any]) -> FailureDiagnosis:
    """Coerce a raw model response into a diagnosis without ever raising.

    A malformed field should cost us that field, not the whole demo.  Values
    outside the vocabulary become ``unknown``; parameters outside the
    whitelist are dropped; out-of-range confidences are pinned to [0, 1].
    Numeric ranges are left alone — clamping is the mapper's call.
    """

    def as_enum(enum_cls, value, fallback):
        try:
            return enum_cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return fallback

    def as_confidence(value):
        """A confidence outside [0, 1] means the critic misbehaved.

        Pinning such a value to 1.0 would label our least trustworthy
        response as our most trusted one, so anything unparseable or out of
        range reads as zero instead.  The dashboard treats confidence below
        ``CONFIDENCE_FLOOR`` as "critic unsure" rather than as a real number.
        """
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return confidence if 0.0 <= confidence <= 1.0 else 0.0

    causes = []
    for item in raw.get("estimated_causes") or []:
        if not isinstance(item, dict) or "cause" not in item:
            continue
        causes.append(
            EstimatedCause(
                cause=str(item["cause"]),
                confidence=as_confidence(item.get("confidence")),
            )
        )

    changes = []
    for item in raw.get("recommended_sim_changes") or []:
        if not isinstance(item, dict):
            continue
        parameter = as_enum(SimParameter, item.get("parameter"), None)
        if parameter is None:
            continue  # not on the whitelist; the mapper would reject it anyway
        try:
            lo, hi = float(item["min"]), float(item["max"])
        except (KeyError, TypeError, ValueError):
            continue
        changes.append(SimChange(parameter=parameter, min=min(lo, hi), max=max(lo, hi)))

    success = bool(raw.get("success", False))
    return FailureDiagnosis(
        success=success,
        stage=as_enum(Stage, raw.get("stage"), Stage.COMPLETE if success else Stage.UNKNOWN),
        failure=as_enum(FailureMode, raw.get("failure"), FailureMode.NONE if success else FailureMode.UNKNOWN),
        confidence=as_confidence(raw.get("confidence")),
        estimated_causes=causes,
        recommended_sim_changes=changes,
        summary=str(raw.get("summary") or ""),
    )


class MetricRow(BaseModel):
    """One row of the before/after table.

    Values are strings so a row can hold ``"90%"``, ``"FAIL"`` or ``None``.
    Section 12: never invent a number, show ``None`` until it is measured.
    """

    label: str
    v0: str | None = None
    v1: str | None = None


class PolicyMetrics(BaseModel):
    """The improvement panel. Held-out set is the row that actually matters."""

    rows: list[MetricRow] = Field(default_factory=list)
    held_out_scenarios: int = 0
    initial_demos: int = 0
    corrective_demos: int = 0
