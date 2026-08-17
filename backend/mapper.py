"""Turns a critic diagnosis into simulator parameters that are safe to run.

This is deliberately boring, deterministic Python — no model in the loop.
Plan section 10: the critic may *recommend* anything, the mapper decides what
is *legal*. Two independent inputs feed it:

1.  The critic's own ``recommended_sim_changes``.
2.  A fixed rule table keyed by failure mode, so a diagnosis still produces a
    useful curriculum even when the model returns no recommendations at all.

Everything is then clamped into ``PARAMETER_BOUNDS``. A clamped range is
flagged rather than silently narrowed, because the dashboard shows the
guardrail firing — that is the honest answer to "how do you know the critic
is right?"
"""

from __future__ import annotations

from schemas import (
    PARAMETER_BOUNDS,
    FailureDiagnosis,
    FailureMode,
    SimChange,
    SimParameter,
)

#: Nominal scenario values — what the baseline training distribution used.
#: The dashboard renders these as the "before" side of `0.60 -> 0.20-0.50`.
#: Replace with the real Antioch scenario defaults once they are known.
BASELINE_SCENARIO: dict[SimParameter, float] = {
    SimParameter.OBJECT_FRICTION: 0.60,
    SimParameter.OBJECT_MASS: 1.00,
    SimParameter.OBJECT_X: 0.00,
    SimParameter.OBJECT_Y: 0.00,
    SimParameter.OBJECT_YAW: 0.00,
    SimParameter.GRASP_POSE_NOISE: 0.00,
    SimParameter.CAMERA_POSE_NOISE: 0.00,
    SimParameter.ACTION_DELAY: 0.00,
    SimParameter.JOINT_TARGET_NOISE: 0.00,
}

#: Fallback curriculum per failure mode, from plan section 10's example rules.
#: Used when the critic recommends nothing, and merged with its suggestions
#: when it does. Ranges are intentionally modest: the point is targeted
#: coverage of the observed weakness, not blanket randomisation.
RULES: dict[FailureMode, list[tuple[SimParameter, float, float]]] = {
    FailureMode.OBJECT_SLIP: [
        (SimParameter.OBJECT_FRICTION, 0.20, 0.50),
        (SimParameter.OBJECT_MASS, 0.80, 1.30),
        (SimParameter.GRASP_POSE_NOISE, 0.0, 8.0),
    ],
    FailureMode.FAILED_GRASP: [
        (SimParameter.OBJECT_YAW, -20.0, 20.0),
        (SimParameter.OBJECT_X, -0.04, 0.04),
        (SimParameter.GRASP_POSE_NOISE, 0.0, 10.0),
    ],
    FailureMode.MISSED_OBJECT: [
        (SimParameter.OBJECT_X, -0.06, 0.06),
        (SimParameter.OBJECT_Y, -0.06, 0.06),
    ],
    FailureMode.BAD_ALIGNMENT: [
        (SimParameter.CAMERA_POSE_NOISE, 0.0, 6.0),
        (SimParameter.OBJECT_YAW, -25.0, 25.0),
    ],
    FailureMode.PREMATURE_RELEASE: [
        (SimParameter.ACTION_DELAY, 0.0, 60.0),
        (SimParameter.GRASP_POSE_NOISE, 0.0, 6.0),
    ],
    FailureMode.COLLISION: [
        (SimParameter.ACTION_DELAY, 0.0, 40.0),
        (SimParameter.JOINT_TARGET_NOISE, 0.0, 2.0),
    ],
    FailureMode.UNREACHABLE_POSE: [
        (SimParameter.OBJECT_X, -0.05, 0.05),
        (SimParameter.OBJECT_Y, -0.05, 0.05),
    ],
    FailureMode.PLACEMENT_ERROR: [
        (SimParameter.OBJECT_X, -0.05, 0.05),
        (SimParameter.JOINT_TARGET_NOISE, 0.0, 2.5),
    ],
}

#: Causes the critic names in prose, mapped to the knob they implicate. Lets a
#: confident cause pull in a parameter the failure-mode rule alone would miss.
CAUSE_HINTS: dict[str, SimParameter] = {
    "low_friction": SimParameter.OBJECT_FRICTION,
    "high_friction": SimParameter.OBJECT_FRICTION,
    "friction_mismatch": SimParameter.OBJECT_FRICTION,
    "heavy_object": SimParameter.OBJECT_MASS,
    "grasp_offset": SimParameter.GRASP_POSE_NOISE,
    "grasp_position": SimParameter.GRASP_POSE_NOISE,
    "camera_offset": SimParameter.CAMERA_POSE_NOISE,
    "calibration_error": SimParameter.CAMERA_POSE_NOISE,
    "object_pose": SimParameter.OBJECT_X,
    "object_rotation": SimParameter.OBJECT_YAW,
    "actuator_lag": SimParameter.ACTION_DELAY,
    "timing": SimParameter.ACTION_DELAY,
}

#: A cause below this confidence does not get to add a parameter of its own.
CAUSE_THRESHOLD = 0.15

#: A critic recommendation covering more than this fraction of a parameter's
#: legal span is treated as uninformative and loses to the targeted rule.
#: The whole point of failure-conditioned curriculum is to spend simulation on
#: the observed weakness; a request for "vary mass across its entire range"
#: is blanket randomisation wearing a diagnosis as a hat.
MAX_RECOMMENDED_SPAN = 0.6


def clamp(parameter: SimParameter, low: float, high: float) -> SimChange:
    """Pull a requested range inside the legal bounds, recording if we had to."""
    floor, ceiling, _unit = PARAMETER_BOUNDS[parameter]
    low, high = min(low, high), max(low, high)
    clamped_low = max(floor, min(low, ceiling))
    clamped_high = max(floor, min(high, ceiling))
    was_clamped = clamped_low != low or clamped_high != high
    return SimChange(
        parameter=parameter,
        min=round(clamped_low, 4),
        max=round(clamped_high, 4),
        clamped=was_clamped,
    )


def map_diagnosis(diagnosis: FailureDiagnosis) -> list[SimChange]:
    """Produce the clamped curriculum for one diagnosis.

    Precedence, highest first: an explicit critic recommendation, then the
    failure-mode rule, then a confident named cause. Each parameter appears
    at most once — the first source to claim it wins, so the model's own
    numbers are preferred over our defaults when it bothered to give them.
    """
    if diagnosis.success:
        return []

    proposals: dict[SimParameter, tuple[float, float]] = {}
    rule_for = {parameter: (low, high) for parameter, low, high in RULES.get(diagnosis.failure, [])}
    # Parameters where we overrode the critic outright, as opposed to merely
    # trimming its range. Surfaced the same way as a clamp, because from the
    # dashboard's point of view both mean "the mapper did not do as it was told".
    overridden: set[SimParameter] = set()

    for change in diagnosis.recommended_sim_changes:
        floor, ceiling, _ = PARAMETER_BOUNDS[change.parameter]
        legal_span = ceiling - floor
        requested_span = abs(change.max - change.min)

        # A near-full-range request tells us nothing about where the policy is
        # weak, so fall through to the targeted rule if one exists.
        if legal_span > 0 and requested_span / legal_span > MAX_RECOMMENDED_SPAN:
            if change.parameter in rule_for:
                overridden.add(change.parameter)
                continue
        proposals.setdefault(change.parameter, (change.min, change.max))

    for parameter, (low, high) in rule_for.items():
        proposals.setdefault(parameter, (low, high))

    for cause in diagnosis.estimated_causes:
        if cause.confidence < CAUSE_THRESHOLD:
            continue
        parameter = CAUSE_HINTS.get(cause.cause.strip().lower())
        if parameter is None or parameter in proposals:
            continue
        # No rule fired for this parameter, so widen from the baseline by an
        # amount proportional to how strongly the critic believes the cause.
        floor, ceiling, _ = PARAMETER_BOUNDS[parameter]
        centre = BASELINE_SCENARIO.get(parameter, (floor + ceiling) / 2)
        span = (ceiling - floor) * 0.25 * cause.confidence
        proposals[parameter] = (centre - span, centre + span)

    changes = []
    for parameter, (low, high) in proposals.items():
        change = clamp(parameter, low, high)
        if parameter in overridden:
            change.clamped = True
        changes.append(change)
    return changes


def baseline_for(changes: list[SimChange]) -> dict[str, float]:
    """The 'before' values the dashboard shows beside each new range."""
    return {
        change.parameter.value: BASELINE_SCENARIO[change.parameter]
        for change in changes
        if change.parameter in BASELINE_SCENARIO
    }
