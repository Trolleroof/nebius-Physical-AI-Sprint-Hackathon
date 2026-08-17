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

import random

from schemas import (
    PARAMETER_BOUNDS,
    FailureDiagnosis,
    FailureMode,
    SimChange,
    SimParameter,
)

#: Nominal scenario values — the defaults on ``so101_pick_place``. The
#: dashboard renders these as the "before" side of `0.32 -> 0.24-0.40`.
BASELINE_SCENARIO: dict[SimParameter, float] = {
    SimParameter.BLOCK_AZIMUTH: -0.3157,
    SimParameter.TRAY_AZIMUTH: 0.5225,
}

#: Fallback curriculum per failure mode, in the spirit of plan section 10 but
#: restricted to knobs the scenario actually has. Used when the critic
#: recommends nothing, and merged with its suggestions when it does. Ranges
#: are deliberately modest: targeted coverage of the observed weakness, not
#: blanket randomisation.
#:
#: Several failure modes map to nothing, because the scenario exposes only
#: geometry — there is no friction, mass, yaw or noise to vary. Those return
#: an empty curriculum, and the dashboard says so rather than inventing work.
#: Extending coverage means adding a typed parameter to the scenario first.
RULES: dict[FailureMode, list[tuple[SimParameter, float, float]]] = {
    # The scenario declares four sweep cases from -0.42 to -0.12 rad, which is
    # the bearing band the expert is known to reach. Rules stay near it: the
    # legal range is +/-0.9, but sampling the whole span mostly produces poses
    # the arm cannot make, which is noise rather than curriculum.
    FailureMode.FAILED_GRASP: [
        (SimParameter.BLOCK_AZIMUTH, -0.42, -0.12),
    ],
    FailureMode.MISSED_OBJECT: [
        (SimParameter.BLOCK_AZIMUTH, -0.52, -0.05),
    ],
    FailureMode.BAD_ALIGNMENT: [
        (SimParameter.BLOCK_AZIMUTH, -0.40, -0.20),
    ],
    FailureMode.UNREACHABLE_POSE: [
        (SimParameter.BLOCK_AZIMUTH, -0.62, 0.05),
    ],
    FailureMode.PLACEMENT_ERROR: [
        (SimParameter.TRAY_AZIMUTH, 0.40, 0.64),
    ],
}

#: Failure modes the scenario cannot currently reproduce, and what it would
#: take to cover them. Surfaced to the operator instead of silently producing
#: an empty curriculum — an honest "we can't simulate this yet" is a better
#: answer to a judge than a curriculum that does not address the diagnosis.
UNMAPPABLE: dict[FailureMode, str] = {
    FailureMode.OBJECT_SLIP: "needs a friction or mass parameter on the simulated scene",
    FailureMode.PREMATURE_RELEASE: "needs a grip-force or actuation-timing parameter",
    FailureMode.COLLISION: "needs a traverse-height parameter",
}

#: Causes the critic names in prose, mapped to the knob they implicate. Lets a
#: confident cause pull in a parameter the failure-mode rule alone would miss.
#: Only bearings can vary, so a cause implying radial distance ("too far to
#: reach") has no knob and is deliberately absent rather than mapped to the
#: nearest thing that happens to exist.
CAUSE_HINTS: dict[str, SimParameter] = {
    "object_pose": SimParameter.BLOCK_AZIMUTH,
    "block_position": SimParameter.BLOCK_AZIMUTH,
    "lateral_offset": SimParameter.BLOCK_AZIMUTH,
    "bearing_error": SimParameter.BLOCK_AZIMUTH,
    "tray_position": SimParameter.TRAY_AZIMUTH,
    "placement_offset": SimParameter.TRAY_AZIMUTH,
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
        # weak. Prefer the targeted rule; if there is no rule, drop it rather
        # than accept it. "Vary this across its entire range" is not a weaker
        # curriculum than nothing, it is worse than nothing — it spends the
        # whole batch on scenarios the diagnosis never implicated.
        if legal_span > 0 and requested_span / legal_span > MAX_RECOMMENDED_SPAN:
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


def unmappable_reason(diagnosis: FailureDiagnosis) -> str | None:
    """Why this diagnosis produced no curriculum, if it produced none.

    Returning a reason is the point: a failure the simulator cannot reproduce
    is a real finding about the scenario, and hiding it behind an empty panel
    would make the loop look broken rather than honest.
    """
    if diagnosis.success or map_diagnosis(diagnosis):
        return None
    return UNMAPPABLE.get(
        diagnosis.failure, "no parameter on so101_pick_place covers this failure mode"
    )


def sample_curriculum(changes: list[SimChange], n: int, seed: int = 0) -> list[dict[str, float]]:
    """Turn ranges into n concrete parameter sets, one per scenario run.

    The mapper reasons in ranges because that is what a diagnosis implies,
    but a simulator takes one concrete value per run, so a curriculum of
    thirty scenarios is thirty samples from those ranges.

    Sampling is stratified rather than uniform: each parameter's range is cut
    into n even bands and one value drawn per band, then shuffled. Uniform
    sampling of thirty points routinely leaves gaps and clusters, and a gap
    at the hard end of the friction range is exactly the case we are trying
    to cover. Seeded, so a demo can be rehearsed against fixed scenarios.
    """
    if not changes or n <= 0:
        return []

    rng = random.Random(seed)
    columns: dict[str, list[float]] = {}

    for change in changes:
        width = (change.max - change.min) / n
        band_samples = [
            round(change.min + width * (i + rng.random()), 4) for i in range(n)
        ]
        rng.shuffle(band_samples)
        columns[change.parameter.value] = band_samples

    return [{name: values[i] for name, values in columns.items()} for i in range(n)]


def baseline_for(changes: list[SimChange]) -> dict[str, float]:
    """The 'before' values the dashboard shows beside each new range."""
    return {
        change.parameter.value: BASELINE_SCENARIO[change.parameter]
        for change in changes
        if change.parameter in BASELINE_SCENARIO
    }
