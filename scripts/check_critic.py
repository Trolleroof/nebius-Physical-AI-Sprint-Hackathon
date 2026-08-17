"""Check a critic against the contract before wiring it into the demo.

Two ways to run it.

Validate raw model output — no code needed, just paste what the model
returned into a file:

    backend/.venv/bin/python scripts/check_critic.py --json response.json

Or exercise a real implementation end to end:

    CRITIC_BACKEND=nvidia backend/.venv/bin/python scripts/check_critic.py \
        --video artifacts/videos/v0_real_failure.mp4

The point is to fail here, in one command, rather than in front of a judge.
The parser downstream is deliberately forgiving — it drops fields it does not
recognise instead of raising — which is right for demo day but means a
mistake is silent. This script is where that silence gets broken.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from mapper import map_diagnosis, sample_curriculum, unmappable_reason  # noqa: E402
from schemas import (  # noqa: E402
    CONFIDENCE_FLOOR,
    PARAMETER_BOUNDS,
    FailureMode,
    SimParameter,
    Stage,
    parse_diagnosis,
)

PASS, WARN, FAIL = "  ok  ", " warn ", " FAIL "
problems = 0
warnings = 0


def report(level: str, check: str, detail: str = "", fix: str = "") -> None:
    global problems, warnings
    if level is FAIL:
        problems += 1
    if level is WARN:
        warnings += 1
    print(f"[{level}] {check}" + (f" — {detail}" if detail else ""))
    if fix and level is not PASS:
        print(f"         fix: {fix}")


def check(raw: dict, latency_ms: int | None = None) -> None:
    print(f"\nraw response has {len(raw)} top-level fields\n")

    # --- structure ----------------------------------------------------------

    missing = [f for f in ("success", "stage", "failure", "confidence") if f not in raw]
    report(
        FAIL if missing else PASS,
        "required fields present",
        f"missing {', '.join(missing)}" if missing else "success, stage, failure, confidence",
        "the model must emit all four; constrain it with the JSON schema",
    )

    diagnosis = parse_diagnosis(raw)

    # --- vocabularies -------------------------------------------------------

    raw_stage = str(raw.get("stage", "")).strip().lower()
    if raw_stage and diagnosis.stage.value != raw_stage:
        report(
            FAIL,
            "stage is in the vocabulary",
            f"{raw.get('stage')!r} is not a valid stage, degraded to 'unknown'",
            f"use one of: {', '.join(s.value for s in Stage)}",
        )
    else:
        report(PASS, "stage is in the vocabulary", diagnosis.stage.value)

    raw_failure = str(raw.get("failure", "")).strip().lower()
    if raw_failure and diagnosis.failure.value != raw_failure:
        report(
            FAIL,
            "failure is in the vocabulary",
            f"{raw.get('failure')!r} is not a valid failure mode, degraded to 'unknown'",
            f"use one of: {', '.join(f.value for f in FailureMode)}",
        )
    else:
        report(PASS, "failure is in the vocabulary", diagnosis.failure.value)

    # --- internal consistency ----------------------------------------------

    if diagnosis.success and diagnosis.failure not in (FailureMode.NONE, FailureMode.UNKNOWN):
        report(FAIL, "success and failure agree", "success=true but a failure mode was named",
               "a successful rollout must report failure='none'")
    elif not diagnosis.success and diagnosis.failure is FailureMode.NONE:
        report(FAIL, "success and failure agree", "success=false but failure='none'",
               "name the failure, or set success=true")
    else:
        report(PASS, "success and failure agree")

    if not diagnosis.success and diagnosis.stage is Stage.COMPLETE:
        report(FAIL, "stage matches the outcome", "failed rollout reported stage='complete'",
               "report the stage the attempt failed at")
    else:
        report(PASS, "stage matches the outcome")

    # --- confidence ---------------------------------------------------------

    if "confidence" in raw and diagnosis.confidence == 0.0 and raw["confidence"] not in (0, 0.0):
        report(FAIL, "confidence is usable", f"{raw['confidence']!r} is outside 0.0–1.0, read as 0",
               "emit a float between 0 and 1, not a percentage")
    elif diagnosis.confidence < CONFIDENCE_FLOOR:
        report(WARN, "confidence is usable",
               f"{diagnosis.confidence} is below the {CONFIDENCE_FLOOR} floor",
               "fine if the model is genuinely unsure; the UI will say 'critic unsure'")
    else:
        report(PASS, "confidence is usable", str(diagnosis.confidence))

    # --- causes -------------------------------------------------------------

    raw_causes = raw.get("estimated_causes") or []
    if not diagnosis.estimated_causes:
        report(WARN, "estimated causes present", "none returned",
               "not fatal — the mapper falls back to its failure-mode rules")
    elif len(diagnosis.estimated_causes) < len(raw_causes):
        report(WARN, "estimated causes present",
               f"{len(raw_causes) - len(diagnosis.estimated_causes)} dropped as malformed",
               "each cause needs a 'cause' string and a 'confidence' float")
    else:
        report(PASS, "estimated causes present", f"{len(diagnosis.estimated_causes)} returned")

    # --- parameters: the one that fails silently ---------------------------

    raw_params = [
        str(c.get("parameter"))
        for c in (raw.get("recommended_sim_changes") or [])
        if isinstance(c, dict)
    ]
    legal = {p.value for p in SimParameter}
    unknown = [p for p in raw_params if p not in legal]

    if unknown:
        report(FAIL, "every parameter is on the whitelist",
               f"dropped silently: {', '.join(unknown)}",
               f"only these exist on the scenario: {', '.join(sorted(legal))}")
    elif raw_params:
        report(PASS, "every parameter is on the whitelist", ", ".join(raw_params))
    else:
        report(WARN, "every parameter is on the whitelist", "none recommended",
               "fine — the mapper's rules cover it, as long as the failure mode is mappable")

    # --- does it actually drive the simulator? ------------------------------

    curriculum = map_diagnosis(diagnosis)
    reason = unmappable_reason(diagnosis)

    if curriculum:
        report(PASS, "produces a runnable curriculum",
               ", ".join(f"{c.parameter.value} {c.min}–{c.max}" for c in curriculum))

        illegal = []
        for row in sample_curriculum(curriculum, 30, seed=7):
            for name, value in row.items():
                low, high, _ = PARAMETER_BOUNDS[SimParameter(name)]
                if not low <= value <= high:
                    illegal.append(f"{name}={value}")
        report(FAIL if illegal else PASS, "all 30 dispatches are inside scenario bounds",
               ", ".join(illegal[:3]) if illegal else "verified")
    elif reason:
        report(WARN, "produces a runnable curriculum", f"none — {reason}",
               "the dashboard reports this honestly, but this diagnosis will not "
               "improve the policy; pick a failure mode the scenario can reproduce")
    elif diagnosis.success:
        report(PASS, "produces a runnable curriculum", "rollout succeeded, nothing to correct")

    # --- latency ------------------------------------------------------------

    if latency_ms is not None:
        level = PASS if latency_ms < 15_000 else WARN
        report(level, "latency is demo-safe", f"{latency_ms / 1000:.1f}s",
               "over ~15s feels broken while a judge waits; send a shorter clip")


async def from_backend(video: str) -> tuple[dict, int]:
    from critic import DEFAULT_TASK, get_critic

    critic = get_critic()
    print(f"running {type(critic).__name__}.analyze({video!r})")
    loop = asyncio.get_running_loop()
    started = loop.time()
    diagnosis = await critic.analyze(video, DEFAULT_TASK)
    return diagnosis.model_dump(mode="json"), int((loop.time() - started) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="a file holding the raw model response")
    parser.add_argument("--video", default="artifacts/videos/v0_real_failure.mp4")
    args = parser.parse_args()

    if args.json:
        raw, latency = json.loads(args.json.read_text()), None
    else:
        raw, latency = asyncio.run(from_backend(args.video))

    check(raw, latency)

    print()
    if problems:
        print(f"{problems} contract violation(s). Do not wire this in yet.")
        sys.exit(1)
    print(f"Contract satisfied{f' with {warnings} warning(s)' if warnings else ''}. Safe to wire in.")


if __name__ == "__main__":
    main()
