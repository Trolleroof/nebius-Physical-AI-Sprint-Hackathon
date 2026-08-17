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
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Protocol

from schemas import FailureDiagnosis, parse_diagnosis

ROOT = Path(__file__).resolve().parent.parent


def _load_root_env() -> None:
    """Read the repo-root .env into os.environ, without overriding real env.

    Keeps the API key out of the repo (the file is gitignored) while letting
    both uvicorn and the standalone scripts pick it up with zero setup.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))

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
  block_x, block_y (block position on the table, metres)
  tray_x, tray_y (tray centre position, metres)
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
                    {"parameter": "block_y", "min": -0.17, "max": -0.10},
                    # Deliberately near-full-range: exercises the mapper's
                    # rejection of uninformative recommendations on every run.
                    {"parameter": "block_x", "min": 0.10, "max": 0.50},
                ],
                "summary": (
                    "The gripper closed beside the block rather than around "
                    "it, catching only its near edge."
                ),
            }
        )


class GeminiCritic:
    """Gemini Robotics ER 2 as the embodied reasoning critic.

    Sends the rollout video to ``gemini-robotics-er-2-preview`` with the
    response constrained to the FailureDiagnosis schema, so the model cannot
    answer outside the closed vocabulary. The reply still goes through
    ``parse_diagnosis`` — the schema constrains, the parser decides.

    Any failure after construction degrades to a zero-confidence "unknown"
    diagnosis rather than raising: the module contract is that a dead critic
    reads as "critic unsure" on the dashboard, never as a dead demo.
    """

    MODEL = os.environ.get("GEMINI_MODEL", "gemini-robotics-er-2-preview")
    #: Above this the request goes through the Files API instead of inline.
    INLINE_LIMIT = 19 * 1024 * 1024
    #: The API samples video at 1 fps by default, which can miss a fast grasp.
    FPS = 3.0
    TIMEOUT_S = float(os.environ.get("CRITIC_TIMEOUT_S", "75"))

    def __init__(self) -> None:
        _load_root_env()
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                "CRITIC_BACKEND=gemini needs GEMINI_API_KEY — create one at "
                "aistudio.google.com/apikey and put it in the repo-root .env."
            )
        # Imported here so the mock path never needs the SDK installed.
        from google import genai

        self._client = genai.Client()

    @staticmethod
    def _resolve(video_path: str) -> Path:
        """Map an event-stream URL like /artifacts/videos/x.mp4 to a real file."""
        candidates = (Path(video_path), ROOT / video_path.lstrip("/\\"))
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"no rollout video at {video_path}")

    async def analyze(self, video_path: str, task: str = DEFAULT_TASK) -> FailureDiagnosis:
        try:
            return await asyncio.wait_for(self._analyze(video_path, task), self.TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — degrade, never kill the run
            print(f"[critic] {type(exc).__name__}: {exc}", file=sys.stderr)
            return parse_diagnosis(
                {
                    "success": False,
                    "stage": "unknown",
                    "failure": "unknown",
                    "confidence": 0.0,
                    "summary": f"Critic unavailable ({type(exc).__name__}) — rollout is undiagnosed.",
                }
            )

    async def _analyze(self, video_path: str, task: str) -> FailureDiagnosis:
        from google.genai import types

        path = self._resolve(video_path)
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        metadata = types.VideoMetadata(fps=self.FPS)

        data = path.read_bytes()
        if len(data) <= self.INLINE_LIMIT:
            video = types.Part(
                inline_data=types.Blob(data=data, mime_type=mime), video_metadata=metadata
            )
        else:
            upload = await self._client.aio.files.upload(file=path)
            while upload.state and upload.state.name == "PROCESSING":
                await asyncio.sleep(2.0)
                upload = await self._client.aio.files.get(name=upload.name)
            if upload.state and upload.state.name == "FAILED":
                raise RuntimeError(f"Gemini rejected the upload of {path.name}")
            video = types.Part(
                file_data=types.FileData(file_uri=upload.uri, mime_type=upload.mime_type or mime),
                video_metadata=metadata,
            )

        response = await self._client.aio.models.generate_content(
            model=self.MODEL,
            # Video first, instruction after — the documented best practice.
            contents=[video, "Analyse this rollout and return the diagnosis."],
            config=types.GenerateContentConfig(
                system_instruction=CRITIC_PROMPT.format(task=task),
                response_mime_type="application/json",
                response_schema=FailureDiagnosis,
                # "medium" measured ~26 s on a 5 s clip, "low" ~4 s. A judge
                # is waiting, so demo day can flip this without touching code.
                thinking_config=types.ThinkingConfig(
                    thinking_level=os.environ.get("GEMINI_THINKING", "medium")
                ),
            ),
        )
        return parse_diagnosis(json.loads(response.text or "{}"))


def get_critic() -> Critic:
    """Pick the critic implementation.

    CRITIC_BACKEND=mock (default) needs nothing; CRITIC_BACKEND=gemini uses
    Gemini Robotics ER 2 and needs GEMINI_API_KEY in the env or root .env.
    """
    backend = os.environ.get("CRITIC_BACKEND", "mock").lower()
    if backend == "mock":
        return MockCritic()
    if backend in {"gemini", "er2", "gemini-er2"}:
        return GeminiCritic()
    raise NotImplementedError(
        f"CRITIC_BACKEND={backend!r} has no implementation yet. "
        "Add it in backend/critic.py; it must satisfy the Critic protocol."
    )
