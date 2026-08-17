"""FastAPI app. The dashboard's only backend.

The UI/backend contract is deliberately tiny (plan section 15): one event
stream plus a handful of POSTs that kick off phases. The dashboard is an
event renderer, so nothing here returns state the stream would not also
carry — the POSTs return an acknowledgement and the truth arrives as events.

Run:  backend/.venv/bin/uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orchestrator import orchestrator
from schemas import PolicyMetrics
from state import RunStatus, bus

app = FastAPI(title="Continual Embodied Learning", version="0.1.0")

#: Rollout videos, batch thumbnails and saved diagnoses.
#:
#: Antioch's own artifact URLs are signed and expire, so anything the
#: dashboard displays is downloaded here first and served by us. That also
#: means the demo keeps working with no network, which matters at a booth.
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
for subdir in ("videos", "demo/batch", "diagnoses", "evals"):
    (ARTIFACTS / subdir).mkdir(parents=True, exist_ok=True)

# The dashboard runs on a different port in dev, and at a booth it may be
# opened from a second machine, so origins stay open on a local network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/artifacts", StaticFiles(directory=ARTIFACTS), name="artifacts")


class Ack(BaseModel):
    ok: bool = True
    detail: str = ""


# --- event stream -----------------------------------------------------------


@app.get("/api/events")
async def events() -> StreamingResponse:
    """Server-sent events: the run so far, then everything that follows."""

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for payload in bus.subscribe():
                yield f"data: {json.dumps(payload)}\n\n".encode()
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this a proxy will buffer the stream and the dashboard
            # will appear frozen until the run finishes.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status", response_model=RunStatus)
async def status() -> RunStatus:
    return bus.status


@app.get("/api/history")
async def history() -> dict:
    """Whole run as one payload — same shape as the fixture file.

    Lets the dashboard fall back to a plain fetch if SSE is unavailable, and
    makes it trivial to save a real run as a replayable artifact.
    """
    return {
        "meta": {
            "source": "live",
            "run_id": bus.status.run_id,
            "event_count": len(bus.history),
        },
        "events": bus.history,
    }


# --- phases -----------------------------------------------------------------


class SimRequest(BaseModel):
    policy: str = "v0"
    n_episodes: int = 30


@app.post("/api/sim/run", response_model=Ack)
async def sim_run(body: SimRequest) -> Ack:
    orchestrator.launch(orchestrator.run_sim(body.policy, body.n_episodes))
    return Ack(detail=f"simulating {body.n_episodes} episodes with {body.policy}")


class DeployRequest(BaseModel):
    policy: str = "v0"
    redeploy: bool = False


@app.post("/api/deploy", response_model=Ack)
async def deploy(body: DeployRequest) -> Ack:
    orchestrator.launch(orchestrator.deploy(body.policy, body.redeploy))
    return Ack(detail=f"deploying {body.policy} to SO-101")


class AnalyzeRequest(BaseModel):
    video_path: str | None = None


@app.post("/api/critic/analyze", response_model=Ack)
async def analyze(body: AnalyzeRequest) -> Ack:
    orchestrator.launch(orchestrator.analyze(body.video_path))
    return Ack(detail="critic analysing rollout")


class CurriculumRequest(BaseModel):
    n_scenarios: int = 30


@app.post("/api/curriculum/generate", response_model=Ack)
async def curriculum(body: CurriculumRequest) -> Ack:
    if orchestrator.diagnosis is None:
        raise HTTPException(status_code=409, detail="No diagnosis yet; run /api/critic/analyze.")
    await orchestrator.generate_curriculum(body.n_scenarios)
    return Ack(detail=f"{len(orchestrator.curriculum)} parameters mapped")


@app.post("/api/sim/batch", response_model=Ack)
async def sim_batch(body: CurriculumRequest) -> Ack:
    if not orchestrator.curriculum:
        raise HTTPException(status_code=409, detail="No curriculum yet.")
    orchestrator.launch(orchestrator.run_targeted_batch(body.n_scenarios))
    return Ack(detail=f"running {body.n_scenarios} targeted scenarios")


class PolicyRequest(BaseModel):
    policy: str
    checkpoint: str | None = None


@app.post("/api/policy/load", response_model=Ack)
async def policy_load(body: PolicyRequest) -> Ack:
    await orchestrator.load_policy(body.policy, body.checkpoint)
    return Ack(detail=f"loaded {body.policy}")


@app.post("/api/metrics", response_model=Ack)
async def metrics(body: PolicyMetrics) -> Ack:
    """Person 2 posts the real v0/v1 figures here when they exist."""
    await orchestrator.publish_metrics(body)
    return Ack(detail="metrics published")


# --- demo control -----------------------------------------------------------


@app.post("/api/demo/run", response_model=Ack)
async def demo_run() -> Ack:
    if bus.status.busy:
        raise HTTPException(status_code=409, detail="A run is already in progress.")
    orchestrator.launch(orchestrator.run_full_demo())
    return Ack(detail="running the full loop")


@app.post("/api/demo/reset", response_model=Ack)
async def demo_reset() -> Ack:
    # Stop the run before wiping it, or it keeps publishing into the next one.
    await orchestrator.cancel_running()
    orchestrator.reset()
    return Ack(detail="demo reset")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
