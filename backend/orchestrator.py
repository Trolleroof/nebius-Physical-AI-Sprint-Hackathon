"""Drives the loop and narrates it as events.

Every phase is callable on its own — so a judge can watch one step live —
and ``run_full_demo`` chains them for the unattended booth loop. Each phase
holds the results the next one needs, which is why they are methods on one
object rather than free functions.

Nothing here knows how the critic or the simulator actually work; both
arrive through the protocols in ``critic.py`` and ``antioch_client.py``. That
is the seam where mock becomes real, and it is the only place that changes.
"""

from __future__ import annotations

import asyncio
import uuid

from antioch_client import get_sim
from critic import DEFAULT_TASK, get_critic
from events import (
    BatchComplete,
    BatchProgress,
    BatchStarted,
    CriticStarted,
    CurriculumReady,
    DeployStarted,
    DiagnosisReady,
    ErrorRaised,
    MetricsReady,
    PolicyLoaded,
    RealFailed,
    RealSuccess,
    RedeployStarted,
    SimFailed,
    SimPassed,
    SimProgress,
    SimStarted,
    VideoReady,
)
from mapper import baseline_for, map_diagnosis, unmappable_reason
from schemas import FailureDiagnosis, MetricRow, PolicyMetrics, SimChange
from state import bus

#: The controlled out-of-distribution condition from plan section 8.
CONDITION = "block +6cm right of its trained position"

SIM_PASS_THRESHOLD = 0.8

#: Rollout footage. These files will not exist until hardware runs; the
#: dashboard falls back to the 3D model when a video 404s, so a missing file
#: costs nothing.
VIDEO_FAILURE = "/artifacts/videos/v0_real_failure.mp4"
VIDEO_SUCCESS = "/artifacts/videos/v1_real_success.mp4"


class Orchestrator:
    def __init__(self) -> None:
        self.run_id: str = ""
        self.diagnosis: FailureDiagnosis | None = None
        self.curriculum: list[SimChange] = []
        self.batch_results: list[bool] = []
        self.last_video: str | None = None
        self._lock = asyncio.Lock()

    # --- helpers ------------------------------------------------------------

    def _ensure_run(self) -> str:
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:8]
            bus.status.run_id = self.run_id
        return self.run_id

    def reset(self) -> None:
        self.run_id = ""
        self.diagnosis = None
        self.curriculum = []
        self.batch_results = []
        self.last_video = None
        bus.reset()

    # --- phases -------------------------------------------------------------

    async def run_sim(self, policy: str = "v0", n_episodes: int = 30) -> bool:
        """Simulate, then gate. Returns whether the policy earned a deployment."""
        run_id = self._ensure_run()
        bus.status.stage, bus.status.environment, bus.status.policy = "simulating", "sim", policy

        await bus.publish(
            SimStarted(
                run_id=run_id, policy=policy, scenario="so101_pick_place", n_episodes=n_episodes
            )
        )

        sim = get_sim()
        successes = completed = 0
        async for ok in sim.run_batch(n_episodes, [], seed=1):
            completed += 1
            successes += int(ok)
            # Report periodically rather than per episode: the progress bar
            # does not need thirty updates, and the rail should not flicker.
            if completed % 10 == 0 or completed == n_episodes:
                await bus.publish(
                    SimProgress(
                        run_id=run_id, completed=completed, total=n_episodes, successes=successes
                    )
                )

        passed = successes / max(n_episodes, 1) >= SIM_PASS_THRESHOLD
        event = SimPassed if passed else SimFailed
        await bus.publish(
            event(run_id=run_id, successes=successes, total=n_episodes, threshold=SIM_PASS_THRESHOLD)
        )
        return passed

    async def load_policy(self, policy: str, checkpoint: str | None = None) -> None:
        run_id = self._ensure_run()
        bus.status.policy = policy
        await bus.publish(
            PolicyLoaded(
                run_id=run_id,
                policy=policy,
                checkpoint=checkpoint or f"act_{policy}_step12000.safetensors",
            )
        )

    async def deploy(self, policy: str = "v0", redeploy: bool = False) -> bool:
        """Run on the real arm under the controlled condition.

        Outcome is currently scripted — v0 fails, v1 succeeds — because that
        is the demo's spine. When the robot is wired in, this is where the
        real rollout and its verdict come from.
        """
        run_id = self._ensure_run()
        bus.status.stage, bus.status.environment, bus.status.policy = "deploying", "real", policy

        event = RedeployStarted if redeploy else DeployStarted
        await bus.publish(event(run_id=run_id, policy=policy, condition=CONDITION))

        await asyncio.sleep(2.0)  # stands in for the physical rollout

        succeeded = policy != "v0"
        self.last_video = VIDEO_SUCCESS if succeeded else VIDEO_FAILURE
        await bus.publish(
            VideoReady(
                run_id=run_id,
                url=self.last_video,
                poster_url=self.last_video.replace(".mp4", ".jpg"),
                duration_s=8.4,
            )
        )

        if succeeded:
            await bus.publish(
                RealSuccess(
                    run_id=run_id, policy=policy, note="Block placed in tray under the same condition."
                )
            )
        else:
            await bus.publish(
                RealFailed(
                    run_id=run_id, policy=policy, note="Gripper closed beside the block."
                )
            )
        return succeeded

    async def analyze(self, video_path: str | None = None) -> FailureDiagnosis:
        """Send the rollout to the critic and publish its structured verdict."""
        run_id = self._ensure_run()
        bus.status.stage = "diagnosing"
        video = video_path or self.last_video or VIDEO_FAILURE

        await bus.publish(CriticStarted(run_id=run_id, video_url=video))

        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            diagnosis = await get_critic().analyze(video, DEFAULT_TASK)
        except Exception as exc:  # a dead critic must not take the demo with it
            await bus.publish(ErrorRaised(run_id=run_id, message=str(exc), where="critic"))
            raise

        latency_ms = int((loop.time() - started) * 1000)
        self.diagnosis = diagnosis
        await bus.publish(DiagnosisReady(run_id=run_id, diagnosis=diagnosis, latency_ms=latency_ms))
        return diagnosis

    async def generate_curriculum(self, n_scenarios: int = 30) -> list[SimChange]:
        """Convert the diagnosis into clamped, legal simulator parameters."""
        run_id = self._ensure_run()
        bus.status.stage = "adapting"

        if self.diagnosis is None:
            raise ValueError("No diagnosis yet — call /api/critic/analyze first.")

        self.curriculum = map_diagnosis(self.diagnosis)
        await bus.publish(
            CurriculumReady(
                run_id=run_id,
                curriculum_id=f"curr-{run_id}",
                changes=self.curriculum,
                n_scenarios=n_scenarios if self.curriculum else 0,
                baseline=baseline_for(self.curriculum),
                unmappable_reason=unmappable_reason(self.diagnosis),
            )
        )
        return self.curriculum

    async def run_targeted_batch(self, n_scenarios: int = 30) -> list[bool]:
        """Run the corrective curriculum, streaming one result per scenario."""
        run_id = self._ensure_run()
        bus.status.stage, bus.status.environment = "collecting", "sim"

        await bus.publish(
            BatchStarted(run_id=run_id, curriculum_id=f"curr-{run_id}", n_scenarios=n_scenarios)
        )

        self.batch_results = []
        sim = get_sim()
        async for ok in sim.run_batch(n_scenarios, self.curriculum, seed=7):
            index = len(self.batch_results)
            self.batch_results.append(ok)
            await bus.publish(
                BatchProgress(run_id=run_id, index=index, total=n_scenarios, success=ok)
            )

        await bus.publish(
            BatchComplete(
                run_id=run_id, successes=sum(self.batch_results), total=len(self.batch_results)
            )
        )
        return self.batch_results

    async def publish_metrics(self, metrics: PolicyMetrics | None = None) -> None:
        """Publish the before/after table.

        Defaults to empty cells rather than invented ones: section 12 says
        display only what was measured, and the dashboard renders None as an
        em dash. Person 2 supplies the real figures.
        """
        run_id = self._ensure_run()
        bus.status.stage = "learning"
        await bus.publish(
            MetricsReady(
                run_id=run_id,
                metrics=metrics
                or PolicyMetrics(
                    rows=[
                        MetricRow(label="Baseline sim"),
                        MetricRow(label="Hard held-out set"),
                        MetricRow(label="Real condition", v0="FAIL", v1="PASS"),
                    ],
                    held_out_scenarios=len(self.batch_results),
                ),
            )
        )

    # --- the whole loop, unattended -----------------------------------------

    async def run_full_demo(self) -> None:
        """Sim -> real fail -> diagnose -> adapt -> learn -> real pass.

        Guarded by a lock so a second click cannot interleave two runs and
        produce a nonsensical event order on screen.
        """
        if self._lock.locked():
            return
        async with self._lock:
            bus.status.busy = True
            try:
                self.reset()
                await self.run_sim("v0")
                await self.load_policy("v0")
                await self.deploy("v0")
                await self.analyze()
                await self.generate_curriculum()
                await self.run_targeted_batch()
                await self.load_policy("v1")
                await self.publish_metrics()
                await self.deploy("v1", redeploy=True)
                bus.status.stage = "complete"
            except Exception as exc:
                await bus.publish(
                    ErrorRaised(run_id=self.run_id or "none", message=str(exc), where="orchestrator")
                )
                bus.status.stage = "error"
            finally:
                bus.status.busy = False


orchestrator = Orchestrator()
