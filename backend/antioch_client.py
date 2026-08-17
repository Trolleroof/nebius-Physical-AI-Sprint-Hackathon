"""Antioch / Isaac execution, behind one swappable interface.

Two operations matter to the demo: run a batch of episodes under a given
parameter distribution, and report each one's success. Everything else the
simulator can do is out of scope today.

To go live, implement ``AntiochClient.run_batch`` — most likely shelling out
to the Antioch CLI with a scenario file written from ``changes`` — and select
it in ``get_sim``. Yielding results one at a time (rather than returning a
list) is what lets the dashboard's grid fill cell by cell.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import AsyncIterator, Protocol

from schemas import SimChange


class AntiochClient(Protocol):
    async def run_batch(
        self, n: int, changes: list[SimChange], seed: int = 0
    ) -> AsyncIterator[bool]: ...


class MockAntioch:
    """Plausible batch behaviour with no simulator attached.

    Success rate degrades as the curriculum gets harder, so the grid does not
    come back all green and the numbers stay believable. Seeded, so a given
    run is reproducible — a demo that shuffles its own results between takes
    is impossible to rehearse against.
    """

    seconds_per_episode = 0.35

    async def run_batch(
        self, n: int, changes: list[SimChange], seed: int = 0
    ) -> AsyncIterator[bool]:
        rng = random.Random(seed or len(changes))
        # Each additional perturbed parameter costs a little success rate.
        base_rate = max(0.55, 0.92 - 0.06 * len(changes))

        for _ in range(n):
            await asyncio.sleep(self.seconds_per_episode)
            yield rng.random() < base_rate


class AntiochCLI:
    """Drives the real `antioch` CLI.

    Confirmed against antioch-sim's help output:

        antioch scenario run --scenario NAME --set K=V ... --queue --json
        antioch scenario show RUN_ID --json

    `--set` takes single values, so the curriculum's ranges are sampled into
    one concrete parameter set per run (see mapper.sample_curriculum).

    TWO THINGS STILL NEED CONFIRMING FROM WHOEVER OWNS THE SIM:
      * SCENARIO — the authored scenario's name.
      * _succeeded() — which key in a finished run's JSON means "passed".
        The guesses below cover the obvious shapes; check one real run with
        `antioch scenario show <id> --json` and fix it in one place.
    """

    #: Authored scenario to run. Override with ANTIOCH_SCENARIO.
    SCENARIO = os.environ.get("ANTIOCH_SCENARIO", "cube_to_tray")
    #: How long to wait for a queued run before giving up on it.
    TIMEOUT_S = float(os.environ.get("ANTIOCH_TIMEOUT_S", "300"))
    POLL_S = 3.0

    async def _cli(self, *args: str) -> dict | list:
        process = await asyncio.create_subprocess_exec(
            "antioch",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"antioch {' '.join(args)} failed: {stderr.decode().strip()}")
        return json.loads(stdout.decode() or "{}")

    @staticmethod
    def _run_ids(queued: dict | list) -> list[str]:
        """Pull run IDs out of `--queue --json` output, whatever it wraps them in."""
        if isinstance(queued, dict):
            queued = queued.get("runs") or queued.get("scenario_runs") or [queued]
        ids = []
        for entry in queued:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict):
                found = entry.get("id") or entry.get("run_id") or entry.get("scenario_run_id")
                if found:
                    ids.append(str(found))
        return ids

    @staticmethod
    def _succeeded(run: dict) -> bool | None:
        """None while still running; True/False once the run has settled."""
        status = str(run.get("status") or run.get("state") or "").lower()
        if status in {"queued", "running", "pending", "dispatched", ""}:
            return None
        if status in {"failed", "error", "cancelled", "timeout"}:
            return False

        results = run.get("results") or {}
        for key in ("success", "passed", "task_success", "cube_in_tray"):
            if key in results:
                return bool(results[key])
            if key in run:
                return bool(run[key])
        # Settled with no recognisable success key: treat completion as pass
        # and let the check above be corrected once a real run is inspected.
        return status in {"succeeded", "success", "passed", "completed", "complete"}

    async def run_batch(
        self, n: int, changes: list[SimChange], seed: int = 0
    ) -> AsyncIterator[bool]:
        from mapper import sample_curriculum

        samples = sample_curriculum(changes, n, seed) or [{} for _ in range(n)]

        run_ids: list[str] = []
        for sample in samples:
            args = ["scenario", "run", "--scenario", self.SCENARIO]
            for key, value in sample.items():
                args += ["--set", f"{key}={value}"]
            args += ["--queue", "--json"]
            run_ids += self._run_ids(await self._cli(*args))

        # Yield in submission order so the dashboard grid fills predictably,
        # even though the machines finish out of order.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.TIMEOUT_S
        for run_id in run_ids:
            while True:
                run = await self._cli("scenario", "show", run_id, "--json")
                verdict = self._succeeded(run if isinstance(run, dict) else {})
                if verdict is not None:
                    yield verdict
                    break
                if loop.time() > deadline:
                    yield False  # never stall the demo on a hung machine
                    break
                await asyncio.sleep(self.POLL_S)


def get_sim() -> AntiochClient:
    backend = os.environ.get("SIM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockAntioch()
    if backend in {"antioch", "cli"}:
        return AntiochCLI()
    raise NotImplementedError(
        f"SIM_BACKEND={backend!r} has no implementation yet. "
        "Add it in backend/antioch_client.py; it must satisfy the AntiochClient protocol."
    )
