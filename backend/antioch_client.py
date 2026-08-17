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


def get_sim() -> AntiochClient:
    backend = os.environ.get("SIM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockAntioch()
    raise NotImplementedError(
        f"SIM_BACKEND={backend!r} has no implementation yet. "
        "Add it in backend/antioch_client.py; it must satisfy the AntiochClient protocol."
    )
