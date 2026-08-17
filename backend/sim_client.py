"""Simulator execution, behind one swappable interface.

Two operations matter to the demo: run a batch of episodes under a given
parameter distribution, and report each one's success. Everything else a
simulator can do is out of scope.

The Antioch/Isaac client that used to live here has been removed — the
project is moving to MuJoCo. Everything upstream of this file is unaffected,
which is the point of the protocol: the orchestrator, the mapper, the event
stream and the dashboard never knew which simulator was behind it.

To wire MuJoCo in, implement ``run_batch`` and register it in ``get_sim``.
The only contract is that it yields one bool per episode, as it finishes,
rather than returning a list — that is what lets the dashboard's grid fill
cell by cell instead of appearing all at once.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import AsyncIterator, Protocol

from schemas import SimChange


class SimClient(Protocol):
    async def run_batch(
        self, n: int, changes: list[SimChange], seed: int = 0
    ) -> AsyncIterator[bool]: ...


class MockSim:
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


def get_sim() -> SimClient:
    """Pick the simulator implementation.

    ``SIM_BACKEND=mock`` (the default) needs nothing. A MuJoCo branch goes
    here when it exists; nothing else in the codebase changes when it does.
    """
    backend = os.environ.get("SIM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockSim()
    raise NotImplementedError(
        f"SIM_BACKEND={backend!r} has no implementation. Only 'mock' exists — "
        "add a MuJoCo client here satisfying the SimClient protocol."
    )
