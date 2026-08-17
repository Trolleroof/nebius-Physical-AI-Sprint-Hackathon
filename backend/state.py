"""Run state and the event bus the dashboard subscribes to.

Everything the UI knows arrives as an event, in order, with a sequence
number. The bus keeps full history so a browser that connects late — or
reconnects after the laptop sleeps at the booth — replays from the start
rather than showing a half-built screen.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from pydantic import BaseModel

from events import BaseEvent


class RunStatus(BaseModel):
    """The small snapshot GET /api/status returns."""

    stage: str = "idle"
    policy: str | None = None
    environment: str = "sim"
    run_id: str | None = None
    busy: bool = False
    event_count: int = 0


class EventBus:
    """Fan-out of events to any number of connected dashboards."""

    def __init__(self) -> None:
        self._history: list[dict] = []
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._seq = 0
        self.status = RunStatus()

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    async def publish(self, event: BaseEvent) -> dict:
        """Stamp an event with sequence and time, store it, fan it out."""
        event.seq = self._seq
        event.ts = time.time()
        self._seq += 1

        payload = event.model_dump(mode="json")
        self._history.append(payload)
        self.status.event_count = len(self._history)

        for queue in list(self._subscribers):
            # A dashboard that has stopped reading must not stall the run.
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)

        return payload

    async def subscribe(self) -> AsyncIterator[dict]:
        """Yield the whole run so far, then everything that follows."""
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        try:
            for payload in self._history:
                yield payload
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    def reset(self) -> None:
        """Wipe the run. Backs the dashboard's RESET DEMO control."""
        self._history.clear()
        self._seq = 0
        self.status = RunStatus()


bus = EventBus()
