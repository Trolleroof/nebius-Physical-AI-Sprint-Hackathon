"use client";

/**
 * Picks the dashboard's data source and hides the difference.
 *
 * On mount it pings the backend. If FastAPI answers, the dashboard runs live
 * off SSE; if it does not, the recorded fixture plays instead. Same reducer,
 * same components, same controls — only the source of events differs.
 *
 * That is section 21's fallback ladder as a single boolean: the demo cannot
 * enter a state where there is nothing to show. Unplug the backend mid-run
 * and a reload still tells the whole story.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { RobotEvent } from "./contract";
import { DashboardState, initialState, reduce } from "./reducer";
import { ReplayControls, useReplay } from "./useReplay";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const PROBE_TIMEOUT_MS = 1200;

export type Source = "probing" | "live" | "fixture";

export interface RunControls extends ReplayControls {
  source: Source;
}

function useLive(enabled: boolean) {
  const [state, setState] = useState<DashboardState>(initialState);
  const [events, setEvents] = useState<RobotEvent[]>([]);
  const stream = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const source = new EventSource(`${API}/api/events`);
    stream.current = source;

    source.onmessage = (message) => {
      const event: RobotEvent = JSON.parse(message.data);
      setEvents((previous) => [...previous, event]);
      setState((previous) => reduce(previous, event));
    };
    // A dropped stream is not fatal: whatever has already arrived stays on
    // screen, and EventSource retries on its own.
    source.onerror = () => {};

    return () => source.close();
  }, [enabled]);

  const post = useCallback((path: string) => {
    fetch(`${API}${path}`, { method: "POST" }).catch(() => {});
  }, []);

  const restart = useCallback(() => {
    setState(initialState());
    setEvents([]);
    post("/api/demo/reset");
    post("/api/demo/run");
  }, [post]);

  return { state, events, restart, run: () => post("/api/demo/run") };
}

export function useRun(): RunControls {
  const [source, setSource] = useState<Source>("probing");

  useEffect(() => {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), PROBE_TIMEOUT_MS);

    fetch(`${API}/healthz`, { signal: abort.signal })
      .then((r) => setSource(r.ok ? "live" : "fixture"))
      .catch(() => setSource("fixture"))
      .finally(() => clearTimeout(timer));

    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, []);

  const fixture = useReplay();
  const live = useLive(source === "live");

  if (source !== "live") {
    return { ...fixture, source, isFixture: source === "fixture" && fixture.isFixture };
  }

  // Live mode has no timeline to scrub, so the transport controls become
  // run/reset against the backend and the speed control is inert.
  return {
    ...fixture,
    source,
    state: live.state,
    events: live.events,
    index: live.events.length,
    isFixture: false,
    playing: true,
    finished: false,
    play: live.run,
    pause: () => {},
    toggle: live.run,
    restart: live.restart,
    jumpToEnd: () => {},
  };
}
