"use client";

/**
 * Plays a recorded run into the reducer.
 *
 * The fixture holds ~150 s of real-world timing, which is far too slow for a
 * booth where a judge gives you 60–90 s. Gaps are divided by `speed` and then
 * capped, so long waits (a 13 s simulation batch) compress hard while short
 * beats keep their rhythm. Nothing about the event data changes — only the
 * pacing of delivery.
 *
 * When the backend exists, swap `loadFixture` for an EventSource and feed the
 * same `reduce` call. Nothing else in the UI changes.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { FixtureFile, RobotEvent } from "./contract";
import { DashboardState, initialState, reduce } from "./reducer";

const DEFAULT_SPEED = 3;
/** No single beat waits longer than this, however long the real gap was. */
const MAX_GAP_MS = 900;

export interface ReplayControls {
  state: DashboardState;
  events: RobotEvent[];
  index: number;
  playing: boolean;
  finished: boolean;
  isFixture: boolean;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  restart: () => void;
  jumpToEnd: () => void;
  speed: number;
  setSpeed: (s: number) => void;
}

export function useReplay(src = "/fixtures/demo_run.json"): ReplayControls {
  const [events, setEvents] = useState<RobotEvent[]>([]);
  const [isFixture, setIsFixture] = useState(false);
  const [state, setState] = useState<DashboardState>(initialState);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(DEFAULT_SPEED);

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(src)
      .then((r) => r.json())
      .then((file: FixtureFile) => {
        if (cancelled) return;
        setEvents(file.events);
        setIsFixture(file.meta?.source === "fixture");
        setPlaying(true); // attract mode: start as soon as it loads
      })
      .catch(() => {
        /* leave the dashboard in its empty state; panels show placeholders */
      });
    return () => {
      cancelled = true;
    };
  }, [src]);

  // Step the timeline. Each tick applies one event and schedules the next.
  useEffect(() => {
    if (!playing || events.length === 0 || index >= events.length) return;

    const event = events[index];
    const previous = index > 0 ? events[index - 1] : null;
    const rawGapMs = previous ? (event.ts - previous.ts) * 1000 : 0;
    const delay = Math.min(rawGapMs / speed, MAX_GAP_MS);

    timer.current = setTimeout(() => {
      setState((s) => reduce(s, event));
      setIndex((i) => i + 1);
    }, delay);

    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [playing, events, index, speed]);

  const finished = events.length > 0 && index >= events.length;

  // Hold on the final frame rather than looping: the payoff screen
  // (FAIL -> PASS) has to stay up for whoever walks past next.
  useEffect(() => {
    if (finished) setPlaying(false);
  }, [finished]);

  const play = useCallback(() => setPlaying(true), []);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => setPlaying((p) => !p), []);

  const restart = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    setState(initialState());
    setIndex(0);
    setPlaying(true);
  }, []);

  const jumpToEnd = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    setState(events.reduce(reduce, initialState()));
    setIndex(events.length);
    setPlaying(false);
  }, [events]);

  return {
    state,
    events,
    index,
    playing,
    finished,
    isFixture,
    play,
    pause,
    toggle,
    restart,
    jumpToEnd,
    speed,
    setSpeed,
  };
}
