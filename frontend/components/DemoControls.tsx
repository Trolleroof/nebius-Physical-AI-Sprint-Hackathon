"use client";

/**
 * Operator controls, deliberately quiet.
 *
 * Section 21 wants these reachable but not visible: a judge should not be
 * reading a "load recorded failure" button while you describe the run. They
 * sit at low contrast until hovered, and every action has a key, so the whole
 * demo can be driven without a pointer ever appearing on screen.
 *
 *   space  play / pause        r  restart / re-run
 *   e      jump to end         f  drop to the recorded run
 *   1/2/3  replay speed        l  return to live
 *
 * `f` is the one that matters. If the live run wedges in front of a judge,
 * it switches to the recorded story mid-sentence with no reload.
 */

import { useEffect } from "react";
import { RunControls } from "@/lib/useRun";

const SPEEDS = [1, 3, 6];

export function DemoControls({ replay }: { replay: RunControls }) {
  const {
    toggle,
    restart,
    jumpToEnd,
    setSpeed,
    forceSource,
    playing,
    speed,
    finished,
    index,
    events,
    source,
  } = replay;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      switch (e.key.toLowerCase()) {
        case " ":
          e.preventDefault();
          toggle();
          break;
        case "r":
          restart();
          break;
        case "e":
          jumpToEnd();
          break;
        case "f":
          forceSource("fixture");
          break;
        case "l":
          forceSource("live");
          break;
        case "1":
        case "2":
        case "3":
          setSpeed(SPEEDS[Number(e.key) - 1]);
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, restart, jumpToEnd, setSpeed, forceSource]);

  const live = source === "live";
  const progress = events.length ? index / events.length : 0;

  return (
    <div className="pointer-events-auto absolute right-3 bottom-3 z-20 flex items-center gap-2 opacity-25 transition-opacity duration-300 hover:opacity-100">
      {!live && (
        <div className="h-px w-16 overflow-hidden bg-[var(--color-line)]">
          <div
            className="h-full bg-[var(--color-accent)] transition-[width] duration-200"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      )}

      <button onClick={toggle} className="label hover:text-[var(--color-accent)]">
        {live ? "run" : playing ? "pause" : finished ? "done" : "play"}
      </button>
      <button onClick={restart} className="label hover:text-[var(--color-accent)]">
        restart
      </button>

      {!live && (
        <>
          <button onClick={jumpToEnd} className="label hover:text-[var(--color-accent)]">
            end
          </button>
          <div className="flex items-center gap-1">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`label ${
                  speed === s ? "text-[var(--color-accent)]" : "hover:text-[var(--color-ink)]"
                }`}
              >
                {s}×
              </button>
            ))}
          </div>
        </>
      )}

      <span className="h-3 w-px bg-[var(--color-line-strong)]" />

      <button
        onClick={() => forceSource(live ? "fixture" : "live")}
        className="label hover:text-[var(--color-warn)]"
        title="Switch between the live backend and the recorded run"
      >
        {live ? "→ recorded" : "→ live"}
      </button>
    </div>
  );
}
