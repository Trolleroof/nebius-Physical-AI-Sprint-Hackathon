"use client";

/**
 * Operator controls, deliberately quiet.
 *
 * Section 21 wants these reachable but not visible: a judge should not be
 * reading a "load recorded failure" button while you describe the run. They
 * sit at low contrast until hovered, and every action also has a key, so the
 * whole demo can be driven without the pointer ever appearing on screen.
 *
 *   space  play / pause      r  restart      e  jump to end      1/2/3  speed
 */

import { useEffect } from "react";
import { ReplayControls } from "@/lib/useReplay";

const SPEEDS = [1, 3, 6];

export function DemoControls({ replay }: { replay: ReplayControls }) {
  const { toggle, restart, jumpToEnd, setSpeed, playing, speed, finished, index, events } = replay;

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
        case "1":
        case "2":
        case "3":
          setSpeed(SPEEDS[Number(e.key) - 1]);
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, restart, jumpToEnd, setSpeed]);

  const progress = events.length ? index / events.length : 0;

  return (
    <div className="group pointer-events-auto absolute right-3 bottom-3 z-20 flex items-center gap-2 opacity-25 transition-opacity duration-300 hover:opacity-100">
      <div className="h-px w-16 overflow-hidden bg-[var(--color-line)]">
        <div
          className="h-full bg-[var(--color-accent)] transition-[width] duration-200"
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      <button onClick={toggle} className="label hover:text-[var(--color-accent)]">
        {playing ? "pause" : finished ? "done" : "play"}
      </button>
      <button onClick={restart} className="label hover:text-[var(--color-accent)]">
        restart
      </button>
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
    </div>
  );
}
