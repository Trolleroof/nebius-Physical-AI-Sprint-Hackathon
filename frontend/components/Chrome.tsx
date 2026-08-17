"use client";

/**
 * Fixed page chrome: header, vertical side rails, running clock.
 *
 * Pure instrument-panel dressing — it carries the identity of the system so
 * the data panels can stay plain. The clock mounts empty and fills in on the
 * client, because rendering a time on the server guarantees a hydration
 * mismatch.
 */

import { useEffect, useState } from "react";
import { DashboardState } from "@/lib/reducer";
import { StatusDot } from "./ui";

function useClock() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export function Header({ state, isFixture }: { state: DashboardState; isFixture: boolean }) {
  return (
    <header className="flex shrink-0 items-center gap-4 border-b border-[var(--color-line)] px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-[13px] tracking-[0.3em] text-[var(--color-ink)] glow">
          CEL
        </span>
        <span className="label hidden sm:inline">Continual Embodied Learning</span>
      </div>

      <div className="hatch h-2 flex-1 opacity-25" />

      <div className="hidden items-center gap-2 md:flex">
        <StatusDot tone={state.outcome === "running" ? "run" : "idle"} />
        <span className="label">SO-101 · ACT · Antioch · Embodied critic</span>
      </div>

      <div className="hatch h-2 w-16 opacity-25" />

      {isFixture && (
        <span className="rounded-[1px] border border-[var(--color-warn)]/40 bg-[var(--color-warn)]/10 px-2 py-0.5 font-mono text-[9px] tracking-[0.16em] text-[var(--color-warn)] uppercase">
          demo data
        </span>
      )}

      <span className="font-mono text-[11px] tracking-[0.16em] text-[var(--color-ink)]">
        POLICY {state.policy ? state.policy.toUpperCase() : "—"}
      </span>
    </header>
  );
}

export function LeftRail({ state }: { state: DashboardState }) {
  return (
    <div className="hidden w-8 shrink-0 flex-col items-center justify-between py-6 lg:flex">
      <div className="flex flex-col items-center gap-1">
        <span className="size-1.5 rounded-full bg-[var(--color-line-strong)]" />
        <span className="size-1.5 rounded-full bg-[var(--color-dim)]" />
      </div>
      <span className="vrail">{state.environment === "real" ? "hardware live" : "simulation"}</span>
      <span className="vrail">{state.runId ?? "no run"}</span>
    </div>
  );
}

export function RightRail() {
  const now = useClock();

  return (
    <div className="hidden w-8 shrink-0 flex-col items-center justify-between py-6 lg:flex">
      <span className="vrail">act policy</span>
      <span className="vrail tabular-nums">
        {now ? now.toLocaleTimeString("en-GB") : "--:--:--"}
      </span>
      <span className="vrail tabular-nums">
        {now ? now.toISOString().slice(0, 10) : "----------"}
      </span>
    </div>
  );
}
