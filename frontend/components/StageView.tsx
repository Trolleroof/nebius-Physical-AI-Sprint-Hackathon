"use client";

/**
 * The main viewport: rollout footage when it exists, the SO-101 model when it
 * does not.
 *
 * The fixture references videos that will not exist until the hardware runs,
 * so a failed load silently falls back to the 3D arm rather than showing a
 * broken player. That fallback is also the demo-day safety net if a file is
 * missing from the booth machine.
 */

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { Robot } from "./Robot";
import { DashboardState } from "@/lib/reducer";
import { SourceBadge, StatusDot } from "./ui";
import { Origin } from "@/lib/provenance";

function OverlayRow({ k, v, tone }: { k: string; v: string; tone?: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="label w-24 shrink-0">{k}</span>
      <span className={`font-mono text-[11px] tracking-wide ${tone ?? "text-[var(--color-ink)]"}`}>
        {v}
      </span>
    </div>
  );
}

export function StageView({ state, origin }: { state: DashboardState; origin?: Origin }) {
  const [videoFailed, setVideoFailed] = useState(false);

  // A new clip deserves a fresh attempt at loading.
  useEffect(() => setVideoFailed(false), [state.video?.url]);

  const showVideo = state.video && !videoFailed;

  const outcomeTone =
    state.outcome === "failure"
      ? "text-[var(--color-fail)]"
      : state.outcome === "success"
        ? "text-[var(--color-ok)]"
        : "text-[var(--color-accent)]";

  const outcomeLabel =
    state.outcome === "failure"
      ? "✗ FAILURE"
      : state.outcome === "success"
        ? "✓ SUCCESS"
        : state.outcome === "running"
          ? "RUNNING"
          : "STANDBY";

  return (
    <div className="panel bracket relative flex min-h-0 flex-1 overflow-hidden">
      {/* Base layer: always mounted so the fallback is instant. */}
      <Robot className="absolute inset-0" />

      <AnimatePresence>
        {showVideo && (
          <motion.video
            key={state.video!.url}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="absolute inset-0 size-full object-cover"
            src={state.video!.url}
            poster={state.video!.poster ?? undefined}
            autoPlay
            muted
            loop
            playsInline
            onError={() => setVideoFailed(true)}
          />
        )}
      </AnimatePresence>

      {/* Vignette keeps overlay text legible over any footage. */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-transparent to-black/45" />

      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <StatusDot
              tone={
                state.outcome === "failure"
                  ? "fail"
                  : state.outcome === "success"
                    ? "ok"
                    : state.outcome === "running"
                      ? "run"
                      : "idle"
              }
            />
            <span className="label text-[var(--color-ink)]">
              {state.environment === "real" ? "Real SO-101" : "Simulation"}
            </span>
            {origin && <SourceBadge origin={origin} />}
          </div>
          {state.criticRunning && (
            <span className="label blink text-[var(--color-accent)]">critic analysing…</span>
          )}
        </div>

        <div className="space-y-1">
          <OverlayRow k="environment" v={state.environment === "real" ? "REAL" : "SIMULATION"} />
          <OverlayRow k="policy" v={state.policy ? `ACT ${state.policy}` : "—"} />
          <OverlayRow k="task" v="cube → tray" />
          {state.condition && <OverlayRow k="condition" v={state.condition} />}
          <OverlayRow k="status" v={outcomeLabel} tone={outcomeTone} />
        </div>
      </div>
    </div>
  );
}
