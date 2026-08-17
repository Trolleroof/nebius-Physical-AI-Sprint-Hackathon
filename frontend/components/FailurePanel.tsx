"use client";

/**
 * The critic's read of the rollout.
 *
 * The stage checklist is the most legible possible rendering of "the model
 * understood what happened" — a judge gets it without reading JSON. The raw
 * payload sits behind a toggle for the technical judges who want to see the
 * schema is real and constrained.
 */

import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { CONFIDENCE_FLOOR, stageTimeline } from "@/lib/contract";
import { DashboardState } from "@/lib/reducer";
import { Awaiting, Meter, Panel } from "./ui";
import { Origin } from "@/lib/provenance";

const MARK = { passed: "✓", failed: "✗", not_reached: "–" } as const;
const TONE = {
  passed: "text-[var(--color-ok)]",
  failed: "text-[var(--color-fail)]",
  not_reached: "text-[var(--color-dim)]",
} as const;

export function FailurePanel({ state, origin }: { state: DashboardState; origin?: Origin }) {
  const [showRaw, setShowRaw] = useState(false);
  const d = state.diagnosis;

  return (
    <Panel
      title={d?.success ? "Rollout verdict" : "Real-world failure"}
      origin={origin}
      live={state.criticRunning}
      right={
        d && (
          <button
            onClick={() => setShowRaw((v) => !v)}
            className="label transition-colors hover:text-[var(--color-accent)]"
          >
            {showRaw ? "panel" : "json"}
          </button>
        )
      }
    >
      {!d ? (
        <Awaiting label={state.criticRunning ? "critic running…" : "awaiting rollout"} />
      ) : showRaw ? (
        <pre className="h-full overflow-auto font-mono text-[9px] leading-relaxed text-[var(--color-muted)]">
          {JSON.stringify(d, null, 1)}
        </pre>
      ) : (
        <div className="flex h-full flex-col justify-between gap-3">
          <ul className="space-y-1">
            {stageTimeline(d).map(([stage, status], i) => (
              <motion.li
                key={stage}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07, duration: 0.3 }}
                className="flex items-center justify-between border-b border-[var(--color-line)] pb-1"
              >
                <span
                  className={`font-mono text-[10px] tracking-[0.2em] uppercase ${
                    status === "not_reached" ? "text-[var(--color-dim)]" : "text-[var(--color-ink)]"
                  }`}
                >
                  {stage}
                </span>
                <span className={`text-[11px] ${TONE[status]}`}>{MARK[status]}</span>
              </motion.li>
            ))}
          </ul>

          <AnimatePresence mode="wait">
            <motion.div
              key={d.failure}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-2"
            >
              <p className="text-[13px] leading-snug text-[var(--color-ink)]">
                {d.summary || d.failure.replace(/_/g, " ")}
              </p>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="label">
                    {d.confidence >= CONFIDENCE_FLOOR ? "critic confidence" : "critic unsure"}
                  </span>
                  <span className="font-mono text-[11px] text-[var(--color-accent)]">
                    {d.confidence >= CONFIDENCE_FLOOR ? `${Math.round(d.confidence * 100)}%` : "low"}
                  </span>
                </div>
                <Meter value={d.confidence} tone={d.confidence >= CONFIDENCE_FLOOR ? "accent" : "dim"} />
              </div>

              {state.diagnosisLatencyMs != null && (
                <div className="flex justify-between">
                  <span className="label">diagnosis latency</span>
                  <span className="font-mono text-[10px] text-[var(--color-muted)]">
                    {(state.diagnosisLatencyMs / 1000).toFixed(1)}s
                  </span>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      )}
    </Panel>
  );
}
