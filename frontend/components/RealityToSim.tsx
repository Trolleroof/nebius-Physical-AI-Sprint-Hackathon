"use client";

/**
 * Where one physical failure becomes a simulation curriculum.
 *
 * Two things are deliberate here. Baselines are shown as `0.60 →` before each
 * new range, because a bare range reads as static config while an arrow proves
 * the diagnosis changed something. And clamped parameters are called out: the
 * critic may recommend anything, the mapper decides what is legal, and a judge
 * asking "how do you know the critic is right?" gets to watch the guardrail
 * fire instead of hearing a reassurance.
 */

import { motion } from "motion/react";
import { PARAMETER_META } from "@/lib/contract";
import { DashboardState } from "@/lib/reducer";
import { Awaiting, Meter, Panel } from "./ui";

export function RealityToSim({ state }: { state: DashboardState }) {
  const causes = state.diagnosis?.estimated_causes ?? [];
  const curriculum = state.curriculum;

  return (
    <Panel
      title="Reality → simulation"
      right={
        curriculum && (
          <span className="label text-[var(--color-accent)]">{curriculum.nScenarios} scenarios</span>
        )
      }
    >
      {causes.length === 0 && !curriculum ? (
        <Awaiting label="awaiting diagnosis" />
      ) : (
        <div className="flex h-full flex-col gap-3 overflow-auto">
          {causes.length > 0 && (
            <div className="space-y-1.5">
              <span className="label">detected weakness</span>
              {causes.map((c, i) => (
                <motion.div
                  key={c.cause}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.08 }}
                  className="space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-[var(--color-ink)]">
                      {c.cause.replace(/_/g, " ")}
                    </span>
                    <span className="font-mono text-[10px] text-[var(--color-muted)]">
                      {Math.round(c.confidence * 100)}%
                    </span>
                  </div>
                  <Meter value={c.confidence} tone={i === 0 ? "accent" : "dim"} />
                </motion.div>
              ))}
            </div>
          )}

          {curriculum && (
            <div className="space-y-2 border-t border-[var(--color-line)] pt-2">
              <span className="label">new curriculum</span>
              {curriculum.changes.map((change, i) => {
                const meta = PARAMETER_META[change.parameter];
                const before = curriculum.baseline[change.parameter];
                return (
                  <motion.div
                    key={change.parameter}
                    initial={{ opacity: 0, x: 8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 + i * 0.1, duration: 0.4 }}
                    className="flex items-baseline gap-2 font-mono text-[10px]"
                  >
                    <span className="w-20 shrink-0 text-[var(--color-muted)]">{meta.label}</span>
                    {before != null && (
                      <>
                        <span className="text-[var(--color-dim)] line-through">{before}</span>
                        <span className="text-[var(--color-dim)]">→</span>
                      </>
                    )}
                    <span className="text-[var(--color-accent)]">
                      {change.min}–{change.max}
                      <span className="ml-0.5 text-[var(--color-dim)]">{meta.unit}</span>
                    </span>
                    {change.clamped && (
                      <span
                        className="ml-auto text-[var(--color-warn)]"
                        title="The critic requested a wider range than the simulator accepts; the mapper clamped it."
                      >
                        ⚠ clamped
                      </span>
                    )}
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
