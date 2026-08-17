"use client";

/**
 * The payoff, and the only panel that must never clear.
 *
 * A judge who watches nothing else still leaves having seen 42% → 81% and
 * FAIL → PASS. Section 31's checklist calls for the result to be visible
 * permanently on screen; this is that requirement in component form.
 *
 * Unmeasured cells render as "—" rather than a zero. Section 12: never invent
 * a number.
 */

import { motion } from "motion/react";
import { DashboardState } from "@/lib/reducer";
import { SourceBadge, Ticker } from "./ui";
import { Origin } from "@/lib/provenance";

function cellTone(value: string | null) {
  if (!value) return "text-[var(--color-dim)]";
  if (/fail/i.test(value)) return "text-[var(--color-fail)]";
  if (/pass|success/i.test(value)) return "text-[var(--color-ok)]";
  return "text-[var(--color-ink)]";
}

export function ImprovementStrip({ state, origin }: { state: DashboardState; origin?: Origin }) {
  const metrics = state.metrics;
  const rows = metrics?.rows ?? [
    { label: "Baseline sim", v0: null, v1: null },
    { label: "Hard held-out set", v0: null, v1: null },
    { label: "Real condition", v0: null, v1: null },
  ];

  return (
    <section className="panel bracket flex shrink-0 items-stretch gap-6 px-4 py-2.5">
      <div className="flex shrink-0 flex-col justify-center">
        <div className="flex items-center gap-2">
          <span className="label">policy improvement</span>
          {origin && <SourceBadge origin={origin} />}
        </div>
        <span className="font-mono text-[10px] text-[var(--color-dim)]">
          {metrics
            ? `${metrics.initial_demos} + ${metrics.corrective_demos} demos · ${metrics.held_out_scenarios}-scenario held-out set`
            : "same held-out set, both policies"}
        </span>
      </div>

      <div className="hatch w-px opacity-30" />

      <div className="grid flex-1 grid-cols-3 gap-x-6">
        {rows.map((row, i) => (
          <motion.div
            key={row.label}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1, duration: 0.4 }}
            className="flex flex-col justify-center gap-0.5"
          >
            <span className="label truncate">{row.label}</span>
            <div className="flex items-baseline gap-2 font-mono">
              <Ticker
                value={row.v0 ?? "—"}
                className={`text-[15px] ${cellTone(row.v0)}`}
              />
              <span className="text-[11px] text-[var(--color-dim)]">→</span>
              <Ticker
                value={row.v1 ?? "—"}
                className={`text-[19px] font-medium ${cellTone(row.v1)} ${
                  row.v1 ? "glow" : ""
                }`}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <div className="hatch w-px opacity-30" />

      <div className="flex shrink-0 flex-col justify-center text-right">
        <span className="label">v0</span>
        <span className="label text-[var(--color-accent)]">→ v1</span>
      </div>
    </section>
  );
}
